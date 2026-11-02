# SPDX-License-Identifier: AGPL-3.0-only
# serve-nemotron-amd.ps1 - single serve entry for Nemotron 3.5 Super VL on
# AMD Strix Halo (gfx1151) under Windows. Windows twin of
# scripts/serve-nemotron-gb10.sh; same golden-check contract, same release
# profile (32K / prefix / 0.85 util), HIP build.
#
#   .\serve-nemotron-amd.ps1                    # serve spec=g8 on :8391
#   .\serve-nemotron-amd.ps1 -Spec off          # no speculation
#   .\serve-nemotron-amd.ps1 -Build             # materialize + build first
#   .\serve-nemotron-amd.ps1 -Check             # boot, golden compare, stop
#   .\serve-nemotron-amd.ps1 -Detach            # ssh-surviving detached serve
#
# MODEL / DRAFTER env vars override the weight paths (winbox defaults).

param(
    [ValidateSet('off','g3','g8')]
    [string]$Spec = 'g8',
    [int]$Port = 8391,
    [switch]$Check,
    [switch]$Build,
    [switch]$Detach,
    [switch]$AllowDirty
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent          # scripts/.. = repo root
$Name = 'nemotron-3.5-super-vl'
$Model   = if ($env:MODEL)   { $env:MODEL }   else { "$env:USERPROFILE\models\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4" }
$Drafter = if ($env:DRAFTER) { $env:DRAFTER } else { "$env:USERPROFILE\models\NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr" }
$Exe  = "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe"
$Shim = "$Repo\target\x86_64-pc-windows-msvc\release\nvcuda.dll"

function Fail($msg, $code) { Write-Error $msg; exit $code }

# --- refusals -----------------------------------------------------------------
$gfx = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
       Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -First 1
if (-not $gfx -or $gfx.Name -notmatch '8060S|Strix|AI MAX') {
    Fail "GPU is '$($gfx.Name)' - expected a Strix Halo part (Radeon 8060S / gfx1151)." 10
}
$commitLimitGB = [math]::Round(((Get-Counter '\Memory\Commit Limit').CounterSamples[0].CookedValue)/1GB,1)
if ($commitLimitGB -lt 120) {
    Fail "commit limit ${commitLimitGB} GiB < 120 GiB - the model OOMs below it. Fix: fixed pagefile ~95 GiB (registry PagingFiles 'C:\pagefile.sys 97280 97280', AutomaticManagedPagefile=0), reboot." 11
}
if (-not $Build) {
    & python "$PSScriptRoot\strix-windows\materialize_symlinks.py" $Repo --check | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "strix-hip symlink copies stale - run scripts\strix-windows\materialize_symlinks.py $Repo (or pass -Build)." 12 }
}
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$Tag  = (& git -C $Repo describe --tags --exact-match HEAD 2>$null)
$ErrorActionPreference = $eap
if (-not $Tag) { $Tag = 'untagged' }
if (-not $AllowDirty) {
    # Materialized symlink copies legitimately show as modified; allow exactly
    # those paths (mode 120000 in the index) and nothing else.
    $links = & git -C $Repo ls-files -s |
             Where-Object { $_ -match '^120000 ' -and $_ -match 'kernels/strix-hip/' } |
             ForEach-Object { ($_ -split "\t")[1] }
    $linkSet = @{}; foreach ($l in $links) { $linkSet[$l] = $true }
    $dirty = & git -C $Repo status --porcelain |
             Where-Object {
                 $_ -notmatch '^\?\?' -and
                 -not $linkSet.ContainsKey(($_.Substring(3) -replace '.* -> ',''))
             }
    if ($dirty) { Fail ("dirty tracked files outside materialized-symlink paths:`n" + ($dirty -join "`n") + "`n(pass -AllowDirty to override)") 13 }
}
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$BuildHead = (& git -C $Repo log -1 --format=%H -- . ':(exclude)scripts/golden' ':(exclude)*.md' ':(exclude)docs')
$ErrorActionPreference = $eap
if (-not $BuildHead) { $BuildHead = $Head }
$marker = "$Repo\.built-$BuildHead"
if ((-not (Test-Path $Exe) -or -not (Test-Path $marker)) -and -not $Build) {
    Fail "spark.exe or .built-$BuildHead missing - pass -Build to build." 14
}

# --- build: the exact env/flags of the verified p013 build ----------------------
if ($Build) {
    & python "$PSScriptRoot\strix-windows\materialize_symlinks.py" $Repo
    if ($LASTEXITCODE -ne 0) { Fail 'materialize failed' 15 }
    Get-ChildItem "$Repo\.built-*" -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Host "=== build $Repo ($Head, strix-hip all targets) ==="
    # first_run.ps1 -Phase build env:
    $env:HIP_PATH = 'C:\TheRock\10.0.0'
    $env:ATLAS_TARGET_HW = 'strix-hip'; $env:ATLAS_TARGET_MODEL = '*'; $env:ATLAS_TARGET_QUANT = 'nvfp4'
    $env:CUDARC_CUDA_VERSION = '12080'; $env:ATLAS_NO_RDMA = '1'
    $env:ATLAS_HIP_COMPAT_INCLUDE = "$Repo\crates\atlas-kernels\hip\compat"
    $env:ATLAS_HIPCC = 'C:\TheRock\10.0.0\bin\hipcc.exe'
    $env:PATH = "C:\TheRock\10.0.0\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
    # MSVC toolchain on PATH for the host link (vcvars):
    $vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    & cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        $kv = $_ -split '=',2; if ($kv.Count -eq 2) { [Environment]::SetEnvironmentVariable($kv[0], $kv[1]) }
    }
    Push-Location $Repo
    try {
        $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        # --target triple (cargo-native path is target\x86_64-pc-windows-msvc\release\);
        # --no-default-features --features cuda avoids pulling in nccl.
        cargo build --release -p spark-server --target x86_64-pc-windows-msvc --no-default-features --features cuda
        $code = $LASTEXITCODE
        $ErrorActionPreference = $eap
        if ($code -ne 0) { Fail "cargo build failed ($code)" 16 }
    } finally { Pop-Location }
    # Stage the ROCm runtime dlls beside the exe (first_run's staging; amdhip64 loads
    # its comgr by plain name so a stale System32 copy must be shadowed locally).
    $hipBin = Join-Path $env:HIP_PATH 'bin'
    Get-ChildItem $hipBin -Filter *.dll -EA SilentlyContinue |
        Where-Object { $_.Name -match '^(amdhip64(?:_\d+)?|amd_comgr(?:_\d+)?|hiprtc.*)\.dll$' } |
        ForEach-Object { Copy-Item $_.FullName (Split-Path $Exe) -Force }
    # Stage the HIP shim dlls (nvcuda.dll / cuda.dll) from the kernel build out dir.
    $shimOut = Get-ChildItem (Join-Path (Split-Path $Exe) 'build\atlas-kernels-*\out\nvcuda.dll') |
               Sort-Object LastWriteTime -Desc | Select-Object -First 1
    if ($shimOut) {
        Copy-Item $shimOut.FullName (Split-Path $Exe) -Force
        Copy-Item (Join-Path $shimOut.DirectoryName 'cuda.dll') (Split-Path $Exe) -Force -ErrorAction SilentlyContinue
    } else { Fail 'nvcuda.dll shim not found in build output' 18 }
    New-Item $marker -ItemType File -Force | Out-Null
}

# --- banner -------------------------------------------------------------------
$BinSha  = (Get-FileHash $Exe  -Algorithm SHA256).Hash
$ShimSha = (Get-FileHash $Shim -Algorithm SHA256).Hash
$banner  = "serve-nemotron-amd tag=$Tag sha=$Head binsha=$BinSha shimsha=$ShimSha spec=$Spec port=$Port model=$Model drafter=$Drafter commit_limit_GiB=$commitLimitGB"

# --- serve args: the verified 32K fit profile -----------------------------------
$a = @('serve', $Model, '--model-name', $Name, '--bind', '127.0.0.1', '--port', "$Port",
       '--no-tui', '--no-fast-load', '--max-seq-len', '32768', '--max-prefill-tokens', '4096',
       '--kv-cache-dtype', 'bf16', '--gpu-memory-utilization', '0.85', '--max-num-seqs', '1',
       '--max-batch-size', '1', '--ssm-cache-slots', '8', '--enable-prefix-caching', 'true',
       '--request-timeout', '0', '--disable-thinking')
if ($Spec -ne 'off') {
    $a += @('--dflash', '--draft-model', $Drafter, '--dflash-gamma', ($Spec -replace 'g',''))
}

# --- env: exactly the verified p013nat block ------------------------------------
$env:HIP_PATH = 'C:\TheRock\10.0.0'                # HIP SDK root
$env:HF_HUB_OFFLINE = '1'                          # no hub fetches at serve time
$env:RUST_LOG = 'info'                             # verified log level
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = '0'            # verified fit profile
$env:ATLAS_OOM_WATCHDOG_MB = '400'                 # verified fit profile
$env:ATLAS_MEM_PROFILE = '1'                       # memprofile/memdiag lines
$env:ATLAS_MTP_ACCEPT_DEBUG = '1'                  # engagement logging
$env:ATLAS_W4A16_DP4A = '0'                        # verified fit profile
$env:ATLAS_NO_MAMBA_VERIFY_FUSED = '1'             # verified fit profile
$env:ATLAS_SSM_DECODE_RING = '0'                   # ring=0 of the fit profile

# --- -Detach: re-invoke as a WmiPrvSE child so the serve survives ssh teardown ----
$stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = "$env:USERPROFILE\n35vl-runs"; New-Item -ItemType Directory -Force $logDir | Out-Null
$readyBody0 = "{`"model`":`"$Name`",`"messages`":[{`"role`":`"user`",`"content`":`"Reply OK.`"}],`"max_tokens`":2,`"temperature`":0}"
if ($Detach -and $Check) { Fail '-Detach and -Check are incompatible (Check stops the server when done).' 19 }
if ($Detach) {
    $log = "$logDir\serve-amd-$Spec-$stamp.log"
    $args_ = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -Spec $Spec -Port $Port"
    if ($AllowDirty) { $args_ += ' -AllowDirty' }
    $res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "powershell $args_"; CurrentDirectory = $Repo }
    if ($res.ReturnValue -ne 0) { Fail "detached launch failed rc=$($res.ReturnValue)" 17 }
    $launched = Get-Date
    $banner | Tee-Object $log
    Write-Host "detached wrapper pid=$($res.ProcessId); polling for READY (up to 62 min)"
    # The child writes serve-amd-<spec>-<stamp>.pid once spark.exe starts.
    $deadline = (Get-Date).AddMinutes(62)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        $child = Get-Process -Id $res.ProcessId -ErrorAction SilentlyContinue
        if (-not $child) { Fail "detached child exited before READY - see $log" 17 }
        $pidf = Get-ChildItem "$logDir\serve-amd-$Spec-*.pid" -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -gt $launched } |
                Sort-Object LastWriteTime | Select-Object -Last 1
        if ($pidf) {
            $spid = [int](Get-Content $pidf -TotalCount 1)
            try {
                $m = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 5
                if ($m.StatusCode -eq 200) {
                    $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $readyBody0 -TimeoutSec 300
                    if ($r.Content -match '"choices"') {
                        $lg = (Get-Content $pidf -TotalCount 2)[1]
                        Write-Host "READY spark_pid=$spid log=$lg"
                        exit 0
                    }
                }
            } catch { }
        }
    }
    Fail 'detached serve timed out (62 min) waiting for READY' 17
}

# --- boot, with 719 retry --------------------------------------------------------
$bannerFile = "$logDir\serve-amd-$Spec-$stamp.banner.txt"
$banner | Tee-Object $bannerFile
Write-Host "banner: $bannerFile"
$readyBody = $readyBody0
$ready = $false
$attempt = 0
$died719 = 0
$proc = $null
$serveOut = $null; $serveErr = $null
while (-not $ready -and $attempt -lt 3) {
    $attempt++
    $serveOut = "$logDir\serve-amd-$Spec-$stamp-a$attempt.out.log"
    $serveErr = "$logDir\serve-amd-$Spec-$stamp-a$attempt.err.log"
    $proc = Start-Process -FilePath $Exe -ArgumentList $a -WorkingDirectory $Repo -NoNewWindow -PassThru `
            -RedirectStandardOutput $serveOut -RedirectStandardError $serveErr
    Set-Content "$logDir\serve-amd-$Spec-$stamp.pid" "$($proc.Id)`n$serveOut"
    Write-Host "spark pid=$($proc.Id) out=$serveOut err=$serveErr"
    for ($i = 0; $i -lt 240 -and -not $ready -and -not $proc.HasExited; $i++) {
        Start-Sleep -Seconds 5
        try {
            $m = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 5
            if ($m.StatusCode -eq 200) {
                $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $readyBody -TimeoutSec 300
                if ($r.Content -match '"choices"') { $ready = $true }
            }
        } catch { }
    }
    if ($ready) { break }
    if ($proc.HasExited) {
        $tails = ((Get-Content $serveOut -Tail 40 -ErrorAction SilentlyContinue) -join "`n") + "`n" +
                 ((Get-Content $serveErr -Tail 40 -ErrorAction SilentlyContinue) -join "`n")
        if ($tails -match 'status 719|hipErrorLaunchFailure|HSA_STATUS_ERROR_EXCEPTION') {
            $died719++
            Write-Host "boot attempt ${attempt}: HIP 719 wedge; killing pid $($proc.Id), retry in 30s"
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep 30
            continue
        }
        Write-Host "serve died before READY (exit $($proc.ExitCode)); log tail:"
        Get-Content $serveErr -Tail 20 -ErrorAction SilentlyContinue
        exit 4
    }
}
if (-not $ready) {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    if ($died719 -ge 3) { Write-Host 'reboot required - 3x HIP wedges'; exit 5 }
    Write-Host 'serve timeout; log tail:'
    Get-Content $serveErr -Tail 20 -ErrorAction SilentlyContinue
    exit 5
}
Write-Host "READY pid=$($proc.Id)"

# --- --check: golden compare ------------------------------------------------------
if ($Check) {
    $py = @'
import json, sys, glob, urllib.request, os
repo, mode, port, model = sys.argv[1:5]
pd = f"{repo}/scripts/golden/nemotron-gb10/prompts"
gd = f"{repo}/scripts/golden/nemotron-amd/atlas"
fail = 0
for pf in sorted(glob.glob(pd + "/*.json")):
    name = os.path.basename(pf)[:-5]
    body = json.load(open(pf, encoding="utf-8"))
    body["model"] = model
    body["temperature"] = 0
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
            timeout=600)
        resp = json.load(r)
    except Exception as e:
        print(f"{name}: FAIL request {e}"); fail = 1; continue
    msg = resp["choices"][0]["message"]
    got = {"content": msg.get("content"), "tool_calls": msg.get("tool_calls")}
    gf = f"{gd}/{name}.json"
    if not os.path.exists(gf):
        print(f"{name}: FAIL no golden {gf}"); fail = 1; continue
    want = json.load(open(gf, encoding="utf-8"))
    if got == want:
        print(f"{name}: PASS"); continue
    fail = 1
    aa = json.dumps(got, sort_keys=True); bb = json.dumps(want, sort_keys=True)
    i = next((j for j, (u, v) in enumerate(zip(aa, bb)) if u != v), min(len(aa), len(bb)))
    print(f"{name}: FAIL first-diff-char {i}")
    print(f"  got : {aa[max(0,i-80):i+80]}")
    print(f"  want: {bb[max(0,i-80):i+80]}")
sys.exit(fail)
'@
    $pyf = "$env:TEMP\amd_golden_check.py"; Set-Content $pyf $py -Encoding utf8
    try {
        & python $pyf $Repo 'atlas' "$Port" $Name
        $rc = $LASTEXITCODE
    } finally {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    exit $rc
}

# --- hold the console (default) ---------------------------------------------------
try {
    Wait-Process -Id $proc.Id
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
