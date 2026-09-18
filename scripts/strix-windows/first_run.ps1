# First run of Atlas on Strix Halo (gfx1151) under Windows, native HIP.
#
# This is the ONE entry point for docs/porting/STRIX_WINDOWS_HIP.md.
#
# TWO WAYS TO USE IT
#
#   1. JUST TEST IT (no toolchain, no build, ~5 min + download)
#      Grab the prebuilt `spark-windows-x86_64-amd-hip` zip from any green run of
#      the release matrix, unzip it KEEPING THE DLLs BESIDE spark.exe, then:
#
#        $env:ATLAS_BIN = "C:\path\to\unzipped\spark.exe"
#        powershell -ExecutionPolicy Bypass -File scripts\strix-windows\first_run.ps1
#
#      With ATLAS_BIN set this skips the toolchain check, the symlink repair and
#      the build entirely -- it only needs the AMD driver and the weights. This is
#      the answer to "OK, how do I test it?".
#
#   2. BUILD IT FROM SOURCE (contributors)
#      Leave ATLAS_BIN unset and it checks the toolchain, repairs the kernel
#      symlinks a Windows clone breaks, builds, then serves:
#
#        powershell -ExecutionPolicy Bypass -File scripts\strix-windows\first_run.ps1
#
# Either way it ends by serving `nvidia/Qwen3.8-27B-NVFP4` and firing a smoke
# request at it, so a successful run prints a real completion.
#
# The serve flags and env below are the fingerprinted fp8d recipe behind the
# 2026-09-13 ST-995 record (995/995, zero faults, ROCm 10) -- NOT the earlier
# preservation profile, whose bf16-GDN kernel path is a different arm set and
# is a live suspect in a WDDM bugcheck on a second box. Read the comments at
# each one before changing it.
#
# MUST run from PowerShell, NOT Git Bash: under bash, Git's /usr/bin precedes MSVC
# on PATH and rustc invokes the coreutils `link.exe` instead of the MSVC linker.
#
# Every phase is idempotent -- re-run after a failure and it resumes.
#
# Optional (all have working defaults):
#   ATLAS_BIN        prebuilt spark.exe. Set it to skip straight to serving.
#   ATLAS_REPO       repo root.        Default: the checkout this script lives in.
#   ATLAS_MODEL_DIR  weights snapshot. Default: $env:USERPROFILE\models\<leaf of
#                    ATLAS_MODEL_NAME> -- nvidia-Qwen3.8-27B-NVFP4 by default.
#   ATLAS_MODEL_NAME served model id. Default: nvidia/Qwen3.8-27B-NVFP4.
#                    This drives kernel-target resolution -- see $ModelName below.
#   HIP_PATH         ROCm SDK root.    Default: newest under C:\TheRock or C:\Program Files\AMD\ROCm
#   ATLAS_MAX_SEQ_LEN --max-seq-len. Default 4096, matching the validated gate.
#   ATLAS_MAX_PREFILL_TOKENS --max-prefill-tokens. Default 2048.
#   ATLAS_GPU_UTIL   --gpu-memory-utilization. Default 0.95, matching the
#                    fingerprinted fp8d recipe -- on a ~63 GB gfx1151 the
#                    nvidia checkpoint leaves ~5 GB for KV only at >=0.93.
#   ATLAS_PORT       serve port.       Default 8081.
#   ATLAS_BIND       serve host.       Default 127.0.0.1.
#
# Parameters:
#   -Phase   all (default) | check | symlinks | build | serve
#   -NoSmokeTest   skip the post-startup completion probe.
#
# Example:
#   powershell -ExecutionPolicy Bypass -File scripts\strix-windows\first_run.ps1 -Phase check
[CmdletBinding()]
param(
    [ValidateSet('all', 'check', 'symlinks', 'build', 'serve')]
    [string]$Phase = 'all',
    [switch]$NoSmokeTest
)
$ErrorActionPreference = 'Stop'

# Default the repo root to the checkout this script lives in, so the script works
# from any working directory and from a moved clone.
$RepoRoot = if ($env:ATLAS_REPO) { $env:ATLAS_REPO }
            else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }

# The served model name resolves the exact Qwen3.8 kernel target. This branch is
# Qwen3.8-specific, so custom snapshot-directory names must not silently select
# Qwen3.6; ATLAS_MODEL_NAME remains an explicit override.
$ModelName = if ($env:ATLAS_MODEL_NAME) { $env:ATLAS_MODEL_NAME }
             else { 'nvidia/Qwen3.8-27B-NVFP4' }
# The default weights dir derives from the checkpoint id so ATLAS_MODEL_NAME
# and ATLAS_MODEL_DIR cannot drift apart: serving an unsloth snapshot (per-row
# FP8 GDN weights) under the nvidia name keeps NVFP4 + transposed + prefill +
# rowwise copies resident at once and exhausts the KV budget. The on-disk
# convention is org-dash-repo (`hf download org/repo --local-dir models\org-repo`).
$ModelDir = if ($env:ATLAS_MODEL_DIR) { $env:ATLAS_MODEL_DIR }
            else { "$env:USERPROFILE\models\$($ModelName -replace '/', '-')" }
# DFLASH=1 swaps the default MTP spec decode for the DFlash2 drafter,
# mirroring serve-amd.sh (DFlash and --speculative are mutually exclusive).
$Dflash     = $env:DFLASH -eq '1'
$DraftModel = if ($env:DRAFT_MODEL) { $env:DRAFT_MODEL } else { 'incoai/Qwen3.8-27B-DFlash2' }
# DFlash needs headroom for the resident drafter + SSM verify pool; 0.80 is
# the measured gfx1151 util under DFlash (serve-amd.sh), 0.95 under MTP.
$GpuUtil  = if ($env:ATLAS_GPU_UTIL) { $env:ATLAS_GPU_UTIL } elseif ($Dflash) { '0.80' } else { '0.95' }
# The validated Qwen3.8 gate uses 4096 context and a 2048-token prefill arena.
$MaxSeqLen = if ($env:ATLAS_MAX_SEQ_LEN) { $env:ATLAS_MAX_SEQ_LEN } else { '4096' }
$MaxPrefill = if ($env:ATLAS_MAX_PREFILL_TOKENS) { $env:ATLAS_MAX_PREFILL_TOKENS } else { '2048' }
$Port     = if ($env:ATLAS_PORT) { $env:ATLAS_PORT } else { '8081' }
$BindHost = if ($env:ATLAS_BIND) { $env:ATLAS_BIND } else { '127.0.0.1' }

# ATLAS_BIN points at a prebuilt spark.exe (the CI zip). When it is set there is
# nothing to build, so the binary's own directory takes the place of target/ and
# the toolchain checks are skipped -- a tester needs the driver and weights only.
$Prebuilt = [bool]$env:ATLAS_BIN

# Which toolchain pieces this invocation actually needs. Only compiling requires
# MSVC / hipcc / Rust; serving an existing binary requires none of them, and
# demanding them anyway is how a "just run it" path turns into a yak shave.
# 'check' counts as needing them: with ATLAS_BIN unset it means "can this box
# build and run Atlas?", which is the question someone auditing before they start
# is actually asking. '-Phase serve' on an existing binary does not.
$NeedsBuild = (-not $Prebuilt) -and ($Phase -in @('all', 'build', 'check'))

if ($Prebuilt) {
    if (-not (Test-Path $env:ATLAS_BIN)) { throw "ATLAS_BIN does not exist: $env:ATLAS_BIN" }
    $ReleaseDir = Split-Path (Resolve-Path $env:ATLAS_BIN).Path -Parent
} else {
    $TargetRoot = if ($env:CARGO_TARGET_DIR) {
        if ([System.IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) { $env:CARGO_TARGET_DIR }
        else { Join-Path $RepoRoot $env:CARGO_TARGET_DIR }
    } else { Join-Path $RepoRoot 'target' }
    $ReleaseDir = Join-Path $TargetRoot 'x86_64-pc-windows-msvc\release'
}

$fails = @()
function Ok   ($m) { Write-Host "  [ ok ] $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Bad  ($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; $script:fails += $m }
function Head ($m) { Write-Host ""; Write-Host "=== $m ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------
function Phase-Check {
    Head 'Preflight'

    if ($env:MSYSTEM) { Bad 'Running under MSYS/Git Bash. Use a plain PowerShell window.' }

    # Prebuilt path: nothing gets compiled, so MSVC / Rust / hipcc are all
    # irrelevant. Check only what running actually needs -- the exe, the DLLs
    # that must sit beside it, and the GPU.
    if ($Prebuilt) {
        Ok "prebuilt binary: $env:ATLAS_BIN"
        # cudarc dlopens nvcuda.dll from the exe's OWN directory, and that DLL
        # imports the versioned HIP runtime. Unzipping the exe on its own, or
        # copying it somewhere tidy and leaving the DLLs behind, fails at cuInit
        # with a missing-module error that does not name the DLL.
        foreach ($d in @('nvcuda.dll', 'cuda.dll')) {
            if (Test-Path (Join-Path $ReleaseDir $d)) { Ok "  $d beside the exe" }
            else { Bad "$d is NOT beside spark.exe -- keep the whole unzipped folder together" }
        }
        # amdhip64 is version-suffixed and the suffix tracks the SDK
        # (6.x -> amdhip64_6.dll, 7.x -> amdhip64_7.dll), so match on the
        # pattern rather than a literal name -- a literal 6.x name reports a
        # bogus failure on a bundle staged from a 7.x SDK.
        $hipRt = Get-ChildItem $ReleaseDir -Filter 'amdhip64_*.dll' -EA SilentlyContinue
        if ($hipRt) { foreach ($h in $hipRt) { Ok ("  " + $h.Name + " beside the exe") } }
        else { Bad 'amdhip64_<ver>.dll is NOT beside spark.exe -- keep the whole unzipped folder together' }
        Check-Gpu
        if ($fails.Count) { Write-Host ''; Write-Host 'Preflight failed.' -ForegroundColor Red; exit 1 }
        Ok 'preflight clean'
        return
    }

    # VCToolsInstallDir is not only for cl.exe: atlas-kernels/build.rs uses it to
    # locate dumpbin.exe, which generates the HIP shim's export .def. Without
    # dumpbin the shim builds but exports nothing, and cudarc finds no symbols.
    if ($NeedsBuild) {
        $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vsw) {
            $script:VsPath = & $vsw -products * -latest -format value -property installationPath
            if ($script:VsPath -and (Test-Path (Join-Path $script:VsPath 'VC\Auxiliary\Build\vcvars64.bat'))) {
                Ok "MSVC: $script:VsPath"
            } else { Bad "vswhere found no VC toolset. Install the 'Desktop development with C++' workload." }
        } else { Bad 'MSVC Build Tools not installed (no vswhere.exe).' }
    }

    if (-not $env:HIP_PATH) {
        $rocm = @('C:\Program Files\AMD\ROCm', 'C:\TheRock') |
                Where-Object { Test-Path $_ } |
                ForEach-Object { Get-ChildItem $_ -Directory -ErrorAction SilentlyContinue } |
                Where-Object {
                    (Test-Path (Join-Path $_.FullName 'bin\hipcc.bin.exe')) -or
                    (Test-Path (Join-Path $_.FullName 'bin\hipcc.exe'))
                } |
                Sort-Object {
                    $vf = Join-Path $_.FullName '.info\version'
                    if (Test-Path $vf) { [version](Get-Content $vf -First 1) }
                    else { try { [version]$_.Name } catch { [version]'0.0' } }
                } -Descending | Select-Object -First 1
        if ($rocm) { $env:HIP_PATH = $rocm.FullName }
    }
    if ($env:HIP_PATH -and (Test-Path $env:HIP_PATH)) {
        $h = Join-Path $env:HIP_PATH 'bin\hipcc.bin.exe'
        if (-not (Test-Path $h)) { $h = Join-Path $env:HIP_PATH 'bin\hipcc.exe' }
        if (Test-Path $h) { $script:Hipcc = $h; Ok "HIP SDK: $env:HIP_PATH" }
        elseif ($NeedsBuild) { Bad "hipcc not found under $env:HIP_PATH\bin" }
    } elseif ($NeedsBuild) {
        Bad 'ROCm SDK not found under C:\TheRock or C:\Program Files\AMD\ROCm'
    }
    # Serving needs no SDK at all -- the runtime DLLs sit beside the exe.

    # Rust is only needed to COMPILE. Serving an already-built spark.exe must not
    # require a working toolchain -- gating it on one turns "run the server" into
    # "fix your rustup first" for no reason.
    if ($NeedsBuild) {
        # `cargo` is normally a rustup shim: it resolves on PATH and then refuses
        # to run when no default toolchain is set, so check it actually runs.
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
            $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
        }
        if (Get-Command cargo -ErrorAction SilentlyContinue) {
            # Probe from INSIDE the repo. rust-toolchain.toml pins the channel
            # (1.93.1) and there is deliberately no global rustup default, so
            # `cargo --version` run from anywhere else fails with "could not
            # choose a version of cargo" and rustup helpfully suggests
            # `rustup default stable` -- which is the wrong fix here, and
            # installs a toolchain the build will not use.
            #
            # Push-Location alone is NOT enough: it moves PowerShell's own
            # location but leaves [Environment]::CurrentDirectory where the shell
            # started, and that is the working directory a spawned rustup
            # inherits and searches upward from for rust-toolchain.toml. Without
            # the explicit sync this check fails for anyone who launches the
            # script from outside the repo -- i.e. exactly the copy-paste case.
            $ErrorActionPreference = 'Continue'
            $prevCwd = [Environment]::CurrentDirectory
            Push-Location $RepoRoot
            [Environment]::CurrentDirectory = (Get-Location).Path
            # Let the command run to completion and read $LASTEXITCODE before
            # touching the output. Piping straight into `Select-Object -First 1`
            # stops the pipeline as soon as one object arrives, which terminates
            # the native process early and leaves a bogus exit code behind.
            $out = cargo --version 2>$null
            $rc = $LASTEXITCODE
            $v = ($out | Select-Object -First 1)
            Pop-Location
            [Environment]::CurrentDirectory = $prevCwd
            $ErrorActionPreference = 'Stop'
            if ($rc -eq 0 -and $v) { Ok "rust: $v" }
            else { Bad "cargo will not run in $RepoRoot -- is rust-toolchain.toml's channel installed? Try: rustup toolchain install" }
        } else { Bad 'Rust not installed (https://rustup.rs).' }
    }

    Check-Gpu

    if ($fails.Count) { Write-Host ''; Write-Host 'Preflight failed.' -ForegroundColor Red; exit 1 }
    Ok 'preflight clean'
}

# The kernels here are gfx1151. Another AMD part builds and loads fine right up
# until it looks for a matching code object -- and on WDDM a wrong-arch or
# faulting compute launch can escalate to a kernel-mode page fault that
# BUGCHECKS the whole machine (0x119 VIDEO_SCHEDULER_INTERNAL_ERROR, observed
# 2026-09-14 on a second Strix-class box). So a detected wrong GPU is a hard
# FAIL, not a warning -- a first run must never be the thing that bluescreens
# someone. ATLAS_SKIP_GPU_CHECK=1 overrides for porting work on other arches.
# hipInfo only exists with the HIP SDK installed, which a prebuilt-zip tester has
# no reason to have -- fall back to the adapter name from the driver.
function Check-Gpu {
    if ($env:ATLAS_SKIP_GPU_CHECK -eq '1') { Warn 'GPU check skipped (ATLAS_SKIP_GPU_CHECK=1)'; return }
    $hipInfo = if ($env:HIP_PATH) { Join-Path $env:HIP_PATH 'bin\hipInfo.exe' } else { $null }
    if ($hipInfo -and (Test-Path $hipInfo)) {
        $gfx = & $hipInfo 2>$null | Select-String 'gcnArchName:\s*(\S+)' |
               ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1
        if ($gfx -eq 'gfx1151') { Ok 'GPU: gfx1151 (Strix Halo)'; return }
        elseif ($gfx) { Bad "GPU reports '$gfx', not gfx1151 -- this tree ships gfx1151 kernels only. Serving wrong-arch code on WDDM can bugcheck the OS."; return }
    }
    $amd = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
           Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -First 1
    if ($amd) {
        if ($amd.Name -match '8060S|Strix|AI MAX') { Ok "GPU: $($amd.Name)" }
        else { Bad "GPU is '$($amd.Name)' -- expected a Strix Halo part (Radeon 8060S / gfx1151). Serving wrong-arch code on WDDM can bugcheck the OS." }
    } else { Warn 'No AMD adapter found. Is the Adrenalin driver installed?' }
}

# ---------------------------------------------------------------------------
# kernels/ leans on git symlinks that CHAIN (strix-hip/foo.cu -> ../../strix/foo.cu
# -> ../../gb10/foo.cu). Git for Windows defaults to core.symlinks=false and checks
# each one out as a ~32 byte TEXT FILE containing the link target path, so hipcc is
# handed a few hundred path strings where kernel sources should be. build.rs stops
# the build and names core.symlinks when it sees one, but detecting is not fixing.
#
# Cloning with `-c core.symlinks=true` needs Developer Mode or an elevated shell and
# fails silently when it does not take, so repair from the git object store instead:
# that is correct no matter what state the working tree is in, and is re-runnable.
function Phase-Symlinks {
    Head 'Kernel symlinks'
    Push-Location $RepoRoot
    try {
        $links = @{}
        foreach ($line in (git ls-files -s kernels/)) {
            if ($line -match '^120000 ([0-9a-f]{40}) \d+\s+(.+)$') {
                $links[$matches[2]] = (git cat-file blob $matches[1])
            }
        }
        if ($links.Count -eq 0) { Ok 'no symlink entries under kernels/'; return }

        $written = 0; $ok = 0; $failed = @()
        foreach ($link in $links.Keys) {
            # Follow the chain to a path that is not itself a link entry. A
            # single-hop resolve just leaves a stub pointing at another stub.
            $cur = $link; $hops = 0
            while ($links.ContainsKey($cur)) {
                $dir = Split-Path $cur -Parent
                $joined = if ($dir) { "$dir/$($links[$cur])" } else { $links[$cur] }
                $parts = New-Object System.Collections.Generic.List[string]
                foreach ($p in $joined -split '[\\/]+') {
                    if ($p -eq '.' -or $p -eq '') { continue }
                    elseif ($p -eq '..') { if ($parts.Count) { $parts.RemoveAt($parts.Count - 1) } }
                    else { $parts.Add($p) }
                }
                $cur = ($parts -join '/')
                if (++$hops -gt 10) { break }
            }
            if ($links.ContainsKey($cur)) { $failed += "$link (cycle)"; continue }

            # Content comes from the index, not the working tree: on a broken clone
            # the destination may itself still be a stub.
            $sha = (git ls-files -s -- $cur) -replace '^\d+ ([0-9a-f]{40}).*$', '$1'
            if ($sha -notmatch '^[0-9a-f]{40}$') { $failed += "$link -> $cur (no blob)"; continue }

            # LF, UTF-8, no BOM, matching what git stores. Set-Content/Out-File
            # would write CRLF and a BOM and leave all of them permanently dirty.
            $bytes = (New-Object System.Text.UTF8Encoding($false)).GetBytes(
                        (@(git cat-file blob $sha) -join "`n") + "`n")
            $dest = Join-Path $RepoRoot ($link -replace '/', '\')
            if ((Test-Path $dest) -and
                ([System.IO.File]::ReadAllBytes($dest).Length -eq $bytes.Length) -and
                (-not (Compare-Object ([System.IO.File]::ReadAllBytes($dest)) $bytes -SyncWindow 0))) {
                $ok++; continue
            }
            [System.IO.File]::WriteAllBytes($dest, $bytes)
            $written++
        }
        if ($failed.Count) { $failed | ForEach-Object { Bad "symlink: $_" }; throw 'symlink repair failed' }
        Ok "$written materialised, $ok already correct (of $($links.Count))"
        Warn "these now show as modified in 'git status' -- real content where git expects a link blob. Do NOT commit them."
    } finally { Pop-Location }
}

# ---------------------------------------------------------------------------
function Phase-Build {
    Head 'Build'

    # A running server holds an exclusive lock on spark.exe, so the build does all
    # the work and then dies at the final link with "failed to remove file ...
    # Access is denied. (os error 5)", which names neither the server nor the lock.
    $running = Get-Process -Name spark -ErrorAction SilentlyContinue
    if ($running) { throw "spark.exe is running (pid $($running.Id -join ',')) and locks the build output. Stop it first." }

    $vcvars = Join-Path $script:VsPath 'VC\Auxiliary\Build\vcvars64.bat'
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    $env:ATLAS_HIPCC = $script:Hipcc

    # MODEL and QUANT are MANDATORY: the default target is a qwen3-next-80b kernel
    # dir that does not exist under strix-hip, and build.rs panics resolving it.
    $env:ATLAS_TARGET_HW     = 'strix-hip'
    # Overridable so the same script serves other targets in this tree
    # (e.g. ATLAS_TARGET_MODEL=qwen3.8-27b, or '*' to build every one).
    if (-not $env:ATLAS_TARGET_MODEL) { $env:ATLAS_TARGET_MODEL = 'qwen3.8-27b' }
    $env:ATLAS_TARGET_QUANT  = 'nvfp4'
    $env:CUDARC_CUDA_VERSION = '12080'

    Push-Location $RepoRoot
    try {
        # cargo writes ordinary progress ("Compiling foo v1.2.3") to STDERR. Under
        # PowerShell 5.1 with ErrorActionPreference=Stop every stderr line from a
        # native exe becomes a terminating NativeCommandError, which aborts the
        # build on its first crate. Gate on the exit code instead.
        $ErrorActionPreference = 'Continue'
        # --no-default-features --features cuda avoids pulling in nccl, which does
        # not link on a single-GPU box.
        cargo build --release -p spark-server --target x86_64-pc-windows-msvc `
            --no-default-features --features cuda
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($code -ne 0) { throw "cargo build failed ($code)" }
    } finally { Pop-Location }

    Ok "built $ReleaseDir\spark.exe"

    # 97 is the number to check -- main alone builds 91. A lower count means
    # kernels silently failed to resolve.
    #
    # amdhip64_6.dll loads its code-object manager by PLAIN NAME, so if a driver
    # install left an older amd_comgr_2.dll in System32, HIP binds the stale one
    # and every hipModuleLoadData fails as CUDA_ERROR_OUT_OF_MEMORY -- which reads
    # as a KV-sizing bug and sends you tuning --gpu-memory-utilization for hours.
    # The build stages the ROCm copies beside the exe; confirm the size.
    # HIP 7.x RENAMED these: amdhip64_6 -> amdhip64_7, amd_comgr_2 -> amd_comgr_3,
    # and added hiprtc*.dll. Matching on literal 6.x names silently stages nothing
    # on 7.x, and any stale 6.x copy left here SHADOWS the newer runtime. So
    # discover whatever this SDK actually ships and copy it, then drop the others.
    $hipBin = Join-Path $env:HIP_PATH 'bin'
    $wanted = Get-ChildItem $hipBin -Filter *.dll -EA SilentlyContinue |
              Where-Object { $_.Name -match '^(amdhip64(?:_\d+)?|amd_comgr(?:_\d+)?|hiprtc.*)\.dll$' }
    foreach ($f in $wanted) {
        Copy-Item $f.FullName $ReleaseDir -Force
        Ok ("staged {0} ({1:N1} MB)" -f $f.Name, ($f.Length / 1MB))
    }
    if (-not $wanted) { Warn "no HIP runtime DLLs found under $hipBin -- serve may bind a stale system copy" }

    # The HIP shim itself (cuda.dll, and its nvcuda.dll copy) is built into
    # atlas-kernels' OUT_DIR and recorded in ATLAS_HIP_RUNTIME_DIR *for the
    # release PACKAGING step to bundle*. Nothing puts it beside spark.exe, so a
    # source build produces an exe that cannot start: it links cuda.lib and
    # cudarc dlopens nvcuda.dll, and neither is on the loader's search path.
    # The failure is silent and very hard to read -- the process fail-fasts with
    # 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) before writing a single byte to
    # stdout or stderr, and logs no WER entry, so it looks like a corrupt build
    # rather than a missing DLL. Stage them exactly like the SDK runtime DLLs.
    $shimOut = Get-ChildItem (Join-Path $TargetRoot 'x86_64-pc-windows-msvc\release\build') `
                   -Directory -Filter 'atlas-kernels-*' -EA SilentlyContinue |
               ForEach-Object { Join-Path $_.FullName 'out' } |
               Where-Object { Test-Path (Join-Path $_ 'cuda.dll') } |
               Sort-Object { (Get-Item (Join-Path $_ 'cuda.dll')).LastWriteTime } -Descending |
               Select-Object -First 1
    if ($shimOut) {
        foreach ($n in @('cuda.dll', 'nvcuda.dll')) {
            $src = Join-Path $shimOut $n
            if (Test-Path $src) {
                Copy-Item $src $ReleaseDir -Force
                Ok ("staged {0} (HIP shim, {1:N2} MB)" -f $n, ((Get-Item $src).Length / 1MB))
            } else { Bad "shim $n missing from $shimOut" }
        }
    } else {
        Bad 'no atlas-kernels OUT_DIR contains cuda.dll -- spark.exe will fail-fast at startup'
    }
    # Remove runtime DLLs from a DIFFERENT SDK version left over by an earlier build.
    Get-ChildItem $ReleaseDir -Filter *.dll -EA SilentlyContinue |
        Where-Object { $_.Name -match '^(amdhip64(?:_\d+)?|amd_comgr(?:_\d+)?|hiprtc.*)\.dll$' -and
                       $wanted.Name -notcontains $_.Name } |
        ForEach-Object { Remove-Item $_.FullName -Force; Warn ("removed stale {0} from a previous SDK" -f $_.Name) }
}

# ---------------------------------------------------------------------------
function Phase-Serve {
    Head 'Serve'
    $exe = if ($Prebuilt) { (Resolve-Path $env:ATLAS_BIN).Path } else { Join-Path $ReleaseDir 'spark.exe' }
    if (-not (Test-Path $exe)) { throw "spark.exe not found -- run -Phase build first, or set ATLAS_BIN to a prebuilt one" }
    if (-not (Test-Path $ModelDir)) {
        throw "no weights at $ModelDir. Fetch with: hf download $ModelName --local-dir `"$ModelDir`""
    }

    # cudarc dlopens nvcuda.dll from the EXE's directory, and that DLL imports
    # amdhip64_6.dll -- so run from there.
    Set-Location $ReleaseDir
    $env:PATH = "$ReleaseDir;$env:HIP_PATH\bin;$env:PATH"

    # Serve profile: the fp8d recipe — the fingerprinted configuration behind
    # the 2026-09-13 ST-995 record (995/995 samples, zero faults, 3.6 h on
    # ROCm 10; see kernels/strix-hip/qwen3.8-27b/BENCH.toml +
    # scripts/strix-windows/win_serve_qwen38_nvfp4.ps1). It replaces the older
    # preservation profile (ATLAS_GDN_BF16_WEIGHTS / ATLAS_FP8_DEQUANT_*),
    # which is a DIFFERENT kernel path — GDN runs FP8-weight decode here —
    # and which bugchecked a second Strix-class box on first inference
    # (0x119, 2026-09-14). If a box cannot serve this recipe, that is a bug
    # to report, not a knob to turn back: the fallback profile is exactly
    # the path under suspicion.
    $env:ATLAS_W4A16_DP4A = '1'
    $env:ATLAS_W4A16_VARIANT = 'v1'
    $env:ATLAS_FORCE_GLOBAL_GDN = '1'
    $env:ATLAS_GDN_FP8_WEIGHTS = '1'
    $env:ATLAS_GDN_FP8_DECODE = '1'
    $env:ATLAS_SSM_TAIL_PROTECT = '1'
    $env:ATLAS_SSM_TAIL_LEASE_TTL = '128'
    $env:ATLAS_SSM_TAIL_MIDCHUNK = '0'
    $env:ATLAS_MTP_GATE_REPROBE = '64'
    $env:ATLAS_MTP_ACCEPT_DEBUG = '1'
    $env:ATLAS_TRACKED_MEMINFO = '1'

    # 0, NOT the 6 this doc carried before runtime. cuMemGetInfo_v2 now synthesises
    # a truthful free figure from tracked allocations, and that tracker reports
    # Atlas-own bytes only -- so build.rs's co-tenant discount double-counts and
    # oversizes the KV pool (it allocated 11.3 GB of KV, then died on a later 24 MB
    # alloc). 0 fails build.rs's `.filter(|gb| gb > 0.0)` and falls through to the
    # AUTO path (baseline_free - free_now), which is correct given the fixed shim.
    $env:ATLAS_KV_EXTERNAL_RESERVE_GB = '0'

    if (-not $NoSmokeTest) {
        # The server owns this console until Ctrl-C, so probe from a detached
        # process. Log lands beside the exe (inside target/ for a source build,
        # which is gitignored).
        #
        # The probe writes its OWN file rather than being launched with
        # `Start-Process -RedirectStandardOutput`. That switch reassigns the
        # parent's standard handles, and when this script's stdout is itself
        # redirected -- which is exactly what happens when you tee a first run to
        # a log -- everything the SERVER prints afterwards is silently swallowed.
        # The server runs fine and looks dead, which is the worst failure to hand
        # someone testing a port for the first time.
        $smokeLog = Join-Path $ReleaseDir 'first_run_smoke.log'
        $probe = Join-Path $env:TEMP 'atlas_first_run_smoke.ps1'
        @"
`$out = '$smokeLog'
`$u = 'http://${BindHost}:$Port'
`$dl = (Get-Date).AddMinutes(15)
while ((Get-Date) -lt `$dl) {
    try { Invoke-WebRequest -Uri "`$u/v1/models" -TimeoutSec 3 -UseBasicParsing | Out-Null; break }
    catch { Start-Sleep -Seconds 5 }
}
`$b = @{ model='$ModelName'
        prompt="<|im_start|>user``nName three primary colors.<|im_end|>``n<|im_start|>assistant``n"
        max_tokens=64; temperature=0.001 } | ConvertTo-Json -Compress
try {
    `$r = Invoke-RestMethod -Uri "`$u/v1/completions" -Method Post -ContentType 'application/json' -Body `$b -TimeoutSec 300
    @("SMOKE OK  finish=`$(`$r.choices[0].finish_reason)  tokens=`$(`$r.usage.completion_tokens)",
      "TEXT: `$(`$r.choices[0].text.Trim())") | Out-File -FilePath `$out -Encoding ascii
} catch { "SMOKE FAILED: `$_" | Out-File -FilePath `$out -Encoding ascii }
"@ | Out-File -FilePath $probe -Encoding ascii
        if (Test-Path $smokeLog) { Remove-Item $smokeLog -Force -ErrorAction SilentlyContinue }
        Start-Process powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$probe) `
            -WindowStyle Hidden
        Write-Host "  smoke result -> $smokeLog"
    }

    Write-Host "  serving on http://${BindHost}:$Port  (Ctrl-C to stop)"
    Write-Host ''
    $ErrorActionPreference = 'Continue'

    # Windows ROCm reports a 76.9 GB process-local pool. With the 2048-token
    # arena, the fully-NVFP4 checkpoint uses 70.4 GB pre-KV; 0.99 leaves about
    # 3.8 GB of BF16 KV after the inference reserve, enough for the 32K limit.
    #
    # --no-fast-load is no longer required (the Unix-only O_DIRECT loader now warns
    # and falls back instead of hard-erroring); passing it just silences the warning.
    # The kernel audit's 87 optional-arm probes are all declared in this target's
    # MODEL.toml [expected_absent] with per-entry reasons, so the boot gate passes
    # WITHOUT --dangerously-allow-unresolved-kernel-lookups -- and the flag is
    # deliberately absent here: a NEW unresolved lookup is exactly the signal the
    # gate exists to catch, and a first-run script must not ship the escape hatch.
    # Built as one array and splatted in a single position. Splatting mid-way
    # through a backtick-continued native call emits a stray bare '-' argument
    # and spark rejects the whole command line.
    #
    # Remaining deltas vs the proven recipe are sizing-only and deliberate:
    # max-seq 4096 keeps the first-run memory footprint conservative
    # (env-overridable); util 0.95 matches the fingerprinted recipe — the
    # ~63 GB gfx1151 budget only clears the ~14.6 GB inference reserve plus a
    # usable KV slice at >=0.93 on the nvidia checkpoint. --disable-thinking
    # keeps the smoke answer inside 64 tokens. Drafts=3 and ssm-cache-slots=64
    # match the proven profile — those select the exercised kernel arms, not
    # just capacity.
    $serveArgs = @('serve', $ModelDir)
    $serveArgs += @(
        '--no-fast-load'
        '--model-name', $ModelName, '--host', $BindHost, '--port', $Port
        '--max-seq-len', $MaxSeqLen, '--max-prefill-tokens', $MaxPrefill
        '--gpu-memory-utilization', $GpuUtil, '--kv-cache-dtype', 'bf16', '--lm-head-dtype', 'bf16'
        '--max-batch-size', '1', '--vision-max-pixels', '262144'
        '--disable-tool-grammar', 'true'
        '--request-timeout', '0'
        '--disable-thinking'
    )
    if ($Dflash) {
        # Option-B is the measured gfx1151 path (serve-amd.sh default);
        # ATLAS_DFLASH_OPTION_B=0 restores the legacy path. slots=20 /
        # interval=128 is the verified DFlash SSM recipe — the drafter
        # reserves pool slots for gamma/verify. DFLASH_GAMMA overrides the
        # MODEL.toml gamma when set.
        if (-not $env:ATLAS_DFLASH_OPTION_B) { $env:ATLAS_DFLASH_OPTION_B = '1' }
        $serveArgs += @(
            '--dflash', '--draft-model', $DraftModel
            '--ssm-cache-slots', '20', '--ssm-checkpoint-interval', '128'
        )
        if ($env:DFLASH_GAMMA) { $serveArgs += @('--dflash-gamma', $env:DFLASH_GAMMA) }
    } else {
        $serveArgs += @(
            '--speculative', '--num-drafts', '3'
            '--mtp-quantization', 'bf16', '--mtp-vocab', '100000'
            '--ssm-cache-slots', '64', '--ssm-checkpoint-interval', '16'
        )
    }
    & $exe @serveArgs
}

switch ($Phase) {
    'check'    { Phase-Check }
    'symlinks' { Phase-Symlinks }
    'build'    { if ($Prebuilt) { throw 'ATLAS_BIN is set, so there is nothing to build. Unset it to build from source.' }
                 Phase-Check; Phase-Build }
    'serve'    { Phase-Check; Phase-Serve }
    # With a prebuilt binary there is no source to repair and nothing to compile,
    # so "all" is just check + serve.
    'all'      { if ($Prebuilt) { Phase-Check; Phase-Serve }
                 else { Phase-Check; Phase-Symlinks; Phase-Build; Phase-Serve } }
}
