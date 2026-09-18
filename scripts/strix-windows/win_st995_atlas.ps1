$env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_W4A16_DP4A = "1"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"
$env:ATLAS_FP8_DEQUANT_ATTN_TO_BF16 = "1"
$env:ATLAS_FP8_DEQUANT_FFN_TO_BF16 = "1"
$env:ATLAS_GDN_BF16_WEIGHTS = "1"
# Frozen preservation path used by the Linux gate.
$ErrorActionPreference = "Continue"
$Repo = "C:\Users\azeez\code\atlas-inf-pr9"
$Rocm10Bin = "$Repo\target-rocm10\x86_64-pc-windows-msvc\release\spark.exe"
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } elseif (Test-Path $Rocm10Bin) { $Rocm10Bin } else { "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe" }
if ($Bin -eq $Rocm10Bin) {
    $env:HIP_PATH = "C:\TheRock\10.0.0"
    $env:PATH = "$env:HIP_PATH\bin;$env:PATH"
}
$RunTag = if ($env:ATLAS_RUN_TAG) { $env:ATLAS_RUN_TAG } else { "rocm10-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
$Log = "C:\Users\azeez\q38-win-st995-atlas-$RunTag.log"
$ModelDir = "$env:USERPROFILE\models\Qwen3.8-27B-NVFP4"
$env:HOME = "C:\Users\azeez"
$env:ATLAS_HOME = "C:\Users\azeez\.atlas"
$Fingerprint = "C:\Users\azeez\q38-win-st995-atlas-$RunTag-fingerprint.txt"
@(
    "date=" + (Get-Date).ToUniversalTime().ToString("o")
    "binary=" + $Bin
    "binary_sha256=" + (Get-FileHash -Algorithm SHA256 $Bin).Hash
    "hip_path=" + $env:HIP_PATH
    "model=unsloth/Qwen3.8-27B-NVFP4"
    "model_dir=" + $ModelDir
    "serve=preservation util=0.70 seq=4096 prefill=2048 kv=bf16 lm_head=bf16 batch=1 drafts=0 thinking=off grammar=off slots=0"
    "harness=bfcl-subset pinned defaults samples=995 seed=42 temperature=0 no_param_overrides"
) | Out-File $Fingerprint -Encoding utf8
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | Format-List | Out-File $Fingerprint -Append -Encoding utf8
Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5
$proc = Start-Process -FilePath $Bin -ArgumentList @(
    "serve", $ModelDir,
    "--model-name", "unsloth/Qwen3.8-27B-NVFP4",
    "--host", "127.0.0.1", "--port", "8081",
    "--max-seq-len", "4096", "--max-prefill-tokens", "2048",
    "--gpu-memory-utilization", "0.70",
    "--kv-cache-dtype", "bf16", "--lm-head-dtype", "bf16",
    "--max-batch-size", "1",
    "--disable-tool-grammar", "true",
    "--ssm-cache-slots", "0",
    "--num-drafts", "0",
    "--disable-thinking",
    "--dangerously-allow-unresolved-kernel-lookups",
    "--no-fast-load"
) -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru -NoNewWindow
$up = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { "SERVER DIED"; Get-Content $Log -Tail 8; exit 1 }
    try { Invoke-WebRequest "http://127.0.0.1:8081/v1/models" -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { "SERVER NOT UP"; Get-Content $Log -Tail 8; exit 1 }
"server up - Atlas-instrument ST-995 (same runner as the Linux 84.22 baseline)"
$BenchLog = "C:\Users\azeez\q38-win-st995-atlas-$RunTag-bench.log"
& $Bin benchmark run bfcl-subset --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 2>&1 | ForEach-Object { "$_" } | Out-File $BenchLog -Encoding utf8
$BenchExit = $LASTEXITCODE
"ST995 EXIT: $BenchExit"
Get-Content $BenchLog -Tail 45
$Faults = Select-String -Path $Log -Pattern "status 719|hipErrorLaunchFailure|context is destroyed" -ErrorAction SilentlyContinue
$Invalid = Select-String -Path $BenchLog -Pattern "Error during inference|connection was forcibly closed|actively refused" -ErrorAction SilentlyContinue
Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
if ($BenchExit -ne 0 -or $Faults -or $Invalid) { "WIN ST995 INVALID"; exit 1 }
"WIN ST995 ATLAS DONE"
