# SPDX-License-Identifier: AGPL-3.0-only
# Windows BFCL-70 gate on the mirrored Qwen3.8-27B-NVFP4 (unsloth) port.
# Same 70-row draw as the Linux gate (non_live 4 / live 1 / hallucination 1,
# floor 2, temp 0, max_new_tokens 1024) for a like-for-like comparison.
param([string]$Port = "8081")
$ErrorActionPreference = "Stop"
$env:HOME = "C:\Users\azeez"
$env:USERPROFILE = "C:\Users\azeez"
$env:ATLAS_HOME = "C:\Users\azeez\.atlas"
$Repo = "C:\Users\azeez\code\atlas-inf-pr9"
$Rocm10Bin = "$Repo\target-rocm10\x86_64-pc-windows-msvc\release\spark.exe"
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } elseif (Test-Path $Rocm10Bin) { $Rocm10Bin } else { "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe" }
if ($Bin -eq $Rocm10Bin) {
    $env:HIP_PATH = "C:\TheRock\10.0.0"
    $env:PATH = "$env:HIP_PATH\bin;$env:PATH"
}
$RunTag = if ($env:ATLAS_RUN_TAG) { $env:ATLAS_RUN_TAG } else { "rocm10-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
$Log = "C:\Users\azeez\q38-win-bfcl70-$RunTag.log"
$Fingerprint = "C:\Users\azeez\q38-win-bfcl70-$RunTag-fingerprint.txt"
$ModelDir = "$env:USERPROFILE\models\Qwen3.8-27B-NVFP4"
@(
    "date=" + (Get-Date).ToUniversalTime().ToString("o")
    "binary=" + $Bin
    "binary_sha256=" + (Get-FileHash -Algorithm SHA256 $Bin).Hash
    "hip_path=" + $env:HIP_PATH
    "model=unsloth/Qwen3.8-27B-NVFP4"
    "model_dir=" + $ModelDir
    "serve=preservation util=0.70 seq=4096 prefill=2048 kv=bf16 lm_head=bf16 batch=1 drafts=0 thinking=off grammar=off slots=0"
    "harness=bfcl-subset non_live=4 live=1 hallucination=1 floor=2 max_new_tokens=1024 temperature=0 timeout=600 samples=70"
) | Out-File $Fingerprint -Encoding utf8
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | Format-List | Out-File $Fingerprint -Append -Encoding utf8

$env:ATLAS_FP8_DEQUANT_ATTN_TO_BF16 = "1"
$env:ATLAS_FP8_DEQUANT_FFN_TO_BF16 = "1"
$env:ATLAS_GDN_BF16_WEIGHTS = "1"
$env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_W4A16_DP4A = "1"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"

Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5

$proc = Start-Process -FilePath $Bin -ArgumentList @(
    "serve", $ModelDir,
    "--model-name", "unsloth/Qwen3.8-27B-NVFP4",
    "--host", "127.0.0.1", "--port", $Port,
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
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { Write-Output "SERVER DIED during load:"; Get-Content $Log -Tail 10; exit 1 }
    try { Invoke-WebRequest "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { Write-Output "SERVER NOT UP"; Get-Content $Log -Tail 10; exit 1 }
Write-Output "server up"

# PowerShell 5.1 + $ErrorActionPreference=Stop turns a native program's STDERR
# progress lines into a terminating NativeCommandError, so run the benchmark
# via Start-Process with file redirection instead of a pipeline.
$benchLog = "C:\Users\azeez\q38-win-bfcl70-$RunTag-bench.log"
$benchErr = "C:\Users\azeez\q38-win-bfcl70-$RunTag-bench.err"
$bench = Start-Process -FilePath $Bin -ArgumentList @(
    "benchmark", "run", "bfcl-subset",
    "--url", "http://127.0.0.1:$Port",
    "--model", "unsloth/Qwen3.8-27B-NVFP4",
    "--param", "non_live_pct=4", "--param", "live_pct=1", "--param", "hallucination_pct=1",
    "--param", "subset_floor=2", "--param", "max_new_tokens=1024", "--param", "temperature=0",
    "--param", "min_overall=0", "--param", "min_normalized=0", "--param", "request_timeout_s=600"
) -RedirectStandardOutput $benchLog -RedirectStandardError $benchErr -PassThru -NoNewWindow -Wait
Write-Output "benchmark exit: $($bench.ExitCode)"
Get-Content $benchLog -Tail 60
Get-Content $benchErr -Tail 20
$Faults = Select-String -Path $Log -Pattern "status 719|hipErrorLaunchFailure|context is destroyed" -ErrorAction SilentlyContinue
$Invalid = Select-String -Path $benchErr -Pattern "Error during inference|connection was forcibly closed|actively refused" -ErrorAction SilentlyContinue

Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
if ($bench.ExitCode -ne 0 -or $Faults -or $Invalid) { Write-Output "WIN BFCL70 INVALID"; exit 1 }
Write-Output "WIN BFCL70 DONE"
