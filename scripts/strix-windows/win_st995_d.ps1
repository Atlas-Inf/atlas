$env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_W4A16_DP4A = "1"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"
# Destructive requant path + minimized SSM snapshot activity: the crash is
# always in the SSM pool lifecycle at prefill start (cuMemsetD8Async in
# zero_slot), so reduce snapshot frequency and increase pool slots.
$ErrorActionPreference = "Continue"
$Repo = "C:\Users\azeez\code\atlas-inf-pr9"
$Bin = "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe"
$Log = "C:\Users\azeez\q38-win-st995-d.log"
$ModelDir = "$env:USERPROFILE\models\Qwen3.8-27B-NVFP4"
$env:HOME = "C:\Users\azeez"
$env:ATLAS_HOME = "C:\Users\azeez\.atlas"
Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5
$proc = Start-Process -FilePath $Bin -ArgumentList @(
    "serve", $ModelDir,
    "--model-name", "unsloth/Qwen3.8-27B-NVFP4",
    "--host", "127.0.0.1", "--port", "8081",
    "--max-seq-len", "4096", "--max-prefill-tokens", "2048",
    "--gpu-memory-utilization", "0.78",
    "--kv-cache-dtype", "bf16", "--lm-head-dtype", "bf16",
    "--max-batch-size", "1",
    "--disable-tool-grammar", "true",
    "--ssm-cache-slots", "4",
    "--ssm-checkpoint-interval", "4096",
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
"server up - ST-995 attempt 4 (destructive, util 0.74, ssm-interval 4096, slots 4)"
& $Bin benchmark run bfcl-subset --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 2>&1 | ForEach-Object { "$_" } | Out-File C:\Users\azeez\q38-win-st995-d-bench.log -Encoding utf8
"ST995 EXIT: $LASTEXITCODE"
Get-Content C:\Users\azeez\q38-win-st995-d-bench.log -Tail 45
Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
"WIN ST995D DONE"
