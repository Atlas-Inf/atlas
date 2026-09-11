$env:ATLAS_FP8_DEQUANT_ATTN_TO_BF16 = "1"
$env:ATLAS_FP8_DEQUANT_FFN_TO_BF16 = "1"
$env:ATLAS_GDN_BF16_WEIGHTS = "1"
$env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_W4A16_DP4A = "1"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"
$ErrorActionPreference = "Continue"
$Repo = "C:\Users\azeez\code\atlas-inf-pr9"
$Bin = "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe"
$Log = "C:\Users\azeez\q38-win-bfcl70b.log"
$ModelDir = "$env:USERPROFILE\models\Qwen3.8-27B-NVFP4"
Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5
$proc = Start-Process -FilePath $Bin -ArgumentList @(
    "serve", $ModelDir,
    "--model-name", "unsloth/Qwen3.8-27B-NVFP4",
    "--host", "127.0.0.1", "--port", "8081",
    "--max-seq-len", "32768", "--max-prefill-tokens", "2048",
    "--gpu-memory-utilization", "0.99",
    "--kv-cache-dtype", "bf16", "--lm-head-dtype", "bf16",
    "--max-batch-size", "1",
    "--disable-tool-grammar", "true", "--enable-prefix-caching",
    "--ssm-cache-slots", "0", "--ssm-checkpoint-interval", "16",
    "--speculative", "--num-drafts", "2", "--mtp-quantization", "bf16", "--mtp-vocab", "100000",
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
if (-not $up) { "SERVER NOT UP"; exit 1 }
"server up (0.99/32K/K2/prefix-caching profile)"
& $Bin benchmark run bfcl-subset --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 --param non_live_pct=4 --param live_pct=1 --param hallucination_pct=1 --param subset_floor=2 --param max_new_tokens=1024 --param temperature=0 --param min_overall=0 --param min_normalized=0 --param request_timeout_s=600 2>&1 | ForEach-Object { "$_" } | Select-Object -Last 40
Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
"WIN BFCL70B DONE"
