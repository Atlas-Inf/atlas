# SPDX-License-Identifier: AGPL-3.0-only
# Serves nvidia/Qwen3.8-27B-NVFP4 on Windows gfx1151 with the measured fp8d
# recipe — the exact serve profile behind the 2026-09-13 ST-995 record
# (overall 83.32 / normalized 78.70, run-1789300938267487600) and the 17.25
# tok/s decode headline. This is NOT the unsloth preservation recipe: native
# FP8 GDN decode, speculative decode on (K=4), prefix caching, 64K ctx.
param(
    [string]$Tag = ("nv4-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$ModelDir = "$env:USERPROFILE\models\nvidia-Qwen3.8-27B-NVFP4",
    [string]$Port = "8081"
)
$ErrorActionPreference = "Stop"
$Repo = "C:\Users\azeez\code\atlas-inf-pr9"
$Rocm10Bin = "$Repo\target-rocm10\x86_64-pc-windows-msvc\release\spark.exe"
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } elseif (Test-Path $Rocm10Bin) { $Rocm10Bin } else { "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe" }
if ($Bin -eq $Rocm10Bin) {
    $env:HIP_PATH = "C:\TheRock\10.0.0"
    $env:PATH = "$env:HIP_PATH\bin;$env:PATH"
}
$Log = "C:\Users\azeez\q38-win-serve-nv4-$Tag.log"
$env:HOME = "C:\Users\azeez"
$env:ATLAS_HOME = "C:\Users\azeez\.atlas"

# Measured recipe env (ha20-serve-nv4 derivation, 2026-09-12/13).
$env:ATLAS_W4A16_DP4A = "1"
$env:ATLAS_FORCE_GLOBAL_GDN = "1"
$env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_SSM_TAIL_PROTECT = "1"
$env:ATLAS_SSM_TAIL_LEASE_TTL = "128"
$env:ATLAS_MTP_GATE_REPROBE = "64"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"
$env:ATLAS_SSM_TAIL_MIDCHUNK = "0"
$env:ATLAS_MTP_ACCEPT_DEBUG = "1"
$env:ATLAS_GDN_FP8_WEIGHTS = "1"
$env:ATLAS_GDN_FP8_DECODE = "1"
$env:ATLAS_TRACKED_MEMINFO = "1"

$Fingerprint = "C:\Users\azeez\q38-win-serve-nv4-$Tag-fingerprint.txt"
@(
    "date=" + (Get-Date).ToUniversalTime().ToString("o")
    "binary=" + $Bin
    "binary_sha256=" + (Get-FileHash -Algorithm SHA256 $Bin).Hash
    "hip_path=" + $env:HIP_PATH
    "model=nvidia/Qwen3.8-27B-NVFP4"
    "model_dir=" + $ModelDir
    "config_sha256=" + (Get-FileHash -Algorithm SHA256 (Join-Path $ModelDir "config.json")).Hash
    "index_sha256=" + (Get-FileHash -Algorithm SHA256 (Join-Path $ModelDir "model.safetensors.index.json")).Hash
    "serve=fp8d util=0.95 seq=65536 prefill=8192 kv=bf16 lm_head=bf16 batch=1 drafts=3 mtp=bf16 mtp_vocab=100k prefix_cache=on slots=64 grammar=off thinking=on"
) | Out-File $Fingerprint -Encoding utf8

Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5

# NOTE: the fingerprinted ST-995 run (2026-09-13) carried
# `--dangerously-allow-unresolved-kernel-lookups` because the 87 optional-arm
# probes were undeclared then. MODEL.toml now declares all 87 in
# [expected_absent], so the boot audit passes WITHOUT the flag — do not re-add
# it; a new unresolved lookup is exactly the signal the gate exists to catch.
$proc = Start-Process -FilePath $Bin -ArgumentList @(
    "serve", $ModelDir,
    "--no-fast-load",
    "--model-name", "nvidia/Qwen3.8-27B-NVFP4",
    "--host", "0.0.0.0", "--port", $Port,
    "--max-seq-len", "65536", "--max-prefill-tokens", "8192",
    "--gpu-memory-utilization", "0.95",
    "--kv-cache-dtype", "bf16", "--lm-head-dtype", "bf16",
    "--max-batch-size", "1",
    "--vision-max-pixels", "262144",
    "--disable-tool-grammar", "true",
    "--enable-prefix-caching",
    "--ssm-cache-slots", "64", "--ssm-checkpoint-interval", "16",
    "--request-timeout", "0",
    "--speculative", "--num-drafts", "3",
    "--mtp-quantization", "bf16", "--mtp-vocab", "100000"
) -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru -NoNewWindow

$up = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { "SERVER DIED"; Get-Content $Log -Tail 8; exit 1 }
    try { Invoke-WebRequest "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { "SERVER NOT UP"; Get-Content $Log -Tail 8; exit 1 }
"server up - nvidia/Qwen3.8-27B-NVFP4 fp8d recipe on :$Port"
"LOG=$Log"
"FINGERPRINT=$Fingerprint"
