# SPDX-License-Identifier: AGPL-3.0-only
# Serves nvidia/Qwen3.8-27B-NVFP4 + incoai/Qwen3.8-27B-DFlash2 on Windows
# gfx1151 (ROCm 10) — the DFlash gamma=8 Option-B profile from win-integ.ps1:
# util 0.80, seq 8192, prefill 2048, kv/lm_head bf16, batch 1, thinking off,
# grammar off, ssm-cache-slots 20 / ssm-checkpoint-interval 128.
#
# This replaces the box-local DFlash edit of first_run.ps1 — everything the
# DFlash serve needs is here. Build the tree first:
#   powershell -ExecutionPolicy Bypass -File scripts\strix-windows\first_run.ps1 -Phase build
# then:
#   powershell -ExecutionPolicy Bypass -File scripts\strix-windows\win_serve_dflash2.ps1
#
# Optional:
#   -Repo / ATLAS_REPO   repo root. Default: the checkout this script lives in.
#   -ModelDir            target weights snapshot dir.
#   -DrafterDir          DFlash2 drafter weights dir (incoai/Qwen3.8-27B-DFlash2).
#   -Port / -Gamma       serve port (8093) / dflash gamma (8).
#   ATLAS_BIN            prebuilt spark.exe — skips the repo-binary lookup.
param(
    [string]$Repo = "",
    [string]$ModelDir = "$env:USERPROFILE\models\nvidia-Qwen3.8-27B-NVFP4",
    [string]$DrafterDir = "C:\Users\azeez\models\dflash2",
    [string]$Port = "8093",
    [string]$Gamma = "8"
)
$ErrorActionPreference = "Stop"
if (-not $Repo) { $Repo = if ($env:ATLAS_REPO) { $env:ATLAS_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path } }
$Rocm10Bin = "$Repo\target-rocm10\x86_64-pc-windows-msvc\release\spark.exe"
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } elseif (Test-Path $Rocm10Bin) { $Rocm10Bin } else { "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe" }
if (-not (Test-Path $Bin)) { "FAIL: no spark.exe — run first_run.ps1 -Phase build (or set ATLAS_BIN)"; exit 6 }
if (-not (Test-Path $DrafterDir)) { "FAIL: drafter dir missing: $DrafterDir"; exit 6 }
if ($Bin -eq $Rocm10Bin) {
    $env:HIP_PATH = "C:\TheRock\10.0.0"
    $env:PATH = "$env:HIP_PATH\bin;$env:PATH"
}
$Tag = "dflash2-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$Log = "C:\Users\azeez\q38-win-serve-$Tag.log"
$env:HOME = "C:\Users\azeez"
$env:ATLAS_HOME = "C:\Users\azeez\.atlas"

# fp8d recipe env (the fingerprinted ST-995 set — same as first_run.ps1's
# serve phase), plus the DFlash2 switches.
$env:ATLAS_W4A16_DP4A = "1"
$env:ATLAS_W4A16_VARIANT = "v1"
$env:ATLAS_FORCE_GLOBAL_GDN = "1"
$env:ATLAS_GDN_FP8_WEIGHTS = "1"
$env:ATLAS_GDN_FP8_DECODE = "1"
$env:ATLAS_SSM_TAIL_PROTECT = "1"
$env:ATLAS_SSM_TAIL_LEASE_TTL = "128"
$env:ATLAS_SSM_TAIL_MIDCHUNK = "0"
$env:ATLAS_MTP_GATE_REPROBE = "64"
$env:ATLAS_MTP_ACCEPT_DEBUG = "1"
$env:ATLAS_TRACKED_MEMINFO = "1"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"
$env:ATLAS_DFLASH_OPTION_B = "1"

$Fingerprint = "C:\Users\azeez\q38-win-serve-$Tag-fingerprint.txt"
@(
    "date=" + (Get-Date).ToUniversalTime().ToString("o")
    "binary=" + $Bin
    "binary_sha256=" + (Get-FileHash -Algorithm SHA256 $Bin).Hash
    "git=" + (git -C $Repo rev-parse HEAD 2>$null)
    "model=nvidia/Qwen3.8-27B-NVFP4"
    "model_dir=" + $ModelDir
    "drafter=" + $DrafterDir + " gamma=" + $Gamma
    "serve=dflash option_b util=0.80 seq=8192 prefill=2048 kv=bf16 lm_head=bf16 batch=1 thinking=off grammar=off slots=20 ssm-int=128"
) | Out-File $Fingerprint -Encoding utf8

Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5

# NOTE: --dflash replaces --speculative/--num-drafts (DFlash is its own
# proposer — do not pass both). util 0.80 not 0.95: the drafter's weights,
# KV and ctx accumulator sit inside the same ~63 GB gfx1151 budget. The
# boot kernel audit passes without --dangerously-allow-unresolved-kernel-
# lookups — do not re-add it; a NEW unresolved lookup is the gate's signal.
$proc = Start-Process -FilePath $Bin -ArgumentList @(
    "serve", $ModelDir,
    "--no-fast-load",
    "--model-name", "nvidia/Qwen3.8-27B-NVFP4",
    "--host", "127.0.0.1", "--port", $Port,
    "--max-seq-len", "8192", "--max-prefill-tokens", "2048",
    "--gpu-memory-utilization", "0.80",
    "--kv-cache-dtype", "bf16", "--lm-head-dtype", "bf16",
    "--max-batch-size", "1", "--max-num-seqs", "1",
    "--vision-max-pixels", "262144",
    "--disable-tool-grammar", "true",
    "--ssm-cache-slots", "20", "--ssm-checkpoint-interval", "128",
    "--request-timeout", "0",
    "--disable-thinking",
    "--dflash", "--draft-model", $DrafterDir, "--dflash-gamma", $Gamma
) -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru -NoNewWindow

$up = $false
for ($i = 0; $i -lt 450; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { "SERVER DIED"; Get-Content $Log -Tail 8; exit 1 }
    try { Invoke-WebRequest "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { "SERVER NOT UP"; Get-Content $Log -Tail 8; exit 1 }
"server up - nvidia/Qwen3.8-27B-NVFP4 + DFlash2 gamma=$Gamma Option-B on :$Port"

# Boot evidence: the drafter loaded, the selector bound, and the kernel audit
# had zero unresolved lookups.
Select-String -Path $Log -Pattern 'DFlash speculative decoding|DFlash2 candidate selector loaded|propose lanes' | Select-Object -First 4 | ForEach-Object { "    " + $_.Line.Trim().Substring([Math]::Max(0, $_.Line.Trim().Length - 110)) }
$unres = (Select-String -Path $Log -Pattern 'unresolved kernel lookup' | Measure-Object).Count
"kernel audit: unresolved lookups = $unres (want 0)"

# One blocking greedy probe — proves the drafter path actually decodes.
$body = @{ model = "nvidia/Qwen3.8-27B-NVFP4"; messages = @(@{ role = "user"; content = "Reply with exactly one word: the capital of France." }); max_tokens = 32; temperature = 0.0 } | ConvertTo-Json -Compress
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    "SMOKE OK finish=$($r.choices[0].finish_reason) tokens=$($r.usage.completion_tokens) text=$($r.choices[0].message.content.Trim().Substring(0, [Math]::Min(48, $r.choices[0].message.content.Trim().Length)))"
} catch { "SMOKE FAILED: $_" }

"LOG=$Log"
"FINGERPRINT=$Fingerprint"
