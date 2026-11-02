# SPDX-License-Identifier: AGPL-3.0-only
# Serves nvidia/Qwen3.8-27B-NVFP4 + incoai/Qwen3.8-27B-DFlash2 on Windows
# gfx1151 (Strix Halo) - the configuration behind the 2026-09-22 clean
# agentic leg (winbox r3: 672/672 turns, 0 errors, score 0.6183, mean_na
# ~2.7 flat through hour 4).
#
# What makes this different from a naive DFlash serve - the two knobs that
# fixed the 2026-09-15 Linux collapse (604/1007, mean_na 0.496):
#
#   * ATLAS_DFLASH_CTX_WINDOW must cover the full transcript. The captured
#     target-hidden-state accumulator is indexed by ABSOLUTE prompt position
#     and hard-capped at ctx_window; the 4096 default conditioned the drafter
#     on positions 0..4095 forever (frozen at the prompt head) -> 0% accept
#     past ~12K. This script sets it to SeqLen.
#   * --dflash-window-size 0 resolves to "no explicit sliding window"; the
#     checkpoint declares no swa_window_size, so the drafter runs full-prefix
#     attention (KV pool sized to max_seq_len). Long-range signal comes
#     through the ctx conditioning - verified: mean_na 6.18 on a predictable
#     decode at seq_len 28K.
#
# MEMORY FACTS (winbox, updated 2026-09-24 post-merge):
#   * GPU exposes ~89.5 GB this boot. Pre-merge, util 0.90 OOM'd the
#     watchdog during drafter scratch alloc and 0.80 was the tested value.
#   * The 2026-09-24 merge (ctx-acc pool primed before KV sizing, startup
#     balloons held across model build, contiguous Marconi snapshot blobs)
#     raises pre-KV footprint to ~74.5 GB - the primed 2.5 GB ctx-acc plus
#     ~6 GB of balloons now count inside it. util 0.80's budget (~71.6 GB)
#     is BELOW that, so kv_budget=0 and startup bails.
#   * util 0.90 is now the tested value: KV lands ~6.1 GB / ~99K tokens,
#     32 Marconi slots, one transient watchdog trip during pool alloc but
#     the balloons carried it through - server live + smoke verified.
#   * The agentic dataset peaks ~25.2K ISL, so 32768 also works if
#     headroom is tighter on a different boot.
#
#   powershell -ExecutionPolicy Bypass -File .\win_serve_dflash2_nvfp4.ps1 -BindHost 127.0.0.1
[CmdletBinding()]
param(
    [string]$BindHost = '127.0.0.1',
    [string]$Port = '8096',
    [string]$SeqLen = '49152',
    [string]$Util = '0.90',
    [string]$Gamma = '8',
    [string]$SsmSlots = '24',
    [string]$SsmInterval = '128',
    [string]$ModelDir = "$env:USERPROFILE\models\nvidia-Qwen3.8-27B-NVFP4",
    [string]$DraftDir = "$env:USERPROFILE\models\dflash2",
    # default|off|on -- ATLAS_THINKING (from serve-amd.ps1 -Thinking) or
    # DISABLE_THINKING=1 alias 'off' when unset.
    [ValidateSet('default', 'off', 'on')]
    [string]$Thinking = ''
)
$ErrorActionPreference = 'Stop'
# pwsh 7.3+ defaults $PSNativeCommandArgumentPassing to 'Windows' (Standard
# escaping), so spark.exe would see the literal backslashes and the JSON parse
# fails fast. On Windows PowerShell 5.1, pwsh < 7.3, or an explicit Legacy
# opt-in, embedded double quotes are stripped and the \" escape is required.
# (about_Parsing: native argument passing; about_Pwsh: PSNativeCommandArgumentPassing)
function Get-ThinkingOnKwarg {
    $legacy = ($PSVersionTable.PSVersion.Major -lt 7) -or
              ($PSVersionTable.PSVersion -lt [version]'7.3') -or
              ($PSNativeCommandArgumentPassing -eq 'Legacy')
    if ($legacy) { '{\"enable_thinking\":true}' } else { '{"enable_thinking":true}' }
}
# Thinking resolution: explicit -Thinking param > ATLAS_THINKING env >
# DISABLE_THINKING=1 alias > shipped default (off).
if (-not $Thinking) {
    if ($env:ATLAS_THINKING) { $Thinking = $env:ATLAS_THINKING }
    elseif ($env:DISABLE_THINKING -eq '1') { $Thinking = 'off' }
    else { $Thinking = 'default' }
}
if ($Thinking -notin @('default','off','on')) { throw "ATLAS_THINKING/-Thinking must be default|off|on, got '$Thinking'" }
if (-not $env:HOME) { $env:HOME = $env:USERPROFILE }
if (-not $env:ATLAS_HOME) { $env:ATLAS_HOME = Join-Path $env:HOME '.atlas' }
$Repo = if ($env:ATLAS_REPO) { $env:ATLAS_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Repo 'target\x86_64-pc-windows-msvc' }
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } else { Join-Path $TargetDir 'release\spark.exe' }
if (-not $env:HIP_PATH) {
    if (Test-Path 'C:\TheRock\10.0.0') { $env:HIP_PATH = 'C:\TheRock\10.0.0' }
    else { throw 'Set HIP_PATH to the ROCm SDK/runtime root.' }
}
if (-not (Test-Path $Bin)) { throw "spark.exe not found: $Bin (build with build-amd.ps1)" }
if (-not (Test-Path $ModelDir)) { throw "model not found: $ModelDir" }
if (-not (Test-Path $DraftDir)) { throw "drafter not found: $DraftDir" }
if (Get-Process spark -ErrorAction SilentlyContinue) { throw 'spark.exe is already running' }
$ReleaseDir = Split-Path $Bin -Parent
$env:PATH = "$ReleaseDir;$env:HIP_PATH\bin;$env:PATH"
$env:HF_HUB_OFFLINE = '1'
$env:RUST_LOG = 'info'
$env:ATLAS_DFLASH_OPTION_B = '1'
$env:ATLAS_DFLASH_CTX_WINDOW = $SeqLen   # must cover the transcript; see header
$env:ATLAS_MTP_ACCEPT_DEBUG = '1'
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = Join-Path $Repo "out\dflash2-serve-$Stamp.log"
$Fingerprint = Join-Path $Repo "out\dflash2-serve-$Stamp-fingerprint.txt"
New-Item -ItemType Directory -Force (Join-Path $Repo 'out') | Out-Null
@(
    "date_utc=" + (Get-Date).ToUniversalTime().ToString('o')
    "binary=" + $Bin
    "binary_sha256=" + (Get-FileHash -Algorithm SHA256 $Bin).Hash
    "commit=" + (git -C $Repo rev-parse HEAD)
    "hip_path=" + $env:HIP_PATH
    "model=nvidia/Qwen3.8-27B-NVFP4 dir=$ModelDir"
    "drafter=incoai/Qwen3.8-27B-DFlash2 dir=$DraftDir gamma=$Gamma option_b=1"
    "bind=" + $BindHost + ':' + $Port
    "serve=max_seq_len:$SeqLen max_prefill_tokens:2048 gpu_util:$Util kv:bf16 lm_head:bf16 batch:1 dflash_window:0(full) dflash_ctx_window:$SeqLen ssm_slots:$SsmSlots ssm_ckpt:$SsmInterval prefix_cache:true thinking:" + $(if ($Thinking -eq 'default') { 'default(off)' } else { $Thinking }) + " request_timeout:900"
) | Out-File $Fingerprint -Encoding utf8
$HashFiles = @(
    (Join-Path $ModelDir 'config.json'),
    (Join-Path $DraftDir 'dflash_config.json'),
    (Join-Path $ReleaseDir 'cuda.dll'),
    (Join-Path $ReleaseDir 'nvcuda.dll')
) + @(Get-ChildItem $ReleaseDir -File | Where-Object { $_.Name -match '^(amdhip64_|amd_comgr|hiprtc)' } | ForEach-Object FullName)
$HashFiles | Where-Object { Test-Path $_ } | ForEach-Object { Get-FileHash -Algorithm SHA256 $_ } |
    Format-Table -AutoSize | Out-File $Fingerprint -Append -Encoding utf8
$ServeArgs = @(
    'serve', $ModelDir,
    '--model-name', 'nvidia/Qwen3.8-27B-NVFP4',
    '--host', $BindHost, '--port', $Port,
    '--no-tui', '--no-fast-load',
    '--max-seq-len', $SeqLen, '--max-prefill-tokens', '2048',
    '--gpu-memory-utilization', $Util,
    '--kv-cache-dtype', 'bf16', '--lm-head-dtype', 'bf16',
    '--max-batch-size', '1',
    '--dflash', '--draft-model', $DraftDir,
    '--dflash-gamma', $Gamma, '--dflash-window-size', '0',
    '--ssm-cache-slots', $SsmSlots, '--ssm-checkpoint-interval', $SsmInterval,
    '--enable-prefix-caching',
    '--request-timeout', '900'
)
# off/default keep the shipped --disable-thinking; 'on' makes thinking the
# SERVER default instead. The escaped-quote form is required: PS 5.1 strips
# embedded double quotes from native-exe arguments.
switch ($Thinking) {
    'on'      { $ServeArgs += '--default-chat-template-kwargs', (Get-ThinkingOnKwarg) }
    default   { $ServeArgs += '--disable-thinking' }
}
& $Bin @ServeArgs 2>&1 | Tee-Object -FilePath $Log
