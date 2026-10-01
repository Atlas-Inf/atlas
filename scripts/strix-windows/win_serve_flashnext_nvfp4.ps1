# SPDX-License-Identifier: AGPL-3.0-only
# Serves nvidia/Qwen3.8-Flash-Next-NVFP4 on Windows gfx1151 (Strix Halo) with
# the measured boot-7g recipe -- the profile behind the 2026-09-20 MTP headline
# (17.0 tok/s decode, mean_na 0.95; serial arm ~8 tok/s after correcting the
# watchdog-rollback rework in its Done: lines, i.e. ~2.1x -- see the doc).
#
# MEMORY FACTS this recipe is built around (winbox, 2026-09-19/20):
#  * VGM MUST be 32 GB -- set it with vgmctl (scripts/strix-windows/vgmctl/).
#    On this runtime VGM 96 GB leaves only 48 GiB usable, VGM 0.5 GB leaves
#    ~63 GB; VGM 32 GB exposes 96 GB to the HIP shim.
#  * The resident commit wall is ~84.5 GiB (32 GiB carve-out + ~52.5 GiB WDDM
#    shared). cuMemAlloc may SUCCEED past it and poison the context with a
#    sticky 719 on the next submission -- the wall is real even when the API
#    reports more free.
#  * ATLAS_UMA_COMMIT_LIMIT_GB is an OPERATOR-ASSERTED ceiling: the HIP shim
#    reports it as totalGlobalMem (env > 0 overrides, no min vs the real
#    property). 96 is what VGM 32 exposes; it is NOT a safe budget -- the
#    recipe keeps committed under the 84.5 wall instead.
#  * At Server live the serve process holds ~81 GiB WorkingSet and the host
#    has ~12 GiB free. NOTHING else memory-heavy may run beside it.
#  * --ssm-cache-slots 0 drops the 1.8 GB Marconi snapshot pool (dead weight
#    without --enable-prefix-caching). The KV budget is self-relative -- it
#    absorbs whatever you free, so utilization must also stay low.
#  * SERIAL=1 drops --speculative AND lowers util to 0.86: serial pre-KV is
#    ~4 GB lower than the MTP arm, so at 0.90 the KV budget absorbs the slack
#    (~7.1 GB) and crosses the wall during pool allocation. Measured.
#  * DISABLE_THINKING=1 appends --disable-thinking: the remote-driven perf leg
#    serves with reasoning off. Default unchanged -- thinking stays at the
#    checkpoint default, which is what the Windows ST-995 records used.
#  * -BindHost defaults to 127.0.0.1; set a Tailscale address when the run is
#    driven from another box (winbox's "Atlas agentic perf" + Tailscale-In
#    rules allow it). Wildcard binds are refused.
param(
    [string]$Tag = ("fnext-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$ModelDir = "$env:USERPROFILE\models\nvidia-Qwen3.8-Flash-Next-NVFP4",
    [string]$Port = "8095",
    [string]$BindHost = "127.0.0.1"
)
$ErrorActionPreference = "Stop"
if ($BindHost -in @('0.0.0.0', '::', '*')) { throw 'BindHost must be a specific local or Tailscale address.' }
$Repo = if ($env:ATLAS_REPO) { $env:ATLAS_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } else { "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe" }
if (-not $env:HIP_PATH) {
    if (Test-Path 'C:\TheRock\10.0.0') { $env:HIP_PATH = 'C:\TheRock\10.0.0' }
    else { throw 'Set HIP_PATH to the ROCm SDK/runtime root.' }
}
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"
if (-not $env:HOME) { $env:HOME = $env:USERPROFILE }
New-Item -ItemType Directory -Force (Join-Path $Repo 'out') | Out-Null
$Log = Join-Path $Repo "out\serve-fnext-$Tag.log"

# Measured recipe env (boot-7g, 2026-09-20).
$env:HF_HUB_OFFLINE = "1"
$env:RUST_LOG = "info"
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = "0"
$env:ATLAS_OOM_WATCHDOG_MB = "400"
$env:ATLAS_MEM_PROFILE = "1"
$env:ATLAS_UMA_COMMIT_LIMIT_GB = "96"
$env:ATLAS_MTP_ACCEPT_DEBUG = "1"
# MTP past the QSA inert bound (~2k context). The engine defaults this ON only
# for NVIDIA ("unmeasured" on gfx1151), which leaves every agentic-length turn
# on serial decode here. It is measured on gfx1151: both Windows ST-995 records
# on #80 (84.22 / 83.61) and the 10/10 agentic-webserver leg at 32k ran with it
# on, so the recipe serves it by default. An explicit ATLAS_QSA_VERIFY_ACTIVE=0
# still opts out.
if (-not $env:ATLAS_QSA_VERIFY_ACTIVE) { $env:ATLAS_QSA_VERIFY_ACTIVE = "1" }

$Serial = ($env:SERIAL -eq "1")
$Util = if ($env:ATLAS_UTIL) { $env:ATLAS_UTIL } elseif ($Serial) { "0.86" } else { "0.90" }  # see header: serial pre-KV is ~4 GB lower
$SeqLen = if ($env:SEQ_LEN) { $env:SEQ_LEN } else { "8192" }
$PrefillTokens = if ($env:PREFILL_TOKENS) { $env:PREFILL_TOKENS } else { "2048" }
$SsmSlots = if ($env:SSM_SLOTS) { $env:SSM_SLOTS } else { "0" }
# ATLAS_THINKING (from serve-amd.ps1 -Thinking) or DISABLE_THINKING=1 alias:
# off -> --disable-thinking, on -> --default-chat-template-kwargs, default ->
# the checkpoint's template default. Resolved before the fingerprint line.
$Thinking = if ($env:ATLAS_THINKING) { $env:ATLAS_THINKING }
            elseif ($env:DISABLE_THINKING -eq "1") { 'off' }
            else { 'default' }

$Fingerprint = Join-Path $Repo "out\serve-fnext-$Tag-fingerprint.txt"
@(
    "date=" + (Get-Date).ToUniversalTime().ToString("o")
    "binary=" + $Bin
    "binary_sha256=" + (Get-FileHash -Algorithm SHA256 $Bin).Hash
    "hip_path=" + $env:HIP_PATH
    "model=nvidia/Qwen3.8-Flash-Next-NVFP4"
    "model_dir=" + $ModelDir
    "config_sha256=" + (Get-FileHash -Algorithm SHA256 (Join-Path $ModelDir "config.json")).Hash
    "serve=boot7g util=$Util seq=$SeqLen prefill=2048 kv=bf16 batch=1 drafts=" + $(if ($Serial) { "0" } else { "1" }) + " ssm_slots=0 serial=$Serial vgm=32GB commit_limit=96 qsa_verify_active=$env:ATLAS_QSA_VERIFY_ACTIVE bind=$BindHost thinking=" + $(if ($env:DISABLE_THINKING -eq "1") { "off" } else { "checkpoint-default" })
) | Out-File $Fingerprint -Encoding utf8

Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 5

# NOTE: boots 4-7 carried `--dangerously-allow-unresolved-kernel-lookups`
# because the 62 optional-arm probes were undeclared then. MODEL.toml now
# declares all 62 in [expected_absent], so the boot audit passes WITHOUT the
# flag -- do not re-add it; a new unresolved lookup is exactly the signal the
# gate exists to catch.
$Args = @(
    "serve", $ModelDir,
    "--no-fast-load",
    "--no-tui",
    "--model-name", "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "--host", $BindHost, "--port", $Port,
    "--max-seq-len", $SeqLen, "--max-prefill-tokens", $PrefillTokens,
    "--max-batch-size", "1", "--max-num-seqs", "1",
    "--gpu-memory-utilization", $Util,
    "--kv-cache-dtype", "bf16",
    "--request-timeout", "0",
    "--ssm-cache-slots", $SsmSlots
)
if ($env:PREFIX_CACHE -eq "1") { $Args += "--enable-prefix-caching" }
# The 'on' argument passes through Start-Process -ArgumentList, where PS
# joins elements and strips embedded quotes, so the JSON is escaped as
# {\"...\"}: spark.exe's argv parser then sees {"enable_thinking":true}
# (and startup would fail fast on malformed JSON).
if ($Thinking -eq 'off') { $Args += "--disable-thinking" }
if ($Thinking -eq 'on') { $Args += @("--default-chat-template-kwargs", '{\"enable_thinking\":true}') }
if ($Thinking -notin @('default','off','on')) { throw "ATLAS_THINKING must be default|off|on, got '$Thinking'" }
if (-not $Serial) { $Args += @("--speculative", "--num-drafts", "1") }

$proc = Start-Process -FilePath $Bin -ArgumentList $Args -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru -NoNewWindow

$up = $false
for ($i = 0; $i -lt 1350; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { "SERVER DIED"; Get-Content $Log -Tail 8; exit 1 }
    try { Invoke-WebRequest "http://${BindHost}:$Port/v1/models" -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { "SERVER NOT UP"; Get-Content $Log -Tail 8; exit 1 }
"server up - nvidia/Qwen3.8-Flash-Next-NVFP4 boot-7g recipe on ${BindHost}:$Port (serial=$Serial util=$Util)"
"LOG=$Log"
"FINGERPRINT=$Fingerprint"
