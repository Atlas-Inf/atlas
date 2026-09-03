# SPDX-License-Identifier: AGPL-3.0-only
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BindHost,
    [string]$Port = '8081'
)
$ErrorActionPreference = 'Stop'
if ($BindHost -in @('0.0.0.0', '::', '*')) { throw 'BindHost must be a specific local or Tailscale address.' }
$env:HOME = 'C:\Users\azeez'
$env:USERPROFILE = 'C:\Users\azeez'
$env:ATLAS_HOME = 'C:\Users\azeez\.atlas'
$Repo = if ($env:ATLAS_REPO) { $env:ATLAS_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Repo 'target-rocm10' }
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } else { Join-Path $TargetDir 'x86_64-pc-windows-msvc\release\spark.exe' }
$ModelDir = if ($env:ATLAS_MODEL_DIR) { $env:ATLAS_MODEL_DIR } else { 'C:\Users\azeez\models\Qwen3.8-27B-NVFP4' }
$ModelName = 'unsloth/Qwen3.8-27B-NVFP4'
if (-not $env:HIP_PATH) {
    if (Test-Path 'C:\TheRock\10.0.0') { $env:HIP_PATH = 'C:\TheRock\10.0.0' }
    else { throw 'Set HIP_PATH to the ROCm SDK/runtime root.' }
}
if (-not (Test-Path $Bin)) { throw "spark.exe not found: $Bin" }
if (-not (Test-Path $ModelDir)) { throw "model not found: $ModelDir" }
if (Get-Process spark -ErrorAction SilentlyContinue) { throw 'spark.exe is already running' }
$ReleaseDir = Split-Path $Bin -Parent
$env:PATH = "$ReleaseDir;$env:HIP_PATH\bin;$env:PATH"
$env:ATLAS_W4A16_VARIANT = 'v1'
$env:ATLAS_W4A16_DP4A = '1'
$env:ATLAS_KV_EXTERNAL_RESERVE_GB = '0'
$env:ATLAS_FP8_DEQUANT_ATTN_TO_BF16 = '1'
$env:ATLAS_FP8_DEQUANT_FFN_TO_BF16 = '1'
$env:ATLAS_GDN_BF16_WEIGHTS = '1'
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Log = "C:\Users\azeez\q38-agentic-perf-$Stamp-serve.log"
@(
    "date_utc=" + (Get-Date).ToUniversalTime().ToString('o')
    "binary=" + $Bin
    "binary_sha256=" + (Get-FileHash -Algorithm SHA256 $Bin).Hash
    "hip_path=" + $env:HIP_PATH
    "model=" + $ModelName
    "bind=" + $BindHost + ':' + $Port
    'serve=max_seq_len:32768 max_prefill_tokens:2048 gpu_util:0.70 kv:bf16 lm_head:bf16 batch:1 drafts:1 prefix:true ssm_slots:16 interval:16'
) | Out-File "C:\Users\azeez\q38-agentic-perf-$Stamp-fingerprint.txt" -Encoding utf8
$Fingerprint = "C:\Users\azeez\q38-agentic-perf-$Stamp-fingerprint.txt"
$HashFiles = @(
    (Join-Path $ModelDir 'config.json')
    (Join-Path $ModelDir 'model.safetensors.index.json')
    (Join-Path $ReleaseDir 'cuda.dll')
    (Join-Path $ReleaseDir 'nvcuda.dll')
) + @(Get-ChildItem $ReleaseDir -File | Where-Object { $_.Name -match '^(amdhip64_|amd_comgr|hiprtc)' } | ForEach-Object FullName)
$HashFiles | Where-Object { Test-Path $_ } | ForEach-Object { Get-FileHash -Algorithm SHA256 $_ } |
    Format-Table -AutoSize | Out-File $Fingerprint -Append -Encoding utf8
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate |
    Format-List | Out-File $Fingerprint -Append -Encoding utf8
Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion,ReleaseDate |
    Format-List | Out-File $Fingerprint -Append -Encoding utf8
$ServeArgs = @(
    'serve', $ModelDir,
    '--model-name', $ModelName,
    '--host', $BindHost, '--port', $Port,
    '--max-seq-len', '32768', '--max-prefill-tokens', '2048',
    '--gpu-memory-utilization', '0.70',
    '--kv-cache-dtype', 'bf16', '--lm-head-dtype', 'bf16',
    '--max-batch-size', '1',
    '--speculative', '--num-drafts', '1', '--mtp-quantization', 'bf16', '--mtp-vocab', '100000',
    '--enable-prefix-caching', '--ssm-cache-slots', '16', '--ssm-checkpoint-interval', '16',
    '--disable-tool-grammar', 'true', '--disable-thinking',
    '--dangerously-allow-unresolved-kernel-lookups', '--no-fast-load'
)
& $Bin @ServeArgs 2>&1 | Tee-Object -FilePath $Log
exit $LASTEXITCODE
