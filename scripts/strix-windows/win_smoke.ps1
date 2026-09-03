# SPDX-License-Identifier: AGPL-3.0-only
# Windows parity smoke for the mirrored Qwen3.8-27B-NVFP4 (unsloth) port:
# serves the FROZEN accuracy recipe (mirrors q38_bfcl70.sh on strix) and fires
# the plain-recall control plus one relevant tool control.
param(
    [string]$Port = "8091",
    [string]$ModelDir = "$env:USERPROFILE\models\Qwen3.8-27B-NVFP4"
)
$ErrorActionPreference = "Stop"
$Repo = "C:\Users\azeez\code\atlas-inf-pr9"
$Bin = if ($env:ATLAS_BIN) { $env:ATLAS_BIN } else { "$Repo\target\x86_64-pc-windows-msvc\release\spark.exe" }
$Log = "C:\Users\azeez\q38-win-smoke.log"

# Frozen preservation env (same as the Linux fingerprint).
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
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { Write-Output "SERVER DIED during load:"; Get-Content $Log -Tail 12; exit 1 }
    try { Invoke-WebRequest "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { Write-Output "SERVER NOT UP in 180s"; Get-Content $Log -Tail 12; exit 1 }
Write-Output "server up after ~$($i * 2)s"

$body = @{
    model = "unsloth/Qwen3.8-27B-NVFP4"; stream = $false
    temperature = 0.0; max_tokens = 8
    messages = @(@{ role = "user"; content = "Answer exactly: Paris" })
} | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
Write-Output ("RECALL: " + $r.choices[0].message.content)

$toolJson = @'
{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "stream": false,
    "temperature": 0.0,
    "max_tokens": 256,
    "tool_choice": "auto",
    "messages": [{"role": "user", "content": "What is the current weather in Shanghai, China? Use the weather tool."}],
    "tools": [
        {"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}},
        {"type": "function", "function": {"name": "get_time_zone", "description": "Get the time zone for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}
    ]
}
'@
$r2 = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $toolJson -ContentType "application/json" -TimeoutSec 600
Write-Output ("TOOL_CALLS: " + ($r2.choices[0].message.tool_calls | ConvertTo-Json -Depth 5 -Compress))
Write-Output ("CONTENT_HEAD: " + ($r2.choices[0].message.content | Out-String).Substring(0, [Math]::Min(160, ($r2.choices[0].message.content | Out-String).Length)))

Get-Process spark -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Output "WIN SMOKE DONE"
