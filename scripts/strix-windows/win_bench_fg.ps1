$env:ATLAS_FP8_DEQUANT_ATTN_TO_BF16 = "1"
$env:ATLAS_FP8_DEQUANT_FFN_TO_BF16 = "1"
$env:ATLAS_GDN_BF16_WEIGHTS = "1"
$ErrorActionPreference = "Continue"
& "C:\Users\azeez\code\atlas-inf-pr9\target\x86_64-pc-windows-msvc\release\spark.exe" benchmark run bfcl-subset --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 --param non_live_pct=4 --param live_pct=1 --param hallucination_pct=1 --param subset_floor=2 --param max_new_tokens=1024 --param temperature=0 --param min_overall=0 --param min_normalized=0 --param request_timeout_s=600
Write-Output ("EXIT: " + $LASTEXITCODE)
