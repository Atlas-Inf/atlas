#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# The MLPerf ST-996 leg (harness bfcl_v4, hallucination 12 / live 23 /
# non_live 46, floor 25, n~1004, seeds 42, temp 0, offline/max-throughput)
# for Qwen3.8-27B-NVFP4 (unsloth) on AzeezStrix — the same draw and instrument
# as the previous working unsloth-3.6 run (results_st996_unsloth_ccdaab7e:
# overall 78.59 / normalized 80.45), served under the submission profile
# mapped to Qwen3.8's AMD dtypes (per-row FP8 -> BF16 preservation, rest
# NVFP4, KV bf16, MTP bf16).
set -euo pipefail
SNAP=/home/azeez/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
D=/home/azeez/endpoints/examples/10_Edge_Agentic_Example
RD=results_st996_qwen38_unsloth_strix

mkdir -p /home/azeez/endpoints/$RD
# Derive from the PROVEN st996 config (results_st996_unsloth_ccdaab7e) —
# only the model name, tokenizer, and report_dir change.
sed -e "s|unsloth/Qwen3.6-27B-NVFP4|unsloth/Qwen3.8-27B-NVFP4|" \
    -e "s|tokenizer_name: .*|tokenizer_name: $SNAP|" \
    -e "s|results_st996_unsloth_ccdaab7e|$RD|" \
    /home/azeez/endpoints/results_st996_unsloth_ccdaab7e/config.yaml \
    > $D/st996_qwen38_unsloth.yaml
echo "=== derived config (must show qwen38 + 12/23/46) ==="
grep -nE 'name: "unsloth|hallucination: 12|live: 23|non_live: 46|report_dir' $D/st996_qwen38_unsloth.yaml

# Serve: submission profile + preservation dtype mapping.
cd ~/atlas-inf-pr8
SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1
export ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_KV_OVERCOMMIT=1
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
pkill -f '[s]park serve' 2>/dev/null && sleep 8 || true
target/release/spark serve "$SNAP" \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port 8081 \
  --max-seq-len 65536 --max-prefill-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype bf16 --lm-head-dtype bf16 \
  --max-batch-size 1 \
  --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots 16 --ssm-checkpoint-interval 16 \
  --disable-thinking \
  --dangerously-allow-unresolved-kernel-lookups \
  >"$HOME/q38-st996-serve.log" 2>&1 &
SPARK_PID=$!
for i in $(seq 1 150); do
  curl -s -m2 http://127.0.0.1:8081/v1/models >/dev/null 2>&1 && break
  kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED:"; tail -8 "$HOME/q38-st996-serve.log"; exit 1; }
  sleep 2
done
echo "server up (submission profile, preservation dtypes)"

cd ~/endpoints
. .venv/bin/activate
LOG=/home/azeez/st996_qwen38_$(date +%Y%m%d_%H%M%S).log
echo "=== MLPerf ST-996 | bfcl_v4 12/23/46 | qwen38 unsloth | submission profile | START $(date) ===" > "$LOG"
inference-endpoint benchmark from-config --config "$D/st996_qwen38_unsloth.yaml" --accuracy-only >> "$LOG" 2>&1
echo "RC=$? $(date)" >> "$LOG"
python3 - <<PY
import json
r = json.load(open("/home/azeez/endpoints/$RD/results.json"))
s = r["accuracy_scores"]["bfcl_v4::function_calling"]["score"]
print("ST-996 qwen38-unsloth:", json.dumps(s, indent=1))
PY
echo "ST996 DONE rd=/home/azeez/endpoints/$RD"
kill "$SPARK_PID" 2>/dev/null || true
