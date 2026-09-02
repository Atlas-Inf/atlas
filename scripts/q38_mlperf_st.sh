#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# The MLPerf v6.1 edge-agentic ST accuracy leg (bfcl_v4, 62/10/10, floor 25,
# n=995, seed 42, temp 0) for Qwen3.8-27B-NVFP4 (unsloth) on AzeezStrix —
# the same instrument and draw as the previous working unsloth runs
# (mlperf_unsloth_32slot 0.7648 / mlperf_unsloth_ccdaab7e 0.7196), served
# under the submission profile mapped to Qwen3.8's AMD dtypes.
set -euo pipefail
cd ~/endpoints-mlperf
. .venv/bin/activate
D=examples/11_Edge_Agentic_Example
sed -e 's|"Qwen3.6-27B-Q4_K_M"|"unsloth/Qwen3.8-27B-NVFP4"|' \
    -e 's|http://localhost:8080|http://localhost:8081|' \
    -e 's|results/edge_agentic_full_run/|results/mlperf_qwen38_unsloth_strix/|' \
    "$D/online_edge_full_run.yaml" > "$D/golden_qwen38_unsloth_strix.yaml"
echo "=== derived config (must show unsloth + 8081 + 62/10/10) ==="
grep -nE 'name: "unsloth|localhost:8081|non_live: 62|live: 10|hallucination: 10|report_dir|temperature|seed:' "$D/golden_qwen38_unsloth_strix.yaml"

# Serve: the submission profile mapped to Qwen3.8 (see q38_replay1007.sh).
cd ~/atlas-inf-pr8
SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1
export ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_KV_OVERCOMMIT=1
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
pkill -f 'spark serve' 2>/dev/null && sleep 8 || true
target/release/spark serve ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108 \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port 8081 \
  --max-seq-len 65536 --max-prefill-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype bf16 --lm-head-dtype bf16 \
  --max-batch-size 1 \
  --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots 32 --ssm-checkpoint-interval 16 \
  --disable-thinking \
  --dangerously-allow-unresolved-kernel-lookups \
  >"$HOME/q38-mlperf-st-serve.log" 2>&1 &
SPARK_PID=$!
for i in $(seq 1 120); do
  curl -s -m3 -o /dev/null -w "" http://127.0.0.1:8081/v1/models 2>/dev/null && break
  kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED:"; tail -8 "$HOME/q38-mlperf-st-serve.log"; exit 1; }
  sleep 2
done
echo "server up (submission profile)"

cd ~/endpoints-mlperf
LOG=/home/azeez/mlperf_qwen38_unsloth_$(date +%Y%m%d_%H%M%S).log
echo "$LOG" > /home/azeez/mlperf_current.txt
echo "=== MLPerf-edge BFCL accuracy | 62/10/10 | qwen38 unsloth | submission profile | START $(date) ===" > "$LOG"
inference-endpoint benchmark from-config --config "$D/golden_qwen38_unsloth_strix.yaml" --accuracy-only >> "$LOG" 2>&1
echo "MLPERF_RC=$? $(date)" >> "$LOG"
tail -5 "$LOG"
echo "MLPERF_ST_DONE"
kill "$SPARK_PID" 2>/dev/null || true
