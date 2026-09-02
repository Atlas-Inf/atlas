#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# BFCL-70 focused accuracy gate on the FIXED preservation build (post
# tc->pipelined FFN dispatch). Same 70-row draw as the recorded Kristianpaul
# diagnostics (non_live 4 / live 1 / hallucination 1, floor 2, temp 0).
set -euo pipefail
cd ~/atlas-inf-pr8
PORT=8081
LOG="$HOME/q38-bfcl70-postfix.log"
SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"

SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_VARIANT=v1 ATLAS_W4A16_DP4A=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1

pkill -f 'spark serve' 2>/dev/null && sleep 8 || true
target/release/spark serve "$SNAP" \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port "$PORT" \
  --max-seq-len 4096 --max-prefill-tokens 2048 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype bf16 --lm-head-dtype bf16 \
  --max-batch-size 1 \
  --disable-tool-grammar true \
  --ssm-cache-slots 0 \
  --disable-thinking \
  --dangerously-allow-unresolved-kernel-lookups \
  >"$LOG" 2>&1 &
SPARK_PID=$!
for i in $(seq 1 90); do
  curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED:"; tail -5 "$LOG"; exit 1; }
  sleep 2
done
echo "server up — running BFCL-70"
target/release/spark benchmark run bfcl-subset \
  --url "http://127.0.0.1:$PORT" \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --param non_live_pct=4 --param live_pct=1 --param hallucination_pct=1 \
  --param subset_floor=2 --param max_new_tokens=1024 --param temperature=0 \
  --param min_overall=0 --param min_normalized=0 --param request_timeout_s=600 \
  2>&1 | sed "s/\x1b\[[0-9;]*m//g" | tail -60
echo "-- serve log tail --"
grep -E "watchdog fired" "$LOG" | sed "s/\x1b\[[0-9;]*m//g" | wc -l
kill "$SPARK_PID" 2>/dev/null || true
sleep 5
echo "BFCL70 DONE"
