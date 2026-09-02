#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# The PINNED ST-995 accuracy gate (the MLPerf-edge golden draw) on the frozen
# Qwen3.8-27B-NVFP4 (unsloth) preservation serve — the completely unmodified
# default command with NO --param overrides, per the handoff §11. Do not add
# percentage, floor, temperature, or threshold overrides here.
set -euo pipefail
cd ~/atlas-inf-pr8
PORT=8081
LOG="$HOME/q38-st995-postfix.log"
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
echo "server up — running PINNED ST-995 (no param overrides)"
target/release/spark benchmark run bfcl-subset \
  --url "http://127.0.0.1:$PORT" \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  2>&1 | sed "s/\x1b\[[0-9;]*m//g" | tail -80
echo "-- serve log tail --"
grep -cE "watchdog fired" "$LOG" || true
kill "$SPARK_PID" 2>/dev/null || true
sleep 5
echo "ST995 DONE"
