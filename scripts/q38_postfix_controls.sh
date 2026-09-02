#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Serve the preservation build (post tc->pipelined FFN fix) and run the full
# 14-case targeted relevant/irrelevant tool-control battery from the handoff.
set -euo pipefail
cd ~/atlas-inf-pr8
PORT=8091
LOG="$HOME/q38-postfix-serve.log"
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
echo "server up — running targeted controls"
python3 "$HOME/q38_targeted_controls_stream.py" "http://127.0.0.1:$PORT" \
  "unsloth/Qwen3.8-27B-NVFP4" "$HOME/.atlas/artifacts/bfcl" 0.0 false \
  2>&1 | sed "s/\x1b\[[0-9;]*m//g"
echo "-- serve log extract --"
grep -E "Prefill first token|watchdog fired|Done:" "$LOG" | sed "s/\x1b\[[0-9;]*m//g" | tail -40
kill "$SPARK_PID" 2>/dev/null || true
sleep 5
echo "CONTROLS DONE"
