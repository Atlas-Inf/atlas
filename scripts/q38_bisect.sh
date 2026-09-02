#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# One-variable serve bisection for the Qwen3.8 Unsloth BF16-preservation
# prefill pathology on AzeezStrix.
#
# Serves the authoritative checkpoint with the 2026-09-01 preservation
# fingerprint's exact flags, varying ONLY the three per-row-FP8 preservation
# env flags, then probes one relevant BFCL control and extracts the server's
# own prefill timing, first predicted token, and loop-watchdog lines.
#
# Usage: q38_bisect.sh ROW_NAME "FLAG1=1 FLAG2=1" [SAMPLE_ID] [MAX_TOKENS]
set -euo pipefail
cd ~/atlas-inf-pr8

ROW="${1:?row name}"
FLAGS="${2:?space-separated preservation flags, may be empty}"
SAMPLE="${3:-live_multiple_4-2-1}"
MAXTOK="${4:-256}"
PORT=8091
LOG="$HOME/q38-bisect-$ROW.log"
PROBE_LOG="$HOME/q38-bisect-$ROW.probe.txt"
SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"

SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_VARIANT=v1 ATLAS_W4A16_DP4A=1
export ATLAS_KV_EXTERNAL_RESERVE_GB=0 ATLAS_MEM_PROFILE=1
# The variable under test: only the named flags are set this row.
for f in ATLAS_FP8_DEQUANT_ATTN_TO_BF16 ATLAS_FP8_DEQUANT_FFN_TO_BF16 ATLAS_GDN_BF16_WEIGHTS; do
  unset "$f" || true
done
for f in $FLAGS; do export "$f"; done

echo "== ROW $ROW flags: ${FLAGS:-<none, destructive requant>} =="
pkill -f 'spark serve' 2>/dev/null && sleep 8 || true

env -u ATLAS_FP8_DEQUANT_ATTN_TO_BF16 -u ATLAS_FP8_DEQUANT_FFN_TO_BF16 -u ATLAS_GDN_BF16_WEIGHTS \
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
  if grep -q "READY\|server ready\|listening" "$LOG" 2>/dev/null \
     || curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SPARK_PID" 2>/dev/null; then
    echo "SERVER DIED during load:"; tail -20 "$LOG"; exit 1
  fi
  sleep 2
done
echo "server up after ~$((i * 2))s"

python3 "$HOME/q38_bisect_probe.py" "http://127.0.0.1:$PORT" \
  "unsloth/Qwen3.8-27B-NVFP4" "$HOME/.atlas/artifacts/bfcl" "$SAMPLE" "$MAXTOK" \
  | tee "$PROBE_LOG"

echo "-- server log extract --"
grep -E "Chunked prefill start|Prefill first token|Prefilled|watchdog fired|prompt tokens" "$LOG" \
  | sed "s/\x1b\[[0-9;]*m//g" | tail -8

kill "$SPARK_PID" 2>/dev/null || true
sleep 6
echo "== ROW $ROW done =="
