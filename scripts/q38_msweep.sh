#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# M-sweep against the live all-BF16 preservation serve: plain-text prompts at
# increasing token counts, no tools, to attribute the prefill pathology to a
# per-token serial cost vs a threshold effect, and to test tool-independence.
set -euo pipefail
cd ~/atlas-inf-pr8
PORT=8091
LOG="$HOME/q38-msweep.log"
SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"

SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_VARIANT=v1 ATLAS_W4A16_DP4A=1
export ATLAS_KV_EXTERNAL_RESERVE_GB=0
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
echo "server up"

for M in 16 128 256 512 513 1024 2048; do
  # ~1 token per word plus slack; the exact count is read back from the log.
  WORDS=$((M))
  PROMPT=$(python3 -c "print(' '.join(['alpha','bravo','charlie','delta'] * $((WORDS / 4 + 1)))[:$WORDS * 6])")
  WALL=$(python3 - "$PORT" "$M" "$PROMPT" <<'PY'
import json, sys, time, urllib.request
port, m, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "unsloth/Qwen3.8-27B-NVFP4", "stream": False,
        "temperature": 0.0, "max_tokens": 8,
        "messages": [{"role": "user", "content": prompt + "\n\nAnswer with one word: what comes after delta?"}]}
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    wall = time.time() - t0
    text = out["choices"][0]["message"]["content"][:60]
    print(f"{wall:.1f}|{text!r}")
except Exception as e:
    print(f"{time.time()-t0:.1f}|ERROR {e!r}")
PY
)
  echo "M~$M wall|head: $WALL"
done

echo "-- server prefill timings --"
grep -E "Chunked prefill start|Prefill first token|watchdog fired|Session .*prompt tokens" "$LOG" \
  | sed "s/\x1b\[[0-9;]*m//g" | tail -40
kill "$SPARK_PID" 2>/dev/null || true
sleep 5
echo "SWEEP DONE"
