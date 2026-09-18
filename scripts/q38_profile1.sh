#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# One profiled prefill: ATLAS_PROFILE=1 + ATLAS_PROFILE_FIRST=1, one ~615-token
# plain prompt, then extract the per-phase SSM prefill timings.
set -euo pipefail
cd "${ATLAS_REPO:-$HOME/atlas-inf-pr8}"
PORT=8091
LOG="$HOME/q38-profile.log"
SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"

SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_VARIANT=v1 ATLAS_W4A16_DP4A=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
export ATLAS_PROFILE=1 ATLAS_PROFILE_FIRST=1

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

python3 - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
words = " ".join(["alpha", "bravo", "charlie", "delta"] * 160)
body = {"model": "unsloth/Qwen3.8-27B-NVFP4", "stream": False,
        "temperature": 0.0, "max_tokens": 8,
        "messages": [{"role": "user", "content": words + "\n\nAnswer with one word: what comes after delta?"}]}
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    out = json.load(r)
print(f"wall {time.time()-t0:.1f}s head={out['choices'][0]['message']['content'][:40]!r}")
PY

echo "-- profile lines (first SSM layer only, plus attention/prefill lines) --"
grep -E "SSM prefill \[|ATTN prefill|attn.*prefill|Prefill first token|prompt tokens" "$LOG" \
  | sed "s/\x1b\[[0-9;]*m//g" | head -60 || true
kill "$SPARK_PID" 2>/dev/null || true
sleep 5
echo "PROFILE DONE"
