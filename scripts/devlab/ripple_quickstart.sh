#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix: the public Ripple-65B quickstart, verbatim (the command we hand to r/strixhalo), on the workflow's build.
# Checks that DRAFT_MODEL accepts the Hugging Face id (not just a snapshot path), that it serves, answers greedy
# twice identically, makes a tool call, and reports decode tok/s.
#
#   scripts/devlab/ripple_quickstart.sh
set -euo pipefail
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
log "ripple_quickstart $(git rev-parse --short=9 HEAD)"
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub
for m in ricursiveAI/Ripple-65B-Preview-NVFP4 ricursiveAI/Ripple-65B-Preview-DFlash2; do
  python3 -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1])" "$m" >/dev/null || { log "DOWNLOAD FAILED $m"; exit 1; }
done
PORT=8888
# ---- the quickstart command, as published (only ATLAS_BIN/PORT/RUST_LOG added for the harness) ----
( exec env RUST_LOG=info ATLAS_BIN="$PWD/target/release/spark" PORT=$PORT \
    DFLASH=1 DRAFT_MODEL=ricursiveAI/Ripple-65B-Preview-DFlash2 \
    GPU_UTIL=0.95 MAX_SEQ_LEN=8192 MAX_BATCH=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0 \
    ./serve-amd.sh ricursiveAI/Ripple-65B-Preview-NVFP4 --dflash-window-size 16384 ) > "$OUT/serve.log" 2>&1 &
SPID=$!
for _ in $(seq 1 720); do
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { log "serve exited early"; grep -aE "ERROR|Error|error:|panick" "$OUT/serve.log" | tail -8 | tee -a "$RES"; exit 1; }
  sleep 5
done
log "serving on :$PORT; models: $(curl -s http://127.0.0.1:$PORT/v1/models | head -c 300)"
grep -aoE "DFlash[^\"]{0,120}(drafter|draft model|loaded)[^\"]{0,80}" "$OUT/serve.log" | sed "s/\x1b\[[0-9;]*m//g" | head -3 | tee -a "$RES" || true
python3 - "$PORT" <<'EOF' 2>&1 | tee -a "$RES"
import json, sys, time, urllib.request
port = sys.argv[1]
def chat(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800)); return r, time.time() - t
models = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
print(f"  model id: {models}")
q = {"model": models, "messages": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}], "max_tokens": 96, "temperature": 0}
a, ta = chat(q); b, tb = chat(q)
ca, cb = a["choices"][0]["message"].get("content"), b["choices"][0]["message"].get("content")
n = a["usage"]["completion_tokens"]
print(f"  greedy x2 identical: {ca == cb} | {n} tok in {ta:.1f}s ({n / ta:.2f} tok/s incl. prefill) | {ca[:120]!r}")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
t, tt = chat({"model": models, "messages": [{"role": "user", "content": "What's the weather in Austin right now?"}], "tools": tools, "max_tokens": 256, "temperature": 0})
tc = t["choices"][0]["message"].get("tool_calls")
print(f"  tool call: {json.dumps(tc)[:200] if tc else 'NONE: ' + repr(t['choices'][0]['message'].get('content'))[:160]}")
lg, tl = chat({"model": models, "messages": [{"role": "user", "content": "Write a 300-word story about a lighthouse keeper."}], "max_tokens": 400, "temperature": 0})
m = lg["usage"]["completion_tokens"]; print(f"  long gen: {m} tok in {tl:.1f}s -> {m / tl:.2f} tok/s (incl. prefill)")
EOF
kill -INT $SPID 2>/dev/null || true; for _ in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null || true
log "server: $(grep -caE ' ERROR |panicked' "$OUT/serve.log") ERROR/panic lines"
log DONE
