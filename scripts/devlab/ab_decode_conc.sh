#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: concurrent-DECODE A/B of the checked-out ref (already built by the workflow) against a base ref.
#
#   scripts/devlab/ab_decode_conc.sh <model> <base-ref> [rounds] [KEY=VAL ...]
#   e.g.  scripts/devlab/ab_decode_conc.sh nvidia/Qwen3.8-27B-NVFP4 main 2
#
# Builds <base-ref> in a sibling worktree, then serves <model> from each build in turn (base, test, base, ...) at
# MAX_BATCH 8. Each round measures the steady all-decoding aggregate (tok/s) with C = 1 / 4 / 8 concurrent
# 128-token greedy requests. Round 1 also checks, per build, how many of 8 concurrent replies equal the same
# prompt served alone. Results go to $DEVLAB_OUT/RESULTS.txt (the workflow publishes it to the job summary).
# The prefill twin is ab_prefill.sh; this one exists because a C>1 change cannot be judged by a C=1 bench.
set -euo pipefail
MODEL=${1:?model}; BASE=${2:?base ref}; ROUNDS=${3:-2}
# Extra args of the form KEY=VALUE become env for the serves (serve-amd.sh knobs: GPU_UTIL, MAX_BATCH, ...).
for kv in "${@:4}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
# Same memory reasoning as ab_prefill.sh, but the batch-8 SSM/MTP pools need more headroom: at util 0.88 the 27B load
# hit cuMemAlloc failed (2 GB requested, 1.2 GB free; run 41181983380), so the default here is 0.70.
export GPU_UTIL=${GPU_UTIL:-0.70} MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384} ATLAS_KV_EXTERNAL_RESERVE_GB=${ATLAS_KV_EXTERNAL_RESERVE_GB:-0} \
  SSM_SLOTS=${SSM_SLOTS:-8} SSM_CKPT_INTERVAL=${SSM_CKPT_INTERVAL:-512} MAX_BATCH=${MAX_BATCH:-8} NUM_DRAFTS=${NUM_DRAFTS:-0}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
TEST_DIR=$PWD; BASE_DIR="${RUNNER_TEMP:-/tmp}/devlab-base"
log "test $(git -C "$TEST_DIR" rev-parse --short=9 HEAD)  base $BASE  model $MODEL  rounds $ROUNDS  MAX_BATCH=$MAX_BATCH NUM_DRAFTS=$NUM_DRAFTS GPU_UTIL=$GPU_UTIL"

# --- model (through $HF_ENDPOINT into $HF_HOME; persistent scratch) ----------------------------------------
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub
t0=$(date +%s)
python3 -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$MODEL" \
  || { log "MODEL DOWNLOAD FAILED ($MODEL via ${HF_ENDPOINT:-hf.co})"; exit 1; }
log "model ready in $(( $(date +%s) - t0 ))s"

# --- base build ---------------------------------------------------------------------------------------
rm -rf "$BASE_DIR"
git -C "$TEST_DIR" fetch -q --depth=1 https://github.com/Atlas-Inf/atlas.git "$BASE"
git -C "$TEST_DIR" worktree add -q --detach "$BASE_DIR" FETCH_HEAD
log "base sha $(git -C "$BASE_DIR" rev-parse --short=9 HEAD)"
( cd "$BASE_DIR" && ./build-amd.sh ) > "$OUT/base-build.log" 2>&1 || { log "BASE BUILD FAILED"; tail -30 "$OUT/base-build.log"; exit 1; }

PORT=18081
serve() { # $1 = repo dir, $2 = log
  ( cd "$1" && exec env RUST_LOG=info ATLAS_BIN="$1/target/release/spark" PORT=$PORT HOST=127.0.0.1 MODEL_NAME=m \
      ./serve-amd.sh "$MODEL" --disable-thinking --request-timeout 0 ) > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 1 720); do
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || { log "serve exited early ($2)"; tail -20 "$2"; return 1; }
    sleep 5
  done
  log "serve not ready after 60 min ($2)"; return 1
}
stop() { kill -INT $SPID 2>/dev/null || true; for _ in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  kill -9 $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true
  for _ in $(seq 1 60); do [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 40000 ] && break; sleep 2; done; }

client() { # $1 = arm, $2 = round, $3 = identity file (written on round 1)
  python3 - "$PORT" "$1" "$2" "$3" <<'EOF'
import json, sys, threading, time, urllib.request
port, arm, rnd, idf = sys.argv[1:5]
C = open("/usr/share/common-licenses/GPL-3").read().split()
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
def stream(p, mt, rec):
    req = {"model": "m", "temperature": 0, "max_tokens": mt, "stream": True,
           "messages": [{"role": "user", "content": p}]}
    t0 = time.time(); first = None; n = 0; text = []
    with urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(req).encode(),
                                headers={"Content-Type": "application/json"}), timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta", {}).get("content")
            if d:
                now = time.time(); first = first or now; n += 1; text.append(d)
    rec.update(start=first or t0, end=time.time(), n=n, text="".join(text))
def run(c):
    recs = [dict() for _ in range(c)]
    prompts = [" ".join((C * 3)[(i * 500 + c * 97) % 4000:][:1500]) + "\nContinue this text." for i in range(c)]
    ts = [threading.Thread(target=stream, args=(prompts[i], 128, recs[i])) for i in range(c)]
    [t.start() for t in ts]; [t.join() for t in ts]
    lo = max(r["start"] for r in recs); hi = min(r["end"] for r in recs)
    if hi <= lo:
        return float("nan")
    # Steady window: every request is decoding. Each contributes its own rate times its overlap with the window.
    tok = sum(r["n"] * (min(hi, r["end"]) - max(lo, r["start"])) / max(1e-6, r["end"] - r["start"]) for r in recs)
    return tok / (hi - lo)
stream("Say hi.", 4, {})
print(f"  {arm:4s} r{rnd}: " + " | ".join(f"C{c} {run(c):6.1f} tok/s" for c in (1, 4, 8)), flush=True)
if rnd == "1":
    P = ["Explain why the sky is blue in two sentences.", "Write a Python function that reverses a linked list.",
         "List three causes of the French Revolution.", "What is the capital of Australia and why was it chosen?",
         "Describe photosynthesis to a ten-year-old.", "Give a haiku about autumn rain.",
         "How does a hash map handle collisions?", "Summarize the plot of Hamlet in three sentences."]
    solo = []
    for p in P:
        r = {}; stream(p, 64, r); solo.append(r["text"])
    conc = [dict() for _ in P]
    ts = [threading.Thread(target=stream, args=(p, 64, conc[i])) for i, p in enumerate(P)]
    [t.start() for t in ts]; [t.join() for t in ts]
    json.dump({"same": sum(a == b["text"] for a, b in zip(solo, conc))}, open(idf, "w"))
EOF
}

for r in $(seq 1 "$ROUNDS"); do
  for arm in base test; do
    [ $arm = base ] && D=$BASE_DIR || D=$TEST_DIR
    serve "$D" "$OUT/serve-$arm-$r.log" || exit 1
    client $arm $r "$OUT/identity-$arm.json" | tee -a "$RES"
    stop
    log "  $arm r$r faults: ERROR=$(grep -ac ' ERROR ' "$OUT/serve-$arm-$r.log" || true) panics=$(grep -ac 'panicked' "$OUT/serve-$arm-$r.log" || true)"
  done
done
for arm in base test; do
  echo "  $arm: C8 replies identical to solo: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['same'])" "$OUT/identity-$arm.json")/8" | tee -a "$RES"
done
log "DONE"
