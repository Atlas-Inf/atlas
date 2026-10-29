#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: prefill A/B of the checked-out ref (already built by the workflow) against a base ref.
#
#   scripts/devlab/ab_prefill.sh <model> <base-ref> [rounds]
#   e.g.  scripts/devlab/ab_prefill.sh nvidia/Qwen3.8-27B-NVFP4 main 3
#
# Builds <base-ref> in a sibling worktree, then serves <model> from each build in turn (base, test, base, ...),
# measuring cold TTFT at ~2k / ~8k / ~14k tokens with distinct prompts, plus greedy 48-token probes compared
# base vs test. Results go to $DEVLAB_OUT/RESULTS.txt (the workflow publishes it to the job summary and
# uploads the directory as an artifact). Models download through the runner's HF mirror into $HF_HOME.
set -euo pipefail
MODEL=${1:?model}; BASE=${2:?base ref}; ROUNDS=${3:-3}
# Extra args of the form KEY=VALUE become env for the serves (serve-amd.sh knobs: GPU_UTIL, MAX_SEQ_LEN, ...).
for kv in "${@:4}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
# DevLab Strix Halo exposes 62.5 GB to HIP (the GTT limit) against 124 GB of system RAM. Since #240 the KV budget
# counts what Atlas already holds when free > total, so util is a share of 62.5 GB that must cover the ~42 GB of
# 27B weights + pools as well as KV: 0.30 (18.7 GB) now fails with "No memory left for KV cache" (run 40907522061),
# 0.88 serves (run 40846750527). Keep the strix box's proven 27B recipe (SSM_SLOTS=4, SSM_CKPT_INTERVAL=512, MAX_BATCH=1).
export GPU_UTIL=${GPU_UTIL:-0.88} MAX_SEQ_LEN=${MAX_SEQ_LEN:-26624} ATLAS_KV_EXTERNAL_RESERVE_GB=${ATLAS_KV_EXTERNAL_RESERVE_GB:-0} \
  SSM_SLOTS=${SSM_SLOTS:-4} SSM_CKPT_INTERVAL=${SSM_CKPT_INTERVAL:-512} MAX_BATCH=${MAX_BATCH:-1}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
TEST_DIR=$PWD; BASE_DIR="${RUNNER_TEMP:-/tmp}/devlab-base"
log "test $(git -C "$TEST_DIR" rev-parse --short=9 HEAD)  base $BASE  model $MODEL  rounds $ROUNDS  GPU_UTIL=$GPU_UTIL MAX_SEQ_LEN=$MAX_SEQ_LEN"

# --- model: spark serve reads the HF cache only, so fetch it first (through $HF_ENDPOINT into $HF_HOME;
# the cache sits on the runner's persistent scratch, so later runs skip the download) ---------------------
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub
t0=$(date +%s)
python3 - "$MODEL" <<'PY2' || { log "MODEL DOWNLOAD FAILED ($MODEL via ${HF_ENDPOINT:-hf.co})"; exit 1; }
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY2
log "model ready in $(( $(date +%s) - t0 ))s (HF_ENDPOINT=${HF_ENDPOINT:-unset} HF_HOME=${HF_HOME:-unset})"

# --- base build ---------------------------------------------------------------------------------------
rm -rf "$BASE_DIR"
git -C "$TEST_DIR" fetch -q --depth=1 https://github.com/Atlas-Inf/atlas.git "$BASE"
git -C "$TEST_DIR" worktree add -q --detach "$BASE_DIR" FETCH_HEAD
log "base sha $(git -C "$BASE_DIR" rev-parse --short=9 HEAD)"
( cd "$BASE_DIR" && ./build-amd.sh ) > "$OUT/base-build.log" 2>&1 || { log "BASE BUILD FAILED"; tail -30 "$OUT/base-build.log"; exit 1; }
log "builds: base $(sha256sum "$BASE_DIR/target/release/spark" | cut -c1-12)  test $(sha256sum "$TEST_DIR/target/release/spark" | cut -c1-12)"

PORT=18081
serve() { # $1 = repo dir, $2 = log
  ( cd "$1" && exec env RUST_LOG=info ATLAS_BIN="$1/target/release/spark" PORT=$PORT HOST=127.0.0.1 MODEL_NAME=m \
      ./serve-amd.sh "$MODEL" --disable-thinking --request-timeout 0 ) > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 1 720); do  # first serve may download the model through the mirror
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || { log "serve exited early ($2)"; tail -20 "$2"; return 1; }
    sleep 5
  done
  log "serve not ready after 60 min ($2)"; return 1
}
stop() { kill -INT $SPID 2>/dev/null || true; for _ in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  kill -9 $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true
  # wait for the GPU memory to come back before the next serve (a fixed sleep after SIGTERM froze a GB10 once)
  for _ in $(seq 1 60); do [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 40000 ] && break; sleep 2; done; }

client() { # $1 = arm, $2 = round, $3 = probe file (written on round 1)
  python3 - "$PORT" "$1" "$2" "$3" <<'PY'
import json, sys, time, urllib.request
port, arm, rnd, pf = sys.argv[1:5]
C = open("/usr/share/common-licenses/GPL-3").read().split()
def ask(p, mt):
    t = time.time(); o = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "m", "temperature": 0, "max_tokens": mt, "messages": [{"role": "user", "content": p}]}).encode(),
        headers={"Content-Type": "application/json"}), timeout=3600).read())
    return time.time() - t, o["usage"]["prompt_tokens"], o["choices"][0]["message"].get("content") or ""
ask("Say hi.", 4)
row = []
for n in (2048, 8192, 14000):
    s = (100 * int(rnd) + n) * 7919 % len(C); t, pt, _ = ask(" ".join((C * 9)[s:s + int(n * 0.75)]) + "\nFirst word?", 1)
    row.append(f"{pt}tok {t:.2f}s ({pt/t:.0f} tok/s)")
print(f"  {arm:4s} r{rnd}: " + " | ".join(row), flush=True)
if rnd == "1":
    probes = {"short": "Why is the sky blue? Answer in two sentences.",
              "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
              "2k": " ".join(C[:1500]) + "\nSummarize the text above in one sentence.",
              "10k": " ".join((C * 3)[:7500]) + "\nSummarize the text above in one sentence."}
    json.dump({k: ask(v, 48)[2] for k, v in probes.items()}, open(pf, "w"))
PY
}

for r in $(seq 1 "$ROUNDS"); do
  for arm in base test; do
    [ $arm = base ] && D=$BASE_DIR || D=$TEST_DIR
    serve "$D" "$OUT/serve-$arm-$r.log" || exit 1
    client $arm $r "$OUT/probes-$arm.json" | tee -a "$RES"
    stop
    log "  $arm r$r faults: ERROR=$(grep -ac ' ERROR ' "$OUT/serve-$arm-$r.log" || true) panics=$(grep -ac 'panicked' "$OUT/serve-$arm-$r.log" || true)"
  done
done
python3 - "$OUT" <<'PY' | tee -a "$RES"
import json, sys
o = sys.argv[1]; b = json.load(open(f"{o}/probes-base.json")); t = json.load(open(f"{o}/probes-test.json"))
for k in b:
    if b[k] == t[k]: print(f"  probe {k}: IDENTICAL")
    else:
        i = next((j for j, (x, y) in enumerate(zip(b[k], t[k])) if x != y), min(len(b[k]), len(t[k])))
        print(f"  probe {k}: DIVERGES at char {i}  base={b[k][:70]!r}  test={t[k][:70]!r}")
PY
log "DONE"
