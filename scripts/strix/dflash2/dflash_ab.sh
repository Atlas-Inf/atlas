#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DFlash2-vs-MTP-vs-serial sustained A/B, one arm per invocation.
# Same harness shape as sustained_ab_ckpt.sh, but serving through
# ~/atlas-dflash2/serve-amd.sh so the arm is selected by env only.
#   usage: dflash_ab.sh <arm-tag> <outdir>
#   arms: serial | mtp_k4 | dflash_g8 | dflash_g4 | dflash_g6
set -uo pipefail
TAG=$1
OUT=$2
PORT=8093
MODEL=nvidia/Qwen3.8-27B-NVFP4
WT=/home/azeez/atlas-dflash2
BIN=${BIN:-$WT/target/release/spark}
export ATLAS_BIN=$BIN

export HF_HUB_OFFLINE=1 RUST_LOG=info ATLAS_MTP_ACCEPT_DEBUG=1
export GPU_UTIL=0.80 MAX_SEQ_LEN=8192 PORT=$PORT HOST=127.0.0.1 MODEL_NAME=$MODEL ATLAS_DFLASH_STEP_TIMING=1
case "$TAG" in
  serial)    export NUM_DRAFTS=0 ;;
  mtp_k4)    export NUM_DRAFTS=3 ;;
  dflash_g8) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 ;;
  dflash_g4) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 DFLASH_GAMMA=4 ;;
  dflash_g6) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 DFLASH_GAMMA=6 ;;
  ob_h256)   export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 ATLAS_DFLASH_OPTION_B=1 ;;
  ob_h128)   export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 ATLAS_DFLASH_OPTION_B=1 ;;
  legacy_h128) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 ;;
  ob_h128_gemv_g8) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 ;;
  ob_h128_gemv_g7) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 DFLASH_GAMMA=7 ;;
  ob_h128_gemv_g4) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 DFLASH_GAMMA=4 ;;
  ob_h128_gemv_g8_off) export DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 ATLAS_DFLASH_SMALL_M_GEMV=0 ;;
  *) echo "unknown arm $TAG"; exit 1 ;;
esac

LOG="$OUT/serve-$TAG.log"
mkdir -p "$OUT"
{
  echo "arm=$TAG"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "commit=$(cd $WT && git rev-parse HEAD)"
  echo "binary=$BIN"
  echo "binary_sha256=$(sha256sum "$BIN" | awk "{print \$1}")"
  echo "checkpoint=$MODEL"
  echo "env=HF_HUB_OFFLINE=1 RUST_LOG=info ATLAS_MTP_ACCEPT_DEBUG=1 GPU_UTIL=0.80 MAX_SEQ_LEN=8192 arm_env: NUM_DRAFTS=${NUM_DRAFTS:-unset} DFLASH=${DFLASH:-unset} DRAFT_MODEL=${DRAFT_MODEL:-unset} DFLASH_GAMMA=${DFLASH_GAMMA:-unset} ATLAS_DFLASH_OPTION_B=${ATLAS_DFLASH_OPTION_B:-unset} ATLAS_DFLASH_SMALL_M_GEMV=${ATLAS_DFLASH_SMALL_M_GEMV:-unset}"
  echo "serve=serve-amd.sh max-seq=8192 prefill=2048 util=.80 kv=bf16 head=nvfp4 batch=1 ssm-slots=0 checkpoint-interval=16 thinking=off"
  echo "harness=MinHeap warmup16 3x512 + prose/ json 512 temp0 seed0 reasoning none"
} | tee "$OUT/fingerprint-$TAG.txt"

pkill -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null; for _ in 1 2 3 4 5 6 7 8 9 10; do
  pgrep -f '[s]park[_a-zA-Z0-9.-]* serve' >/dev/null || break
  sleep 2
done
pkill -9 -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null; sleep 3
(cd "$WT" && ./serve-amd.sh "$MODEL" --disable-thinking --request-timeout 900) >"$LOG" 2>&1 &
SPARK_PID=$!
READY=0
for _ in $(seq 1 240); do
  kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED ($TAG)"; tail -60 "$LOG"; exit 1; }
  if ss -tlnp 2>/dev/null | grep ":$PORT " | grep -q "pid=$SPARK_PID," \
     && curl -fsS -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED ($TAG)"; tail -60 "$LOG"; exit 1; }
  sleep 2
done
if [ "$READY" != 1 ]; then echo "SERVER NOT READY ($TAG)"; tail -60 "$LOG"; exit 1; fi
echo "$TAG ready temp=$(/opt/rocm/bin/amd-smi metric --temperature 2>/dev/null | sed -n 's/.*EDGE: *//p' | head -1)"

python3 - "$PORT" "$MODEL" "$TAG" "$OUT" <<'PY'
import hashlib, json, statistics, sys, time, urllib.request

port, model, tag, outdir = sys.argv[1:]
MINHEAP = """Implement a complete, production-quality MinHeap class in Python. Include the methods insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, delete, __len__, and a validation method that checks the heap invariant. Include type hints, docstrings, clear error handling, complexity notes, and a compact executable test suite covering empty, singleton, duplicate, negative, and randomized inputs."""
PROSE = "Write a 600-word essay on the history of the printing press and its effect on literacy in Europe."
JSONP = "Return only a JSON object describing 5 fictional cities. Each city has fields: name, population (integer), founded (year), mayor, landmarks (array of exactly 3 strings). No prose, no markdown fences."
results = f"{outdir}/results.jsonl"


def request(prompt, max_tokens, prompt_tag, run):
    body = {
        "model": model, "stream": False, "temperature": 0.0, "seed": 0,
        "max_tokens": max_tokens, "reasoning_effort": "none",
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as response:
        out = json.load(response)
    wall = time.monotonic() - start
    usage = out.get("usage", {})
    details = usage.get("completion_tokens_details") or {}
    text = (out.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    with open(f"{outdir}/{tag}-{prompt_tag}-{run}.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    row = {
        "arm": tag, "prompt": prompt_tag, "run": run, "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "server_tps": usage.get("response_token/s"),
        "ttft_ms": usage.get("time_to_first_token_ms"),
        "accepted": details.get("accepted_prediction_tokens"),
        "finish": (out.get("choices") or [{}])[0].get("finish_reason"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_chars": len(text),
    }
    print(json.dumps(row), flush=True)
    with open(results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


request(MINHEAP, 16, "minheap", 0)          # warmup
rows = [request(MINHEAP, 512, "minheap", i) for i in (1, 2, 3)]
request(PROSE, 512, "prose", 1)
request(JSONP, 512, "json", 1)
tps = [r["server_tps"] for r in rows if isinstance(r["server_tps"], (int, float))]
summary = {
    "type": "summary", "arm": tag,
    "median_server_tps": statistics.median(tps) if tps else None,
    "range_server_tps": [min(tps), max(tps)] if tps else None,
    "text_hashes": [r["text_sha256"] for r in rows],
    "accepted": [r["accepted"] for r in rows],
}
print(json.dumps(summary), flush=True)
with open(results, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(summary) + "\n")
PY

grep -aE "verify_dflash_step|mtp_accept_debug|accept" "$LOG" \
  | sed "s/\x1b\[[0-9;]*m//g" > "$OUT/$TAG-accept.txt" || true
grep -aE "STEP_TIMING" "$LOG" | sed "s/\x1b[[0-9;]*m//g" > "$OUT/$TAG-steptiming.txt" || true

kill "$SPARK_PID" 2>/dev/null
for _ in 1 2 3 4 5 6 7 8; do kill -0 "$SPARK_PID" 2>/dev/null || break; sleep 2; done
kill -9 "$SPARK_PID" 2>/dev/null
wait "$SPARK_PID" 2>/dev/null
echo "=== arm $TAG done ==="
exit 0
