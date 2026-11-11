#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# serve-nemotron-gb10.sh — single serve entry for Nemotron 3.5 Super VL on GB10.
#
#   serve-nemotron-gb10.sh [--spec off|g3|g8] [--mode reference|atlas]
#                          [--port N] [--check] [--allow-dirty]
#
# --check: boot in the chosen mode/spec, run the 3 golden prompts, compare
# content + tool calls against scripts/golden/nemotron-gb10/<mode>/*.json,
# stop the server, exit 0 only if all pass.
set -u

MODE_DEFAULT=atlas

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/target/release/spark"
MODEL=${MODEL:-$HOME/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}
DRAFTER=${DRAFTER:-$HOME/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr}
NAME=nemotron-3.5-super-vl

# Toolchain + runtime env of the verified builds/serves (spark2 build_env.sh):
# CUDA 13 toolchain for the build, NCCL + CUDA libs for the server at runtime.
NCCL_LIB=${NCCL_LIB:-/home/azeez/code/mlperf-venv/lib/python3.12/site-packages/nvidia/nccl/lib}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=13000
export LIBRARY_PATH="$NCCL_LIB:/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NCCL_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

SPEC=g8 MODE=$MODE_DEFAULT PORT=8391 CHECK=0 ALLOW_DIRTY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --spec) SPEC=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --check) CHECK=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "$SPEC" in off|g3|g8) ;; *) echo "--spec must be off|g3|g8" >&2; exit 2 ;; esac
case "$MODE" in reference|atlas) ;; *) echo "--mode must be reference|atlas" >&2; exit 2 ;; esac

HEAD=$(git -C "$REPO" rev-parse HEAD)
TAG=$(git -C "$REPO" describe --tags --exact-match HEAD 2>/dev/null || echo untagged)
if [ "$ALLOW_DIRTY" -eq 0 ] && ! git -C "$REPO" diff --quiet HEAD -- 2>/dev/null; then
  echo "refuse: dirty tracked files in $REPO (pass --allow-dirty to override)" >&2
  exit 3
fi

if [ ! -x "$BIN" ] || [ ! -f "$REPO/.built-$HEAD" ]; then
  rm -f "$REPO"/.built-*
  echo "=== build $REPO ($HEAD, all targets) ===" >&2
  ( cd "$REPO" && ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' \
      cargo build --release -p spark-server ) || { echo "BUILD FAILED" >&2; exit 6; }
  touch "$REPO/.built-$HEAD"
fi
BINSHA=$(sha256sum "$BIN" | awk '{print $1}')

# Exactly serve_n35vl's flags: mode/spec only change the trailing args.
serve_args=(
  serve "$MODEL" --model-name "$NAME"
  --bind 127.0.0.1 --port "$PORT" --no-tui
  --max-seq-len 32768 --max-prefill-tokens 8192 --kv-cache-dtype bf16
  --gpu-memory-utilization 0.85 --max-num-seqs 1 --max-batch-size 1
  --enable-prefix-caching true
)
case "$MODE" in
  reference) serve_args+=(--reference-mode --default-chat-template-kwargs '{"enable_thinking":false}') ;;
  atlas) serve_args+=(--disable-thinking) ;;
esac
case "$SPEC" in
  g3|g8) serve_args+=(--dflash --draft-model "$DRAFTER" --dflash-gamma "${SPEC#g}") ;;
esac

banner() {
  echo "serve-nemotron-gb10 tag=$TAG sha=$HEAD binsha=$BINSHA mode=$MODE spec=$SPEC port=$PORT model=$MODEL drafter=$DRAFTER"
}
banner | tee /dev/stderr

cd "$REPO"  # jinja-templates/ resolve repo-relative

run_serve() {
  exec env RUST_LOG=info ATLAS_MTP_ACCEPT_DEBUG=1 "$BIN" "${serve_args[@]}"
}

if [ "$CHECK" -eq 0 ]; then
  run_serve
fi

# ---- --check: background boot + golden compare ----
LOG=$(mktemp /tmp/n35vl-serve-check.XXXXXX.log)
( run_serve ) > "$LOG" 2>&1 &
SPID=$!
cleanup() { kill -INT "$SPID" 2>/dev/null; sleep 2; kill -9 "$SPID" 2>/dev/null; }
trap cleanup EXIT
ok=0
for _ in $(seq 1 240); do
  if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}],\"max_tokens\":2,\"temperature\":0}" \
      | grep -q '"choices"' && { ok=1; break; }
  fi
  kill -0 "$SPID" 2>/dev/null || { echo "serve died; log:" >&2; tail -20 "$LOG" >&2; exit 4; }
  sleep 5
done
[ "$ok" -eq 1 ] || { echo "serve timeout; log:" >&2; tail -20 "$LOG" >&2; exit 5; }

python3 - "$REPO" "$MODE" "$PORT" "$NAME" <<'PY'
import json, sys, glob, urllib.request, os
repo, mode, port, model = sys.argv[1:5]
pd = f"{repo}/scripts/golden/nemotron-gb10/prompts"
gd = f"{repo}/scripts/golden/nemotron-gb10/{mode}"
fail = 0
for pf in sorted(glob.glob(pd + "/*.json")):
    name = os.path.basename(pf)[:-5]
    body = json.load(open(pf))
    body["model"] = model
    body["temperature"] = 0
    r = urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
        timeout=600)
    resp = json.load(r)
    msg = resp["choices"][0]["message"]
    got = {"content": msg.get("content"), "tool_calls": msg.get("tool_calls")}
    gf = f"{gd}/{name}.json"
    if not os.path.exists(gf):
        print(f"{name}: FAIL no golden {gf}"); fail = 1; continue
    want = json.load(open(gf))
    if got == want:
        print(f"{name}: PASS"); continue
    fail = 1
    a = json.dumps(got, sort_keys=True); b = json.dumps(want, sort_keys=True)
    i = next((j for j, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
    print(f"{name}: FAIL first-diff-char {i}")
    print(f"  got : {a[max(0,i-80):i+80]}")
    print(f"  want: {b[max(0,i-80):i+80]}")
sys.exit(fail)
PY
exit $?
