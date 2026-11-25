#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# serve-inkling2-strix-linux: serve Thinking Machines Inkling 2 Mini on a Strix Halo (gfx1151) under
# native Linux + ROCm, from this checkout (strix-hip target kernels/strix-hip/inkling-2-mini).
#
#   MODE=spark|strict  DSPARK=0|1  PORT=8391  scripts/serve-inkling2-strix-linux.sh [--check]
#
# MODE=spark (default): the checkpoint plus Enntity/Inkling-2-Mini-Spark-NVFP4 (NVFP4 attention, lm_head
#   and layer 20-23 experts for decode rows; the checkpoint's own tensors for prefill rows), assembled
#   as an index-mapped model directory. MODE=strict: the checkpoint as shipped.
# DSPARK=1: the checkpoint's bundled DSpark drafter (mtp.safetensors). Default 0: measured on framework
#   (Strix Halo, C1, 512 tokens), DSpark only gains on Strict code (32.1 vs 28.2 tok/s) and loses
#   elsewhere (Strict prose 17.2; Spark 28.4 / 19.0 / 25.2 vs Spark plain 40.1).
# --check: start, wait for ready, send one request, print it, stop.
#
# Downloads (pinned revisions) go to the Hugging Face cache; DATA_DIR (default ~/.cache/atlas-inkling2)
# holds the assembled Spark directory and the drafter directory. Requires ROCm (hipcc), rustup cargo,
# python3, and an amdgpu with /dev/kfd. Prefix caching is on (Inkling 2 restores conv states from
# snapshots).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/target/release/spark"
MODE=${MODE:-spark} DSPARK=${DSPARK:-0} PORT=${PORT:-8391} NAME=${NAME:-inkling-2-mini}
DATA_DIR=${DATA_DIR:-$HOME/.cache/atlas-inkling2}
CKPT_REPO=thinkingmachines/Inkling-2-Mini CKPT_REV=ed40e7da625f594fc9c4990f5f66be24873c22d7
OV_REPO=Enntity/Inkling-2-Mini-Spark-NVFP4 OV_REV=e9df93dceea2fbf0927caf178104e868452a6292
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/rocm/bin:$PATH"
CHECK=0
for a in "$@"; do case "$a" in --check) CHECK=1 ;; *) echo "unknown arg: $a" >&2; exit 2 ;; esac; done
case "$MODE" in spark|strict) ;; *) echo "MODE must be spark or strict" >&2; exit 2 ;; esac
case "$DSPARK" in 0|1) ;; *) echo "DSPARK must be 0 or 1" >&2; exit 2 ;; esac

# --- pre-flight: GPU and memory ---------------------------------------------------------------------
[ -e /dev/kfd ] || { echo "refuse: /dev/kfd missing (amdgpu not loaded)" >&2; exit 3; }
GFX=$( (rocminfo 2>/dev/null || true) | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')
case " $GFX " in *" gfx1151 "*) ;; *) echo "refuse: no gfx1151 agent (rocminfo: ${GFX:-none})" >&2; exit 3 ;; esac
# GPU-visible memory: the larger of the VRAM carve-out and GTT (unified memory on Strix Halo).
GPU_GIB=0
for d in /sys/class/drm/card*/device; do
  for f in mem_info_vram_total mem_info_gtt_total; do
    [ -r "$d/$f" ] && { g=$(( $(cat "$d/$f") / 1073741824 )); [ "$g" -gt "$GPU_GIB" ] && GPU_GIB=$g; }
  done
done
# 64 GB-class boxes: at 0.88 a load OOM-killed the host; 0.82 is the measured-safe ceiling there.
if [ -z "${GPU_UTIL:-}" ]; then
  if [ "$GPU_GIB" -lt 80 ]; then GPU_UTIL=0.82; else GPU_UTIL=0.80; fi
fi
awk -v u="$GPU_UTIL" -v g="$GPU_GIB" 'BEGIN { exit !(g >= 80 || u <= 0.82) }' \
  || { echo "refuse: GPU_UTIL=$GPU_UTIL above 0.82 on a ${GPU_GIB} GiB box" >&2; exit 3; }
[ "$GPU_GIB" -ge 40 ] || { echo "refuse: ${GPU_GIB} GiB GPU-visible; Inkling 2 Mini needs ~21 GB weights plus KV" >&2; exit 3; }

# --- weights -------------------------------------------------------------------------------------------
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub
fetch() { python3 -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$1" "$2"; }
CKPT=$(fetch "$CKPT_REPO" "$CKPT_REV") || { echo "download failed: $CKPT_REPO" >&2; exit 4; }
MODEL_DIR=$CKPT
mkdir -p "$DATA_DIR"
if [ "$MODE" = spark ]; then
  OV=$(fetch "$OV_REPO" "$OV_REV") || { echo "download failed: $OV_REPO" >&2; exit 4; }
  MODEL_DIR=$DATA_DIR/spark-$OV_REV
  # the checkpoint's files linked, the overlay linked, and an index mapping the overlay's tensors to it
  python3 - "$CKPT" "$OV" "$MODEL_DIR" <<'PY' || exit 4
import json, os, struct, sys
ck, ov, md = sys.argv[1:4]
os.makedirs(md, exist_ok=True)
for f in os.listdir(ck):
    d = os.path.join(md, f)
    if f != "model.safetensors.index.json" and not os.path.lexists(d):
        os.symlink(os.path.realpath(os.path.join(ck, f)), d)
o = os.path.join(md, "spark-nvfp4.safetensors")
if not os.path.lexists(o):
    os.symlink(os.path.realpath(os.path.join(ov, "spark-nvfp4.safetensors")), o)
with open(o, "rb") as h:
    n = struct.unpack("<Q", h.read(8))[0]
    names = [k for k in json.loads(h.read(n)) if k != "__metadata__"]
idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))
idx["weight_map"].update({t: "spark-nvfp4.safetensors" for t in names})
json.dump(idx, open(os.path.join(md, "model.safetensors.index.json"), "w"), indent=1)
print(f"spark model dir: {md} ({len(names)} overlay tensors)", file=sys.stderr)
PY
fi
DRAFT_ARGS=()
if [ "$DSPARK" = 1 ]; then
  # #401's drafter layout: a directory with mtp.safetensors as model.safetensors and a config.json that
  # is the checkpoint's mtp_config plus vocab_size (text_config.unpadded_vocab_size)
  DRAFTER=$DATA_DIR/dspark-$CKPT_REV
  mkdir -p "$DRAFTER"
  ln -sfn "$(readlink -f "$CKPT/mtp.safetensors")" "$DRAFTER/model.safetensors"
  python3 - "$CKPT/config.json" "$DRAFTER/config.json" <<'PY' || exit 4
import json, sys
c = json.load(open(sys.argv[1]))
m = dict(c["mtp_config"]); m["vocab_size"] = c["text_config"]["unpadded_vocab_size"]
json.dump(m, open(sys.argv[2], "w"), indent=1)
PY
  DRAFT_ARGS=(--dflash --draft-model "$DRAFTER" --dflash-gamma 8 --num-drafts 7)
  export ATLAS_DFLASH_BATCH_DEPTH=${ATLAS_DFLASH_BATCH_DEPTH:-1:7,1000000:0}  # one sequence proposes at a time
fi

# --- binary ------------------------------------------------------------------------------------------------
BUILD_HEAD=$(git -C "$REPO" log -1 --format=%H -- . ':(exclude)*.md' ':(exclude)docs' ':(exclude)scripts')
if [ ! -x "$BIN" ] || [ ! -f "$REPO/.built-inkling2-$BUILD_HEAD" ]; then
  echo "=== build $REPO ($BUILD_HEAD, strix-hip inkling-2-mini) ===" >&2
  ( cd "$REPO" && ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL=inkling-2-mini ./build-amd.sh ) \
    || { echo "BUILD FAILED" >&2; exit 6; }
  rm -f "$REPO"/.built-inkling2-*; touch "$REPO/.built-inkling2-$BUILD_HEAD"
fi
# the HIP shims (libcuda / libcudart / libcublasLt) atlas-kernels builds into its OUT_DIR, no rpath
SHIM_DIR=$(ls -dt "$REPO"/target/release/build/atlas-kernels-*/out 2>/dev/null \
  | while read -r d; do [ -f "$d/libcuda.so" ] && { echo "$d"; break; }; done)
[ -n "$SHIM_DIR" ] || { echo "refuse: no HIP shim libcuda.so under target/release/build; rebuild" >&2; exit 6; }
export LD_LIBRARY_PATH="$SHIM_DIR:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "serve-inkling2-strix-linux sha=$(git -C "$REPO" rev-parse --short=9 HEAD) binsha=$(sha256sum "$BIN" | cut -c1-12)" \
     "mode=$MODE dspark=$DSPARK port=$PORT gpu_gib=$GPU_GIB util=$GPU_UTIL model=$MODEL_DIR" >&2
serve_args=(serve "$MODEL_DIR" --model-name "$NAME" --bind 127.0.0.1 --port "$PORT" --no-tui
  --max-seq-len "${MAX_SEQ_LEN:-32768}" --max-batch-size "${MAX_BATCH:-1}" --max-num-seqs "${MAX_BATCH:-1}"
  --kv-cache-dtype bf16 --gpu-memory-utilization "$GPU_UTIL" --max-prefill-tokens 4096
  --enable-prefix-caching true --request-timeout 0 "${DRAFT_ARGS[@]}")
CURL="curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23?\"}],\"max_tokens\":256,\"temperature\":0}'"
cd "$REPO"
run_serve() { exec env RUST_LOG=${RUST_LOG:-info} ATLAS_KV_EXTERNAL_RESERVE_GB=0 ATLAS_OOM_WATCHDOG_MB=400 ${SERVE_PREFIX:-} "$BIN" "${serve_args[@]}"; }
if [ "$CHECK" -eq 0 ]; then
  echo "ready when /v1/models answers; then try: $CURL" >&2
  run_serve
fi

LOG=$(mktemp "${TMPDIR:-/tmp}/inkling2-serve-check.XXXXXX.log")
( run_serve ) > "$LOG" 2>&1 &
SPID=$!
cleanup() { kill -INT "$SPID" 2>/dev/null; for _ in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done; kill -9 "$SPID" 2>/dev/null; }
trap cleanup EXIT
for _ in $(seq 1 360); do
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SPID" 2>/dev/null || { echo "serve died; log tail:" >&2; tail -20 "$LOG" >&2; exit 5; }
  sleep 5
done
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { echo "serve timeout; log tail:" >&2; tail -20 "$LOG" >&2; exit 5; }
echo "READY on :$PORT  ($(grep -aE 'KV cache:.*budget|Prefix caching' "$LOG" | sed -E 's/\x1b\[[0-9;]*m//g; s/^.*INFO [^ ]* //' | tr '\n' ' ' | cut -c1-240))"
echo "try: $CURL"
eval "$CURL" | python3 -c "import json,sys; r=json.load(sys.stdin); m=r['choices'][0]['message']; print('answer:', repr((m.get('content') or '')[:200]), '| usage:', r['usage'].get('completion_tokens'), 'tokens')"
