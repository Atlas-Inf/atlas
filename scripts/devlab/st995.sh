#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: the ST-995 accuracy leg (bfcl-subset: n=995, seed 42, temp 0, no overrides) on the checked-out
# ref (already built by the workflow), for one checkpoint.
#
#   scripts/devlab/st995.sh <model> [KEY=VAL ...]
#   e.g.  scripts/devlab/st995.sh nvidia/Qwen3.8-27B-NVFP4
#         scripts/devlab/st995.sh unsloth/Qwen3.8-27B-NVFP4 LM_HEAD=bf16
#
# Serves <model> through the tree's serve-amd.sh at MAX_BATCH 1 (the leg is sequential), runs
# `spark benchmark run bfcl-subset`, and puts the per-subset table and the overall / normalized line in
# $DEVLAB_OUT/RESULTS.txt. This lets the runner answer checkpoint-vs-engine accuracy questions in parallel with the
# boxes: two dispatches on one commit, one per checkpoint.
set -euo pipefail
MODEL=${1:?model}
for kv in "${@:2}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
export GPU_UTIL=${GPU_UTIL:-0.88} MAX_SEQ_LEN=${MAX_SEQ_LEN:-26624} ATLAS_KV_EXTERNAL_RESERVE_GB=${ATLAS_KV_EXTERNAL_RESERVE_GB:-0} \
  SSM_SLOTS=${SSM_SLOTS:-4} SSM_CKPT_INTERVAL=${SSM_CKPT_INTERVAL:-512} MAX_BATCH=${MAX_BATCH:-1} NUM_DRAFTS=${NUM_DRAFTS:-0}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
log "ref $(git rev-parse --short=9 HEAD)  model $MODEL  LM_HEAD=${LM_HEAD:-default} GPU_UTIL=$GPU_UTIL"

python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub
t0=$(date +%s)
python3 -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$MODEL" \
  || { log "MODEL DOWNLOAD FAILED ($MODEL via ${HF_ENDPOINT:-hf.co})"; exit 1; }
log "model ready in $(( $(date +%s) - t0 ))s"

PORT=18081
( exec env RUST_LOG=info ATLAS_BIN="$PWD/target/release/spark" PORT=$PORT HOST=127.0.0.1 MODEL_NAME=m \
    ./serve-amd.sh "$MODEL" --disable-thinking --request-timeout 0 ) > "$OUT/serve.log" 2>&1 &
SPID=$!
for _ in $(seq 1 720); do
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { log "serve exited early"; tail -20 "$OUT/serve.log"; exit 1; }
  sleep 5
done
log "serving; bfcl-subset starting"
# The CLI links the same HIP shims as the server (libcuda/libcudart/libcublasLt from atlas-kernels' OUT_DIR), which
# serve-amd.sh puts on LD_LIBRARY_PATH for its own process only; same discovery as serve-amd.sh.
SHIM=""
for d in $(ls -dt "$PWD"/target/release/build/atlas-kernels-*/out 2>/dev/null); do
  if ls "$d"/libcuda.so* >/dev/null 2>&1; then SHIM="$d"; break; fi
done
LD_LIBRARY_PATH="${SHIM:-}:${ROCM_PATH:-/opt/rocm}/lib:${LD_LIBRARY_PATH:-}" \
  "$PWD/target/release/spark" benchmark run bfcl-subset --yes --url "http://127.0.0.1:$PORT" --model m > "$OUT/bfcl.log" 2>&1 || true
# bfcl-eval's scorer imports qwen_agent, which imports soundfile, which the provisioned venv does not carry: run
# 41187711625 generated all 995 responses and then failed to score ("No module named 'soundfile'"). The runner keeps
# responses.jsonl and prints the rescore command, so install the module into that venv and rescore in place.
if grep -aq "No module named 'soundfile'" "$OUT/bfcl.log"; then
  V=$(grep -aoE "/[^ ]*/bfcl/venv" "$OUT/bfcl.log" | head -1)
  # The runner prints "...can be rescored: <cmd> failed (exit status: 1):", so cut the trailing status off the command.
  RC=$(sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/bfcl.log" | grep -aoE "can be rescored: .*" | head -1 | sed -E 's/^can be rescored: //; s/ failed \(exit status.*$//')
  log "scorer missing soundfile; installing into $V and rescoring"
  # The provisioned venv may have no pip (run 41212989922: the quiet install changed nothing), so bootstrap it first.
  "$V/bin/python" -m pip --version >/dev/null 2>&1 || "$V/bin/python" -m ensurepip --upgrade >> "$OUT/pip.log" 2>&1 || true
  "$V/bin/python" -m pip install soundfile >> "$OUT/pip.log" 2>&1 \
    || { command -v uv >/dev/null && uv pip install --python "$V/bin/python" soundfile >> "$OUT/pip.log" 2>&1; } || true
  "$V/bin/python" -c "import soundfile" 2>> "$OUT/pip.log" && log "soundfile importable" || { log "soundfile STILL missing; pip log:"; tail -5 "$OUT/pip.log" | tee -a "$RES"; }
  [ -n "$RC" ] && bash -c "$RC" > "$OUT/rescore.log" 2>&1; cat "$OUT/rescore.log" >> "$OUT/bfcl.log" 2>/dev/null || true
fi
sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/bfcl.log" | grep -aE "^\s+[a-z_]+\s+(hallucination|live|non_live)\s+[0-9.]+|▸|overall [0-9.]+" | tee -a "$RES" \
  || { log "no BFCL table in the log; tail:"; tail -15 "$OUT/bfcl.log" | tee -a "$RES"; }
kill -INT $SPID 2>/dev/null || true; for _ in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null || true
log "faults: ERROR=$(grep -ac ' ERROR ' "$OUT/serve.log" || true) panics=$(grep -ac 'panicked' "$OUT/serve.log" || true)"
log "DONE"
