#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: the ST-995 accuracy leg (bfcl-subset: n=995, seed 42, temp 0, no overrides; Atlas's own runner, the
# same leg st995.sh runs against Atlas) against ANOTHER engine's OpenAI-compatible server, for the engine head-to-head.
#
#   scripts/devlab/st995_engine.sh <gufo|llamacpp> [27b|fnext] [KEY=VAL ...]
#   KEYS: WEIGHTS=q4|nvfp4 (llamacpp only; default q4 -- nvfp4 converts NVIDIA's checkpoint with llama.cpp's converter)
# Gufo serves GGUF only, so it runs the Unsloth UD-Q4_K_XL its guide pins. Thinking off, AR (no speculation), 1 session:
# the leg is sequential, and Gufo's quality contract makes greedy speculation output-identical to AR anyway.
set -uo pipefail
ENGINE=${1:?gufo|llamacpp}; MK=fnext; ARGS=("${@:2}")
[[ "${2:-}" =~ ^(27b|fnext)$ ]] && { MK=$2; ARGS=("${@:3}"); }
for kv in "${ARGS[@]}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
WEIGHTS=${WEIGHTS:-q4}; CTX=${CTX:-32768}; PORT=18085
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
model_spec "$MK" || exit 1; pyhf
Q4D=$(hf_get "$Q4_REPO" "$Q4_PAT" "$Q4_REV") || { log "MODEL DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; exit 1; }
TGT=$(first_gguf "$Q4D" "$Q4_PAT")

case $ENGINE in
gufo) gufo_build || exit 1; gufo_up $PORT 1 "$CTX" "$TGT" ;;
llamacpp)
  llama_build || exit 1
  if [ "$WEIGHTS" = nvfp4 ]; then llama_convert_nvfp4 "$NVFP4_REPO" && TGT=$NVFP4_GGUF || { log "no NVFP4 GGUF; stopping"; exit 1; }; fi
  llama_up $PORT 1 "$CTX" "$TGT" ;;
*) log "unknown engine $ENGINE"; exit 1 ;;
esac
log "engine $ENGINE, model $MK, weights $(basename "$TGT") ($(du -shL "$TGT" | cut -f1)), atlas ref $(git rev-parse --short=9 HEAD)"
wait_ready $PORT || { log "SERVER NOT READY"; ENGINE_DOWN "$OUT/serve.log"; tail -20 "$OUT/serve.log" | tee -a "$RES"; exit 1; }
log "serving model id '$MID'; bfcl-subset starting"
SHIM=""; for d in $(ls -dt "$PWD"/target/release/build/atlas-kernels-*/out 2>/dev/null); do ls "$d"/libcuda.so* >/dev/null 2>&1 && { SHIM="$d"; break; }; done
LD_LIBRARY_PATH="${SHIM:-}:${ROCM_PATH:-/opt/rocm}/lib:${LD_LIBRARY_PATH:-}" \
  "$PWD/target/release/spark" benchmark run bfcl-subset --yes --url "http://127.0.0.1:$PORT" --model "$MID" > "$OUT/bfcl.log" 2>&1 || true
if grep -aq "No module named 'soundfile'" "$OUT/bfcl.log"; then
  V=$(grep -aoE "/[^ ]*/bfcl/venv" "$OUT/bfcl.log" | head -1)
  RC=$(sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/bfcl.log" | grep -aoE "can be rescored: .*" | head -1 | sed -E 's/^can be rescored: //; s/ failed \(exit status.*$//')
  "$V/bin/python" -m pip --version >/dev/null 2>&1 || "$V/bin/python" -m ensurepip --upgrade >> "$OUT/pip.log" 2>&1
  "$V/bin/python" -m pip install soundfile >> "$OUT/pip.log" 2>&1
  [ -n "$RC" ] && bash -c "$RC" >> "$OUT/bfcl.log" 2>&1
fi
sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/bfcl.log" | grep -aE "^\s+[a-z_]+\s+(hallucination|live|non_live)\s+[0-9.]+|▸|overall [0-9.]+|overall_accuracy" | tee -a "$RES" \
  || { log "no BFCL table; tail:"; tail -15 "$OUT/bfcl.log" | tee -a "$RES"; }
ENGINE_DOWN "$OUT/serve.log"
log "DONE"
