#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: the ST-995 accuracy leg (bfcl-subset: n=995, seed 42, temp 0, no overrides; Atlas's own runner, the
# same leg st995.sh runs against Atlas) against ANOTHER engine's OpenAI-compatible server, for the engine head-to-head.
#
#   scripts/devlab/st995_engine.sh <atlas|gufo|llamacpp> [27b|fnext] [KEY=VAL ...]
#   KEYS: WEIGHTS=q4|nvfp4 (llamacpp only; default q4 -- nvfp4 converts NVIDIA's checkpoint with llama.cpp's converter)
#         ATLAS_MODE=recipe|refmode|think (atlas; see lib_engines.sh atlas_up), THINK=on (llamacpp: thinking on)
# atlas serves NVIDIA's NVFP4 checkpoint through serve-amd.sh at this ref (MAX_BATCH 1, GPU_UTIL 0.70).
# Gufo serves GGUF only, so it runs the Unsloth UD-Q4_K_XL its guide pins. Thinking off, AR (no speculation), 1 session:
# the leg is sequential, and Gufo's quality contract makes greedy speculation output-identical to AR anyway.
set -uo pipefail
ENGINE=${1:?atlas|gufo|llamacpp}; MK=fnext; ARGS=("${@:2}")
[[ "${2:-}" =~ ^(27b|fnext|ripple)$ ]] && { MK=$2; ARGS=("${@:3}"); }
for kv in "${ARGS[@]}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
WEIGHTS=${WEIGHTS:-q4}; CTX=${CTX:-32768}; PORT=18085
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
# The BFCL venv pip install hit "No space left on device" on the runner (41607542609): keep pip's temp and cache on the scratch disk.
export TMPDIR=${RUNNER_TEMP:-/tmp}/st995-tmp PIP_CACHE_DIR=${RUNNER_TEMP:-/tmp}/st995-pipcache; mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
log "disk: $(df -h "$HOME" /tmp "${RUNNER_TEMP:-/tmp}" 2>/dev/null | awk 'NR>1 {print $6"="$4" free"}' | sort -u | tr '\n' ' ')"
if [ "$MK" = ripple ]; then # atlas only: AR (no drafter), the runner's honest util floor after #351, Ripple's 8192 context
  [ "$ENGINE" = atlas ] || { log "ripple: atlas only"; exit 1; }
  NVFP4_REPO=ricursiveAI/Ripple-65B-Preview-NVFP4; CTX=${CTX_RIPPLE:-8192}; pyhf
else model_spec "$MK" || exit 1; pyhf; fi
if [ "$ENGINE" != atlas ]; then # the GGUF engines' weights; atlas serves NVIDIA's NVFP4 directly
  Q4D=$(hf_get "$Q4_REPO" "$Q4_PAT" "$Q4_REV") || { log "MODEL DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; exit 1; }
  TGT=$(first_gguf "$Q4D" "$Q4_PAT")
fi

case $ENGINE in
atlas)
  [ -x target/release/spark ] || { log "no target/release/spark"; exit 1; }
  if [ "$MK" = fnext ]; then # the mirror refuses the >50 GB PLE file through the hub client (42981375215, 43121079823): curl it
    BIG=model-fp8-mtp-ple.safetensors; pyhf
    M=$("${PY:-python3}" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], ignore_patterns=[sys.argv[2]]))' \
      "$NVFP4_REPO" "$BIG" 2>>"$OUT/download.log") && curl -fsSL -C - --retry 5 --retry-all-errors -o "$M/$BIG" \
      "${HF_ENDPOINT:-https://huggingface.co}/$NVFP4_REPO/resolve/main/$BIG" 2>>"$OUT/download.log" || { log "NVFP4 DOWNLOAD FAILED"; exit 1; }
    TGT=$M/config.json
  else
    hf_get "$NVFP4_REPO" "" >/dev/null || { log "NVFP4 DOWNLOAD FAILED"; exit 1; }
    TGT=$(hf_get "$NVFP4_REPO" "")/config.json
  fi
  # Flash-Next's 92 GB of weights do not fit 0.70 of the runner's 100 GB (43121108957); 0.80 is its carry-check level.
  atlas_up $PORT "$NVFP4_REPO" GPU_UTIL=${GPU_UTIL:-$(case $MK in fnext) echo 0.80 ;; ripple) echo 0.90 ;; *) echo 0.70 ;; esac)} MAX_SEQ_LEN=$CTX MAX_BATCH=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0
  log "atlas mode ${ATLAS_MODE:-recipe}" ;;
gufo) gufo_build || exit 1; gufo_up $PORT 1 "$CTX" "$TGT" ;;
llamacpp)
  llama_build || exit 1
  if [ "$WEIGHTS" = nvfp4 ]; then llama_convert_nvfp4 "$NVFP4_REPO" && TGT=$NVFP4_GGUF || { log "no NVFP4 GGUF; stopping"; exit 1; }; fi
  llama_up $PORT 1 "$CTX" "$TGT" ;;
*) log "unknown engine $ENGINE"; exit 1 ;;
esac
log "engine $ENGINE, model $MK, weights $(basename "$(dirname "$TGT")")/$(basename "$TGT"), atlas ref $(git rev-parse --short=9 HEAD), THINK=${THINK:-off}"
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
