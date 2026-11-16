#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: stock llama.cpp on a GGUF quant of a model, on the same serving protocol Atlas and Gufo are measured
# with (scripts/devlab/gufo_bench.py: pp ~2048 / tg128 at depth 0/8192/16384, C 1/4/8), plus llama-bench.
#
#   scripts/devlab/llamacpp_gguf.sh <hf-gguf-repo> <quant-subdir> [KEY=VAL ...]
#   e.g.  scripts/devlab/llamacpp_gguf.sh unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL   (the workflow's args filter allows no '*')
#   KEYS: LLAMA_REF (default latest release), REV (repo revision), CTX (default 32768), PARALLEL (default 8), OT (an -ot override;
#         default: none, then per_layer_token_embd=CPU if the all-GPU load fails)
# Flash-Next's n-gram embedding table (per_layer_token_embd) alone is ~100 GB in BF16; a GGUF quant plus that table may exceed
# the 100 GB this runner maps to the GPU, so the script first tries everything on the GPU and then the CPU-resident table.
set -uo pipefail
REPO=${1:?gguf repo}; SUB=${2:?quant subdir}; GLOB="$SUB/*.gguf"
for kv in "${@:3}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,*=+-]*$ ]] && export "$kv"; done
CTX=${CTX:-32768}; PARALLEL=${PARALLEL:-8}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
HERE=$(cd "$(dirname "$0")" && pwd)
W=${RUNNER_TEMP:-/tmp}/llamacpp-gguf; mkdir -p "$W"

REF=${LLAMA_REF:-$(curl -s -m 30 https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")}
rm -rf "$W/llama.cpp"; git clone -q --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp.git "$W/llama.cpp" || { log "CLONE FAILED ($REF)"; exit 1; }
log "llama.cpp $REF $(git -C "$W/llama.cpp" rev-parse --short=9 HEAD)"
( cd "$W/llama.cpp" && cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$OUT/cmake.log" 2>&1 \
  && cmake --build build -j "$(nproc)" --target llama-bench llama-server > "$OUT/build.log" 2>&1 ) && log "BUILD OK" \
  || { log "BUILD FAILED: $(grep -m3 -i error "$OUT/build.log" "$OUT/cmake.log" | cut -c1-200)"; exit 1; }
B="$W/llama.cpp/build/bin"

python3 -m venv "$W/venv" && . "$W/venv/bin/activate" && pip install -q huggingface_hub hf_transfer > "$OUT/pip.log" 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
t0=$(date +%s)
D=$(python3 -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=(sys.argv[3] or None), allow_patterns=[sys.argv[2]]))" "$REPO" "$GLOB" "${REV:-}" 2>"$OUT/download.log") \
  || { log "DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; exit 1; }
deactivate
GG=$(ls "$D"/$GLOB 2>/dev/null | sort | head -1); [ -n "$GG" ] || { log "no GGUF matched $GLOB in $D"; exit 1; }
log "model ready in $(( $(date +%s) - t0 ))s: $(du -shL "$(dirname "$GG")" | cut -f1) ($(basename "$GG"))"

PORT=18084
start_server() { # $1 = -ot value or ""
  "$B/llama-server" -m "$GG" -ngl 99 ${1:+-ot "$1"} -c $(( CTX * PARALLEL )) -np "$PARALLEL" --jinja --host 127.0.0.1 --port $PORT \
    --chat-template-kwargs '{"enable_thinking":false}' > "$OUT/server${1:+-ot}.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && return 0; kill -0 $SP 2>/dev/null || return 1; sleep 5; done
  return 1
}
OTV=${OT:-}
if start_server "$OTV"; then log "llama-server up (-ngl 99${OTV:+ -ot $OTV})"
else
  log "llama-server failed to load with -ngl 99${OTV:+ -ot $OTV}: $(grep -aiE 'error|alloc|out of memory|failed' "$OUT"/server*.log | tail -2 | cut -c1-200)"
  kill -9 $SP 2>/dev/null; sleep 5; OTV="per_layer_token_embd=CPU"
  start_server "$OTV" && log "llama-server up (-ngl 99 -ot $OTV)" || { log "llama-server failed again: $(tail -3 "$OUT/server-ot.log" | cut -c1-200)"; exit 1; }
fi
python3 "$HERE/gufo_bench.py" $PORT "llamacpp" "$OUT/llamacpp.json" --depths 0,8192,16384 --cs 1,4,8 --kinds mixed 2>&1 | grep -aE "^\s+\[" | tee -a "$RES"
kill -INT $SP 2>/dev/null; sleep 8; kill -9 $SP 2>/dev/null || true
"$B/llama-bench" -m "$GG" -ngl 99 ${OTV:+-ot "$OTV"} -p 2048 -n 128 -d 0,8192 -r 2 -o md > "$OUT/bench.md" 2> "$OUT/bench.err" \
  && { log "llama-bench:"; cat "$OUT/bench.md" | tee -a "$RES"; } || { log "llama-bench FAILED:"; tail -5 "$OUT/bench.err" | tee -a "$RES"; }
log "DONE"
