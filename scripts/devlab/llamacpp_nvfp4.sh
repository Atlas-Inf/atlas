#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: can stock llama.cpp run NVIDIA's NVFP4 checkpoint of a model on gfx1151, out of the box?
#
#   scripts/devlab/llamacpp_nvfp4.sh <hf-model> [KEY=VAL ...]
#   e.g.  scripts/devlab/llamacpp_nvfp4.sh nvidia/Qwen3.8-Flash-Next-NVFP4
#   KEYS: LLAMA_REF (default: latest release tag), CONVERT_ARGS (extra convert_hf_to_gguf.py flags, e.g. --no-mtp), FALLBACK_GGUF (repo:file-glob to bench if NVFP4 conversion fails)
#
# Steps, each logged to $DEVLAB_OUT/RESULTS.txt with its outcome:
#   1. build llama.cpp (release tag) with the HIP backend for gfx1151 (stock flags: GGML_HIP=ON, AMDGPU_TARGETS=gfx1151)
#   2. download the checkpoint and convert it with llama.cpp's own convert_hf_to_gguf.py (--outtype auto), then report the
#      tensor types in the GGUF (does NVFP4 survive as NVFP4?)
#   3. if a GGUF exists: llama-bench (pp2048 / tg128 at depth 0 and 8192, all layers on GPU), then one llama-server chat
#      reply as a coherence check
# It is a capability study, so failures are results: every step records what happened and the script continues where it can.
set -uo pipefail
MODEL=${1:?model}
for kv in "${@:2}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,*+-]*$ ]] && export "$kv"; done
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
W=${RUNNER_TEMP:-/tmp}/llamacpp-study; mkdir -p "$W"
S=/mnt/scratch/cache/llamacpp-study; mkdir -p "$S"

# ---- 1. llama.cpp build ------------------------------------------------------------------------------------
# Latest published release (the API's tag_name, e.g. b8xxx); a tag-pattern ls-remote picked up a stray non-release tag.
REF=${LLAMA_REF:-$(curl -s -m 30 https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")}
[ -n "$REF" ] || { log "could not resolve the latest llama.cpp release"; exit 1; }
log "llama.cpp ref $REF"
rm -rf "$W/llama.cpp"
git clone -q --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp.git "$W/llama.cpp" || { log "CLONE FAILED"; exit 1; }
log "llama.cpp sha $(git -C "$W/llama.cpp" rev-parse --short=9 HEAD)"
grep -c "NVFP4" "$W/llama.cpp/ggml/include/ggml.h" | sed 's/^/  ggml.h NVFP4 mentions: /' | tee -a "$RES"
grep -l "NVFP4\|nvfp4" "$W/llama.cpp/ggml/src/ggml-cuda/"*.cu* 2>/dev/null | wc -l | sed 's/^/  ggml-cuda (also the HIP backend) files mentioning nvfp4: /' | tee -a "$RES"
( cd "$W/llama.cpp" && cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$OUT/cmake.log" 2>&1 \
  && cmake --build build -j "$(nproc)" --target llama-bench llama-server llama-cli llama-gguf > "$OUT/build.log" 2>&1 ) \
  && log "BUILD OK" || { log "BUILD FAILED: $(grep -m3 -iE 'error' "$OUT/build.log" "$OUT/cmake.log" | cut -c1-200)"; exit 1; }
B="$W/llama.cpp/build/bin"

# ---- 2. convert NVIDIA's NVFP4 checkpoint ------------------------------------------------------------------
python3 -m venv "$W/venv" && . "$W/venv/bin/activate" && pip install -q --upgrade pip >/dev/null 2>&1
pip install -q -r "$W/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub hf_transfer > "$OUT/pip.log" 2>&1 \
  || log "pip install of convert requirements failed: $(tail -2 "$OUT/pip.log")"
# The checkpoint has single shards past huggingface_hub's plain-download limit ("Use hf_transfer or hf_xet", run 41406025913).
export HF_HUB_ENABLE_HF_TRANSFER=1
t0=$(date +%s)
SRC=$(python3 -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$MODEL" 2>"$OUT/download.log") \
  || { log "MODEL DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; exit 1; }
log "model ready in $(( $(date +%s) - t0 ))s at $SRC ($(du -shL "$SRC" | cut -f1))"
GG="$S/$(basename "$MODEL")-from-nvfp4.gguf"; rm -f "$GG"
t0=$(date +%s)
log "convert args: --outtype auto ${CONVERT_ARGS:-}"
if python3 "$W/llama.cpp/convert_hf_to_gguf.py" "$SRC" --outtype auto ${CONVERT_ARGS:-} --outfile "$GG" > "$OUT/convert.log" 2>&1; then
  log "CONVERT OK in $(( $(date +%s) - t0 ))s: $(du -h "$GG" | cut -f1)"
  python3 - "$GG" <<'PY' | tee -a "$RES"
import sys, collections
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); c = collections.Counter(); b = collections.Counter()
for t in r.tensors:
    c[t.tensor_type.name] += 1; b[t.tensor_type.name] += int(t.n_bytes)
print("  GGUF tensor types (count, GB):", ", ".join(f"{k} {c[k]} {b[k]/1e9:.1f}" for k in sorted(c, key=lambda k: -b[k])))
PY
else
  log "CONVERT FAILED after $(( $(date +%s) - t0 ))s; tail:"; tail -12 "$OUT/convert.log" | tee -a "$RES"; GG=""
fi
deactivate 2>/dev/null || true

# ---- 3. bench + coherence ----------------------------------------------------------------------------------
if [ -z "$GG" ] && [ -n "${FALLBACK_GGUF:-}" ]; then
  repo=${FALLBACK_GGUF%%:*}; glob=${FALLBACK_GGUF#*:}
  . "$W/venv/bin/activate"
  D=$(python3 -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=[sys.argv[2]]))" "$repo" "$glob" 2>>"$OUT/download.log")
  deactivate; GG=$(ls "$D"/$glob 2>/dev/null | sort | head -1); log "FALLBACK GGUF: $repo $glob -> ${GG:-none}"
fi
if [ -n "$GG" ]; then
  "$B/llama-bench" -m "$GG" -ngl 99 -p 2048 -n 128 -d 0,8192 -r 2 -o md > "$OUT/bench.md" 2> "$OUT/bench.err" \
    && { log "llama-bench:"; cat "$OUT/bench.md" | tee -a "$RES"; } \
    || { log "llama-bench FAILED:"; tail -8 "$OUT/bench.err" | tee -a "$RES"; }
  PORT=18082
  "$B/llama-server" -m "$GG" -ngl 99 -c 8192 --jinja --port $PORT --host 127.0.0.1 > "$OUT/server.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In two sentences, why is the sky blue?"}],"max_tokens":96,"temperature":0}' > "$OUT/chat.json" 2>&1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('  chat reply:', repr(d['choices'][0]['message']['content'][:300]))" "$OUT/chat.json" 2>/dev/null | tee -a "$RES" \
    || { log "chat request failed:"; head -c 400 "$OUT/chat.json" | tee -a "$RES"; tail -5 "$OUT/server.log" | tee -a "$RES"; }
  kill -INT $SP 2>/dev/null; sleep 5; kill -9 $SP 2>/dev/null || true
fi
log "DONE"
