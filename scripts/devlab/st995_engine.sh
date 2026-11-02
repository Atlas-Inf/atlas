#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: the ST-995 accuracy leg (bfcl-subset: n=995, seed 42, temp 0, no overrides; Atlas's own runner, the
# same leg st995.sh runs against Atlas) against ANOTHER engine's OpenAI-compatible server, for the engine comparison.
#
#   scripts/devlab/st995_engine.sh <gufo|llamacpp> [KEY=VAL ...]
# Model: Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL at revision 38bb39ee (the file gufo_flashnext.sh / llamacpp_gguf.sh
# benchmarked; Gufo has no NVFP4 path, and llama.cpp's NVFP4 conversion of NVIDIA's checkpoint is 180 GB). Thinking off,
# MAX_BATCH-equivalent 1 session (the leg is sequential). Results: the per-subset table + overall / normalized in RESULTS.txt.
set -uo pipefail
ENGINE=${1:?gufo|llamacpp}
for kv in "${@:2}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
REV=38bb39ee97821de2c9009abb7e93950eec396e66; CTX=${CTX:-32768}; PORT=18085
python3 -c "import huggingface_hub, hf_transfer" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
M=$(python3 -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download("unsloth/Qwen3.8-Flash-Next-GGUF", revision=sys.argv[1], allow_patterns=["UD-Q4_K_XL/*"]))' "$REV" 2>"$OUT/download.log") \
  || { log "MODEL DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; exit 1; }
TGT=$(ls "$M"/UD-Q4_K_XL/*.gguf | sort | head -1)
log "engine $ENGINE, model $(basename "$TGT") ($(du -shL "$M/UD-Q4_K_XL" | cut -f1)), atlas ref $(git rev-parse --short=9 HEAD)"

case $ENGINE in
gufo)
  docker volume create gufo-nix >/dev/null
  P=$(docker run --rm -v gufo-nix:/nix nixos/nix:2.30.2 sh -c 'nix --extra-experimental-features "nix-command flakes" build "github:gufo-org/gufo" --accept-flake-config --option build-users-group "" --option sandbox false --option extra-substituters https://gufo.cachix.org --option extra-trusted-public-keys gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE= --print-out-paths --no-link 2>/dev/null | tail -1')
  log "gufo $P"
  GR=$(getent group render | cut -d: -f3); GV=$(getent group video | cut -d: -f3)
  docker rm -f gufo-st >/dev/null 2>&1
  docker run -d --name gufo-st --network host --device /dev/kfd --device /dev/dri --group-add "$GV" --group-add "$GR" --security-opt seccomp=unconfined \
    -v gufo-nix:/nix -v "${HF_HOME:-$HOME/.cache/huggingface}":"${HF_HOME:-$HOME/.cache/huggingface}":ro nixos/nix:2.30.2 \
    "$P/bin/gufo" serve --host 127.0.0.1 --port $PORT llm --think off --model "$TGT" --sessions 1 --context "$CTX" > /dev/null
  stop() { docker logs gufo-st > "$OUT/serve.log" 2>&1; docker rm -f gufo-st >/dev/null 2>&1; } ;;
llamacpp)
  W=${RUNNER_TEMP:-/tmp}/st995-llamacpp; mkdir -p "$W"
  REF=$(curl -s -m 30 https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
  rm -rf "$W/llama.cpp"; git clone -q --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp.git "$W/llama.cpp"
  ( cd "$W/llama.cpp" && cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null 2>&1 && cmake --build build -j "$(nproc)" --target llama-server >/dev/null 2>&1 ) \
    || { log "llama.cpp BUILD FAILED"; exit 1; }
  log "llama.cpp $REF $(git -C "$W/llama.cpp" rev-parse --short=9 HEAD)${OT:+, -ot $OT}"
  "$W/llama.cpp/build/bin/llama-server" -m "$TGT" -ngl 99 ${OT:+-ot "$OT"} -c "$CTX" -np 1 --jinja --host 127.0.0.1 --port $PORT \
    --chat-template-kwargs '{"enable_thinking":false}' > "$OUT/serve.log" 2>&1 &
  SP=$!; stop() { kill -INT $SP 2>/dev/null; sleep 8; kill -9 $SP 2>/dev/null || true; } ;;
*) log "unknown engine $ENGINE"; exit 1 ;;
esac

MID=""
for _ in $(seq 1 360); do
  MID=$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -n "$MID" ] && curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2}" 2>/dev/null | grep -q choices && break
  MID=""; sleep 5
done
[ -n "$MID" ] || { log "SERVER NOT READY"; stop; tail -20 "$OUT/serve.log" | tee -a "$RES"; exit 1; }
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
stop
log "DONE"
