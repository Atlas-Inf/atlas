#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Shared helpers for the DevLab engine head-to-head scripts (h2h_speed.sh, st995_engine.sh): build Gufo / llama.cpp
# out of the box, fetch weights into the runner's HF cache, start and stop each engine's OpenAI server, wait for it.
# Source it after setting OUT and RES; every function logs through log().

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
HFC=${HF_HOME:-$HOME/.cache/huggingface}

# Weights per model key. 27B: Gufo's guide pins Unsloth UD-Q4_K_XL + the z-lab DFlash2 Q4_K_M draft; NVIDIA's NVFP4 is
# the checkpoint Atlas serves. fnext: the revision Gufo's Flash-Next guide pins.
model_spec() {
  case $1 in
    27b)   Q4_REPO=unsloth/Qwen3.8-27B-GGUF; Q4_PAT=Qwen3.8-27B-UD-Q4_K_XL.gguf; Q4_REV=""
           DRAFT_REPO=z-lab/Qwen3.8-27B-DFlash2-GGUF; DRAFT_PAT=Qwen3.8-27B-DFlash2-Q4_K_M.gguf
           NVFP4_REPO=nvidia/Qwen3.8-27B-NVFP4 ;;
    fnext) Q4_REPO=unsloth/Qwen3.8-Flash-Next-GGUF; Q4_PAT="UD-Q4_K_XL/*"; Q4_REV=38bb39ee97821de2c9009abb7e93950eec396e66
           DRAFT_REPO=""; DRAFT_PAT=""; NVFP4_REPO=nvidia/Qwen3.8-Flash-Next-NVFP4 ;;
    *) log "unknown model key $1"; return 1 ;;
  esac
}

pyhf() { # -> PY, a python with huggingface_hub + hf_transfer (system python if it has pip; else a venv, e.g. the qctl boxes)
  PY=python3
  if ! python3 -c "import huggingface_hub, hf_transfer" 2>/dev/null; then
    if python3 -m pip --version >/dev/null 2>&1; then python3 -m pip install --user -q --break-system-packages huggingface_hub hf_transfer
    else local v=$HOME/.cache/h2h-venv; [ -x "$v/bin/python" ] || python3 -m venv "$v"; "$v/bin/python" -c "import huggingface_hub, hf_transfer" 2>/dev/null || "$v/bin/pip" install -q huggingface_hub hf_transfer; PY=$v/bin/python; fi
  fi
  export HF_HUB_ENABLE_HF_TRANSFER=1
}
hf_get() { # repo pattern [rev] -> prints the snapshot dir
  "${PY:-python3}" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], revision=(sys.argv[3] or None), allow_patterns=([sys.argv[2]] if sys.argv[2] else None)))' "$1" "$2" "${3:-}" 2>>"$OUT/download.log"
}
first_gguf() { ls "$1"/$2 2>/dev/null | sort | head -1; }

gufo_build() { # -> GUFO (store path)
  docker pull -q nixos/nix:2.30.2 >/dev/null 2>&1; docker volume create gufo-nix >/dev/null
  docker run --rm -v gufo-nix:/nix nixos/nix:2.30.2 sh -c 'nix --extra-experimental-features "nix-command flakes" build "github:gufo-org/gufo" --accept-flake-config --option build-users-group "" --option sandbox false --option extra-substituters https://gufo.cachix.org --option extra-trusted-public-keys gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE= --print-out-paths --no-link 2>&1 | tail -3' > "$OUT/nix-build.log" 2>&1
  GUFO=$(grep -aE "^/nix/store/.*gufo" "$OUT/nix-build.log" | tail -1)
  [ -n "$GUFO" ] && log "gufo $GUFO" || { log "GUFO BUILD FAILED: $(tail -3 "$OUT/nix-build.log")"; return 1; }
}
gufo_up() { # port sessions context model [extra gufo serve args...]
  local port=$1 ses=$2 ctx=$3 m=$4; shift 4
  local GR GV; GR=$(getent group render | cut -d: -f3); GV=$(getent group video | cut -d: -f3)
  docker rm -f gufo-h2h >/dev/null 2>&1
  docker run -d --name gufo-h2h --network host --device /dev/kfd --device /dev/dri --group-add "$GV" --group-add "$GR" \
    --security-opt seccomp=unconfined -v gufo-nix:/nix -v "$HFC":"$HFC":ro nixos/nix:2.30.2 \
    "$GUFO/bin/gufo" serve --host 127.0.0.1 --port "$port" llm --think off --model "$m" --sessions "$ses" --context "$ctx" "$@" > /dev/null
  ENGINE_ALIVE() { docker ps -q -f name=gufo-h2h | grep -q .; }
  ENGINE_DOWN() { docker logs gufo-h2h > "$1" 2>&1; docker rm -f gufo-h2h >/dev/null 2>&1; }
}

llama_build() { # -> LB (bin dir), LREF
  local W=${RUNNER_TEMP:-/tmp}/llamacpp-h2h; mkdir -p "$W"
  LREF=${LLAMA_REF:-$(curl -s -m 30 https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")}
  rm -rf "$W/llama.cpp"; git clone -q --depth 1 --branch "$LREF" https://github.com/ggml-org/llama.cpp.git "$W/llama.cpp" || { log "llama.cpp CLONE FAILED ($LREF)"; return 1; }
  ( cd "$W/llama.cpp" && cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$OUT/cmake.log" 2>&1 \
    && cmake --build build -j "$(nproc)" --target llama-bench llama-server > "$OUT/build.log" 2>&1 ) \
    || { log "llama.cpp BUILD FAILED: $(grep -m3 -i error "$OUT/build.log" "$OUT/cmake.log" | cut -c1-200)"; return 1; }
  LB="$W/llama.cpp/build/bin"; LSRC="$W/llama.cpp"
  log "llama.cpp $LREF $(git -C "$W/llama.cpp" rev-parse --short=9 HEAD) (stock: GGML_HIP=ON, AMDGPU_TARGETS=gfx1151)"
}
llama_convert_nvfp4() { # repo -> NVFP4_GGUF (empty on failure); tries --outtype auto, then with --no-mtp
  local src gg venv=${RUNNER_TEMP:-/tmp}/llamacpp-h2h/venv
  python3 -m venv "$venv" && "$venv/bin/pip" install -q -r "$LSRC/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub hf_transfer > "$OUT/pip.log" 2>&1
  src=$(HF_HUB_ENABLE_HF_TRANSFER=1 "$venv/bin/python" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))' "$1" 2>>"$OUT/download.log") \
    || { log "NVFP4 DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; NVFP4_GGUF=""; return 1; }
  gg=${RUNNER_TEMP:-/tmp}/llamacpp-h2h/$(basename "$1")-from-nvfp4.gguf
  for args in "" "--no-mtp"; do
    rm -f "$gg"
    if "$venv/bin/python" "$LSRC/convert_hf_to_gguf.py" "$src" --outtype auto $args --outfile "$gg" > "$OUT/convert${args// /}.log" 2>&1; then
      log "NVFP4 CONVERT OK (--outtype auto $args): $(du -h "$gg" | cut -f1)"
      "$venv/bin/python" - "$gg" "$LSRC/gguf-py" <<'PY' | tee -a "$RES"
import sys, collections; sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); b = collections.Counter()
for t in r.tensors: b[t.tensor_type.name] += int(t.n_bytes)
print("  GGUF tensor types (GB):", ", ".join(f"{k} {v/1e9:.1f}" for k, v in b.most_common()))
PY
      NVFP4_GGUF=$gg; return 0
    fi
    log "NVFP4 CONVERT FAILED (--outtype auto $args): $(grep -aE "Error|error" "$OUT/convert${args// /}.log" | tail -1 | cut -c1-200)"
  done
  NVFP4_GGUF=""; return 1
}
llama_up() { # port parallel ctx-per-slot model [extra llama-server args...]
  local port=$1 np=$2 ctx=$3 m=$4; shift 4
  "$LB/llama-server" -m "$m" -ngl 99 -c $(( ctx * np )) -np "$np" --jinja --host 127.0.0.1 --port "$port" \
    --chat-template-kwargs "{\"enable_thinking\":$([ "${THINK:-off}" = on ] && echo true || echo false)}" "$@" > "$OUT/llama-server.log" 2>&1 &
  LPID=$!
  ENGINE_ALIVE() { kill -0 "$LPID" 2>/dev/null; }
  ENGINE_DOWN() { kill -INT "$LPID" 2>/dev/null; sleep 8; kill -9 "$LPID" 2>/dev/null; cp "$OUT/llama-server.log" "$1" 2>/dev/null; true; }
}

halogen_up() { # port -- Peonist's published halogen-flash-server image at its defaults (EULA s4 permits benchmarking and
  # publishing results), weights fetched by the image itself on first start into the runner's scratch.
  local port=$1 GR GV; GR=$(getent group render | cut -d: -f3); GV=$(getent group video | cut -d: -f3)
  local md=${HALOGEN_MODELS:-$(dirname "$HFC")/halogen-models}; mkdir -p "$md"
  docker rm -f halogen-h2h >/dev/null 2>&1
  docker run -d --name halogen-h2h -p "127.0.0.1:$port:8731" --device /dev/kfd --device /dev/dri --group-add "$GV" --group-add "$GR" \
    --ipc=host --ulimit memlock=-1:-1 -e HALOGEN_DOWNLOAD=${HALOGEN_REPO:-peonist-ai/halogen-qwen3.8-flash-next} \
    -e HALOGEN_KV_SLOTS=${SESSIONS:-8} ${HF_ENDPOINT:+-e HF_ENDPOINT=$HF_ENDPOINT} -v "$md":/models \
    "ghcr.io/peonist-ai/halogen-flash-server:${HALOGEN_TAG:-0.20.1}" > /dev/null
  ENGINE_ALIVE() { docker ps -q -f name=halogen-h2h | grep -q .; }
  ENGINE_DOWN() { docker logs halogen-h2h > "$1" 2>&1; docker rm -f halogen-h2h >/dev/null 2>&1; }
}

atlas_up() { # port model [KEY=VAL env...] -- Atlas through serve-amd.sh at this ref (the binary the workflow just built)
  # ATLAS_MODE: recipe (default; --disable-thinking), refmode (--reference-mode, client sampling + checkpoint template only;
  # thinking kept off through the default template kwarg, which is how the llama.cpp side runs), think (thinking forced on).
  local port=$1 m=$2; shift 2
  local mf=(--disable-thinking)
  case ${ATLAS_MODE:-recipe} in
    refmode) mf=(--reference-mode --default-chat-template-kwargs '{"enable_thinking":false}') ;;
    # dropping --disable-thinking alone leaves the 27B's thinking off (MODEL.toml thinking_default unset; the BFCL leg sends no
    # flag -- strix s200c reproduced the think-off score exactly), so turn it on through the default template kwarg
    think) mf=(--default-chat-template-kwargs '{"enable_thinking":true}') ;;
  esac
  ( exec env RUST_LOG=info ATLAS_BIN="$PWD/target/release/spark" PORT="$port" HOST=127.0.0.1 MODEL_NAME=m "$@" \
      ./serve-amd.sh "$m" "${mf[@]}" --request-timeout 0 --enable-prefix-caching ) > "$OUT/atlas-server.log" 2>&1 &
  APID=$!
  ENGINE_ALIVE() { kill -0 "$APID" 2>/dev/null; }
  ENGINE_DOWN() { kill -INT "$APID" 2>/dev/null; for _ in $(seq 1 90); do kill -0 "$APID" 2>/dev/null || break; sleep 2; done
    kill -9 "$APID" 2>/dev/null; cp "$OUT/atlas-server.log" "$1" 2>/dev/null
    for _ in $(seq 1 60); do [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 40000 ] && break; sleep 2; done; true; }
}

wait_ready() { # port [tries x5s, default 360] -> MID (served model id); 0 when a 2-token chat answers
  local port=$1
  for _ in $(seq 1 "${2:-360}"); do
    MID=$(curl -s -m 10 "http://127.0.0.1:$port/v1/models" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    [ -n "$MID" ] && curl -s -m 120 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}))' "$MID")" 2>/dev/null | grep -q choices && return 0
    ENGINE_ALIVE || return 1; sleep 5
  done
  return 1
}
