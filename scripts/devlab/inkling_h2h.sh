#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: Inkling 2 Mini engine head-to-head on ONE runner, same client and prompts for every engine.
#
#   scripts/devlab/inkling_h2h.sh <engines...>     engines: atlas-spark atlas-strict ciru ciru-nospec llamacpp halogen
#
# Per engine: gufo_bench.py (pp ~2048 / tg128 at depth 0 and 8192, then C 1/2/4/8 aggregate decode, 384 tokens a
# request) and prefill_h2h.py (cold 8K / 32K prompts), thinking off by request (reasoning_effort none).
#   atlas-*      scripts/serve-inkling2-strix-linux.sh at this ref, MODE spark|strict, no drafter, 8 sequences
#   ciru         ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 v1.2.2 (ROCm vLLM 0.34 + its gfx1151 kernels), the
#                unmodified checkpoint (--original-weights), its concurrent mode (DSpark K=7) for every leg
#   ciru-nospec  the same with --no-spec
#   llamacpp     ggml-org/llama.cpp master (Inkling 2 merged in #32694 after the last release), GGML_HIP gfx1151,
#                ggml-org/Inkling-2-Mini-GGUF Q8_0, 8 slots
#   halogen      Peonist's halogen-flash-server asked for Inkling (it ships a Qwen3.8-Flash-Next pack): records the error
set -uo pipefail
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"; pyhf
export H2H_REQ_EXTRA='{"reasoning_effort":"none","chat_template_kwargs":{"reasoning_effort":"none"}}'
PORT=18086; HFC=${HF_HOME:-$HOME/.cache/huggingface}
log "ref $(git rev-parse --short=9 HEAD) engines: $*  host $(hostname) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
legs() { # label
  if ! wait_ready $PORT 720; then log "$1 SERVE FAILED"; ENGINE_DOWN "$OUT/serve-$1.log"; grep -aiE "error|fail|unsupported|not support" "$OUT/serve-$1.log" | tail -6 | cut -c1-300 | tee -a "$RES"; return 1; fi
  log "$1 serving (model id $MID)"
  python3 "$HERE/gufo_bench.py" $PORT "$1" "$OUT/$1.json" --depths 0,8192 --cs 1,2,4,8 --kinds mixed 2>&1 | grep -aE "^\s+\[" | tee -a "$RES"
  python3 "$HERE/prefill_h2h.py" $PORT "$1" 2>&1 | tail -3 | tee -a "$RES"
  ENGINE_DOWN "$OUT/serve-$1.log"
}
for e in "$@"; do
  case $e in
  atlas-spark|atlas-strict)
    ( export MODE=${e#atlas-} DSPARK=0 PORT MAX_BATCH=8 MAX_SEQ_LEN=40960 DATA_DIR=$HFC/atlas-inkling2; exec scripts/serve-inkling2-strix-linux.sh ) > "$OUT/serve-$e.log" 2>&1 &
    APID=$!; ENGINE_ALIVE() { kill -0 $APID 2>/dev/null; }; ENGINE_DOWN() { cp "$OUT/serve-$e.log" "$1" 2>/dev/null; kill -INT $APID 2>/dev/null; pkill -INT -P $APID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $APID 2>/dev/null || break; sleep 2; done; kill -9 $APID 2>/dev/null; sleep 5; true; }
    legs "$e" ;;
  ciru|ciru-nospec)
    CR=${RUNNER_TEMP:-/tmp}/ciru; mkdir -p "$CR"
    if [ ! -x "$CR/opt/serve.sh" ]; then
      command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > "$OUT/uv.log" 2>&1; export PATH=$HOME/.local/bin:$PATH; }
      rm -rf "$CR/src"; git clone -q https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151.git "$CR/src" && git -C "$CR/src" checkout -q v1.2.2 || { log "ciru clone failed"; continue; }
      log "ciru $(git -C "$CR/src" rev-parse --short=9 HEAD) (v1.2.2)"
      ( uv venv -q --python 3.12 "$CR/venv" && uv pip install -q --python "$CR/venv/bin/python" --extra-index-url https://wheels.vllm.ai/rocm/0.34.0/rocm100/ \
          'vllm==0.34.0+rocm100' 'huggingface-hub==2.4.0' 'safetensors==0.8.0' ) > "$OUT/ciru-venv.log" 2>&1 || { log "CIRU VENV FAILED: $(tail -3 "$OUT/ciru-venv.log" | cut -c1-200)"; continue; }
      CK=$(hf_get thinkingmachines/Inkling-2-Mini "*" ed40e7da625f594fc9c4990f5f66be24873c22d7) || { log "ckpt download failed"; continue; }
      ( cd "$CR/src" && bash scripts/install.sh --venv "$CR/venv" --transformers-deps "$CR/tdeps" --prefix "$CR/opt" ) > "$OUT/ciru-install.log" 2>&1 \
        || { log "CIRU INSTALL FAILED: $(grep -aiE 'error|fail' "$OUT/ciru-install.log" | tail -3 | cut -c1-200)"; continue; }
    fi
    CK=$(hf_get thinkingmachines/Inkling-2-Mini "*" ed40e7da625f594fc9c4990f5f66be24873c22d7)
    X=(); [ $e = ciru-nospec ] && X=(--no-spec)
    "$CR/opt/serve.sh" --model "$CK" --original-weights --mode concurrent --seqs 8 --context 65536 --port $PORT --served-model-name main "${X[@]}" > "$OUT/serve-$e.log" 2>&1 &
    CPID=$!; ENGINE_ALIVE() { kill -0 $CPID 2>/dev/null; }; ENGINE_DOWN() { cp "$OUT/serve-$e.log" "$1" 2>/dev/null; kill -INT $CPID 2>/dev/null; pkill -INT -P $CPID 2>/dev/null; sleep 10; pkill -9 -P $CPID 2>/dev/null; kill -9 $CPID 2>/dev/null; sleep 5; true; }
    legs "$e" ;;
  llamacpp)
    LLAMA_REF=master llama_build || continue
    GD=$(hf_get ggml-org/Inkling-2-Mini-GGUF "Inkling-2-Mini-Q8_0.gguf") || { log "gguf download failed"; continue; }
    llama_up $PORT 8 40960 "$GD/Inkling-2-Mini-Q8_0.gguf"
    legs llamacpp-q8 ;;
  halogen)
    HALOGEN_REPO=thinkingmachines/Inkling-2-Mini halogen_up || { log "halogen up failed"; continue; }
    PORT_SAVE=$PORT; PORT=8731; legs halogen-inkling; PORT=$PORT_SAVE ;;
  esac
done
log "DONE"
