#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: Inkling 2 Mini engine head-to-head on ONE runner, same client and prompts for every engine.
#
#   scripts/devlab/inkling_h2h.sh <engines...>     engines: atlas-spark atlas-strict ciru ciru-nospec llamacpp halogen
#
# Per engine: decode_h2h.py (non-streaming C1 decode on prose / code / json, C 1/2/4/8 aggregate) and prefill_h2h.py (cold 8K / 32K prompts), thinking off by request (reasoning_effort none).
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
agentic_leg() { # label model-id
  SHIM_DIR=$(ls -dt target/release/build/atlas-kernels-*/out 2>/dev/null | while read -r d; do [ -f "$d/libcuda.so" ] && { echo "$d"; break; }; done)
  LD_LIBRARY_PATH="$SHIM_DIR:/opt/rocm/lib" target/release/spark benchmark run agentic-webserver --url http://127.0.0.1:$PORT --model "$2" --yes \
    --param iterations=10 --param wall_budget_s=100000 --format json --quiet --no-fail-on-verdict > "$OUT/$1-agentic.json" 2> "$OUT/$1-agentic.stderr" || log "$1 agentic rc!=0: $(tail -2 "$OUT/$1-agentic.stderr" | tr '\n' ' ')"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT', sys.argv[2], d.get('benchmark_id'), d.get('run_id'), [(s['label'], s['value']) for s in d['frame']['summary']])" "$OUT/$1-agentic.json" "$1" 2>&1 | tail -1 | tee -a "$RES"
}
legs() { # label
  if ! wait_ready $PORT 720; then log "$1 SERVE FAILED"; ENGINE_DOWN "$OUT/serve-$1.log"; grep -aiE "error|fail|unsupported|not support" "$OUT/serve-$1.log" | tail -6 | cut -c1-300 | tee -a "$RES"; return 1; fi
  log "$1 serving (model id $MID)"
  python3 "$HERE/decode_h2h.py" $PORT "$1" 2>&1 | grep -aE "^\s+\[|Error|error" | tee -a "$RES"
  python3 "$HERE/prefill_h2h.py" $PORT "$1" 2>&1 | tail -3 | tee -a "$RES"
  ENGINE_DOWN "$OUT/serve-$1.log"
}
for e in "$@"; do
  case $e in
  atlas-spark|atlas-strict)
    ( export MODE=${e#atlas-} DSPARK=0 PORT MAX_BATCH=8 MAX_SEQ_LEN=40960 DATA_DIR=$HFC/atlas-inkling2; exec scripts/serve-inkling2-strix-linux.sh ) > "$OUT/serve-$e.log" 2>&1 &
    APID=$!; ENGINE_ALIVE() { kill -0 $APID 2>/dev/null; }; ENGINE_DOWN() { cp "$OUT/serve-$e.log" "$1" 2>/dev/null; kill -INT $APID 2>/dev/null; pkill -INT -P $APID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $APID 2>/dev/null || break; sleep 2; done; kill -9 $APID 2>/dev/null; sleep 5; true; }
    legs "$e" ;;
  ciru|ciru-conc|ciru-orig|ciru-nospec|ciru-st995|ciru-agentic)
    CR=${RUNNER_TEMP:-/tmp}/ciru; mkdir -p "$CR"
    if [ ! -x "$CR/opt/serve.sh" ]; then
      command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > "$OUT/uv.log" 2>&1; export PATH=$HOME/.local/bin:$PATH; }
      rm -rf "$CR/src"; git clone -q https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151.git "$CR/src" && git -C "$CR/src" checkout -q v1.2.2 || { log "ciru clone failed"; continue; }
      log "ciru $(git -C "$CR/src" rev-parse --short=9 HEAD) (v1.2.2)"
      ( uv venv -q --clear --python 3.12 "$CR/venv" && uv pip install -q --python "$CR/venv/bin/python" --extra-index-url https://wheels.vllm.ai/rocm/0.34.0/rocm100/ \
          'vllm==0.34.0+rocm100' 'huggingface-hub==2.4.0' 'safetensors==0.8.0' \
        && uv pip install -q --python "$CR/venv/bin/python" --target "$CR/tdeps" --no-deps 'huggingface-hub==1.31.0' 'safetensors==0.8.0' ) > "$OUT/ciru-venv.log" 2>&1 || { log "CIRU VENV FAILED: $(tail -3 "$OUT/ciru-venv.log" | cut -c1-200)"; continue; }
      ( cd "$CR/src" && bash scripts/install.sh --venv "$CR/venv" --transformers-deps "$CR/tdeps" --prefix "$CR/opt" ) > "$OUT/ciru-install.log" 2>&1 \
        || { log "CIRU INSTALL FAILED: $(grep -aiE 'error|fail' "$OUT/ciru-install.log" | tail -3 | cut -c1-200)"; continue; }
    fi
    # Ciru's release bundle: the unmodified checkpoint + DSpark draft, plus its fitted expert / final-layer weights
    # (its launcher's default). ciru-orig: the same bundle with --original-weights (the checkpoint's own experts).
    CK=$(hf_get jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 "" edcbab56fedf986ac6d6da5fbe6ede0f163556e5) || { log "ciru bundle download failed: $(tail -2 "$OUT/download.log")"; continue; }
    X=(--mode single); case $e in ciru-conc) X=(--mode concurrent --seqs 8) ;; ciru-orig) X=(--mode single --original-weights) ;;
      ciru-nospec) X=(--mode single --no-spec) ;; ciru-st995|ciru-agentic) X=(--mode single --tools --prefix-cache) ;; esac
    "$CR/opt/serve.sh" --model "$CK" "${X[@]}" --context 65536 --port $PORT --served-model-name main > "$OUT/serve-$e.log" 2>&1 &
    CPID=$!; ENGINE_ALIVE() { kill -0 $CPID 2>/dev/null; }; ENGINE_DOWN() { cp "$OUT/serve-$e.log" "$1" 2>/dev/null; kill -INT $CPID 2>/dev/null; pkill -INT -P $CPID 2>/dev/null; sleep 10; pkill -9 -P $CPID 2>/dev/null; kill -9 $CPID 2>/dev/null; sleep 5; true; }
    if [ $e = ciru-agentic ]; then
      wait_ready $PORT 720 || { log "ciru-agentic SERVE FAILED"; ENGINE_DOWN "$OUT/serve-$e.log"; continue; }
      agentic_leg ciru main; ENGINE_DOWN "$OUT/serve-$e.log"
    elif [ $e = ciru-st995 ]; then
      wait_ready $PORT 720 || { log "ciru-st995 SERVE FAILED"; ENGINE_DOWN "$OUT/serve-$e.log"; continue; }
      SHIM_DIR=$(ls -dt target/release/build/atlas-kernels-*/out 2>/dev/null | while read -r d; do [ -f "$d/libcuda.so" ] && { echo "$d"; break; }; done)
      LD_LIBRARY_PATH="$SHIM_DIR:/opt/rocm/lib" target/release/spark benchmark run bfcl-subset --url http://127.0.0.1:$PORT --model main --yes --format json --quiet --no-fail-on-verdict > "$OUT/ciru-bfcl.json" 2> "$OUT/ciru-bfcl.stderr" || log "ciru bfcl rc!=0: $(tail -2 "$OUT/ciru-bfcl.stderr" | tr '\n' ' ')"
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT ciru', d.get('benchmark_id'), d.get('run_id'), [(s['label'], s['value']) for s in d['frame']['summary']])" "$OUT/ciru-bfcl.json" 2>&1 | tail -1 | tee -a "$RES"
      ENGINE_DOWN "$OUT/serve-$e.log"
    else legs "$e"; fi ;;
  llamacpp|llamacpp-agentic)
    [ $e = llamacpp-agentic ] && LLAMA_AGENTIC=1
    LLAMA_REF=master llama_build || continue
    GD=$(hf_get ggml-org/Inkling-2-Mini-GGUF "Inkling-2-Mini-Q8_0.gguf") || { log "gguf download failed"; continue; }
    llama_up $PORT 8 40960 "$GD/Inkling-2-Mini-Q8_0.gguf"
    if [ "${LLAMA_AGENTIC:-0}" = 1 ]; then wait_ready $PORT 720 && agentic_leg llamacpp-q8 "$MID"; ENGINE_DOWN "$OUT/serve-llamacpp.log"; else legs llamacpp-q8; fi ;;
  halogen)
    HALOGEN_REPO=thinkingmachines/Inkling-2-Mini halogen_up || { log "halogen up failed"; continue; }
    PORT_SAVE=$PORT; PORT=8731; legs halogen-inkling; PORT=$PORT_SAVE ;;
  esac
done
log "DONE"
