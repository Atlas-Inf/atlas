#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: Inkling 2 Mini engine head-to-head on ONE runner, same client and prompts for every engine.
#
#   scripts/devlab/inkling_h2h.sh <engines...>     engines: atlas-check atlas-spark atlas-strict ciru ciru-nospec ciru-conc llamacpp halogen
#
# Per engine: decode_h2h.py (non-streaming C1 decode on prose / code / json, C 1/2/4/8 aggregate) and prefill_h2h.py (cold 8K / 32K prompts), thinking off by request (reasoning_effort none).
#   atlas-check  scripts/serve-inkling2-strix-linux.sh --check as a user runs it, MODE strict then spark (rc + answer line)
#   atlas-*      scripts/serve-inkling2-strix-linux.sh at this ref, MODE spark|strict, no drafter, 8 sequences
#   ciru         ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 v1.5.0 (all-HIP gfx1151 engine on its pinned
#                ciru-ai/vllm b1dcc9e6, ROCm vLLM 0.34), --mode single (its default DSpark)
#   ciru-nospec  the same with --no-spec
#   ciru-conc    the same with --mode concurrent --seqs 8
#   llamacpp     ggml-org/llama.cpp master (Inkling 2 merged in #32694 after the last release), GGML_HIP gfx1151,
#                ggml-org/Inkling-2-Mini-GGUF Q8_0, 8 slots
#   llamacpp-st995  the same llama.cpp serve (template defaults, tools through --jinja), scored on the
#                bfcl-subset ST leg by this build's own `spark benchmark run` + the soundfile rescore fallback
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
  grep -m1 "binsha=" "$OUT/serve-$1.log" 2>/dev/null | cut -c1-220 | sed "s/^/  $1 script: /" | tee -a "$RES"
  python3 "$HERE/decode_h2h.py" $PORT "$1" 2>&1 | grep -aE "^\s+\[|Error|error" | tee -a "$RES"
  python3 "$HERE/prefill_h2h.py" $PORT "$1" 2>&1 | tail -3 | tee -a "$RES"
  ENGINE_DOWN "$OUT/serve-$1.log"
}
for e in "$@"; do
  case $e in
  atlas-check)
    for m in strict spark; do
      ( export MODE=$m PORT=18391 DATA_DIR=$HFC/atlas-inkling2; exec scripts/serve-inkling2-strix-linux.sh --check ) > "$OUT/check-$m.log" 2>&1
      rc=$?
      log "atlas-check MODE=$m rc=$rc $(grep -m1 'binsha=' "$OUT/check-$m.log" | cut -c1-160)"
      grep -m1 "answer:" "$OUT/check-$m.log" | cut -c1-160 | sed "s/^/  atlas-check $m /" | tee -a "$RES"
      [ $rc -ne 0 ] && tail -8 "$OUT/check-$m.log" | cut -c1-200 | sed "s/^/  atlas-check $m FAIL /" | tee -a "$RES"
    done ;;
  atlas-spark|atlas-strict)
    ( export MODE=${e#atlas-} DSPARK=0 PORT MAX_BATCH=8 MAX_SEQ_LEN=40960 DATA_DIR=$HFC/atlas-inkling2; exec scripts/serve-inkling2-strix-linux.sh ) > "$OUT/serve-$e.log" 2>&1 &
    APID=$!; ENGINE_ALIVE() { kill -0 $APID 2>/dev/null; }; ENGINE_DOWN() { cp "$OUT/serve-$e.log" "$1" 2>/dev/null; kill -INT $APID 2>/dev/null; pkill -INT -P $APID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $APID 2>/dev/null || break; sleep 2; done; kill -9 $APID 2>/dev/null; sleep 5; true; }
    legs "$e" ;;
  ciru|ciru-conc|ciru-orig|ciru-nospec|ciru-st995|ciru-agentic)
    CR=${RUNNER_TEMP:-/tmp}/ciru; mkdir -p "$CR"
    if [ ! -x "$CR/opt/serve.sh" ]; then
      command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > "$OUT/uv.log" 2>&1; export PATH=$HOME/.local/bin:$PATH; }
      rm -rf "$CR/src"; git clone -q https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151.git "$CR/src" && git -C "$CR/src" checkout -q v1.5.0 || { log "ciru clone failed"; continue; }
      log "ciru $(git -C "$CR/src" rev-parse --short=9 HEAD) (v1.5.0) $(grep -m1 '^vllm_revision=' "$CR/src/scripts/install.sh" | cut -d= -f2 | sed 's/^/vllm /')"
      ( uv venv -q --clear --python 3.12 "$CR/venv" && uv pip install -q --python "$CR/venv/bin/python" --extra-index-url https://wheels.vllm.ai/rocm/0.34.0/rocm100/ \
          'vllm==0.34.0+rocm100' 'huggingface-hub==2.4.0' 'safetensors==0.8.0' \
        && uv pip install -q --python "$CR/venv/bin/python" --target "$CR/tdeps" --no-deps 'huggingface-hub==1.31.0' 'safetensors==0.8.0' ) > "$OUT/ciru-venv.log" 2>&1 || { log "CIRU VENV FAILED: $(tail -3 "$OUT/ciru-venv.log" | cut -c1-200)"; continue; }
      ( cd "$CR/src" && bash scripts/install.sh --venv "$CR/venv" --transformers-deps "$CR/tdeps" --prefix "$CR/opt" ) > "$OUT/ciru-install.log" 2>&1 \
        || { log "CIRU INSTALL FAILED: $(grep -aiE 'error|fail' "$OUT/ciru-install.log" | tail -3 | cut -c1-200)"; continue; }
    fi
    # Ciru's release bundle: the unmodified checkpoint + DSpark draft, plus its fitted expert / final-layer weights
    # (its launcher's default). ciru-orig: the same bundle with --original-weights (the checkpoint's own experts).
    CK=$(hf_get jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 "" edcbab56fedf986ac6d6da5fbe6ede0f163556e5) || { log "ciru bundle download failed: $(tail -2 "$OUT/download.log")"; continue; }
    [ "$e" = ciru ] && log "ciru bundle jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151@$(basename "$CK")"
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
      python3 - "$OUT/ciru-bfcl.json" > "$OUT/rescore.sh" <<'PY2'
import json, re, sys
r = (json.load(open(sys.argv[1]))["frame"].get("verdict") or {}).get("reason", "")
m = re.search(r"rescored: (\S+/python) (\S+/score\.py) --dataset (\S+) --responses (\S+) failed", r)
if m:
    py, sc, ds, rs = m.groups()
    print(f'"{py}" -m pip install -q soundfile && cp "{rs}" "$OUT/ciru-bfcl-responses.jsonl" && "{py}" "{sc}" --dataset "{ds}" --responses "{rs}"')
PY2
      if [ -s "$OUT/rescore.sh" ]; then log "ciru bfcl rescoring"; OUT=$OUT bash "$OUT/rescore.sh" > "$OUT/rescore.out" 2>&1; log "rescore rc=$?"; tail -3 "$OUT/rescore.out" | cut -c1-600 | tee -a "$RES"; fi
      ENGINE_DOWN "$OUT/serve-$e.log"
    else legs "$e"; fi ;;
  llamacpp|llamacpp-agentic|llamacpp-st995)
    [ $e = llamacpp-agentic ] && LLAMA_AGENTIC=1
    [ $e = llamacpp-st995 ] && LLAMA_ST995=1
    LLAMA_REF=master llama_build || continue
    GD=$(hf_get ggml-org/Inkling-2-Mini-GGUF "Inkling-2-Mini-Q8_0.gguf") || { log "gguf download failed"; continue; }
    log "ggml-org/Inkling-2-Mini-GGUF@$(basename "$GD")"
    llama_up $PORT 8 40960 "$GD/Inkling-2-Mini-Q8_0.gguf"
    log "llama-server cmd: $(tr '\0' ' ' < /proc/$LPID/cmdline 2>/dev/null | cut -c1-400)"
    if [ "${LLAMA_AGENTIC:-0}" = 1 ]; then
      wait_ready $PORT 720 && agentic_leg llamacpp-q8 "$MID"; ENGINE_DOWN "$OUT/serve-llamacpp.log"
    elif [ "${LLAMA_ST995:-0}" = 1 ]; then
      wait_ready $PORT 720 || { log "llamacpp-st995 SERVE FAILED"; ENGINE_DOWN "$OUT/serve-llamacpp.log"; continue; }
      SHIM_DIR=$(ls -dt target/release/build/atlas-kernels-*/out 2>/dev/null | while read -r d; do [ -f "$d/libcuda.so" ] && { echo "$d"; break; }; done)
      LD_LIBRARY_PATH="$SHIM_DIR:/opt/rocm/lib" target/release/spark benchmark run bfcl-subset --url http://127.0.0.1:$PORT --model "$MID" --yes --format json --quiet --no-fail-on-verdict > "$OUT/llamacpp-bfcl.json" 2> "$OUT/llamacpp-bfcl.stderr" || log "llamacpp bfcl rc!=0: $(tail -2 "$OUT/llamacpp-bfcl.stderr" | tr '\n' ' ')"
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT llamacpp', d.get('benchmark_id'), d.get('run_id'), [(s['label'], s['value']) for s in d['frame']['summary']])" "$OUT/llamacpp-bfcl.json" 2>&1 | tail -1 | tee -a "$RES"
      python3 - "$OUT/llamacpp-bfcl.json" > "$OUT/rescore-llamacpp.sh" <<'PY2'
import json, re, sys
r = (json.load(open(sys.argv[1]))["frame"].get("verdict") or {}).get("reason", "")
m = re.search(r"rescored: (\S+/python) (\S+/score\.py) --dataset (\S+) --responses (\S+) failed", r)
if m:
    py, sc, ds, rs = m.groups()
    print(f'"{py}" -m pip install -q soundfile && cp "{rs}" "$OUT/llamacpp-bfcl-responses.jsonl" && "{py}" "{sc}" --dataset "{ds}" --responses "{rs}"')
PY2
      if [ -s "$OUT/rescore-llamacpp.sh" ]; then log "llamacpp bfcl rescoring"; OUT=$OUT bash "$OUT/rescore-llamacpp.sh" > "$OUT/rescore-llamacpp.out" 2>&1; log "rescore rc=$?"; tail -3 "$OUT/rescore-llamacpp.out" | cut -c1-600 | tee -a "$RES"; fi
      ENGINE_DOWN "$OUT/serve-llamacpp.log"
    else legs llamacpp-q8; fi ;;
  halogen)
    HALOGEN_REPO=thinkingmachines/Inkling-2-Mini halogen_up || { log "halogen up failed"; continue; }
    PORT_SAVE=$PORT; PORT=8731; legs halogen-inkling; PORT=$PORT_SAVE ;;
  esac
done
log "DONE"
