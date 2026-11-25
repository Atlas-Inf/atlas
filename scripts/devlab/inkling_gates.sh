#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: Inkling 2 Mini quality / agentic gates on the checked-out ref.
#
#   scripts/devlab/inkling_gates.sh <strict|spark> <legs...>     legs: st995 agentic poison xcont csweep ttft dfloor
#
# Serves through scripts/serve-inkling2-strix-linux.sh (which builds the inkling-2-mini target and
# downloads through $HF_ENDPOINT), DSPARK=0, prefix caching on, up to 4 sequences, then runs each leg
# with this build's own `spark benchmark run` and no parameter overrides except the agentic leg's
# iterations=10. Results go to $DEVLAB_OUT/RESULTS.txt.
set -uo pipefail
MODE=${1:?strict|spark}; shift; LEGS=("$@")
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
export MODE DSPARK=0 PREFIX_CACHE=${PREFIX_CACHE:-false} PORT=18091 MAX_BATCH=${MAX_BATCH:-4} DATA_DIR=${HF_HOME:-$HOME/.cache}/atlas-inkling2
log "ref $(git rev-parse --short=9 HEAD) mode=$MODE legs=${LEGS[*]} max_batch=$MAX_BATCH"
scripts/serve-inkling2-strix-linux.sh > "$OUT/serve.log" 2>&1 &
SPID=$!
for _ in $(seq 1 720); do
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { log "serve exited early"; tail -30 "$OUT/serve.log" | tee -a "$RES"; exit 1; }
  sleep 5
done
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || { log "serve not ready after 60 min"; exit 1; }
grep -aE "serve-inkling2-strix-linux|KV cache:.*budget|Prefix caching" "$OUT/serve.log" | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-240 | tee -a "$RES"
B=target/release/spark
# the client binary links the same HIP shims as the server (no rpath)
SHIM_DIR=$(ls -dt target/release/build/atlas-kernels-*/out 2>/dev/null | while read -r d; do [ -f "$d/libcuda.so" ] && { echo "$d"; break; }; done)
export LD_LIBRARY_PATH="$SHIM_DIR:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bench() { local id=$1; shift
  "$B" benchmark run "$id" --url "http://127.0.0.1:$PORT" --model inkling-2-mini --yes "$@" --format json --quiet --no-fail-on-verdict > "$OUT/$id.json" 2> "$OUT/$id.stderr" || log "$id exited nonzero; stderr tail: $(tail -3 "$OUT/$id.stderr" | tr "\n" " " | cut -c1-300)"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT', d.get('benchmark_id'), d.get('run_id'), [(s['label'], s['value']) for s in d['frame']['summary']], (d.get('verdict') or {}).get('kind'))" "$OUT/$id.json" 2>&1 | tee -a "$RES"; }
for leg in "${LEGS[@]}"; do
  case $leg in
    st995) bench bfcl-subset
       # bfcl-eval's qwen-agent imports soundfile, which the pinned venv lacks on this image: generation finishes and
       # scoring dies (run 44992539775). Keep the responses, add soundfile, and re-run the scorer command it printed.
       python3 - "$OUT/bfcl-subset.json" > "$OUT/rescore.sh" <<'PY2'
import json, re, sys
r = (json.load(open(sys.argv[1]))["frame"].get("verdict") or {}).get("reason", "")
m = re.search(r"rescored: (\S+/python) (\S+/score\.py) --dataset (\S+) --responses (\S+) failed", r)
if m:
    py, sc, ds, rs = m.groups()
    print(f'"{py}" -m pip install -q soundfile && cp "{rs}" "$OUT/bfcl-responses.jsonl" && "{py}" "{sc}" --dataset "{ds}" --responses "{rs}"')
PY2
       if [ -s "$OUT/rescore.sh" ]; then log "bfcl scoring failed in-run; rescoring the kept responses"; OUT=$OUT bash "$OUT/rescore.sh" > "$OUT/rescore.out" 2>&1; log "rescore rc=$?"; tail -25 "$OUT/rescore.out" | tee -a "$RES"; fi ;;
    agentic) bench agentic-webserver --param iterations=10 --param wall_budget_s=100000 ;;
    poison) bench ssm-state-poisoning-gate ;;
    xcont) bench cross-contamination ;;
    csweep) bench concurrency-sweep ;;
    ttft) bench ttft-cold-gate; bench ttft-warm-gate ;;
    dfloor) bench decode-floor ;;
    *) log "unknown leg $leg" ;;
  esac
done
kill -INT $SPID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null
log "serve faults: ERROR=$(grep -ac ' ERROR ' "$OUT/serve.log") panics=$(grep -ac panicked "$OUT/serve.log") hipError=$(grep -aci hiperror "$OUT/serve.log")"
log "DONE"
