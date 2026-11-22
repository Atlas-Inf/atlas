#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: Inkling 2 Mini quality / agentic gates on the checked-out ref.
#
#   scripts/devlab/inkling_gates.sh <strict|spark> <legs...>     legs: st995 agentic poison
#
# Serves through scripts/serve-inkling2-strix-linux.sh (which builds the inkling-2-mini target and
# downloads through $HF_ENDPOINT), DSPARK=0, prefix caching on, up to 4 sequences, then runs each leg
# with this build's own `spark benchmark run` and no parameter overrides except the agentic leg's
# iterations=10. Results go to $DEVLAB_OUT/RESULTS.txt.
set -uo pipefail
MODE=${1:?strict|spark}; shift; LEGS=("$@")
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
export MODE DSPARK=0 PORT=18091 MAX_BATCH=${MAX_BATCH:-4} DATA_DIR=${HF_HOME:-$HOME/.cache}/atlas-inkling2
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
bench() { local id=$1; shift
  "$B" benchmark run "$id" --url "http://127.0.0.1:$PORT" --model inkling-2-mini --yes "$@" --format json --quiet --no-fail-on-verdict > "$OUT/$id.json" 2> "$OUT/$id.stderr" || true
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT', d.get('benchmark_id'), d.get('run_id'), [(s['label'], s['value']) for s in d['frame']['summary']], (d.get('verdict') or {}).get('kind'))" "$OUT/$id.json" 2>&1 | tee -a "$RES"; }
for leg in "${LEGS[@]}"; do
  case $leg in
    st995) bench bfcl-subset ;;
    agentic) bench agentic-webserver --param iterations=10 --param wall_budget_s=100000 ;;
    poison) bench ssm-state-poisoning ;;
    *) log "unknown leg $leg" ;;
  esac
done
kill -INT $SPID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null
log "serve faults: ERROR=$(grep -ac ' ERROR ' "$OUT/serve.log") panics=$(grep -ac panicked "$OUT/serve.log") hipError=$(grep -aci hiperror "$OUT/serve.log")"
log "DONE"
