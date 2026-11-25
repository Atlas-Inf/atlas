#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: paired A/B framework gate for an Inkling 2 PR on ONE runner.
#
#   scripts/devlab/inkling_ab_gate.sh <parent_ref> [legs...]
#
# The job's checkout is the PR tree (ref B). `parent_ref` is fetched from origin into a git
# worktree at ./base and built there (ref A). For each binary and each serve MODE
# (spark, strict) the script serves via scripts/serve-inkling2-strix-linux.sh with the
# f373 flags (DSPARK=0, prefix caching on, MAX_BATCH=4) and runs:
#   greedy_eq   greedy C1 prompts; `compare A B` reports identity (both modes)
#   lp_dump     prompt-logprob dumps (the pd check); compare gives per-prompt mean|delta|
#   nll_probe   teacher-forced NLL, code + prose; compare gives mean per category
#   quality_probe + tool_probe   the task_probe legs (both modes)
#   decode_h2h  C1 decode on prose/code/json + C1..C8 aggregate
# lp_dump and nll_probe run in spark mode only (numerics checks, once per arm).
# Vendored drivers live in scripts/devlab/sqbench/ (SparkQwen bench, AGPL).
set -uo pipefail
PARENT=${1:?usage: inkling_ab_gate.sh <parent_ref>}; shift || true
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); SQ="$HERE/sqbench"
HFC=${HF_HOME:-$HOME/.cache/huggingface}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
log "ab-gate: B=$(git rev-parse --short=9 HEAD) vs A=$PARENT  host $(hostname)"

# --- build the parent worktree ---------------------------------------------------------------------
git fetch --depth=50 origin "$PARENT" || { log "fetch $PARENT failed"; exit 2; }
git worktree add --detach base FETCH_HEAD || { log "worktree add failed"; exit 2; }
log "base worktree at $(git -C base rev-parse --short=9 HEAD); building A"
( cd base && ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL=inkling-2-mini ./build-amd.sh ) > "$OUT/build-A.log" 2>&1 \
    || { log "BUILD A FAILED"; tail -15 "$OUT/build-A.log" | tee -a "$RES"; exit 6; }
log "build A done; B builds lazily in its serve"

wait_ready() { # port pid
  for _ in $(seq 1 720); do
    curl -s -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

serve_legs() { # dir tag mode
  local dir=$1 tag=$2 mode=$3 port
  port=$([ "$tag" = A ] && echo 18101 || echo 18102)
  [ "$mode" = strict ] && port=$((port + 10))
  ( cd "$dir" && export MODE=$mode DSPARK=0 PORT=$port MAX_BATCH=4 MAX_SEQ_LEN=40960 \
      DATA_DIR=$HFC/atlas-inkling2; exec scripts/serve-inkling2-strix-linux.sh ) \
      > "$OUT/serve-$tag-$mode.log" 2>&1 &
  local spid=$!
  if ! wait_ready $port $spid; then
    log "$tag-$mode SERVE FAILED"; tail -10 "$OUT/serve-$tag-$mode.log" | cut -c1-240 | tee -a "$RES"; return 1
  fi
  log "$tag-$mode serving ($(grep -m1 'binsha=' "$OUT/serve-$tag-$mode.log" | grep -oE 'binsha=[0-9a-f]+'))"
  export SQ_URL="http://127.0.0.1:$port" SQ_MODEL=inkling-2-mini
  python3 "$SQ/greedy_eq.py" run "$tag-$mode" 4 > "$OUT/greedy-$tag-$mode.log" 2>&1 || log "$tag-$mode greedy_eq rc!=0"
  [ -f "greedy-$tag-$mode.json" ] && mv "greedy-$tag-$mode.json" "$OUT/"
  if [ "$mode" = spark ]; then
    python3 "$SQ/lp_dump.py" "$tag" > "$OUT/lpd-$tag.log" 2>&1 || log "$tag lpd rc!=0"
    [ -f "lpd-$tag.json" ] && mv "lpd-$tag.json" "$OUT/"
    python3 "$SQ/nll_probe.py" "$tag" > "$OUT/nll-$tag.log" 2>&1 || log "$tag nll rc!=0"
    [ -f "nll-$tag.json" ] && mv "nll-$tag.json" "$OUT/"
  fi
  python3 "$SQ/quality_probe.py" "$tag-$mode" 4 > "$OUT/qprobe-$tag-$mode.log" 2>&1 || log "$tag-$mode qprobe rc!=0"
  [ -f "qprobe-$tag-$mode.jsonl" ] && mv "qprobe-$tag-$mode.jsonl" "$OUT/"
  python3 "$SQ/tool_probe.py" "$tag-$mode" > "$OUT/tools-$tag-$mode.log" 2>&1 || log "$tag-$mode tools rc!=0"
  [ -f "tools-$tag-$mode.json" ] && mv "tools-$tag-$mode.json" "$OUT/"
  python3 "$HERE/decode_h2h.py" $port "$tag-$mode" 2>&1 | grep -aE "^\s+\[|Error" | tee -a "$RES"
  local errs
  errs=$(grep -acE ' ERROR |panicked|hipError' "$OUT/serve-$tag-$mode.log" 2>/dev/null)
  log "$tag-$mode serve error/panic/hipError lines: ${errs:-0}"
  kill -INT $spid 2>/dev/null; for _ in $(seq 1 60); do kill -0 $spid 2>/dev/null || break; sleep 2; done
  kill -9 $spid 2>/dev/null; pkill -9 -f "port $port" 2>/dev/null; sleep 10
}

serve_legs base A spark
serve_legs base A strict
serve_legs . B spark
serve_legs . B strict

# --- paired compares -------------------------------------------------------------------------------
for mode in spark strict; do
  if [ -f "$OUT/greedy-A-$mode.json" ] && [ -f "$OUT/greedy-B-$mode.json" ]; then
    log "RESULT greedy-$mode:"; ( cd "$OUT" && python3 "$SQ/greedy_eq.py" compare "A-$mode" "B-$mode" ) 2>&1 | tee -a "$RES"
  fi
done
if [ -f "$OUT/lpd-A.json" ] && [ -f "$OUT/lpd-B.json" ]; then
  log "RESULT lp_dump:"; ( cd "$OUT" && python3 "$SQ/lp_dump.py" compare A B ) 2>&1 | tee -a "$RES"
fi
if [ -f "$OUT/nll-A.json" ] && [ -f "$OUT/nll-B.json" ]; then
  log "RESULT nll_probe:"; ( cd "$OUT" && python3 "$SQ/nll_probe.py" compare A B ) 2>&1 | tee -a "$RES"
fi
log "DONE"
