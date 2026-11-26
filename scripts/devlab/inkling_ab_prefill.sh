#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: paired prefill A/B for an Inkling 2 PR on ONE runner.
#
#   scripts/devlab/inkling_ab_prefill.sh <parent_ref>
#
# The job's checkout is the PR tree (B); <parent_ref> is fetched into a git
# worktree at ./base and built there (ref A). Each binary serves in spark mode
# through scripts/serve-inkling2-strix-linux.sh (DSPARK=0, prefix caching on,
# MAX_BATCH=4) and runs: prefill_h2h (cold 7.7K / 31K), decode_h2h C1..C8, and
# greedy_eq for the identity compare. Results to $DEVLAB_OUT/RESULTS.txt.
set -uo pipefail
PARENT=${1:?usage: inkling_ab_prefill.sh <parent_ref>}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); SQ="$HERE/sqbench"
HFC=${HF_HOME:-$HOME/.cache/huggingface}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
log "ab-prefill: B=$(git rev-parse --short=9 HEAD) vs A=$PARENT  host $(hostname)"

# --- ISA stats for the touched kernels under this runner's hipcc --------------------------------------
COMPAT=$PWD/crates/atlas-kernels/hip/compat
HIPCC=$(command -v hipcc || echo /opt/rocm/bin/hipcc)
for FP in kernels/strix-hip/inkling-2-mini/nvfp4/inkling2_moe_gemm.cu; do
  [ -f "$FP" ] || continue
  T=$(mktemp -d)
  "$HIPCC" -x hip --genco --offload-arch=gfx1151 -O3 -I"$COMPAT" \
    -include hip/hip_runtime.h -include atlas_hip_cache_hints.h -DHDIM=128 \
    --save-temps=obj "$FP" -o "$T/x.co" >"$T/cc.log" 2>&1 || {
      log "$(basename "$FP") compile FAILED: $(tail -3 "$T/cc.log")"; continue; }
  S=$(ls "$T"/*-hip*.s 2>/dev/null | head -1)
  [ -n "$S" ] && python3 - "$S" >> "$RES" <<'PY'
import re,sys
cur=None
for ln in open(sys.argv[1]):
    m=re.search(r'\.globl\s+(\S+)',ln)
    if m: cur=m.group(1)
    if cur and re.search(r'(vgpr_count|spill_count|group_segment_fixed_size|private_segment_fixed_size):',ln):
        print(f"  isa {cur} {ln.strip()}")
PY
done

git fetch --depth=50 origin "$PARENT" || { log "fetch $PARENT failed"; exit 2; }
git worktree add --detach base FETCH_HEAD || { log "worktree add failed"; exit 2; }
log "base worktree at $(git -C base rev-parse --short=9 HEAD); building A"
( cd base && ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL=inkling-2-mini ./build-amd.sh ) > "$OUT/build-A.log" 2>&1 \
    || { log "BUILD A FAILED"; tail -15 "$OUT/build-A.log" | tee -a "$RES"; exit 6; }
log "build A done"

wait_ready() { # port pid
  for _ in $(seq 1 720); do
    curl -s -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

serve_legs() { # dir tag
  local dir=$1 tag=$2 port
  port=$([ "$tag" = A ] && echo 18101 || echo 18102)
  ( cd "$dir" && export MODE=spark DSPARK=0 PORT=$port MAX_BATCH=4 MAX_SEQ_LEN=40960 \
      DATA_DIR=$HFC/atlas-inkling2; exec scripts/serve-inkling2-strix-linux.sh ) \
      > "$OUT/serve-$tag.log" 2>&1 &
  local spid=$!
  if ! wait_ready $port $spid; then
    log "$tag SERVE FAILED"; tail -10 "$OUT/serve-$tag.log" | cut -c1-240 | tee -a "$RES"; return 1
  fi
  log "$tag serving ($(grep -m1 'binsha=' "$OUT/serve-$tag.log" | grep -oE 'binsha=[0-9a-f]+'))"
  export SQ_URL="http://127.0.0.1:$port" SQ_MODEL=inkling-2-mini
  python3 "$HERE/prefill_h2h.py" $port "$tag" 2>&1 | tail -4 | tee -a "$RES"
  python3 "$HERE/decode_h2h.py" $port "$tag" 2>&1 | grep -aE "^\s+\[|Error" | tee -a "$RES"
  python3 "$SQ/greedy_eq.py" run "$tag" 4 > "$OUT/greedy-$tag.log" 2>&1 || log "$tag greedy_eq rc!=0"
  [ -f "greedy-$tag.json" ] && mv "greedy-$tag.json" "$OUT/"
  local errs
  errs=$(grep -acE ' ERROR |panicked|hipError' "$OUT/serve-$tag.log" 2>/dev/null)
  log "$tag serve error/panic/hipError lines: ${errs:-0}"
  kill -INT $spid 2>/dev/null; for _ in $(seq 1 60); do kill -0 $spid 2>/dev/null || break; sleep 2; done
  kill -9 $spid 2>/dev/null; sleep 10
}

serve_legs base A
serve_legs . B

if [ -f "$OUT/greedy-A.json" ] && [ -f "$OUT/greedy-B.json" ]; then
  log "RESULT greedy:"; ( cd "$OUT" && python3 "$SQ/greedy_eq.py" compare A B ) 2>&1 | tee -a "$RES"
fi
log "DONE"
