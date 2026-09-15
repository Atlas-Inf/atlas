#!/usr/bin/env bash
# JOB: dflash-g16-syncdebug — name the culprit launch with the in-tree tool.
#
# `spark-runtime/src/kernel_args.rs::KernelLaunch::launch` has a PCND debug
# lever, ATLAS_DEBUG_SYNC_KERNELS=1: synchronize after EVERY launch and, on a
# fault, return an error carrying grid/block AND a forced Rust backtrace of the
# launch site ("LAUNCH BACKTRACE"). That names the `ops::*` wrapper and the
# dflash_head caller — which CUDA_LAUNCH_BLOCKING (067-I: grid [32,1,1]×[128])
# and memcheck (044/081: no device error; 081 saw [16,1,1]×[256] with 719)
# could not do unambiguously, because a blocking launch may also surface the
# PREVIOUS launch's execution fault.
# Eager path (PROPOSE_WARMUP_N huge), γ=16, Option B, 3 MinHeap requests —
# the fault is at request 2's first propose (second sequence).
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
export ATLAS_DFLASH_OPTION_B=1 ATLAS_MTP_ACCEPT_DEBUG=1
RESULTS_HEADER=$'label\tcompletion_tokens\twall_s\ttok_s_incl_ttft\tsha\tfinish'
[ -f "$OUT/RESULTS.tsv" ] || echo "$RESULTS_HEADER" > "$OUT/RESULTS.tsv"

label=K_eager_syncdebug
log "=== LEG $label: ATLAS_DEBUG_SYNC_KERNELS=1 RUST_BACKTRACE=1 eager γ=16 ==="
if GAMMA=16 serve_dflash "$OUT/serve-$label.log" \
        ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000 ATLAS_DEBUG_SYNC_KERNELS=1 RUST_BACKTRACE=1; then
    probe_minheap "$label" 3
fi
safe_kill "$SRV_PID"; sleep 4
log "--- first ATLAS_DEBUG_SYNC_KERNELS fault + LAUNCH BACKTRACE (first 60 lines)"
n=$(grep -n -m1 "ATLAS_DEBUG_SYNC_KERNELS: async GPU fault" "$OUT/serve-$label.log" | cut -d: -f1 || true)
if [ -n "$n" ]; then
    sed -n "$(( n > 6 ? n - 6 : 1 )),$(( n + 60 ))p" "$OUT/serve-$label.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260
else
    log "  (no ATLAS_DEBUG_SYNC_KERNELS fault line — first fault of any kind:)"
    grep -n -m1 -A20 "ILLEGAL_ADDRESS\|status 7[0-9][0-9]\|failed:" "$OUT/serve-$label.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260
fi
log "== RESULTS =="; cat "$OUT/RESULTS.tsv"
log "JOB DONE"
