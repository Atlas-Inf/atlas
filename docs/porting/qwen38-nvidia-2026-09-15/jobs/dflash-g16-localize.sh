#!/usr/bin/env bash
# JOB: dflash-g16-localize — name the faulting launch behind the γ=16 status-700.
#
# 044 (compute-sanitizer memcheck) reported NO device memory error: the serve
# process died outright during the generation-2 piecewise capture (serve log
# ends at "piecewise capture: starting", sanitizer: "process didn't terminate
# successfully", 194 boot-time API-noise entries only). So memcheck cannot
# see it. 040-C proved the fault also fires on the EAGER path (no capture),
# and 040-E proved CUDA_LAUNCH_BLOCKING=1 surfaces it at the launch — but E
# ran with capture on, so the launch was a whole graph ("cuGraphLaunch failed").
# Combine the two: eager-only + launch-blocking ⇒ the failing launch is a
# single kernel and the Rust error names the op wrapper.
#   I  γ=16, ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000 CUDA_LAUNCH_BLOCKING=1 RUST_LOG=debug
#   J  γ=16, ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000 ATLAS_DFLASH_OPTION_B_DIAG=1
#      (layer-0 diag D2H+sync inside pre_attn — a second, independent fence)
# Same serve profile as 040 (8K seq, util 0.78, pcache OFF, OB=1), 3 MinHeap
# requests (the fault is at request 2's first propose). Output per leg: the
# first ERROR/WARN line that names a kernel/op, plus 30 lines of context.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
export ATLAS_DFLASH_OPTION_B=1 ATLAS_MTP_ACCEPT_DEBUG=1
RESULTS_HEADER=$'label\tcompletion_tokens\twall_s\ttok_s_incl_ttft\tsha\tfinish'
[ -f "$OUT/RESULTS.tsv" ] || echo "$RESULTS_HEADER" > "$OUT/RESULTS.tsv"

leg() {  # leg <label> [env words...]
    local label="$1"; shift
    log "=== LEG $label env: $* ==="
    if GAMMA=16 serve_dflash "$OUT/serve-$label.log" "$@"; then
        probe_minheap "$label" 3
    fi
    safe_kill "$SRV_PID"; sleep 4
    log "--- $label: first fault line + 30 lines before it"
    local n
    n=$(grep -n -m1 "ILLEGAL_ADDRESS\|status 700\|failed:" "$OUT/serve-$label.log" | cut -d: -f1 || true)
    if [ -n "$n" ]; then
        sed -n "$(( n > 30 ? n - 30 : 1 )),$(( n + 3 ))p" "$OUT/serve-$label.log" | cut -c1-330
    else
        log "  (no fault line found — leg survived?)"
    fi
}

leg I_eager_launchblocking ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000 CUDA_LAUNCH_BLOCKING=1 RUST_LOG=debug
leg J_eager_optionbdiag    ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000 ATLAS_DFLASH_OPTION_B_DIAG=1
log "== RESULTS =="; cat "$OUT/RESULTS.tsv"
log "JOB DONE"
