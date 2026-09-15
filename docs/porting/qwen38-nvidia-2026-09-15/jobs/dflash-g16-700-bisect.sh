#!/usr/bin/env bash
# JOB: dflash-g16-700-bisect — localize the CUDA_ERROR_ILLEGAL_ADDRESS (700)
# that killed 027/028/030/031/038/039 (exit 70).
#
# What we know (2026-09-15, from the queue logs):
#   * 038/039 ran the gate serve with enable_prefix_caching=false and STILL
#     died -> the prefix-caching hypothesis (036's premise) is REFUTED.
#   * 026 γ-sweep under Option B: γ=6/8/10/12 ran 4/4 requests clean;
#     γ=16 ran req0 clean (256 tok) and died on req1 at the first propose of
#     the new sequence, right after "piecewise capture: complete" for
#     SequenceGeneration{slot:0, generation:2}. Identical signature to the
#     gate serves, whose --dflash-gamma default is 16 (MODEL.toml says 8 but
#     clap's default wins; serve log: "ENABLED (γ=16 ...)").
#   * Pool sizing differs at γ=16: 258 blocks/4128 slots vs 257/4112 at γ≤12.
#
# Matrix (fresh serve per leg, 3 MinHeap requests each, same fingerprint as
# 025/026: 8K seq, util 0.78, bf16 kv+head, pcache OFF, OPTION_B=1):
#   A  γ=16                                   expect: req1 dies (repro)
#   B  γ=15                                   does the gate's effective γ die too?
#   C  γ=16 + PROPOSE_WARMUP_N=1000000        never captures -> eager only.
#                                             Survives => piecewise capture path.
#   D  γ=16 + DEBUG_FULL_PRECOMPUTE=1         incremental precompute ruled in/out
#   E  γ=16 + CUDA_LAUNCH_BLOCKING=1          fault surfaces AT the launch; the
#                                             Rust error names the op.
#   F  γ=16 + ATLAS_DFLASH_CTX_WINDOW=4080    pool 4080+17 -> 257 blocks: is it
#                                             the 258-block / slot-4112.. edge?
# Each leg: SURVIVED or DIED, plus the first ERROR/700 line. Exit code = bitmask
# of legs that died (A=1,B=2,C=4,D=8,E=16,F=32). All legs dying = 63.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
export ATLAS_DFLASH_OPTION_B=1 ATLAS_MTP_ACCEPT_DEBUG=1

RESULTS_HEADER=$'label\tcompletion_tokens\twall_s\ttok_s_incl_ttft\tsha\tfinish'
[ -f "$OUT/RESULTS.tsv" ] || echo "$RESULTS_HEADER" > "$OUT/RESULTS.tsv"
: > "$OUT/BISECT.tsv"

leg() {  # leg <bit> <label> <gamma> [env words...]
    local bit="$1" label="$2" gamma="$3"; shift 3
    log "=== LEG $label γ=$gamma env: $* ==="
    local died=0
    if GAMMA="$gamma" serve_dflash "$OUT/serve-$label.log" "$@"; then
        probe_minheap "$label" 3
        kill -0 "$SRV_PID" 2>/dev/null || died=1
        grep -q "ILLEGAL_ADDRESS\|status 700\|CUDA context is destroyed" "$OUT/serve-$label.log" && died=1
    else
        died=1
    fi
    local first
    first=$(grep -m1 "ERROR\|ILLEGAL_ADDRESS\|status 7[0-9][0-9]" "$OUT/serve-$label.log" | cut -c1-300 || true)
    local nreq_ok
    nreq_ok=$(grep -c "^$label\.req.*finish=length" "$OUT/RESULTS.tsv" || true)
    printf '%s\tgamma=%s\tdied=%s\treq_ok=%s\tenv=%s\tfirst_err=%s\n' \
        "$label" "$gamma" "$died" "$nreq_ok" "$*" "$first" >> "$OUT/BISECT.tsv"
    log "=== LEG $label died=$died req_ok=$nreq_ok ==="
    safe_kill "$SRV_PID"; sleep 4
    [ "$died" = 1 ] && RC=$((RC | bit))
    return 0
}

RC=0
leg 1  A_g16          16
leg 2  B_g15          15
leg 4  C_g16_nocap    16 ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000
leg 8  D_g16_fullpre  16 ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE=1
leg 16 E_g16_lb       16 CUDA_LAUNCH_BLOCKING=1
leg 32 F_g16_cw4080   16 ATLAS_DFLASH_CTX_WINDOW=4080

log "== BISECT =="; cat "$OUT/BISECT.tsv"
log "JOB DONE rc=$RC (bitmask of legs that died: A=1 B=2 C=4 D=8 E=16 F=32)"
exit "$RC"
