#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: engine head-to-head speed, on the protocol Atlas's headline uses (gufo_bench.py: pp ~2048 / tg128 at
# depth 0/8192/16384, then C 1/4/8 steady aggregate decode), so rows compare directly with the Atlas numbers.
#
#   scripts/devlab/h2h_speed.sh <atlas|gufo|llamacpp|halogen> <27b|fnext|ripple> [KEY=VAL ...]
# ripple: llamacpp only (community Q4_K_M GGUF on llama.cpp's Ripple port; Atlas's Ripple rows come from ripple_bench.sh).     KEYS: SESSIONS (8), CONTEXT (32768)
# atlas arms:   b1 (MAX_BATCH 1, MTP: the single-user recipe) and b8 (MAX_BATCH 8) on NVIDIA's NVFP4 via serve-amd.sh, GPU_UTIL
#               0.70 (the runner's batch-8 27B ceiling, run 41181983380), so all three engines share one machine.
# gufo arms:     ar, and its speculative leg (DFlash2 for 27B, MTP for Flash-Next) -- each engine's own performance leg
# halogen arm:   fnext only; Peonist's image at its defaults (drafting on), thinking off per request (it thinks by default).
# llamacpp arms: nvfp4 (NVIDIA's checkpoint through llama.cpp's own converter, if it converts) and q4 (the Unsloth
#                UD-Q4_K_XL Gufo uses); stock llama.cpp has no speculative leg for these models.
set -uo pipefail
ENGINE=${1:?atlas|gufo|llamacpp|halogen}; MK=${2:?27b|fnext}
for kv in "${@:3}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
# ripple: the Atlas Ripple recipe's shape (1 session, 8192 context, depths 0/4096, C1), so rows compare with ripple_bench.sh.
[ "$MK" = ripple ] && { SESSIONS=${SESSIONS:-1}; CONTEXT=${CONTEXT:-8192}; DEPTHS=${DEPTHS:-0,4096}; CS=${CS:-1}; }
SESSIONS=${SESSIONS:-8}; CONTEXT=${CONTEXT:-32768}; DEPTHS=${DEPTHS:-0,8192,16384}; PORT=18086
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
model_spec "$MK" || exit 1; pyhf
[ "$ENGINE" = halogen ] || { t0=$(date +%s); Q4D=$(hf_get "$Q4_REPO" "$Q4_PAT" "$Q4_REV") || { log "Q4 DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; exit 1; }
Q4=$(first_gguf "$Q4D" "$Q4_PAT"); log "$MK q4: $Q4_REPO $(basename "$Q4") ($(du -shL "$Q4" | cut -f1)) in $(( $(date +%s) - t0 ))s"; }

bench_arm() { # arm-name [gufo_bench concurrency list, default 1,4,8] [wait_ready tries]
  if ! wait_ready $PORT "${3:-360}"; then log "$1 SERVE FAILED:"; ENGINE_DOWN "$OUT/serve-$1.log"; grep -aiE "error|fail|memory|alloc" "$OUT/serve-$1.log" | tail -5 | tee -a "$RES"; return 1; fi
  log "$1 serving (model id $MID)"
  python3 "$HERE/gufo_bench.py" $PORT "$1" "$OUT/$1.json" --depths "$DEPTHS" --cs "${2:-${CS:-1,4,8}}" --kinds mixed 2>&1 | grep -aE "^\s+\[" | tee -a "$RES"
  ENGINE_DOWN "$OUT/serve-$1.log"
}

case $ENGINE in
atlas)
  [ -x target/release/spark ] || { log "no target/release/spark (run with job=bench, which builds it)"; exit 1; }
  # The 53.7 GB model-fp8-mtp-ple.safetensors is over huggingface_hub's plain-HTTP cap and the mirror serves no xet,
  # so pull everything else through the hub cache and curl that one file (resumable) into the snapshot dir.
  BIG=model-fp8-mtp-ple.safetensors
  NV=$("${PY:-python3}" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], ignore_patterns=[sys.argv[2]]))' "$NVFP4_REPO" "$BIG" 2>>"$OUT/download.log") \
    && { [ "$MK" != fnext ] || curl -fsSL -C - --retry 5 --retry-all-errors -o "$NV/$BIG" "${HF_ENDPOINT:-https://huggingface.co}/$NVFP4_REPO/resolve/main/$BIG" 2>>"$OUT/download.log"; } \
    || { log "NVFP4 DOWNLOAD FAILED: $(grep -av "^ " "$OUT/download.log" | tail -8)"; exit 1; }
  log "atlas $(git rev-parse --short=9 HEAD), $NVFP4_REPO ($(du -shL "$NV" | cut -f1))"
  AENV=(GPU_UTIL=${GPU_UTIL:-0.70} MAX_SEQ_LEN=32768 ATLAS_KV_EXTERNAL_RESERVE_GB=0)
  if [ "${PROF:-0}" = 1 ]; then   # PROF=1: only the b8 arm at C4 (the Flash-Next batched-MTP-verify collapse), under rocprofv3
    command -v rocprofv3 >/dev/null || PATH=$PATH:/opt/rocm/bin; W="$OUT/spark-prof"; mkdir -p "$OUT/prof"
    printf '#!/usr/bin/env bash\nexec rocprofv3 --kernel-trace --output-format csv -d %q -o run -- %q "$@"\n' "$OUT/prof" "$PWD/target/release/spark" > "$W"; chmod +x "$W"
    atlas_up $PORT "$NVFP4_REPO" "${AENV[@]}" MAX_BATCH=8 ATLAS_BIN="$W"; bench_arm atlas_b8_prof 4 720
    KT=$(find "$OUT/prof" -name '*kernel_trace.csv' | head -1)
    [ -n "$KT" ] && python3 - "$KT" <<'PY' | tee -a "$RES"
import csv, sys, collections
tot, cnt = collections.Counter(), collections.Counter()
for r in csv.DictReader(open(sys.argv[1])):
    n = (r.get("Kernel_Name") or r.get("KernelName") or "?").split("(")[0][:90]
    tot[n] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"]); cnt[n] += 1
s = sum(tot.values()); print(f"  [prof] {len(tot)} kernels, {s / 1e9:.2f} s GPU total")
for n, t in tot.most_common(40): print(f"  [prof] {t / 1e6:10.1f} ms {100 * t / s:5.1f}% x{cnt[n]:<7} {n}")
PY
  else
    atlas_up $PORT "$NVFP4_REPO" "${AENV[@]}" MAX_BATCH=1; bench_arm atlas_b1 1
    atlas_up $PORT "$NVFP4_REPO" "${AENV[@]}" MAX_BATCH=8; bench_arm atlas_b8
  fi ;;
gufo)
  gufo_build || exit 1
  gufo_up $PORT "$SESSIONS" "$CONTEXT" "$Q4"; bench_arm gufo_ar
  if [ -n "$DRAFT_REPO" ]; then
    DD=$(hf_get "$DRAFT_REPO" "$DRAFT_PAT") && DR=$(first_gguf "$DD" "$DRAFT_PAT")
    if [ -n "${DR:-}" ]; then gufo_up $PORT "$SESSIONS" "$CONTEXT" "$Q4" --speculative dflash2 --dflash-model "$DR"; bench_arm gufo_dflash2
    else log "draft download failed: $(tail -2 "$OUT/download.log")"; fi
  else
    MD=$(hf_get "$Q4_REPO" "MTP/*" "$Q4_REV") && MT=$(first_gguf "$MD" "MTP/*.gguf")
    [ -n "${MT:-}" ] && { gufo_up $PORT "$SESSIONS" "$CONTEXT" "$Q4" --speculative mtp --mtp-model "$MT"; bench_arm gufo_mtp; }
  fi ;;
llamacpp)
  llama_build || exit 1
  if [ "$MK" != ripple ] && llama_convert_nvfp4 "$NVFP4_REPO"; then
    llama_up $PORT "$SESSIONS" "$CONTEXT" "$NVFP4_GGUF"; bench_arm llamacpp_nvfp4
    "$LB/llama-bench" -m "$NVFP4_GGUF" -ngl 99 -p 2048 -n 128 -r 2 -o md 2>"$OUT/bench-nvfp4.err" | tee -a "$RES" || tail -3 "$OUT/bench-nvfp4.err" | tee -a "$RES"
  fi
  llama_up $PORT "$SESSIONS" "$CONTEXT" "$Q4"; bench_arm llamacpp_q4
  "$LB/llama-bench" -m "$Q4" -ngl 99 -p 2048 -n 128 -r 2 -o md 2>"$OUT/bench-q4.err" | tee -a "$RES" || tail -3 "$OUT/bench-q4.err" | tee -a "$RES" ;;
halogen)
  [ "$MK" = fnext ] || { log "halogen serves Flash-Next only"; exit 1; }
  docker pull -q "ghcr.io/peonist-ai/halogen-flash-server:${HALOGEN_TAG:-0.20.1}" > "$OUT/pull.log" 2>&1 || { log "PULL FAILED: $(tail -2 "$OUT/pull.log")"; exit 1; }
  log "halogen ${HALOGEN_TAG:-0.20.1} $(docker image inspect -f '{{.Id}}' "ghcr.io/peonist-ai/halogen-flash-server:${HALOGEN_TAG:-0.20.1}" | cut -c8-19)"
  PORT=8731; halogen_up || { log "halogen models dir not writable"; exit 1; }
  export H2H_REQ_EXTRA='{"reasoning_effort":"none"}'
  bench_arm halogen 1,4,8 2160 ;;   # first start downloads ~118 GB: allow 3 h
*) log "unknown engine $ENGINE"; exit 1 ;;
esac
log "DONE"
