#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# devlab prefill profiler: serve atlas-spark under rocprofv3, run the prefill
# h2h probes (7773 / 30993), dump the top kernels, and report the flash
# kernel's register/occupancy stats under THIS runner's hipcc.
#
#   scripts/devlab/inkling_prof.sh
set -u
OUT=${DEVLAB_OUT:-${OUT:-$PWD}}; RES=$OUT/RESULTS.txt; mkdir -p "$OUT"
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
HFC=${HFC:-/mnt/scratch/cache/hf}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
PORT=18086

RP=$(command -v rocprofv3 || command -v /opt/rocm/bin/rocprofv3 || true)
log "host $(hostname) hipcc: $(hipcc --version 2>/dev/null | head -1 || /opt/rocm/bin/hipcc --version | head -1) rocprofv3: ${RP:-MISSING}"

# --- kernel ISA stats under this toolchain ----------------------------------------------------------
FP=$REPO/kernels/strix-hip/inkling-2-mini/nvfp4/inkling2_flash_prefill.cu
COMPAT=$REPO/crates/atlas-kernels/hip/compat
HIPCC=$(command -v hipcc || echo /opt/rocm/bin/hipcc)
if [ -f "$FP" ]; then
  T=$(mktemp -d)
  "$HIPCC" -x hip --genco --offload-arch=gfx1151 -O3 -I"$COMPAT" \
    -include hip/hip_runtime.h -include atlas_hip_cache_hints.h -DHDIM=128 \
    --save-temps=obj "$FP" -o "$T/fp.co" >"$T/cc.log" 2>&1 && {
      S=$(ls "$T"/inkling2_flash_prefill-hip*.s | head -1)
      cp "$S" "$OUT/fp.s" 2>/dev/null || true
      grep -E "vgpr_count|spill_count|group_segment|wavefront_size|max_flat" "$S" \
        | sed 's/^/  isa /' | tee -a "$RES"
    } || log "flash .cu compile FAILED: $(tail -2 "$T/cc.log")"
fi

# --- serve under rocprofv3 (single prefill burst) ---------------------------------------------------
( export MODE=spark DSPARK=0 PORT MAX_BATCH=8 MAX_SEQ_LEN=40960 DATA_DIR=$HFC/atlas-inkling2 \
    SERVE_PREFIX="$RP --kernel-trace --stats --output-format csv -d $OUT/rp -o p --"
  exec scripts/serve-inkling2-strix-linux.sh ) > "$OUT/serve-prof.log" 2>&1 &
APID=$!
for _ in $(seq 1 360); do
  curl -s -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $APID 2>/dev/null || { log "SERVE DIED"; tail -10 "$OUT/serve-prof.log" | cut -c1-200; exit 5; }
  sleep 4
done
log "serving; running prefill probes"
python3 "$HERE/prefill_h2h.py" $PORT prof 2>&1 | tail -6 | tee -a "$RES"
kill -INT $APID 2>/dev/null; pkill -INT -P $APID 2>/dev/null
for _ in $(seq 1 60); do kill -0 $APID 2>/dev/null || break; sleep 2; done
kill -9 $APID 2>/dev/null; sleep 8
log "faults $(grep -acE 'hipError|panic' "$OUT/serve-prof.log" 2>/dev/null || echo 0)"

# --- kernel table ------------------------------------------------------------------------------------
F=$(find "$OUT/rp" -name "*kernel_trace.csv" 2>/dev/null | head -1)
if [ -z "$F" ]; then log "no kernel_trace.csv (rocprofv3 missing or attach failed)"; exit 0; fi
python3 - "$F" >> "$RES" <<'PY'
import csv,sys,collections
rows=[(int(r["Start_Timestamp"]),int(r["End_Timestamp"]),r["Kernel_Name"].split("(")[0][:64])
      for r in csv.DictReader(open(sys.argv[1]))]
rows.sort(); tot=sum(e-s for s,e,_ in rows); wall=rows[-1][1]-rows[0][0]
c=collections.Counter(); t=collections.Counter()
for s,e,n in rows: c[n]+=1; t[n]+=e-s
print(f"  prof: wall {wall/1e6:.2f}s gpu {tot/1e6:.2f}s busy {tot/wall*100:.1f}%")
for n,x in t.most_common(20):
    print(f"  K {x/1e6:8.1f} ms {x/tot*100:5.1f}%  n={c[n]:6d}  {x/c[n]/1e3:8.1f} us  {n}")
PY
