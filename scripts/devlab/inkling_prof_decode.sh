#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# devlab decode profiler (PR-3b gate): serve atlas-spark under rocprofv3,
# send one 512-token JSON decode request, and report per-kernel launches,
# avg us, launches/token and inter-kernel gap stats for C1 decode.
#
#   scripts/devlab/inkling_prof_decode.sh
set -u
OUT=${DEVLAB_OUT:-${OUT:-$PWD}}; RES=$OUT/RESULTS.txt; mkdir -p "$OUT"
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
HFC=${HFC:-/mnt/scratch/cache/hf}
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
PORT=18086

RP=$(command -v rocprofv3 || command -v /opt/rocm/bin/rocprofv3 || true)
log "host $(hostname) hipcc: $(hipcc --version 2>/dev/null | head -1 || /opt/rocm/bin/hipcc --version | head -1) rocprofv3: ${RP:-MISSING}"

# --- ISA stats for the fused-kernel .cu files under this toolchain -----------------------------------
COMPAT=$REPO/crates/atlas-kernels/hip/compat
HIPCC=$(command -v hipcc || echo /opt/rocm/bin/hipcc)
for FP in $REPO/kernels/strix-hip/inkling-2-mini/nvfp4/inkling2_fused.cu \
          $REPO/kernels/gb10/common/w4a16_gemv.cu; do
  [ -f "$FP" ] || continue
  T=$(mktemp -d)
  "$HIPCC" -x hip --genco --offload-arch=gfx1151 -O3 -I"$COMPAT" \
    -include hip/hip_runtime.h -include atlas_hip_cache_hints.h -DHDIM=128 \
    --save-temps=obj "$FP" -o "$T/x.co" >"$T/cc.log" 2>&1 || {
      log "$(basename "$FP") compile FAILED: $(tail -2 "$T/cc.log")"; continue; }
  S=$(ls "$T"/*-hip*.s 2>/dev/null | head -1)
  [ -n "$S" ] || continue
  BN=$(basename "$FP" .cu)
  log "isa $BN: $(grep -cE 'vgpr_count|spill_count' "$S") kernel(s):"
  grep -E "^[[:space:]]*\.(amdhsa_(kernarg)?|vgpr_count|spill_count|group_segment_fixed_size|private_segment|wavefront_size)" "$S" \
    | grep -E "vgpr_count|spill_count|group_segment" | sed "s/^/  isa $BN /" | tee -a "$RES"
done

# --- serve under rocprofv3, one decode burst ---------------------------------------------------------
( export MODE=spark DSPARK=0 PORT MAX_BATCH=1 MAX_SEQ_LEN=32768 DATA_DIR=$HFC/atlas-inkling2 \
    SERVE_PREFIX="$RP --kernel-trace --stats --output-format csv -d $OUT/rp -o d --" \
  ; exec scripts/serve-inkling2-strix-linux.sh ) > "$OUT/serve-prof.log" 2>&1 &
APID=$!
for _ in $(seq 1 720); do
  curl -s -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $APID 2>/dev/null || { log "SERVE DIED"; tail -10 "$OUT/serve-prof.log" | cut -c1-200; exit 5; }
  sleep 4
done
log "serving; decode request (json prompt, 512 tok, temp 0)"
curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"inkling-2-mini","messages":[{"role":"user","content":"Reply with a JSON object {\"task\":\"list five open problems in algebraic topology\",\"problems\":[...]} ."}],
  "max_tokens":512,"temperature":0}' > "$OUT/resp.json" 2>&1
CT=$(python3 -c "import json;print(json.load(open('$OUT/resp.json'))['usage']['completion_tokens'])" 2>/dev/null || echo 0)
log "completion_tokens=$CT"
kill -INT $APID 2>/dev/null; pkill -INT -P $APID 2>/dev/null
for _ in $(seq 1 60); do kill -0 $APID 2>/dev/null || break; sleep 2; done
kill -9 $APID 2>/dev/null; sleep 8
log "faults $(grep -acE 'hipError|panic' "$OUT/serve-prof.log" 2>/dev/null || echo 0)"

# --- kernel table: launches, avg us, launches/token, gap stats ---------------------------------------
F=$(find "$OUT/rp" -name "*kernel_trace.csv" 2>/dev/null | head -1)
if [ -z "$F" ]; then log "no kernel_trace.csv (rocprofv3 missing or attach failed)"; exit 0; fi
python3 - "$F" "$CT" >> "$RES" <<'PY'
import csv,sys,collections
rows=[(int(r["Start_Timestamp"]),int(r["End_Timestamp"]),r["Kernel_Name"].split("(")[0][:72],
       r.get("Grid_Size",""),r.get("Workgroup_Size",""))
      for r in csv.DictReader(open(sys.argv[1]))]
rows.sort(); ct=max(int(sys.argv[2]),1)
# steady-state decode window: last contiguous run of launches; use the tail
# half of the trace (prefill + warmup excluded)
rows=rows[len(rows)//3:]
tot=sum(e-s for s,e,_,_,_ in rows); wall=rows[-1][1]-rows[0][0]
gaps=[rows[i+1][0]-rows[i][1] for i in range(len(rows)-1)]
gaps.sort()
med=gaps[len(gaps)//2]/1e3 if gaps else 0
c=collections.Counter(); t=collections.Counter()
for s,e,n,_,_ in rows: c[n]+=1; t[n]+=e-s
launches=sum(c.values())
print(f"  prof: kernels {launches}  launches/token {launches/ct:.1f}  wall {wall/1e6:.2f}s gpu {tot/1e6:.2f}s busy {tot/wall*100:.1f}%  medgap {med:.2f}us  gap_total {sum(gaps)/1e6:.2f}us-> {(wall-tot)/1e6:.2f}ms/token")
for n,x in t.most_common(30):
    print(f"  K {x/1e6:8.1f} ms {x/tot*100:5.1f}%  n={c[n]:6d}  {x/c[n]/1e3:8.1f} us  {x/ct/1e3:7.1f} us/tok  {n}")
PY
