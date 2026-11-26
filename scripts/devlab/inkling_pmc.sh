#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# devlab PMC profile leg: rocprofv3 hardware counters on the prefill kernels,
# atlas-spark (the ref the workflow built) and Ciru v1.5.0, on ONE runner.
#
#   scripts/devlab/inkling_pmc.sh [atlas] [ciru]      default: both
#
# Per engine: serve under `rocprofv3 --kernel-trace --stats --pmc <counters>`,
# run the 8K prefill probes, stop, then tabulate per-kernel durations (from the
# trace, which also carries grid/workgroup/VGPR/SGPR/LDS per dispatch) and the
# PMC counters present on gfx1151 (VALU/LDS instruction mix, LDS bank
# conflicts, TCC hit/miss, EA read/write requests as the DRAM-byte proxy,
# waves/waits). Dumps `rocprofv3 --list-counters` first so whatever this agent
# does not expose is visible in counters.txt rather than silently absent.
set -u
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"; pyhf
PORT=18086; HFC=${HF_HOME:-$HOME/.cache/huggingface}
DO_ATLAS=1; DO_CIRU=1
[ $# -gt 0 ] && { DO_ATLAS=0; DO_CIRU=0; for a in "$@"; do case $a in atlas) DO_ATLAS=1;; ciru) DO_CIRU=1;; esac; done; }

RP=$(command -v rocprofv3 || command -v /opt/rocm/bin/rocprofv3 || true)
[ -n "${RP:-}" ] || { log "rocprofv3 MISSING; cannot run the PMC leg"; exit 0; }
log "pmc host $(hostname) ref $(git rev-parse --short=9 HEAD) rocprofv3: $RP"

# --- counters available on this agent ---------------------------------------------------
# This rocprofv3 has no --list-counters; probe each candidate against /bin/true
# (a name the tool rejects fails at argument parse before any GPU is touched).
CANDS="SQ_INSTS SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_SMEM SQ_INSTS_LDS SQ_INSTS_FLAT \
SQ_INSTS_EXP SQ_INSTS_MMA SQ_INSTS_VFMA SQ_INSTS_WMMA SQ_WAVES SQ_BUSY_CYCLES SQ_IDLE_ANY \
SQ_WAIT_ANY SQ_WAIT_TOTAL SQ_WAIT_INST_LDS SQ_WAIT_INST_VMEM SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE \
GRBM_COUNT GRBM_GUI_ACTIVE TCC_HIT TCC_MISS TCC_EA_RDREQ TCC_EA_WRREQ \
TCC_EA_RDREQ_32B TCC_EA_RDREQ_64B TCC_EA_WRREQ_32B TCC_EA_WRREQ_64B"
USE=""; : > "$OUT/counters.txt"
for c in $CANDS; do
  if "$RP" --pmc "$c" -- /bin/true > /dev/null 2>>"$OUT/counters.err"; then
    USE="$USE $c"; echo "$c" >> "$OUT/counters.txt"
  fi
done
# too many counters per pass serializes the replay into many passes; cap at 12,
# keeping the mix/shape counters first
USE=$(echo $USE | tr ' ' '\n' | head -12 | tr '\n' ' ')
log "pmc counters:$USE"

# --- per-dispatch kernel table + counters ----------------------------------------------
analyse() { # label dir
  python3 - "$1" "$2" >> "$RES" <<'PY'
import csv, sys, collections, glob, os
label, d = sys.argv[1], sys.argv[2]
def rows(p):
    return list(csv.DictReader(open(p)))
tr = None
for f in sorted(glob.glob(os.path.join(d, "**", "*kernel_trace*.csv"), recursive=True)):
    tr = f
    break
names = collections.Counter(); dur = collections.Counter(); gxs = collections.defaultdict(list)
vg = {}; sg = {}; ldsb = {}; wgs = {}
if tr:
    for r in rows(tr):
        n = r["Kernel_Name"].split("(")[0][:72]
        names[n] += 1
        try:
            dur[n] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        except Exception:
            pass
        try:
            gxs[n].append(int(r.get("Grid_Size_X") or 0))
        except Exception:
            pass
        for k, m in (("VGPR_Count", vg), ("SGPR_Count", sg), ("LDS_Block_Size", ldsb), ("Workgroup_Size", wgs)):
            v = r.get(k)
            if v:
                try:
                    m[n] = int(v)
                except Exception:
                    pass
tot = sum(dur.values())
print(f"  [{label}] trace: {tot/1e6:.1f} ms gpu over {sum(names.values())} launches")
for n, t in dur.most_common(15):
    x = f"  K {t/1e6:8.1f} ms n={names[n]:5d} {t/names[n]/1e3:8.1f} us"
    if n in wgs:
        x += f" wg={wgs[n]} vgpr={vg.get(n,'?')} sgpr={sg.get(n,'?')} lds={ldsb.get(n,'?')}"
    if gxs.get(n):
        x += f" gx~{int(sum(gxs[n])/len(gxs[n]))}"
    print(f"{x}  {n}")
pmc_files = [f for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
             if f != tr and "trace" not in os.path.basename(f) and "agent" not in os.path.basename(f)]
cnt = collections.defaultdict(lambda: collections.Counter())
for f in pmc_files:
    try:
        rs = rows(f)
    except Exception:
        continue
    if not rs:
        continue
    keys = set(rs[0])
    if "Counter_Name" in keys:  # long format
        for r in rs:
            n = (r.get("Kernel_Name") or "?").split("(")[0][:72]
            try:
                cnt[n][r["Counter_Name"]] += float(r["Counter_Value"])
            except Exception:
                pass
    elif "Kernel_Name" in keys:  # wide format: counter names as columns
        for r in rs:
            n = r["Kernel_Name"].split("(")[0][:72]
            for k, v in r.items():
                if k.startswith(("SQ_", "TCC_", "GRBM_", "LDS_", "TCP_", "TA_")):
                    try:
                        cnt[n][k] += float(v)
                    except Exception:
                        pass
for n, _ in dur.most_common(15):
    if n in cnt:
        print(f"  PMC {n}")
        for c, v in cnt[n].most_common():
            print(f"      {c:28s} {v:.0f}   {v/(dur[n]/1e3):8.0f} /us")
PY
}

kill_engine() { # pid log
  kill -INT "$1" 2>/dev/null; pkill -INT -P "$1" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$1" 2>/dev/null || break; sleep 2; done
  kill -9 "$1" 2>/dev/null; pkill -9 -P "$1" 2>/dev/null; sleep 8
  log "$2 faults $(grep -acE 'hipError|panic' "$OUT/$2" 2>/dev/null || echo 0)"
}

# --- atlas: this ref's build under rocprofv3 --------------------------------------------
if [ $DO_ATLAS = 1 ]; then
  ( export MODE=spark DSPARK=0 PORT MAX_BATCH=8 MAX_SEQ_LEN=40960 DATA_DIR=$HFC/atlas-inkling2 \
      SERVE_PREFIX="$RP --kernel-trace --stats --pmc $USE --output-format csv -d $OUT/rpa -o atlas --" ; \
    exec scripts/serve-inkling2-strix-linux.sh ) > "$OUT/serve-pmc-atlas.log" 2>&1 &
  APID=$!
  for _ in $(seq 1 720); do
    curl -s -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $APID 2>/dev/null || { log "ATLAS SERVE DIED"; tail -8 "$OUT/serve-pmc-atlas.log" | cut -c1-200; exit 5; }
    sleep 4
  done
  log "atlas serving; prefill probes under pmc"
  python3 "$HERE/prefill_h2h.py" $PORT atlas 2>&1 | tail -6 | tee -a "$RES"
  kill_engine $APID serve-pmc-atlas.log
  analyse atlas "$OUT/rpa"
fi

# --- ciru v1.5.0 under rocprofv3 ----------------------------------------------------------
if [ $DO_CIRU = 1 ]; then
  CR=${RUNNER_TEMP:-/tmp}/ciru; mkdir -p "$CR"
  if [ ! -x "$CR/opt/serve.sh" ]; then
    rm -rf "$CR/src"; git clone -q https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151.git "$CR/src" \
      && git -C "$CR/src" checkout -q v1.5.0 || { log "ciru clone failed"; exit 0; }
    log "ciru $(git -C "$CR/src" rev-parse --short=9 HEAD) (v1.5.0)"
    if ! command -v uv >/dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$CR/bin" sh > "$OUT/uv-install.log" 2>&1
      export PATH="$CR/bin:$PATH"
    fi
    ( uv venv "$CR/venv" \
        && uv pip install -q --python "$CR/venv/bin/python" --target "$CR/tdeps" --no-deps 'huggingface-hub==1.31.0' 'safetensors==0.8.0' ) \
      > "$OUT/ciru-venv.log" 2>&1 || { log "CIRU VENV FAILED: $(tail -2 "$OUT/ciru-venv.log" | tr '\n' ' ')"; exit 0; }
    ( cd "$CR/src" && bash scripts/install.sh --venv "$CR/venv" --transformers-deps "$CR/tdeps" --prefix "$CR/opt" ) \
      > "$OUT/ciru-install.log" 2>&1 || { log "CIRU INSTALL FAILED: $(grep -aiE 'error|fail' "$OUT/ciru-install.log" | tail -3 | cut -c1-200)"; exit 0; }
  fi
  CK=$(hf_get jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 "" edcbab56fedf986ac6d6da5fbe6ede0f163556e5) \
    || { log "ciru bundle download failed"; exit 0; }
  "$RP" --kernel-trace --stats --pmc $USE --output-format csv -d "$OUT/rpc" -o ciru -- \
    "$CR/opt/serve.sh" --model "$CK" --mode single --context 65536 --port $PORT --served-model-name main \
    > "$OUT/serve-pmc-ciru.log" 2>&1 &
  CPID=$!
  ok=0
  for _ in $(seq 1 720); do
    curl -s -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $CPID 2>/dev/null || break
    sleep 4
  done
  if [ $ok = 1 ]; then
    log "ciru serving; prefill probes under pmc"
    python3 "$HERE/prefill_h2h.py" $PORT ciru 2>&1 | tail -6 | tee -a "$RES"
  else
    log "ciru SERVE FAILED under rocprofv3"
    tail -8 "$OUT/serve-pmc-ciru.log" | cut -c1-200 | tee -a "$RES"
  fi
  kill_engine $CPID serve-pmc-ciru.log
  pkill -f "opt/serve.sh" 2>/dev/null; sleep 3  # vLLM workers spawned below rocprofv3
  analyse ciru "$OUT/rpc"
fi
log "DONE"
