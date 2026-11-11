#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: Ripple-65B (#334, the strix-hip port) on the runner -- boot, greedy probe, speed.
#
#   scripts/devlab/ripple_bench.sh [KEY=VAL ...]     KEYS: GPU_UTIL (0.90), ARMS ("dflash ar"), MAX_SEQ_LEN (8192)
# arms:  dflash  ricursiveAI/Ripple-65B-Preview-NVFP4 + its DFlash2 drafter (the framework f263/f264 recipe, MAX_BATCH 1,
#                --dflash-window-size 16384); ar  the same checkpoint with no drafter.
# Each arm: a 3-prompt greedy probe (outputs kept, compared across arms: DFlash2 is output-preserving at greedy), then
# gufo_bench C1 pp ~2k / tg128 at depth 0 and 4096. Framework needed GPU_UTIL 0.97 with the drafter on 128 GB; the
# runner has 100 GiB GTT, so a SERVE FAILED here at the default is a memory-fit finding, not a port bug.
set -uo pipefail
for kv in "$@"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+\ -]*$ ]] && export "$kv"; done
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
[ -x target/release/spark ] || { log "no target/release/spark (run with job=bench)"; exit 1; }
pyhf; PORT=18087
dl() { "${PY:-python3}" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))' "$1" 2>>"$OUT/download.log"; }
t0=$(date +%s)
RM=$(dl ricursiveAI/Ripple-65B-Preview-NVFP4) && DM=$(dl ricursiveAI/Ripple-65B-Preview-DFlash2) \
  || { log "RIPPLE DOWNLOAD FAILED: $(grep -av '^ ' "$OUT/download.log" | tail -6)"; exit 1; }
log "atlas $(git rev-parse --short=9 HEAD), Ripple $(du -shL "$RM" | cut -f1) + DFlash2 $(du -shL "$DM" | cut -f1) in $(( $(date +%s) - t0 ))s"

probe() { python3 - "$PORT" "$1" "$OUT/probe-$1.json" <<'PY' | tee -a "$RES"
import json, sys, time, urllib.request
port, tag, path = sys.argv[1:4]; out = {}
for name, q in (("fib", "Write a Python function that returns the n-th Fibonacci number."), ("sky", "Explain in three sentences why the sky is blue."),
                ("essay", "Write a 150-word paragraph about the history of the bicycle.")):
    body = {"model": "m", "messages": [{"role": "user", "content": q}], "max_tokens": 160, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800)); dt = time.time() - t
    u = r["usage"]["completion_tokens"]; c = r["choices"][0]["message"].get("content") or ""; out[name] = c
    print(f"  [{tag}] {name}: {u} tok {u / dt:.2f} tok/s | {c[:90]!r}")
json.dump(out, open(path, "w"))
PY
}
for arm in ${ARMS:-dflash ar}; do
  ENV=(GPU_UTIL=${GPU_UTIL:-0.90} MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192} MAX_BATCH=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0)
  extra=(); [ "$arm" = dflash ] && ENV+=(DFLASH=1 DRAFT_MODEL="$DM") && extra=(--dflash-window-size 16384)
  ( exec env RUST_LOG=info ATLAS_BIN="$PWD/target/release/spark" PORT="$PORT" HOST=127.0.0.1 MODEL_NAME=m "${ENV[@]}" \
      ./serve-amd.sh "$RM" "${extra[@]}" --request-timeout 0 ) > "$OUT/serve-ripple-$arm.log" 2>&1 &
  APID=$!; ENGINE_ALIVE() { kill -0 "$APID" 2>/dev/null; }
  if wait_ready "$PORT" 720; then
    log "ripple_$arm serving"
    probe "ripple_$arm"
    python3 "$HERE/gufo_bench.py" "$PORT" "ripple_$arm" "$OUT/ripple_$arm.json" --depths 0,4096 --cs 1 --kinds mixed 2>&1 | grep -aE "^\s+\[" | tee -a "$RES"
  else
    log "ripple_$arm SERVE FAILED: $(grep -aE 'ERROR|allocfail|panick|Error' "$OUT/serve-ripple-$arm.log" | tail -3)"
  fi
  kill -INT "$APID" 2>/dev/null; for _ in $(seq 1 90); do kill -0 "$APID" 2>/dev/null || break; sleep 2; done; kill -9 "$APID" 2>/dev/null
  log "ripple_$arm faults: $(grep -acE 'panicked|illegal|HIP error|watchdog' "$OUT/serve-ripple-$arm.log") | spec: $(grep -aoE 'tok_step=[0-9.]+' "$OUT/serve-ripple-$arm.log" | tail -3 | tr '\n' ' ')"
  for _ in $(seq 1 60); do [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 40000 ] && break; sleep 2; done
done
[ -f "$OUT/probe-ripple_dflash.json" ] && [ -f "$OUT/probe-ripple_ar.json" ] && python3 - "$OUT" <<'PY' | tee -a "$RES"
import json, sys
a, b = (json.load(open(f"{sys.argv[1]}/probe-ripple_{k}.json")) for k in ("ar", "dflash"))
print("  greedy identical ar vs dflash:", {k: a[k] == b[k] for k in a})
PY
log DONE
