#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: why does Atlas fail BFCL live_irrelevance on nvidia/Qwen3.8-27B-NVFP4 (ST-995 38.64) when stock
# llama.cpp on the SAME checkpoint scores 89.77? Serve each engine on the runner, replay the same live_irrelevance items
# (irrelevance_probe.py: tools as BFCL FC passes them, greedy, thinking off), and keep every reply for a side-by-side diff.
#
#   scripts/devlab/irrelevance_probe.sh [KEY=VAL ...]      KEYS: N (items, default 60), NUM_DRAFTS (Atlas; unset = MTP recipe)
set -uo pipefail
for kv in "$@"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
N=${N:-60}; PORT=18087
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
model_spec 27b; pyhf
DATA="$OUT/BFCL_v4_live_irrelevance.json"
curl -sfL -m 120 -o "$DATA" https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_irrelevance.json \
  || { log "could not fetch the BFCL live_irrelevance data"; exit 1; }
log "live_irrelevance items: $(grep -c . "$DATA"), probing the first $N"

# Atlas, the ST-995 recipe (MTP unless NUM_DRAFTS=0), NVIDIA NVFP4
[ -x target/release/spark ] || { log "no target/release/spark"; exit 1; }
hf_get "$NVFP4_REPO" "" >/dev/null || { log "NVFP4 DOWNLOAD FAILED"; exit 1; }
AENV=(GPU_UTIL=0.70 MAX_SEQ_LEN=20480 MAX_BATCH=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0); [ -n "${NUM_DRAFTS:-}" ] && AENV+=(NUM_DRAFTS=$NUM_DRAFTS)
atlas_up $PORT "$NVFP4_REPO" "${AENV[@]}"
if wait_ready $PORT; then python3 "$HERE/irrelevance_probe.py" $PORT "$MID" "$DATA" "$OUT/atlas.jsonl" "$N" | tee -a "$RES"; else log "atlas SERVE FAILED"; fi
ENGINE_DOWN "$OUT/serve-atlas.log"

# llama.cpp, the same checkpoint through its own converter
llama_build && llama_convert_nvfp4 "$NVFP4_REPO" || { log "llama.cpp side unavailable"; exit 1; }
llama_up $PORT 1 20480 "$NVFP4_GGUF"
if wait_ready $PORT; then python3 "$HERE/irrelevance_probe.py" $PORT "$MID" "$DATA" "$OUT/llamacpp.jsonl" "$N" | tee -a "$RES"; else log "llama.cpp SERVE FAILED"; fi
ENGINE_DOWN "$OUT/serve-llamacpp.log"

python3 - "$OUT/atlas.jsonl" "$OUT/llamacpp.jsonl" <<'PY' | tee -a "$RES"
import json, sys
a = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
b = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
diff = [i for i in a if i in b and bool(a[i]["tool_calls"]) != bool(b[i]["tool_calls"])]
print(f"  items where exactly one engine called a tool: {len(diff)}")
for i in diff[:6]:
    print(f"  --- {i}\n    atlas   : calls={a[i]['tool_calls']} finish={a[i].get('finish')} content={a[i].get('content','')[:240]!r}"
          f"\n    llamacpp: calls={b[i]['tool_calls']} finish={b[i].get('finish')} content={b[i].get('content','')[:240]!r}")
PY
log "DONE"
