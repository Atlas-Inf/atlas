#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix: Inkling 2 Mini drafted == plain text on strix-hip (train #421 leg; this alias branch only).
#   scripts/devlab/inkling_exact.sh [modes...]   modes: strict spark (default both)
# Per mode: serve DSPARK=0 then DSPARK=1 (the start script's drafter env) with scripts/serve-inkling2-strix-linux.sh,
# 8 fixed prompts greedy (reasoning low, 320 tokens) at C1 and C8, compare texts per prompt; spark plain adds
# agentic-webserver x10. On GB10 (#421, sparky 187) drafted == plain 8/8 at C1/C2/C8 on both checkpoints.
set -uo pipefail
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
PORT=18087; HFC=${HF_HOME:-$HOME/.cache/huggingface}
log "ref $(git rev-parse --short=9 HEAD) inkling_exact modes: ${*:-strict spark} host $(hostname)"
greedy() { # tag conc
  python3 - "$PORT" "$MID" "$1" "$2" "$OUT" <<'PY'
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
port, mid, tag, conc, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
P = ["Write a calm, detailed essay about how lighthouses were staffed and maintained in the 1800s.",
     "Explain how sourdough starter cultures work, as plain paragraphs for a home baker.",
     "Write a Python function that parses ISO-8601 durations like P3DT4H5M into seconds, with tests.",
     "Return a JSON object describing three fictional planets with name, radius_km, moons and atmosphere.",
     "Describe the history of the printing press and its effect on literacy in Europe, in prose.",
     "Implement a thread-safe LRU cache in Rust with get and put, and explain the design briefly.",
     "List ten practical tips for reducing household energy use, each with a one-sentence reason.",
     "Explain how tides are caused by the moon and the sun, for a curious teenager, in paragraphs."]
def one(p):
    body = {"model": mid, "messages": [{"role": "user", "content": p}], "max_tokens": 320, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    m = r["choices"][0]["message"]; return (m.get("content") or "") + "|" + (m.get("reasoning_content") or m.get("reasoning") or "")
with ThreadPoolExecutor(conc) as ex: rows = list(ex.map(one, P))
json.dump(rows, open(f"{out}/greedy-{tag}.json", "w"))
PY
}
cmp() { python3 -c "
import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
s=sum(x==y for x,y in zip(a,b)); d=[next((i for i in range(min(len(x),len(y))) if x[i]!=y[i]), min(len(x),len(y))) for x,y in zip(a,b) if x!=y]
print(f'{sys.argv[3]}: {s}/{len(a)} identical' + (f' (first divergence chars {d})' if d else ''))" "$OUT/greedy-$1.json" "$OUT/greedy-$2.json" "$1 vs $2" | tee -a "$RES"; }
serve() { # mode dspark
  ( export MODE=$1 DSPARK=$2 PORT MAX_BATCH=8 MAX_SEQ_LEN=40960 DATA_DIR=$HFC/atlas-inkling2; exec scripts/serve-inkling2-strix-linux.sh ) > "$OUT/serve-$1-d$2.log" 2>&1 &
  APID=$!; ENGINE_ALIVE() { kill -0 $APID 2>/dev/null; }
  ENGINE_DOWN() { kill -INT $APID 2>/dev/null; pkill -INT -P $APID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $APID 2>/dev/null || break; sleep 2; done; kill -9 $APID 2>/dev/null; sleep 5; true; }
  wait_ready $PORT 720 || { log "$1 d$2 SERVE FAILED"; grep -aiE "error|fault|panick" "$OUT/serve-$1-d$2.log" | tail -4 | cut -c1-250 | tee -a "$RES"; ENGINE_DOWN; return 1; }
  log "$1 d$2 serving ($MID)"; }
faults() { grep -acE " ERROR |panicked|MEMORY_FAULT|illegal" "$OUT/serve-$1.log"; }
for m in ${*:-strict spark}; do
  if serve $m 0; then greedy $m-plain-c1 1; greedy $m-plain-c8 8
    if [ $m = spark ]; then
      SHIM_DIR=$(ls -dt target/release/build/atlas-kernels-*/out 2>/dev/null | while read -r d; do [ -f "$d/libcuda.so" ] && { echo "$d"; break; }; done)
      LD_LIBRARY_PATH="$SHIM_DIR:/opt/rocm/lib" target/release/spark benchmark run agentic-webserver --url http://127.0.0.1:$PORT --model "$MID" --yes \
        --param iterations=10 --param wall_budget_s=100000 --format json --quiet --no-fail-on-verdict > "$OUT/agentic-$m.json" 2> "$OUT/agentic-$m.stderr"
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('RESULT agentic', sys.argv[2], [(s['label'], s['value']) for s in d['frame']['summary']])" "$OUT/agentic-$m.json" $m 2>&1 | tail -1 | tee -a "$RES"
    fi
    log "$m d0 faults $(faults $m-d0)"; ENGINE_DOWN; fi
  if serve $m 1; then greedy $m-ds-c1 1; greedy $m-ds-c8 8; log "$m d1 faults $(faults $m-d1)"
    grep -aoE "mean_na=[0-9.]+" "$OUT/serve-$m-d1.log" | tail -5 | tr '\n' ' ' | sed "s/^/  $m d1 accept: /" | tee -a "$RES"; echo | tee -a "$RES"; ENGINE_DOWN; fi
  cmp $m-plain-c1 $m-ds-c1; cmp $m-plain-c8 $m-ds-c8; cmp $m-plain-c1 $m-plain-c8
done
log "DONE"
