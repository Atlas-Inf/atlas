#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench (G3): kernel attribution of Atlas Flash-Next prefill under rocprofv3.
#
#   scripts/devlab/fnext_prefill_prof.sh [KEY=VAL ...]      e.g. GPU_UTIL=0.76 (the runner OOMs at 0.92); PHASE=depth profiles
#   1.9k-token extensions of one cached 16k prefix instead of cold prompts
#
# Serves nvidia/Qwen3.8-Flash-Next-NVFP4 from the workflow's build with spark wrapped in rocprofv3 --kernel-trace,
# then sends prefill-only requests (max_tokens 1, distinct prompts): 3 x ~2k, 2 x ~16k tokens. Kernels are
# summed per name over the whole serve and printed top-40; the load-time kernels are named, so read past them.
# Why: the 11-10 runner 3-way puts Atlas pp at 740 / 344 tok/s (depth 0 / 16k) vs Halogen 1658 / 1471.
set -euo pipefail
for kv in "$@"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
[ -x target/release/spark ] || { log "no target/release/spark (run with job=bench)"; exit 1; }
command -v rocprofv3 >/dev/null || PATH=$PATH:/opt/rocm/bin; command -v rocprofv3 >/dev/null || { log "no rocprofv3"; exit 1; }
REPO=${REPO:-nvidia/Qwen3.8-Flash-Next-NVFP4}; BIG=model-fp8-mtp-ple.safetensors; PORT=${PORT:-8093}
"${PY:-python3}" -c "import huggingface_hub" 2>/dev/null || { V="$HFC/.hfvenv"; [ -x "$V/bin/python" ] || python3 -m venv "$V"; "$V/bin/python" -c "import huggingface_hub" 2>/dev/null || "$V/bin/pip" install -q huggingface_hub >>"$OUT/download.log" 2>&1; PY="$V/bin/python"; }
NV=$("${PY:-python3}" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], ignore_patterns=[sys.argv[2]]))' "$REPO" "$BIG" 2>>"$OUT/download.log") \
  && { [[ "$REPO" != *Flash-Next* ]] || curl -fsSL -C - --retry 5 --retry-all-errors -o "$NV/$BIG" "${HF_ENDPOINT:-https://huggingface.co}/$REPO/resolve/main/$BIG" 2>>"$OUT/download.log"; } \
  || { log "DOWNLOAD FAILED: $(grep -av '^ ' "$OUT/download.log" | tail -6)"; exit 1; }

# spark under the profiler: serve-amd.sh execs $ATLAS_BIN, so point it at a wrapper.
W="$OUT/spark-prof"; mkdir -p "$OUT/prof"
if [ "${NOPROF:-0}" = 1 ]; then W="$PWD/target/release/spark"; else printf '#!/usr/bin/env bash\nexec rocprofv3 --kernel-trace --output-format csv -d %q -o run -- %q "$@"\n' "$OUT/prof" "$PWD/target/release/spark" > "$W"; chmod +x "$W"; fi
( exec env RUST_LOG=info ATLAS_BIN="$W" PORT="$PORT" HOST=127.0.0.1 MODEL_NAME=m GPU_UTIL=${GPU_UTIL:-0.76} MAX_SEQ_LEN=32768 \
    ATLAS_KV_EXTERNAL_RESERVE_GB=0 MAX_BATCH=${MAX_BATCH:-1} ./serve-amd.sh "$REPO" --disable-thinking --request-timeout 0 --enable-prefix-caching ) > "$OUT/atlas-server.log" 2>&1 &
APID=$!; ENGINE_ALIVE() { kill -0 "$APID" 2>/dev/null; }
wait_ready "$PORT" 720 || { log "SERVE FAILED: $(grep -aE 'ERROR|allocfail|panick' "$OUT/atlas-server.log" | tail -3)"; kill -9 "$APID" 2>/dev/null; exit 1; }
log "serving under rocprofv3"

python3 - "$PORT" <<'EOF' | tee -a "$RES"
import json, sys, time, urllib.request, random
port = sys.argv[1]
words = "river stone lantern harvest copper meadow signal orbit canvas thunder violet ember glacier compass atlas".split()
def ask(tag, n_words, seed):
    r = random.Random(seed)
    text = " ".join(r.choice(words) for _ in range(n_words))
    body = {"model": "m", "messages": [{"role": "user", "content": f"[{tag}] Summarize: {text}"}], "max_tokens": 1, "temperature": 0}
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=1800)).get("usage", {})
    dt = time.time() - t
    print(f"  [prof] {tag}: prompt {u.get('prompt_tokens')} tok in {dt:.2f}s -> {u.get('prompt_tokens', 0) / dt:.0f} tok/s", flush=True)
import os
if os.environ.get("PHASE") == "depth":
    # Prefix-cached extension (the h2h "depth 16384" row: 1.86k new tokens on a cached 16k prefix, 5.4 s vs Halogen 1.26 s).
    r = random.Random(7); base = " ".join(r.choice(words) for _ in range(16000))
    def ext(tag, tail_seed, tail_words):
        rr = random.Random(tail_seed); tail = " ".join(rr.choice(words) for _ in range(tail_words))
        body = {"model": "m", "messages": [{"role": "user", "content": f"Summarize: {base}"}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": f"[{tag}] {tail}"}], "max_tokens": 1, "temperature": 0}
        t = time.time(); req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        u = json.load(urllib.request.urlopen(req, timeout=1800)).get("usage", {}); dt = time.time() - t
        print(f"  [prof] {tag}: prompt {u.get('prompt_tokens')} tok (cached {u.get('prompt_tokens_details', {}) or {}}) in {dt:.2f}s", flush=True)
    ext("warm-16k", 0, 10)
    for i in range(5): ext(f"ext-{i}", 300 + i, 1900)
elif os.environ.get("PHASE") == "c8":
    # G1: steady C8 decode (8 concurrent 256-token greedy streams on short prompts) to attribute decode kernels.
    import threading
    def gen(i):
        body = {"model": "m", "messages": [{"role": "user", "content": f"[c8-{i}] Write a long story about a {words[i]}."}], "max_tokens": 256, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        json.load(urllib.request.urlopen(req, timeout=1800))
    for rnd in range(2):
        t = time.time(); th = [threading.Thread(target=gen, args=(i,)) for i in range(8)]
        [x.start() for x in th]; [x.join() for x in th]
        print(f"  [prof] c8 round {rnd}: 8x256 tok in {time.time() - t:.2f}s -> {2048 / (time.time() - t):.1f} tok/s agg", flush=True)
else:
    for i in range(3): ask(f"2k-{i}", 1900, 100 + i)
    for i in range(2): ask(f"16k-{i}", 16000, 200 + i)
EOF
kill -INT "$APID" 2>/dev/null; for _ in $(seq 1 120); do kill -0 "$APID" 2>/dev/null || break; sleep 2; done; kill -9 "$APID" 2>/dev/null || true
cp "$OUT/atlas-server.log" "$OUT/serve-prof.log" 2>/dev/null || true

[ "${NOPROF:-0}" = 1 ] && { log DONE; exit 0; }
KT=$(find "$OUT/prof" -name '*kernel_trace.csv' | head -1)
[ -n "$KT" ] || { log "no kernel_trace.csv (profiler did not flush): $(ls -R "$OUT/prof" | head)"; exit 1; }
python3 - "$KT" <<'EOF' | tee -a "$RES"
import csv, sys, collections
tot, cnt = collections.Counter(), collections.Counter()
for r in csv.DictReader(open(sys.argv[1])):
    n = r.get("Kernel_Name") or r.get("KernelName") or "?"
    n = n.split("(")[0][:90]
    tot[n] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"]); cnt[n] += 1
s = sum(tot.values())
print(f"  [prof] {len(tot)} kernels, {s / 1e9:.2f} s GPU total")
for n, t in tot.most_common(40):
    print(f"  [prof] {t / 1e6:10.1f} ms {100 * t / s:5.1f}% x{cnt[n]:<7} {n}")
EOF
log DONE
