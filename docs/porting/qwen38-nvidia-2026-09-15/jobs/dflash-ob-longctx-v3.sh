#!/usr/bin/env bash
# JOB: dflash-ob-longctx (v3: 50 reps ~ 9.7K tokens; v2's 140 reps tokenized to 27,254 > max_seq_len and every request was rejected)
# (50ms flat at ~1.3K ctx on the profile). Prove it holds at depth:
# ~10K-token prompt, dflash+OB vs serial, watch propose/step timing and
# mean_na in the serve logs. Also exercises ctx-K/V accumulation at scale.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh

HDR=$'mode\tcompletion_tokens\twall_s\ttok_s\tsha\tfinish'
[ -f "$OUT/LONGCTX.tsv" ] || echo "$HDR" > "$OUT/LONGCTX.tsv"

# ~10K-token prompt: repeated realistic code block + a question at the end.
python3 - <<'PY' > "$OUT/body-long.json"
import json
block = '''
pub fn process_batch_%d(records: &[Record], cfg: &BatchConfig) -> Result<BatchStats, BatchError> {
    let mut stats = BatchStats::default();
    for (i, rec) in records.iter().enumerate() {
        if rec.is_tombstone() { stats.skipped += 1; continue; }
        let key = rec.derive_key(&cfg.key_schema)?;
        match self.index.entry(key) {
            Entry::Occupied(mut e) => { e.get_mut().merge(rec)?; stats.merged += 1; }
            Entry::Vacant(v) => { v.insert(rec.to_owned()); stats.inserted += 1; }
        }
        if i % cfg.flush_interval == 0 { self.flush_pending()?; }
    }
    stats.elapsed = self.clock.now() - stats.started;
    Ok(stats)
}
'''
body_text = "Below is a Rust source file. Read it, then answer the question at the end.\n\n```rust\n"
for i in range(50):
    body_text += block.replace("%d", str(i))
body_text += """```

Question: In one paragraph, describe what process_batch does when it encounters a tombstone record, and how entries are merged into the index.
"""
body = {"model": "nvidia/Qwen3.8-27B-NVFP4",
 "messages": [{"role": "user", "content": body_text}],
 "temperature": 0, "max_tokens": 256, "presence_penalty": 0,
 "frequency_penalty": 0, "repetition_penalty": 1.0,
 "reasoning_effort": "none", "stream": False}
json.dump(body, open("/dev/stdout", "w"))
PY
log "prompt bytes: $(wc -c < "$OUT/body-long.json")"

serve_long() {
    local slog="$1"; local dflash_on="$2"
    : > "$slog"
    local extra=()
    if [ "$dflash_on" = 1 ]; then
        extra=(--dflash --draft-model "$DRAFTER" --dflash-gamma "${GAMMA:-8}")
    fi
    ( setsid env ATLAS_MTP_ACCEPT_DEBUG=1 ${3:-} "$SPARK" serve "$MODEL" \
        --no-tui --bind 127.0.0.1 --port "$PORT" \
        --max-seq-len 16384 --max-batch-size 1 --max-num-seqs 1 \
        --gpu-memory-utilization 0.78 --kv-cache-dtype bf16 --lm-head-dtype bf16 \
        --enable-prefix-caching false \
        "${extra[@]}" \
        > "$slog" 2>&1 & echo $! > "$OUT/.serve.pid" )
    sleep 2; SRV_PID=$(cat "$OUT/.serve.pid")
    local i
    for i in $(seq 1 900); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { log "  healthy after ${i}s"; return 0; }
        kill -0 "$SRV_PID" 2>/dev/null || { log "  SERVER DIED"; tail -20 "$slog"; return 1; }
        sleep 1
    done
    return 1
}

probe_long() {
    local mode="$1" nreq="${2:-3}" r
    for r in $(seq 0 $((nreq - 1))); do
        local t0 t1
        t0=$(date +%s.%N)
        curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d @"$OUT/body-long.json" \
            > "$OUT/resp-$mode-$r.json" 2>&1
        t1=$(date +%s.%N)
        python3 - "$OUT/resp-$mode-$r.json" "$t0" "$t1" "$mode" <<'PY' >> "$OUT/LONGCTX.tsv"
import json,sys,hashlib
p,t0,t1,mode=sys.argv[1:]
try:
    d=json.load(open(p)); ct=d.get("usage",{}).get("completion_tokens") or 0
    wall=float(t1)-float(t0)
    text=d["choices"][0]["message"].get("content") or ""
    sha=hashlib.sha256(text.encode()).hexdigest()[:12]
    print(f"{mode}\t{ct}\t{wall:.2f}\t{ct/wall:.2f}\t{sha}\t{d['choices'][0].get('finish_reason')}")
except Exception as e:
    print(f"{mode}\tERROR\t{e}")
PY
    done
}

for cfg in serial optionb; do
    log "=== LEG $cfg longctx ==="
    if [ "$cfg" = serial ]; then
        serve_long "$OUT/serve-long-serial.log" 0 || continue
    else
        serve_long "$OUT/serve-long-optionb.log" 1 "ATLAS_DFLASH_OPTION_B=1" || continue
    fi
    probe_long "$cfg" 3
    safe_kill "$SRV_PID"; wait "$SRV_PID" 2>/dev/null || true; sleep 5
done
log "JOB DONE — check serve-long-optionb.log for propose/step timing at ~10K ctx"
