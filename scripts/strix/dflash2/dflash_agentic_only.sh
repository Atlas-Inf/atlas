#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DFlash2 overnight chain on AzeezStrix (gfx1151), port/dflash2-strix-linux:
#   leg 1  ST-995  — bfcl-subset golden draw (62/10/10, n=995, seed 42, temp 0,
#                    NO param overrides) via `spark benchmark run bfcl-subset`
#   leg 2  ST-996  — the local 12/23/46 n~1004 draw via inference-endpoint
#                    (same instrument as results_st996_unsloth_ccdaab7e)
#   leg 3  agentic — MLPerf edge agentic-coding performance leg (2.5h, 20
#                    trajectories / 1007 turns, inline IoU) via inference-endpoint
# Every leg serves nvidia/Qwen3.8-27B-NVFP4 + the incoai/Qwen3.8-27B-DFlash2
# drafter through serve-amd.sh (DFLASH=1) on ONE binary, and writes a
# fingerprint file next to its result. No thermal aborts, no watchdogs
# (AGENTS.md Strix run policy).
#
#   usage: GAMMA=8 strix-dflash2-agentic-only.sh        (run under setsid/nohup)
# This is the leg-3-only variant of strix-dflash2-overnight.sh (ST-996 dropped
# to fit the merge window; ST-995 already ran in dflash-overnight-20260915T1706Z).
set -uo pipefail

WT=/home/azeez/atlas-dflash2
BIN=$WT/target/release/spark
GAMMA="${GAMMA:-8}"
DRAFTER=/home/azeez/.models/dflash2            # incoai/Qwen3.8-27B-DFlash2 (BF16, block_size 8)
SNAP=/home/azeez/.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/dbb8f445b3145f8a4c18ddc769f032d57d32867c
MODEL=nvidia/Qwen3.8-27B-NVFP4
PORT=8093
EP=/home/azeez/endpoints
TS=$(date -u +%Y%m%dT%H%MZ)
OUT=/home/azeez/dp4a-ab/out/dflash-agentic-$TS
mkdir -p "$OUT"
BIN_SHA=$(sha256sum "$BIN" | awk '{print $1}')
COMMIT=$(git -C "$WT" rev-parse HEAD)

log() { echo "=== $(date -u +%FT%TZ) $* ===" | tee -a "$OUT/chain.log"; }

fingerprint() {  # <leg> <max_seq> <extra serve flags...>
  local leg=$1 maxseq=$2; shift 2
  {
    echo "leg=$leg"
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "host=$(hostname)"
    echo "branch=$(git -C "$WT" rev-parse --abbrev-ref HEAD) commit=$COMMIT"
    echo "binary=$BIN sha256=$BIN_SHA"
    echo "checkpoint=$MODEL revision=dbb8f445b3145f8a4c18ddc769f032d57d32867c"
    echo "drafter=incoai/Qwen3.8-27B-DFlash2 path=$DRAFTER gamma=$GAMMA option_b=${OPTION_B:-1} small_m_gemv=${ATLAS_DFLASH_SMALL_M_GEMV:-default}"
    echo "serve=serve-amd.sh DFLASH=1 GPU_UTIL=0.80 request-timeout=900 MAX_SEQ_LEN=$maxseq prefill=2048 kv=bf16 head=nvfp4 batch=1 ssm-slots=${SSM_SLOTS:-0} ssm-checkpoint-interval=${SSM_CKPT_INTERVAL:-16} dflash-window-size=${DFLASH_WINDOW_SIZE:-0}(full) dflash-ctx-window=${DFLASH_CTX_WINDOW:-$maxseq} mtp=off thinking=off(--disable-thinking) extra='$*'"
    echo "env=ATLAS_W4A16_DP4A=1(default) ATLAS_W4A16_VARIANT=v1 ATLAS_KV_EXTERNAL_RESERVE_GB=0 ATLAS_MTP_ACCEPT_DEBUG=1 HF_HUB_OFFLINE=1"
    echo "gpu_temp_edge=$(/opt/rocm/bin/amd-smi metric --temperature 2>/dev/null | sed -n 's/.*EDGE: *//p' | head -1)"
  } | tee "$OUT/$leg-fingerprint.txt"
}

stop_serve() {
  pkill -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null
  for _ in $(seq 1 15); do pgrep -f '[s]park[_a-zA-Z0-9.-]* serve' >/dev/null || break; sleep 2; done
  pkill -9 -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null; sleep 3
}

serve() {  # <serve-log> <max_seq> <extra serve flags...>
  local slog=$1 maxseq=$2; shift 2
  stop_serve
  ( cd "$WT" && HF_HUB_OFFLINE=1 RUST_LOG=info ATLAS_MTP_ACCEPT_DEBUG=1 \
      ATLAS_DFLASH_OPTION_B="${OPTION_B:-1}" \
      ATLAS_DFLASH_CTX_WINDOW="${DFLASH_CTX_WINDOW:-$maxseq}" \
      DFLASH=1 DRAFT_MODEL="$DRAFTER" DFLASH_GAMMA="$GAMMA" GPU_UTIL=0.80 \
      MAX_SEQ_LEN="$maxseq" PORT=$PORT HOST=127.0.0.1 MODEL_NAME="$MODEL" SSM_SLOTS="${SSM_SLOTS:-0}" SSM_CKPT_INTERVAL="${SSM_CKPT_INTERVAL:-16}" \
      ./serve-amd.sh "$MODEL" --disable-thinking --request-timeout 900 \
      --dflash-window-size "${DFLASH_WINDOW_SIZE:-0}" "$@" >"$slog" 2>&1 & echo $! >"$OUT/.serve.pid" )
  sleep 2
  SRV=$(cat "$OUT/.serve.pid")
  for _ in $(seq 1 450); do            # up to 15 min (drafter + 22 GB target)
    kill -0 "$SRV" 2>/dev/null || { log "SERVER DIED during startup ($slog)"; tail -40 "$slog" | tee -a "$OUT/chain.log"; return 1; }
    curl -fsS -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "server ready pid=$SRV"; return 0; }
    sleep 2
  done
  log "SERVER NOT READY after 15 min ($slog)"; return 1
}

accept_summary() {  # <serve-log> <leg>
  grep -E "mean_na|tok_step|DFLASH K=" "$1" | tail -n 2000 > "$OUT/$2-accept.txt" || true
  python3 - "$1" >> "$OUT/$2-accept.txt" <<'PY'
import re, statistics, sys
na, ts = [], []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"mean_na=([0-9.]+)", line)
    if m: na.append(float(m.group(1)))
    m = re.search(r"tok_step=([0-9.]+)", line)
    if m: ts.append(float(m.group(1)))
if na: print(f"ACCEPT SUMMARY: n={len(na)} mean_na avg={statistics.mean(na):.3f} median={statistics.median(na):.3f}")
if ts: print(f"ACCEPT SUMMARY: tok_step avg={statistics.mean(ts):.3f} median={statistics.median(ts):.3f}")
PY
}

# ─────────────────────────── leg 3: MLPerf agentic 2.5h perf leg ─────────────
# Dataset peaks at ~25.2K ISL -> 32768 ctx covers every sample (the 2026-09-15
# leg ran 24576 and dropped 3 prompts + cascades on Prompt-too-long 400s).
# DFlash conditioning now spans the full transcript: ATLAS_DFLASH_CTX_WINDOW
# defaults to maxseq (the 4096 default had frozen the drafter on the prompt's
# first 4K positions — root cause of the acceptance collapse) and
# --dflash-window-size 0 gives the drafter full-prefix attention.
# Prefix caching is required for the leg to
# finish inside its 4h cap (multi-turn replay); PCACHE=0 disables it if the
# pre-probe below shows the dflash+prefix-caching fault reported on GB10.
log "LEG3 MLPerf agentic 2.5h — dflash gamma=$GAMMA"
# 2026-09-15 22:39Z attempt at SSM_SLOTS=16 OOM'd: the engine raises the slot count
# to cover max-seq-len at the recipe's 256 tok/snapshot (16 -> 104 slots x 151 MB
# = 15.7 GB on top of 39 GB pre-KV + 5.9 GB KV + 2.4 GB MTP pools). Interval 128
# (2048 tok/snapshot) -> 20 slots (3.0 GB); everything else unchanged.
SSM_SLOTS=16
SSM_CKPT_INTERVAL=128
PCACHE_FLAGS=(--enable-prefix-caching)
[ "${PCACHE:-1}" = 0 ] && PCACHE_FLAGS=()
fingerprint agentic 32768 "${PCACHE_FLAGS[@]}"
if serve "$OUT/agentic-serve.log" 32768 "${PCACHE_FLAGS[@]}"; then
  # pre-probe: a ~12K-token prompt twice (multi-chunk prefill + cache hit) and a
  # short prompt twice; any server death here is recorded, then the leg runs anyway.
  python3 - "$PORT" "$MODEL" "$OUT" <<'PY'
import json, sys, urllib.request, hashlib
port, model, out = sys.argv[1:4]
filler = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog {i*7%997}" for i in range(1400))
def post(tag, content, max_tokens):
    body = {"model": model, "temperature": 0, "seed": 0, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
        txt = r["choices"][0]["message"]["content"] or ""
        print(f"pcache-probe {tag}: ok usage={r.get('usage',{}).get('prompt_tokens')}/{r.get('usage',{}).get('completion_tokens')} sha={hashlib.sha256(txt.encode()).hexdigest()[:12]}")
    except Exception as e:
        print(f"pcache-probe {tag}: FAILED {e}")
post("short-1", "Name three prime numbers below 20.", 32)
post("short-2", "Name three prime numbers below 20.", 32)
post("long-1", "Summarize this log in one sentence.\n" + filler, 64)
post("long-2", "Summarize this log in one sentence.\n" + filler, 64)
PY
  if ! kill -0 "$(cat "$OUT/.serve.pid")" 2>/dev/null; then
    log "pcache probe KILLED the server — restarting WITHOUT prefix caching for the leg"
    tail -60 "$OUT/agentic-serve.log" > "$OUT/agentic-pcache-crash-tail.log"
    fingerprint agentic 32768
    serve "$OUT/agentic-serve.log" 32768 || { log "LEG3 SKIPPED — serve failed"; stop_serve; exit 0; }
  fi
  RDAG=results_agentic_dflash_$TS
  CFGAG=$EP/examples/10_Edge_Agentic_Example/online_agentic_2.5h_dflash_$TS.yaml
  "$EP/.venv/bin/python" - "$EP/examples/10_Edge_Agentic_Example/online_agentic_2.5h_atlas_strix.yaml" "$CFGAG" "$MODEL" "$RDAG" "$PORT" <<'PY'
import sys, yaml
src, dst, model, rd, port = sys.argv[1:6]
c = yaml.safe_load(open(src))
c["model_params"]["name"] = model
c["report_dir"] = rd + "/"
c["endpoint_config"]["endpoints"] = [f"http://localhost:{port}"]
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print("agentic config ->", dst, "report_dir", rd)
PY
  ( cd "$EP" && . .venv/bin/activate && inference-endpoint benchmark from-config --config "$CFGAG" --report-dir "$RDAG/" ) \
    > "$OUT/agentic-harness.log" 2>&1
  log "agentic harness rc=$?"
  grep -iE "Score for|Completed in|successful|dropped|IoU|Estimated QPS" "$OUT/agentic-harness.log" | tail -20 | tee -a "$OUT/chain.log"
  accept_summary "$OUT/agentic-serve.log" agentic
  grep -ciE "illegal|fault|panicked|CUDA_ERROR|hipError" "$OUT/agentic-serve.log" | xargs -I{} log "agentic serve-log fault-line count: {}"
else
  log "LEG3 SKIPPED — serve failed"
fi
stop_serve
log "CHAIN DONE — $OUT"
