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
#
# 2026-09-24 recipe (carry branch, balloons + contiguous Marconi blobs):
#   GPU_UTIL=0.84 SSM_SLOTS=12 SSM_CKPT_INTERVAL=512 MAX_PREFILL_TOKENS=1024
#   MAX_SEQ_LEN=26624 DFLASH_CTX_WINDOW=12288 --dflash-window-size 2048
# Memory reality on the 61 GB host (learned 2026-09-24 across three legs):
#   pre-KV(~46-48 GB: weights+layers+drafter+balloons+acc+Marconi) + KV +
#   post-KV(~5.5 GB: Marconi blob 1.8 + SSM pools + drafter staging) must fit
#   under ~56 GB (60 GB device cap − 4 GB --oom-guard-mb floor). The
#   util→KV mapping is a knife edge: 0.81 → 2.7K-token pool (leg produced
#   965 EMPTY 200-responses — 'KV cache exhausted' mid-prefill returns a
#   vacuous completion the harness counts as successful), 0.84 → ~45-73K
#   tokens depending on where pre-KV lands this run, ≥0.86 → post-KV allocs
#   hit the guard floor and startup dies. A co-resident harness adds ~3 GB
#   host pressure: 0.84 was host-OOM-killed mid-leg (kern.log "Out of memory:
#   Killed process (spark)", 2026-09-24T05:08Z). For a valid leg either run
#   the harness OFF-BOX (HOST=0.0.0.0 + yaml endpoint to the tailscale IP)
#   or accept the OOM risk — the KV_MIN_TOKENS guard below aborts before a
#   starved pool can silently corrupt a leg.
set -uo pipefail

WT="${WT:-/home/azeez/atlas-dflash2}"
BIN=$WT/target/release/spark
GPU_UTIL="${GPU_UTIL:-0.84}"
KV_MIN_TOKENS="${KV_MIN_TOKENS:-40000}"   # abort leg if serve lands a pool below this
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-1024}"
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
    echo "serve=serve-amd.sh DFLASH=1 GPU_UTIL=$GPU_UTIL request-timeout=900 MAX_SEQ_LEN=$maxseq prefill=$MAX_PREFILL_TOKENS kv=bf16 head=nvfp4 batch=1 ssm-slots=${SSM_SLOTS:-0} ssm-checkpoint-interval=${SSM_CKPT_INTERVAL:-16} dflash-window-size=${DFLASH_WINDOW_SIZE:-0}(full) dflash-ctx-window=${DFLASH_CTX_WINDOW:-$maxseq} mtp=off thinking=off(--disable-thinking) extra='$*'"
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
      DFLASH=1 DRAFT_MODEL="$DRAFTER" DFLASH_GAMMA="$GAMMA" GPU_UTIL="$GPU_UTIL" \
      MAX_SEQ_LEN="$maxseq" MAX_PREFILL_TOKENS="$MAX_PREFILL_TOKENS" PORT=$PORT HOST="${HOST:-127.0.0.1}" MODEL_NAME="$MODEL" SSM_SLOTS="${SSM_SLOTS:-0}" SSM_CKPT_INTERVAL="${SSM_CKPT_INTERVAL:-16}" \
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
# 2026-09-24 recipe (carry branch): with the acc-prime + reserve/drafter
# balloons + contiguous Marconi blob, startup clears at util 0.81-0.84; 0.84
# was host-OOM-killed mid-leg (unified memory: device allocs are host RAM,
# ~56 GB RSS + harness crossed the 61 GB line) — stay at 0.81. 12 slots @
# 512-block ckpt interval = an SSM anchor every 8192 tokens (2x finer than the
# prior leg's 16K), contiguous pool ~1.8 GB. prefill 1024 halves the buffer
# arena. ctx window 12288 and --dflash-window-size 2048 bound the drafter's
# carry footprint. MAX_SEQ_LEN=26624 still covers the ~25.2K ISL peak.
SSM_SLOTS="${SSM_SLOTS:-12}"   # MUST be ≥ the count coverage auto-raises to:
# preflight sizes the reserve balloon from the REQUESTED slots, but the build
# raises slots to cover max_seq — asking for less balloons too small for the
# 1.8 GB contiguous Marconi blob, it spills to the fragmented tail, startup dies
SSM_CKPT_INTERVAL="${SSM_CKPT_INTERVAL:-512}"
DFLASH_CTX_WINDOW="${DFLASH_CTX_WINDOW:-12288}"
DFLASH_WINDOW_SIZE="${DFLASH_WINDOW_SIZE:-2048}"
MAXSEQ="${MAXSEQ:-26624}"
PCACHE_FLAGS=(--enable-prefix-caching)
[ "${PCACHE:-1}" = 0 ] && PCACHE_FLAGS=()
fingerprint agentic "$MAXSEQ" "${PCACHE_FLAGS[@]}"
if serve "$OUT/agentic-serve.log" "$MAXSEQ" "${PCACHE_FLAGS[@]}"; then
  # KV-floor guard: a starved pool silently corrupts the leg — the serve
  # returns EMPTY 200s on 'KV cache exhausted' mid-prefill and the harness
  # counts them successful (the 2026-09-24T1632Z leg: pool=2704 tokens →
  # 965/1007 empty turns → score 0.0048). Refuse to run under the floor.
  KV_TOK=$(grep -aoE "[0-9]+ max KV tokens" "$OUT/agentic-serve.log" | tail -1 | awk '{print $1}')
  if [ -z "${KV_TOK:-}" ] || [ "$KV_TOK" -lt "$KV_MIN_TOKENS" ]; then
    log "LEG3 ABORT — KV pool ${KV_TOK:-unknown} tokens < floor $KV_MIN_TOKENS; raise GPU_UTIL or shrink pre-KV (do NOT run: a starved pool yields an invalid leg)"
    stop_serve
    log "CHAIN DONE — $OUT"
    exit 0
  fi
  log "KV pool ${KV_TOK} tokens ≥ floor $KV_MIN_TOKENS — proceeding"
  # pre-probe: a ~12K-token prompt twice (multi-chunk prefill + cache hit) and a
  # short prompt twice. This also absorbs the one-time kernel JIT / CUDA-graph
  # capture cost — the 2026-09-24 leg without it paid ~594 s TTFT on turn 1,
  # tripped the per-turn timeout, and the harness dropped all 58 turns of that
  # conversation. Any server death here is recorded, then the leg runs anyway.
  python3 - "$PORT" "$MODEL" "$OUT" <<'PY'
import json, sys, urllib.request, hashlib
port, model, out = sys.argv[1:4]
filler = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog {i*7%997}" for i in range(1000))
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
    fingerprint agentic "$MAXSEQ"
    serve "$OUT/agentic-serve.log" "$MAXSEQ" || { log "LEG3 SKIPPED — serve failed"; stop_serve; exit 0; }
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
  # Invalid-leg detector: any mid-prefill KV exhaustion means the harness
  # counted vacuous 200s as successful turns — the leg is NOT a valid result.
  KVERR=$(grep -c "KV cache exhausted" "$OUT/agentic-serve.log" || true)
  [ "${KVERR:-0}" -gt 0 ] && log "LEG3 INVALID — $KVERR 'KV cache exhausted' prefill failures produced empty completions the harness counted as successful; score/turn-rate are void"
  accept_summary "$OUT/agentic-serve.log" agentic
  grep -ciE "illegal|fault|panicked|CUDA_ERROR|hipError" "$OUT/agentic-serve.log" | xargs -I{} log "agentic serve-log fault-line count: {}"
else
  log "LEG3 SKIPPED — serve failed"
fi
stop_serve
log "CHAIN DONE — $OUT"
