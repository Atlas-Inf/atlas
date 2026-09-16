#!/usr/bin/env bash
# JOB: conc-sweep-dflash-ob-g8 — concurrency-sweep gate (C1/4/8/16, ISL 512,
# OSL 320, natural prompts, 1 warm-up) on nvidia/Qwen3.8-27B-NVFP4 with
# DFlash2 + Option B, γ from MODEL.toml (=8), wt-dflash2-gamma build.
#
# Reference on this box/subject (MTP K=4, recipe as pinned):
#   run-1789248105378681119 — PASS, C1 20.9 / C4 41.9 / C8 64.1 / C16 85.2,
#   floors 16.2/33.5/50.5/72.
# 034 (manual C8 probe, MinHeap, auto MTP gate): serial 65 agg vs OB 27 agg —
# the gate shed spec (serial=0.69) and still lost. The recipe pins
# `mtp_gate: force`, so THIS leg measures raw DFlash2 at width, no shedding.
#
# Overrides (each a deliberate, documented deviation from the recipe):
#   max_batch_size=16       recipe pins 128; the drafter's Option-B paged pool is
#                           257 blocks × 5 layers × 64 KB PER SEQ → 128 seqs =
#                           10.8 GB. 16 covers every rung the sweep runs.
#   gpu_memory_utilization  0.85 → 0.78 (drafter build OOMs at 0.85, PR #21 #2)
#   speculative=false dflash=true   the lane under test
# Everything else (fp8 KV, prefix caching, thinking off, ssm_* knobs) as pinned.
# Expect ~5–10 min. A FAIL on floors is the honest C-scaling read for DFlash2.
set -u
source /home/azeez/code/build_env.sh 2>/dev/null || true
export HF_HOME=/home/azeez/code/hf
export RUST_LOG=info
export ATLAS_DFLASH_OPTION_B=1
export ATLAS_MTP_ACCEPT_DEBUG=1
WT=/home/azeez/code/wt-dflash2-gamma
[ -x "$WT/target/release/spark" ] || { echo "REFUSE: $WT binary missing — build job did not run"; exit 7; }
cd "$WT"
git log --oneline -1
# 099 and 103 both died at boot with cuMemAlloc OOM (<1 GB free) because the
# previous job's serve was still releasing ~97 GB when this one started: the
# dispatcher's GPU gate checks processes, not free memory. nvidia-smi reports
# memory.used as [N/A] on the GB10, but the memory is unified, so host
# MemAvailable tracks device frees: wait until MemAvailable > 100 GB (of
# ~122 GB) for 3 consecutive samples (up to 5 min).
ok=0
for i in $(seq 1 60); do
    avail=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
    if [ "$avail" -gt 100000 ]; then ok=$((ok + 1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
    sleep 5
done
echo "== host MemAvailable=${avail} MB after drain wait (${i}×5 s) =="
echo "== $(date -u +%H:%M:%SZ) concurrency-sweep gate: nvidia + dflash + OPTION_B + γ(MODEL.toml) + bs16 =="
./target/release/spark benchmark run concurrency-sweep --pull-request-gate --yes \
    --serve-override gpu_memory_utilization=0.78 \
    --serve-override max_batch_size=16 \
    --serve-override speculative=false \
    --serve-override dflash=true \
    2>&1 | tee "$(dirname "$0")/gate.log"
rc=${PIPESTATUS[0]}
echo "== γ actually served: $(grep -m1 -o 'DFlash speculative decoding: ENABLED (γ=[0-9]*' "$(dirname "$0")/gate.log" || echo 'NOT FOUND') =="
grep -o "DFLASH WIDTH n_active=[0-9]* verify=[0-9]* boot=[0-9]* dspark_batch_ok=[a-z]*" "$(dirname "$0")/gate.log" | sort | uniq -c | sort -rn | head -8
echo "== exit=$rc $(date -u +%H:%M:%SZ) =="
exit "$rc"
