#!/usr/bin/env bash
# JOB: agentic-mtp-27b-ref — the MISSING reference for the DFlash2 agentic
# legs: agentic-webserver on nvidia/Qwen3.8-27B-NVFP4 with the recipe's own
# speculative lane (MTP, `speculative: true num_drafts: 1`), same harness
# params (iterations=50, wall 9000 s), same gate path, same build
# (wt-dflash2-gamma @ a61f0263).
#
# Why: the "agentic 10/10 · 5.5 s/turn" figures in the gap analysis and PR
# comments were measured on the gate's DEFAULT subject (the 35B MoE), not the
# 27B. The only 27B agentic records are DFlash legacy γ=16
# (run-1789388076481949662: 57.3 s/turn) and tonight's DFlash+OB γ=8 (job
# 100). Without an MTP-27B record the "is DFlash2 accelerating" question has
# no same-model baseline. One variable vs job 100: the spec lane.
# gpu_memory_utilization stays at the recipe's 0.85 (no drafter to fit).
set -u
source /home/azeez/code/build_env.sh 2>/dev/null || true
export HF_HOME=/home/azeez/code/hf
export RUST_LOG=info
export ATLAS_MTP_ACCEPT_DEBUG=1
WT=/home/azeez/code/wt-dflash2-gamma
[ -x "$WT/target/release/spark" ] || { echo "REFUSE: $WT binary missing"; exit 7; }
cd "$WT"
git log --oneline -1
# MemAvailable tracks device frees: wait until MemAvailable > 100 GB (of
# ~122 GB) for 3 consecutive samples (up to 5 min).
ok=0
for i in $(seq 1 60); do
    avail=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
    if [ "$avail" -gt 100000 ]; then ok=$((ok + 1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
    sleep 5
done
echo "== host MemAvailable=${avail} MB after drain wait (${i}×5 s) =="
echo "== $(date -u +%H:%M:%SZ) agentic-webserver gate: nvidia 27B + recipe MTP (reference) =="
./target/release/spark benchmark run agentic-webserver \
    --checkpoint nvidia/Qwen3.8-27B-NVFP4 \
    --param iterations=50 --param wall_budget_s=9000 --param s_per_turn_budget=0 \
    --pull-request-gate --yes \
    2>&1 | tee "$(dirname "$0")/gate.log"
rc=${PIPESTATUS[0]}
echo "== spec lane served: $(grep -m1 -o 'MTP speculative decoding: ENABLED[^\n]*\|DFlash speculative decoding: ENABLED[^\n]*' "$(dirname "$0")/gate.log" | cut -c1-80) =="
echo "== exit=$rc $(date -u +%H:%M:%SZ) =="
exit "$rc"
