#!/usr/bin/env bash
# JOB: agentic-dflash-ob-g8 — 2.5 h agentic-webserver gate leg on
# nvidia/Qwen3.8-27B-NVFP4 with DFlash2 + Option B, γ from MODEL.toml (=8) via
# the wt-dflash2-gamma build. Recipe pins otherwise untouched (prefix caching
# ON — the pcache×status-700 hypothesis was refuted by 038/039, which died
# with pcache OFF; the real trigger was γ=16, bisected in 040).
#
# References on this subject/harness (iterations=50, wall 9000 s):
#   * run-1789388076481949662 — DFlash legacy (no Option B), γ=16:
#     49/50 webserver_ok, Σwall 28878 s → perf FAIL, 57.3 s/turn.
#   * MTP gate records: 10/10, ~5.5 s/turn (PR #21/#23 comments).
# This leg is the Option-B measurement 028/031/039 never produced. Read both
# axes: webserver_ok (correctness) and s_per_turn / Σwall vs 9000 s (perf).
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
echo "== $(date -u +%H:%M:%SZ) agentic-webserver gate: nvidia + dflash + OPTION_B + γ(MODEL.toml) =="
./target/release/spark benchmark run agentic-webserver \
    --checkpoint nvidia/Qwen3.8-27B-NVFP4 \
    --param iterations=50 --param wall_budget_s=9000 --param s_per_turn_budget=0 \
    --serve-override gpu_memory_utilization=0.75 \
    --serve-override speculative=false \
    --serve-override dflash=true \
    --pull-request-gate --yes \
    2>&1 | tee "$(dirname "$0")/gate.log"
rc=${PIPESTATUS[0]}
echo "== γ actually served: $(grep -m1 -o 'DFlash speculative decoding: ENABLED (γ=[0-9]*' "$(dirname "$0")/gate.log" || echo 'NOT FOUND') =="
echo "== exit=$rc $(date -u +%H:%M:%SZ) =="
exit "$rc"
