#!/usr/bin/env bash
# JOB: bfcl-dflash-ob-g8-thinkon — ST-995 (bfcl-subset golden draw, n=995,
# seed 42, temp 0) on nvidia/Qwen3.8-27B-NVFP4 with DFlash2 + Option B, γ
# resolved from MODEL.toml (=8) by the wt-dflash2-gamma build, thinking ON.
#
# Why this exact configuration:
#   * 022 is the PASSING reference on this subject: MTP (recipe default),
#     `disable_thinking=false` + reasoning_effort medium → 87.94 / 88.24
#     (run-1789419056919283587). This leg changes ONE variable vs 022: the
#     speculative lane (MTP → DFlash2+OB). Same recipe, same overrides, same
#     draw. Everything else (thinking, pcache ON per recipe, util) identical
#     EXCEPT gpu_memory_utilization: 022 ran the recipe's 0.85 without a
#     drafter; the drafter (3.85 GB BF16) at 0.85 OOMs in drafter build on
#     this 119.6 GB box (PR #21 finding 2), so 0.75 — documented deviation.
#   * 027/030/038 (γ=16 default) all died at the first propose of the second
#     sequence (status 700, bisected in 040). γ=8 ran 4/4 + 8/8 + C8 clean.
#   * Correctness read: spec decode is hard-gated OFF inside <think>, so most
#     tokens here are serial; what this measures is that DFlash2's verify
#     lane does not perturb the emitted answer stream vs 022. Speed is NOT
#     the headline of this record.
# Expected wall ≈ 022's 14924 s (~4.1 h).
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
echo "== $(date -u +%H:%M:%SZ) bfcl-subset gate: nvidia + dflash + OPTION_B + γ(MODEL.toml) + thinking ON =="
./target/release/spark benchmark run bfcl-subset --pull-request-gate \
    --serve-override gpu_memory_utilization=0.75 \
    --serve-override speculative=false \
    --serve-override dflash=true \
    --serve-override disable_thinking=false \
    --serve-override 'default_chat_template_kwargs={"reasoning_effort":"medium"}' \
    2>&1 | tee "$(dirname "$0")/gate.log"
rc=${PIPESTATUS[0]}
echo "== γ actually served: $(grep -m1 -o 'DFlash speculative decoding: ENABLED (γ=[0-9]*' "$(dirname "$0")/gate.log" || echo 'NOT FOUND') =="
echo "== exit=$rc $(date -u +%H:%M:%SZ) =="
exit "$rc"
