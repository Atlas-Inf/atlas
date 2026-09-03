#!/usr/bin/env bash
# Serve a model with Atlas on AMD GPUs. Verified coherent on gfx1151 / Strix
# Halo with Qwen3.8-27B and Qwen3.6-27B. See docs/porting/amd-strix-halo-scale.md.
#
#   ./serve-amd.sh                                  # Qwen3.8-27B-NVFP4 (default)
#   ./serve-amd.sh nvidia/Qwen3.6-27B-NVFP4         # or any local snapshot path
#   PORT=9000 MAX_SEQ_LEN=32768 ./serve-amd.sh
#
# A binary built with ATLAS_TARGET_MODEL='*' (build-amd.sh's default) carries
# every strix kernel target, and resolution picks the right one from the
# checkpoint reference — so the same binary serves 3.6 and 3.8.
#
# Every flag and variable below is the one the measured configuration in
# ../40-bench/RESULTS.md and ../40-bench/BFCL.md actually ran with. If you
# change one, you are no longer running the configuration those numbers
# describe.
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-unsloth/Qwen3.8-27B-NVFP4}"
[ $# -gt 0 ] && shift            # anything left in "$@" is passed through to spark
HW="${ATLAS_TARGET_HW:-strix-hip}"
ROCM_HOME="${ATLAS_ROCM_HOME:-/opt/rocm}"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
BIN="${ATLAS_BIN:-$TARGET_DIR/release/spark}"

# ── gfx1151 runtime shims (each explained in docs §4) ────────────────────────
export ATLAS_W4A16_VARIANT=v1     # BF16-MMA NVFP4 GEMM (SCALE device FP8 encode is broken on gfx1151)
export ATLAS_W4A16_DP4A=1         # int8-DP4A decode GEMV
#
# NOT set, though every earlier Strix doc lists it: ATLAS_FORCE_GLOBAL_GDN=1.
# It has ZERO readers in crates/ on current main (only docs/ still mention it).
# It is unnecessary now because the strix kernel tree ships its own
# kernels/strix-hip/qwen3.6-27b/nvfp4/gated_delta_rule.cu as a model-specific
# override, already written for RDNA3.5's 64 KB LDS budget — the thing the
# lever used to force at dispatch time is now the only kernel there is.

# The NVIDIA-only FP8 LDMAB prefill path is compile-time disabled on native HIP;
# gfx1151 uses the scalar FP8 fallback without requiring a runtime flag.

# ── SSM / MTP levers, unchanged from the certified Qwen3.6 recipe ────────────
# Carried forward verbatim: these are what the 3.6 submission was measured
# under, and keeping them identical is what makes the 3.6-vs-3.8 comparison in
# ../40-bench/RESULTS.md apples-to-apples.
export ATLAS_SSM_TAIL_MIDCHUNK=1  # capture the mid-chunk SSM tail state
export ATLAS_MTP_GATE_REPROBE=64  # re-probe the MTP accept gate every 64 tokens
#
# Three more from the 3.6 recipe are deliberately dropped — all three are
# no-ops on current main, and two of them make the server print a warning:
#   ATLAS_SSM_TAIL_PROTECT=1   renamed 2026-08-05 to the opt-OUT
#                              ATLAS_DISABLE_SSM_TAIL_PROTECT; the lease is now
#                              on by default, so setting the old name does
#                              nothing and the behaviour is unchanged.
#   ATLAS_MTP_DRAFTER_PREFILL=1 } "OBSOLETE and IGNORED — MTP drafter prefill
#   ATLAS_MTP_CARRY_DRAFTER=1   } and cross-turn carry are ON by default"
#                              (spark_model::model::drafter_context). To turn
#                              them off you now set ATLAS_NO_MTP_DRAFTER_CONTEXT=1.

# ── memory: Strix Halo is a unified-memory part ──────────────────────────────
# The GPU allocates from the same RAM the OS uses. GTT reports 60 GB but the
# kernel will not hand over the last few — the measured allocatable ceiling is
# ~55 GB. These defaults are the FROZEN Qwen3.8-27B-NVFP4 (unsloth) accuracy
# recipe — the configuration the targeted controls, BFCL-70 and the pinned
# ST-995 run under (2026-09-02 fingerprint, see the handoff): 0.88 utilization
# is the largest that leaves non-fragmented headroom for the SSM pool after the
# 43.1 GB pre-KV BF16-preservation load; 4096 context matches the gate draws.
GPU_UTIL="${GPU_UTIL:-0.88}"
SSM_SLOTS="${SSM_SLOTS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
# Qwen3.8 tool schemas routinely exceed 2K tokens. The corrected BC=32 paged-
# prefill kernel preserves all query rows across chunks, so the 0.99 GB arena
# can stay at 2048 while long prompts retain single-chunk numerical behavior.
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-2048}"
# 0 selects self-relative accounting (baseline free minus Atlas allocations),
# which measured the unified-memory pool correctly on both Linux and Windows.
export ATLAS_KV_EXTERNAL_RESERVE_GB="${ATLAS_KV_EXTERNAL_RESERVE_GB:-0}"

# ── per-row FP8 preservation (correctness default for the unsloth checkpoint) ─
# The authoritative checkpoint stores attention, GDN, lm_head and the final
# eight FFN layers as per-row-FP8 inside an NVFP4 net. The block-scaled w8a16
# kernels cannot consume [N,1] row scales, and the old fallback requantised
# them to NVFP4 (destructive). These flags dequant once to BF16 instead; the
# policy no-ops for checkpoints whose projections are not all per-row FP8, so
# exporting them unconditionally is safe for every other model. On gfx1151 the
# BF16 FFN prefill routes through the CPU-oracle-validated pipelined WMMA GEMM
# (the `dense_gemm_tc` port writes only part of its output tile there).
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1
export ATLAS_FP8_DEQUANT_FFN_TO_BF16=1
export ATLAS_GDN_BF16_WEIGHTS=1
# Never let a host environment silently re-enable the quarantined tc kernel.
unset ATLAS_FFN_BF16_PREFILL_TC

if [ "$HW" = "strix-hip" ]; then
  # The HIP shims (libcuda/libcudart/libcublasLt) are built into atlas-kernels'
  # OUT_DIR; the loader needs them plus the selected ROCm runtime.
  SHIM=$(ls -dt "$TARGET_DIR"/release/build/atlas-kernels-*/out 2>/dev/null | head -1)
  export PATH="$ROCM_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="${SHIM:-}:$ROCM_HOME/lib:${LD_LIBRARY_PATH:-}"
else
  : "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
  # SCALE libs FIRST so /opt/rocm cannot shadow the fixed libhsa-runtime64 (the
  # gfx1151 queue-create fix lives in SCALE 1.7.1's bundled ROCm 7.2.3):
  export LD_LIBRARY_PATH="$SCALE_HOME/targets/gfx1151/lib:$SCALE_HOME/lib"
  export PATH="$SCALE_HOME/targets/gfx1151/bin:$PATH"
fi

# The gfx1151 kernel set is 94 modules where gb10's is 167, so 92 dispatch sites
# resolve to a fallback and main's kernel audit (#388) refuses to serve without
# this flag. Those fallbacks are PRE-EXISTING — the audit landed on main after
# the Strix branch forked, so the certified 3.6 submission was produced under
# exactly the same ones. See ../30-verify/KERNEL_AUDIT.md before quoting any
# Strix perf number as final.
ALLOW_FALLBACKS="--dangerously-allow-unresolved-kernel-lookups"

# --model-name only matters when MODEL is a local snapshot path and you want the
# API to report the canonical repo id (as the benchmark configs expect).
NAME_ARG=(); [ -n "${MODEL_NAME:-}" ] && NAME_ARG=(--model-name "$MODEL_NAME")

# MTP speculation is OPT-IN (NUM_DRAFTS>0): the frozen Qwen3.8 accuracy recipe
# ran without it (spec decode is hard-gated off under thinking-off tool calls
# anyway, and the accuracy fingerprint must not carry an unused lever).
SPEC_ARGS=()
if [ "${NUM_DRAFTS:-0}" -gt 0 ]; then
  SPEC_ARGS=(--speculative --num-drafts "$NUM_DRAFTS" --mtp-quantization bf16 --mtp-vocab 100000)
fi

GFX=$("$ROCM_HOME/bin/rocminfo" 2>/dev/null | sed -n 's/.*\(gfx[0-9][0-9]*\).*/\1/p' | head -1)
echo "serving $MODEL on ${GFX:-AMD} via $HW"
exec "$BIN" serve "$MODEL" "${NAME_ARG[@]}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8081}" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype bf16 --max-batch-size "${MAX_BATCH:-1}" \
  "${SPEC_ARGS[@]}" \
  --disable-tool-grammar true \
  --ssm-cache-slots "$SSM_SLOTS" --ssm-checkpoint-interval 16 \
  $ALLOW_FALLBACKS \
  --disable-thinking \
  "$@"
