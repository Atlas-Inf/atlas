#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the PTX the Qwen3.8-Flash-Next MoE decode bench JIT-loads (production
# flags, as crates/atlas-kernels compiles them), build the bench, run it.
#   scripts/dev/qwen4exp_moe_decode_bench.sh check|time|sweep [pool] [reps]
# Run from the repository root on a GB10. See qwen4exp_moe_decode_bench.cu.
set -euo pipefail
out=${QX_BENCH_DIR:-/tmp/qx-moe-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr --Werror all-warnings)
pids=()
for stem in moe_shared_expert_fused_t moe_shared_expert_fused_batch2_t moe_shared_expert_fused_batch3_t; do
  "$nvcc" "${flags[@]}" "kernels/gb10/common/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
"$nvcc" "${flags[@]}" -DQX_SWEEP kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_decode.cu \
  -o "$out/qwen4exp_moe_decode.ptx" & pids+=($!)
"$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_moe_decode_bench.cu -lcuda \
  -o "$out/qwen4exp_moe_decode_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/qwen4exp_moe_decode_bench" "$out" "${1:-check}" "${@:2}"
