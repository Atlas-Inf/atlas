#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the PTX the qwen4_exp exact-batching benches JIT-load (production flags,
# as crates/atlas-kernels compiles the qwen3.8-flash-next target), build the
# bench, run it.
#   scripts/dev/qwen4exp_batch_exact_bench.sh check|time|sweep        (BATCH_FAST)
#   scripts/dev/qwen4exp_batch_exact_bench.sh small-check|small-time  (BATCH_SMALL)
# Run from the repository root on a GB10. See qwen4exp_batch_exact_bench.cu and
# qwen4exp_batch_small_bench.cu.
set -euo pipefail
out=${QB_BENCH_DIR:-/tmp/qb-exact-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr --Werror all-warnings)
mode=${1:-check}
pids=()
if [[ $mode == small-* ]]; then
  for stem in moe_topk moe_expert_gemv ssm_preprocess causal_conv1d rms_norm moe_permute; do
    "$nvcc" "${flags[@]}" "kernels/gb10/common/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
  done
  for stem in gated_delta_rule qwen4exp_decode_fuse hyper_connection; do
    "$nvcc" "${flags[@]}" "kernels/gb10/qwen3.8-flash-next/nvfp4/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
  done
  "$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_batch_small_bench.cu -lcuda \
    -o "$out/qwen4exp_batch_small_bench" & pids+=($!)
  for p in "${pids[@]}"; do wait "$p"; done
  exec "$out/qwen4exp_batch_small_bench" "$out" "${mode#small-}" "${2:-4}"
fi
for stem in moe_shared_expert_fused dense_gemv_bf16 dense_gemv_bf16_batchm moe_permute w4a16_gemv w8a16_gemv w8a16_gemv_batch4; do
  "$nvcc" "${flags[@]}" "kernels/gb10/common/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
for stem in qwen4exp_moe_rows; do
  # QU_SWEEP adds the unit kernels' sweep shapes; the production entry points
  # are compiled exactly as without it.
  "$nvcc" "${flags[@]}" -DQU_SWEEP "kernels/gb10/qwen3.8-flash-next/nvfp4/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
"$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_batch_exact_bench.cu -lcuda \
  -o "$out/qwen4exp_batch_exact_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/qwen4exp_batch_exact_bench" "$out" "$mode"
