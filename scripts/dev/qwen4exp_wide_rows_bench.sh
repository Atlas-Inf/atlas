#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the PTX the qwen4_exp wide exact-row bench JIT-loads (production flags,
# as crates/atlas-kernels compiles the qwen3.8-flash-next target), build the
# bench, run it.
#   scripts/dev/qwen4exp_wide_rows_bench.sh check|time|sweep
# Run from the repository root on a GB10. See qwen4exp_wide_rows_bench.cu.
set -euo pipefail
out=${QW_BENCH_DIR:-/tmp/qw-wide-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr --Werror all-warnings)
pids=()
for stem in dense_gemv_bf16 dense_gemv_bf16_batchm w4a16_gemv; do
  "$nvcc" "${flags[@]}" "kernels/gb10/common/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
# QW_SWEEP adds the sweep shapes; the production entry points compile as without it.
"$nvcc" "${flags[@]}" -DQW_SWEEP kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_wide_rows.cu \
  -o "$out/qwen4exp_wide_rows.ptx" & pids+=($!)
"$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_wide_rows_bench.cu -lcuda \
  -o "$out/qwen4exp_wide_rows_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/qwen4exp_wide_rows_bench" "$out" "${1:-check}"
