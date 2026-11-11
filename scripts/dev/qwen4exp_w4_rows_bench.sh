#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the PTX the qwen4_exp NVFP4 draft-head row bench JIT-loads (production flags,
# as crates/atlas-kernels compiles the qwen3.8-flash-next target), build the
# bench, run it.
#   scripts/dev/qwen4exp_w4_rows_bench.sh check|time|sweep
# Run from the repository root on a GB10. See qwen4exp_w4_rows_bench.cu.
set -euo pipefail
out=${QW_BENCH_DIR:-/tmp/qw-w4-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr --Werror all-warnings)
pids=()
"$nvcc" "${flags[@]}" kernels/gb10/common/w4a16_gemv.cu -o "$out/w4a16_gemv.ptx" & pids+=($!)
"$nvcc" "${flags[@]}" -DQW4_SWEEP kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_w4_rows.cu \
  -o "$out/qwen4exp_w4_rows.ptx" & pids+=($!)
"$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_w4_rows_bench.cu -lcuda \
  -o "$out/qwen4exp_w4_rows_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/qwen4exp_w4_rows_bench" "$out" "${1:-check}"
