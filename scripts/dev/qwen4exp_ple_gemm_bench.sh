#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build and run scripts/dev/qwen4exp_ple_gemm_bench.cu (repo root, GB10),
# with the production flags of the qwen3.8-flash-next target.
set -euo pipefail
out=${QW_BENCH_DIR:-/tmp/qw-ple-gemm-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
"$nvcc" --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr \
  kernels/gb10/common/dense_gemm_bf16.cu -o "$out/dense_gemm_bf16.ptx" &
"$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_ple_gemm_bench.cu -lcuda \
  -o "$out/qwen4exp_ple_gemm_bench" &
wait
exec "$out/qwen4exp_ple_gemm_bench" "$out"
