#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the PTX the QSA decode bench JIT-loads (production flags, as
# crates/atlas-kernels compiles the qwen3.8-flash-next target), build the
# bench, run it. From the repository root on a GB10:
#   scripts/dev/qwen4exp_qsa_decode_bench.sh check|time [pos=77000] [rows=4] [nq=12] [nkv=1]
set -euo pipefail
out=${QD_BENCH_DIR:-/tmp/qsa-decode-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false --expt-relaxed-constexpr --Werror all-warnings)
K=kernels/gb10/qwen3.8-flash-next/nvfp4
pids=()
"$nvcc" "${flags[@]}" $K/qsa_indexer.cu -o "$out/qsa_indexer.ptx" & pids+=($!)
"$nvcc" "${flags[@]}" $K/qsa_decode_rows.cu -o "$out/qsa_decode_rows.ptx" & pids+=($!)
"$nvcc" "${flags[@]}" kernels/gb10/common/paged_decode_attn.cu -o "$out/paged_decode_attn.ptx" & pids+=($!)
"$nvcc" -O3 -std=c++17 -arch=sm_121a scripts/dev/qwen4exp_qsa_decode_bench.cu -lcuda -o "$out/bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/bench" "$out" "${1:-check}" "${2:-77000}" "${3:-4}" "${4:-12}" "${5:-1}"
