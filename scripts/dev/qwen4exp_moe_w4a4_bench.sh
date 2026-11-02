#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# scripts/dev/qwen4exp_moe_w4a4_bench.sh [check|time]: build the PTX
# qwen4exp_moe_w4a4_bench.cu loads (production flags) and run it (repo root, GB10).
set -euo pipefail
out=${QB_BENCH_DIR:-/tmp/qmoe-w4a4-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false --Werror all-warnings)
pids=()
for stem in moe_prefill_w4a4 moe_prefill_q38 qwen4exp_moe_tcp; do
  "$nvcc" "${flags[@]}" "kernels/gb10/qwen3.8-flash-next/nvfp4/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
"$nvcc" -O3 -std=c++17 -arch=sm_121a -Xcompiler -ffp-contract=off -Iscripts/dev scripts/dev/qwen4exp_moe_w4a4_bench.cu \
  -lcuda -o "$out/qwen4exp_moe_w4a4_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/qwen4exp_moe_w4a4_bench" "$out" "${1:-time}"
