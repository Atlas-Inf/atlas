#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# scripts/dev/qwen4exp_moe_nodup_bench.sh [tokens=16000]: build the PTX
# qwen4exp_moe_nodup_bench.cu loads (production flags) and run it (repo root,
# GB10). QB_BASE_CU=<file> adds that kernel file (the q38 chain before the NM
# change, e.g. `git show HEAD~:...moe_prefill_q38.cu`) as the baseline arm.
set -euo pipefail
out=${QB_BENCH_DIR:-/tmp/qmoe-nodup-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false)
k=kernels/gb10/qwen3.8-flash-next/nvfp4
pids=()
"$nvcc" "${flags[@]}" "$k/moe_prefill_q38.cu" -o "$out/moe_prefill_q38.ptx" & pids+=($!)
"$nvcc" "${flags[@]}" kernels/gb10/common/moe_transpose_batched.cu -o "$out/moe_transpose_batched.ptx" & pids+=($!)
if [ -n "${QB_BASE_CU:-}" ]; then
  "$nvcc" "${flags[@]}" -I"$k" "$QB_BASE_CU" -o "$out/moe_prefill_q38_base.ptx" & pids+=($!)
fi
"$nvcc" -O3 -std=c++17 -arch=sm_121a -Iscripts/dev scripts/dev/qwen4exp_moe_nodup_bench.cu -lcuda \
  -o "$out/qwen4exp_moe_nodup_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
exec "$out/qwen4exp_moe_nodup_bench" "$out" "${1:-16000}"
