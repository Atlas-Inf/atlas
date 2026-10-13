#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the PTX scripts/dev/qwen4exp_moe_c8_bench.cu JIT-loads (production
# flags, as crates/atlas-kernels compiles the qwen3.8-flash-next target), build
# the bench, run it from the repository root on a GB10:
#   scripts/dev/qwen4exp_moe_c8_bench.sh probe|time|check|tc-units-check|tc-ident [variant filter]
#   (REALBIN=route.bin: real routing; LAYER_PASSES=n: interleaved layer timings, min/median)
#   scripts/dev/qwen4exp_moe_c8_bench.sh tc-time|tc-check   (qwen4exp_moe_c8_tc_bench.cu)
set -euo pipefail
out=${QB_BENCH_DIR:-/tmp/qmc8-bench}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(--ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr --Werror all-warnings)
pids=()
"$nvcc" "${flags[@]}" kernels/gb10/common/moe_shared_expert_fused.cu -o "$out/moe_shared_expert_fused.ptx" & pids+=($!)
for stem in qwen4exp_moe_rows qwen4exp_moe_c8 qwen4exp_moe_c8_tc qwen4exp_moe_c8_tc3 moe_prefill_q38; do
  [[ -f kernels/gb10/qwen3.8-flash-next/nvfp4/$stem.cu ]] || continue
  "$nvcc" "${flags[@]}" "kernels/gb10/qwen3.8-flash-next/nvfp4/$stem.cu" -o "$out/$stem.ptx" & pids+=($!)
done
# The rows pair at other unit shapes (qwen4exp_moe_c8_variants.h ROWS_SHAPES).
for shape in r8c4:8:64:8:4 r8c8:8:64:8:8 r16c4:16:64:8:4 r32c4:32:64:8:4 r32c8:32:64:8:8 r16t32c4:16:32:4:4; do
  IFS=: read -r tag rmax tile warps rc <<<"$shape"
  "$nvcc" "${flags[@]}" -DQU_RMAX=${rmax}u -DQU_SD_TILE=$tile -DQU_SD_WARPS=$warps -DQU_SD_RC=$rc \
    kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_rows.cu -o "$out/qwen4exp_moe_rows_$tag.ptx" & pids+=($!)
done
"$nvcc" -O3 -std=c++17 -arch=sm_121a -Iscripts/dev scripts/dev/qwen4exp_moe_c8_bench.cu -lcuda \
  -o "$out/qwen4exp_moe_c8_bench" & pids+=($!)
"$nvcc" -O3 -std=c++17 -arch=sm_121a -Iscripts/dev scripts/dev/qwen4exp_moe_c8_tc_bench.cu -lcuda \
  -o "$out/qwen4exp_moe_c8_tc_bench" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
case ${1:-time} in
  tc-time|tc-check) exec "$out/qwen4exp_moe_c8_tc_bench" "$out" "$1" ;;
  *) exec "$out/qwen4exp_moe_c8_bench" "$out" "${1:-time}" "${2:-}" ;;
esac
