#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: compile and run every *.hip kernel bench in a directory of the checked-out ref.
#
#   scripts/devlab/kbench.sh <dir> [args passed to each bench]
#   e.g.  scripts/devlab/kbench.sh scripts/devlab/kbench 2048 8192
#
# Each bench is compiled with the same flags and include paths the strix-hip build uses (-ffp-contract=off,
# the HIP compat headers, the strix-hip common + 27B kernel dirs) and run once. Output to $DEVLAB_OUT/RESULTS.txt.
set -euo pipefail
DIR=${1:?bench dir}; shift || true
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
ROOT=$PWD
FLAGS=(--offload-arch=gfx1151 -O3 -ffp-contract=off -include hip/hip_runtime.h
       -include "$ROOT/crates/atlas-kernels/hip/compat/atlas_hip_cache_hints.h"
       -I"$ROOT/crates/atlas-kernels/hip/compat" -I"$ROOT/kernels/strix-hip/common"
       -I"$ROOT/kernels/strix-hip/qwen3.6-27b/nvfp4" -I"$DIR")
echo "kbench $(git rev-parse --short=9 HEAD) dir=$DIR args=$*" | tee -a "$RES"
for src in "$DIR"/*.hip; do
  [ -e "$src" ] || { echo "no .hip files in $DIR" | tee -a "$RES"; exit 1; }
  exe="$OUT/$(basename "$src" .hip)"
  if hipcc "${FLAGS[@]}" "$src" -o "$exe" -lrocblas > "$exe.build.log" 2>&1; then
    echo "== $(basename "$src")" | tee -a "$RES"
    "$exe" "$@" 2>&1 | tee -a "$RES"
  else
    echo "== $(basename "$src"): BUILD FAILED" | tee -a "$RES"; grep -m5 error "$exe.build.log" | tee -a "$RES"
  fi
done
