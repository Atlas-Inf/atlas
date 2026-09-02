#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Kernel-by-kernel drift audit for the Qwen3.8-27B (gfx1151, strix-hip) serve
# path. Runs every existing microtest that covers a kernel this model
# dispatches, captures the drift line, and writes one consolidated log.
set -uo pipefail
cd ~/atlas-inf-pr8
SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
OUT="$HOME/q38-kernel-drift-audit.log"
EXE=target/release/examples
: > "$OUT"
echo "# Qwen3.8-27B kernel drift audit — $(date -u +%Y-%m-%dT%H:%M:%SZ) — AzeezStrix gfx1151 strix-hip" >> "$OUT"
echo "# binary: $(sha256sum target/release/spark | cut -c1-16)... (serve path), examples built from the same tree" >> "$OUT"

run() {
  local name="$1"; shift
  echo "## $name" >> "$OUT"
  timeout 300 "$EXE/$name" "$@" >> "$OUT" 2>&1
  echo "  [exit $?]" >> "$OUT"
  echo >> "$OUT"
}

# ── prefill-path kernels ──
run rmsnorm_vanilla_microtest
run rope_microtest
run conv1d_strided_microtest
run gdn_split4_microtest
run inferspark_attn_microtest
run inferspark_attn_paged_bf16_microtest
run w8a16_microtest
run w8a16t_microtest
run w4a16_parity_microtest
run w4a16_cpu_reference_microtest
run dense_gemm_bf16_oracle 5120 1

# ── decode-path kernels ──
run w4a16_gemv_dp4a_microtest
run w8a16_gemv_batch4_microtest
run dense_gemv_bf16_batchm_microtest
run gemv_fp4_vs_fp8_microtest
run bf16_batch_bitparity_microtest
run w4a16_batch_bitparity_microtest
run w4a16_bf16_microtest
run w4a16_bf16_v2_microtest

echo "AUDIT DONE" >> "$OUT"
grep -cE "PASS|RESULT" "$OUT"
tail -1 "$OUT"
