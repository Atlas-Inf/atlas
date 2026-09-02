#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# One-time Linux→Windows source mirror for the Qwen3.8 Strix port merge gate.
#
# Copies the FROZEN Linux worktree state (port/qwen3.8-strix-linux) over the
# Windows worktree (port/qwen3.8-windows) for every file the port work touched,
# EXCEPT the Windows-owned platform files listed in KEEP_WINDOWS. Run from the
# repo parent (new_atlas/) after the Linux accuracy gates pass.
#
# Deliberate reconciliations (2026-09-02):
#   * kernels/strix-hip/qwen3.8-27b/MODEL.toml takes the LINUX version:
#     authoritative hf_id unsloth/Qwen3.8-27B-NVFP4; the Windows-only
#     template_owns_tool_prompt key has ZERO readers in either tree (grep
#     verified) and is dropped with the kristianpaul hf_id it shipped with.
#     Effective tool-prompt config is identical: template_owns_tool_definitions
#     absent = false = parser-owned definitions on both platforms.
#   * crates/spark-server/src/api/chat/prepare.rs takes the LINUX version: the
#     Windows copy gated parser injection on the dead template_owns_tool_prompt
#     field; the frozen Linux behavior (always inject when tools are active) is
#     what the 4/6 relevant-control and BFCL results were measured under.
#   * crates/atlas-kernels/hip/* shim files and scripts/strix-windows/* stay
#     Windows-owned (platform-specific, last-known-good on winbox).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
L="$ROOT/wt-strix-linux"
W="$ROOT/wt-strix-windows"

COPY=(
  crates/atlas-kernels/build.rs
  crates/atlas-kernels/build_codegen.rs
  crates/atlas-kernels/build_parse_behavior.rs
  crates/atlas-kernels/src/lib.rs
  crates/spark-model/Cargo.toml
  crates/spark-model/examples/dense_gemm_bf16_oracle.rs
  crates/spark-model/examples/w4a16_cpu_reference_microtest.rs
  crates/spark-model/examples/inferspark_attn_paged_bf16_microtest.rs
  crates/spark-model/src/layers/dense_ffn.rs
  crates/spark-model/src/layers/ops/gemm_fp8_prefill.rs
  crates/spark-model/src/layers/qwen3_ssm/trait_prefill.rs
  crates/spark-model/src/layers/qwen3_ssm/trait_prefill_proj.rs
  crates/spark-model/src/weight_loader/qwen35_dense.rs
  crates/spark-model/src/weight_loader/qwen35_dense/fp8_preservation.rs
  crates/spark-model/src/weight_loader/qwen35_dense/fp8_policy_tests.rs
  crates/spark-server/src/api/chat/levers.rs
  crates/spark-server/src/api/chat/prepare.rs
  crates/spark-server/src/main_modules/serve_load.rs
  crates/spark-server/src/tool_parser/prompt_levers.rs
  crates/spark-server/src/tool_parser/qwen3_coder.rs
  crates/spark-server/src/tool_parser/tests/group_b.rs
  kernels/strix-hip/common/prefill_paged_compute.cuh
  kernels/strix-hip/qwen3.8-27b/MODEL.toml
  serve-amd.sh
)

for f in "${COPY[@]}"; do
  if [ ! -f "$L/$f" ]; then echo "SKIP (missing on linux): $f"; continue; fi
  mkdir -p "$W/$(dirname "$f")"
  cp "$L/$f" "$W/$f"
  echo "copied: $f"
done
echo "MIRROR DONE — review with: cd wt-strix-windows && git status --short && git diff --stat"
