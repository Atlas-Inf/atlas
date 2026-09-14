#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 on AMD Strix Halo (gfx1151, ROCm/HIP) — single-node serve.
#
# Runs the native `spark` binary — Strix serves bare-metal, no container.
# The W4A8 integer-DP4A decode arm is ON by default on this target
# (ATLAS_W4A16_DP4A=0 rolls back to the float E2M1-LUT path if ever needed).
#
# Usage:
#   ./scripts/start-qwen38-strix.sh [CHECKPOINT]
#
# CHECKPOINT defaults to the NVIDIA NVFP4 HF repo (downloaded into the HF
# cache on first run); pass a local snapshot path for the requantized or
# unsloth-preservation checkpoint. Tunables are env-overridable below.
#
# Validated fingerprint (see kernels/strix-hip/qwen3.8-27b/BENCH.toml notes):
# AzeezStrix / ROCm 7.13, ~60 GB GTT — 28.3-28.6 tok/s K=4 decode,
# 13.3-13.6 tok/s at 30k ctx, bfcl-subset 995-row 83.02/80.41.
set -euo pipefail

MODEL="${1:-nvidia/Qwen3.8-27B-NVFP4}"
BIN="${BIN:-spark}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_PREFILL="${MAX_PREFILL:-2048}"
MTP_QUANT="${MTP_QUANT:-bf16}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

echo "=== Atlas Qwen3.8-27B NVFP4 — Strix Halo (gfx1151) ==="
echo "Model:      $MODEL"
echo "Binary:     $BIN"
echo "Port:       $PORT"
echo "GPU mem:    $GPU_MEM_UTIL"
echo "Max seq:    $MAX_SEQ_LEN"
echo "Spec decode: K=4 verify (3 drafts, mtp=$MTP_QUANT)"
echo ""

# --mtp-gate is left at its `auto` default: the arbiter keeps K=4 verify
# engaged while it measures net-positive and disarms it if that ever stops
# being true (forcing is a diagnostic, not a production setting).
exec "$BIN" serve "$MODEL" \
  --bind 127.0.0.1 --port "$PORT" \
  --max-seq-len "$MAX_SEQ_LEN" --max-prefill-tokens "$MAX_PREFILL" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype bf16 --lm-head-dtype nvfp4 \
  --speculative --num-drafts 3 \
  --mtp-quantization "$MTP_QUANT" --mtp-vocab 100000 \
  --dangerously-allow-unresolved-kernel-lookups \
  $EXTRA_FLAGS
