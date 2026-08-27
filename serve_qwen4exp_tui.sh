#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (model_type qwen4_exp) — port tracked in Avarok #753.
#
# This serves: mHC highway on all 48 layers, PLE at model layer 1, QSA decode
# and prefill selection, vision, CUDA graphs, prefix caching, C>1.
#
# CONTEXT ABOVE 8192 NEEDS ATLAS_PLE_MAX_TOKENS RAISED. The engine's prefill
# chunk is 8193 (not 8192), and the PLE scratch is sized from
# ATLAS_PLE_MAX_TOKENS, which defaults to 2048 — a 2191-token prompt already
# fails layer 1 without it. Anything chunked needs >= 8193; this script sets
# 9500 whenever MAX_SEQ_LEN is above 8192, and the cap error message says so
# explicitly when it is hit.
#
# PRIMARY CHECKPOINT is the Inferact NVFP4 release. Against RadixArk's it has
# the same architecture and the same per-expert ModelOpt NVFP4 layout, but
# keeps the PLE n-gram tables in BF16 rather than FP8 — simpler to load (no
# dequant) and more accurate (on LongCat, BF16 n-gram rows measured 0.0050
# error against the reference vs FP8's 0.0247). It costs 170 GB on disk
# against 126 GB, but its RESIDENT footprint is smaller (74.9 vs 78.2 GB)
# because its MTP experts are quantized.
#
#   ./serve_qwen4exp_tui.sh                       # Inferact, port 8889
#   QWEN4EXP_PATH=/path/to/radixark ./serve_qwen4exp_tui.sh
#
# ONE Atlas instance at a time: --gpu-memory-utilization RESERVES its whole
# fraction up front, so a second server fails its OOM pre-flight.
set -euo pipefail
cd "$(dirname "$0")"

SNAP="${QWEN4EXP_PATH:-/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/129972269565f7f4f664fdf8dd42268d3bbda9fd}"
if [[ ! -f "$SNAP/config.json" ]]; then
  echo "Qwen3.8-Flash-Next checkpoint not found at: $SNAP" >&2
  echo "Override with QWEN4EXP_PATH=/path/to/snapshot" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ms/nccl/build/lib"
# INFO so the namespace audit, the placeholder-norm warning and the alloc
# ledger are all visible — the whole point of a load-only run.
export RUST_LOG="${RUST_LOG:-info}"
# PLE scratch is sized from this, not from max_position_embeddings.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
if (( MAX_SEQ_LEN > 8192 )); then
  export ATLAS_PLE_MAX_TOKENS="${ATLAS_PLE_MAX_TOKENS:-9500}"
  echo "  chunked prefill possible (max-seq-len $MAX_SEQ_LEN) -> ATLAS_PLE_MAX_TOKENS=$ATLAS_PLE_MAX_TOKENS"
else
  export ATLAS_PLE_MAX_TOKENS="${ATLAS_PLE_MAX_TOKENS:-$MAX_SEQ_LEN}"
fi

echo "Qwen3.8-Flash-Next  ->  port ${PORT:-8889}"
echo "  mHC highway + PLE n-gram LIVE (NFS shard prefetch on: /tank is NFS-mounted)"
echo "  checkpoint: $SNAP"
exec target/release/spark serve \
  --model-from-path "$SNAP" \
  --model-name "${MODEL_NAME:-qwen4exp}" \
  --kernel-target qwen3.8-flash-next \
  --bind "${BIND:-127.0.0.1}" \
  --port "${PORT:-8889}" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-batch-size "${MAX_BATCH_SIZE:-4}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  --default-chat-template-kwargs "${REASONING_KWARGS:-{\"reasoning_effort\":\"low\"}}" \
  "$@"
