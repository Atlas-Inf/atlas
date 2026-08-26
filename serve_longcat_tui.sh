#!/usr/bin/env bash
# Serve LongCat-Flash-Lite with Atlas and bring up the TUI dashboard (the TUI
# is automatic on an interactive terminal — do NOT pipe this, or it disables
# itself and you get the plain log stream).
#
# This is the n-gram-embedding model: 14 checkpoint layers -> 28 engine
# sublayers of MLA + shortcut MoE, with a fused input embedding that mixes the
# base token row with 12 hashed n-gram lookups.
#
# Validated at this config on 2026-08-25 against
# bench/ngram_ref/longcat_forward_golden.npz: the fused embedding matches the
# reference at cos 1.0000 and every one of the 14 checkpoint layers holds
# >= 0.9952 across all 28 sublayers.
#
#   ./serve_longcat_tui.sh                  # port 8888, TUI up
#   PORT=8899 ./serve_longcat_tui.sh        # somewhere else
#   MAX_SEQ_LEN=65536 ./serve_longcat_tui.sh
#
# Port defaults to 8888 because that is what bench/agentic/* expects
# (ATLAS_URL defaults to http://localhost:8888/v1/chat/completions), so the
# agentic harnesses point at this with no extra flags.
#
# ONE Atlas instance at a time: --gpu-memory-utilization RESERVES its whole
# fraction of the box up front, so a second server will fail its OOM
# pre-flight. Kill the running one by PID first.
set -euo pipefail
cd "$(dirname "$0")"

SNAP="${LONGCAT_PATH:-/tank/hf/hub/models--meituan-longcat--LongCat-Flash-Lite/snapshots/b62b68827ead0b7fef3ba98b57f18484acaaec06}"
if [[ ! -f "$SNAP/config.json" ]]; then
  echo "LongCat checkpoint not found at: $SNAP" >&2
  echo "Override with LONGCAT_PATH=/path/to/snapshot" >&2
  exit 1
fi

# NCCL lives outside the default loader path on this box.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ms/nccl/build/lib"
# Serve at INFO — warn/error hides the load ledger and the n-gram cache line.
export RUST_LOG="${RUST_LOG:-info}"

# Resident rows per n-gram table. The 12 tables are 62.8 GB of BF16 on disk and
# are NEVER uploaded: they are served row-by-row off NVMe out of a pinned
# GPU-addressable arena. 65536 slots x 512 B x 12 tables = 403 MB, and that
# frees the rest for KV. Raise it if you see cache thrash on long contexts.
export ATLAS_NGRAM_CACHE_SLOTS="${ATLAS_NGRAM_CACHE_SLOTS:-65536}"

echo "LongCat-Flash-Lite  ->  port ${PORT:-8888}   (TUI: needs an interactive terminal)"
exec target/release/spark serve \
  --model-from-path "$SNAP" \
  --model-name "${MODEL_NAME:-longcat-full}" \
  --kernel-target longcat-flash-lite \
  --bind "${BIND:-127.0.0.1}" \
  --port "${PORT:-8888}" \
  --max-seq-len "${MAX_SEQ_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-batch-size "${MAX_BATCH_SIZE:-16}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
  --disable-thinking \
  "$@"
