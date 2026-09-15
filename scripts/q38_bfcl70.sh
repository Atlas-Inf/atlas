#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# BFCL-70 focused accuracy gate on the FIXED preservation build (post
# tc->pipelined FFN dispatch). Same 70-row draw as the recorded Kristianpaul
# diagnostics (non_live 4 / live 1 / hallucination 1, floor 2, temp 0).
set -euo pipefail
ROOT="${ATLAS_REPO:-$HOME/atlas-inf-pr8}"
cd "$ROOT"
PORT="${ATLAS_PORT:-8081}"
ROCM_HOME="${ATLAS_ROCM_HOME:-/opt/rocm}"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
BIN="${ATLAS_BIN:-$TARGET_DIR/release/spark}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$HOME/q38-bfcl70-$STAMP-serve.log"
BENCH_LOG="$HOME/q38-bfcl70-$STAMP-bench.log"
FINGERPRINT="$HOME/q38-bfcl70-$STAMP-fingerprint.txt"
SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
SHIM=$(ls -dt "$TARGET_DIR"/release/build/atlas-kernels-*/out | head -1)
export PATH="$ROCM_HOME/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$SHIM:$ROCM_HOME/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_VARIANT=v1 ATLAS_W4A16_DP4A=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
EXTRA_ARGS=()
if [[ "${ATLAS_BFCL_MTP:-0}" == 1 ]]; then
  EXTRA_ARGS=(--speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000)
fi
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit=$(git rev-parse HEAD)"
  echo "dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
  echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "rocm_home=$ROCM_HOME"
  echo "target_dir=$TARGET_DIR"
  echo "shim=$SHIM"
  echo "mtp=${ATLAS_BFCL_MTP:-0}"
  echo "checkpoint_revision=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
  echo "harness=bfcl-subset non_live=4 live=1 hallucination=1 floor=2 max_new_tokens=1024 temperature=0 timeout=600 samples=70"
  "$ROCM_HOME/bin/hipcc" --version | head -3
} | tee "$FINGERPRINT"
if pgrep -x spark >/dev/null 2>&1; then
  echo "spark is already running" >&2
  exit 1
fi
SPARK_PID=
cleanup() {
  if [[ -n "${SPARK_PID:-}" ]]; then
    kill "$SPARK_PID" 2>/dev/null || true
    wait "$SPARK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
"$BIN" serve "$SNAP" \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port "$PORT" \
  --max-seq-len 4096 --max-prefill-tokens 2048 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype bf16 --lm-head-dtype bf16 \
  --max-batch-size 1 \
  --disable-tool-grammar true \
  --ssm-cache-slots 0 \
  --disable-thinking \
  --dangerously-allow-unresolved-kernel-lookups \
  "${EXTRA_ARGS[@]}" \
  >"$LOG" 2>&1 &
SPARK_PID=$!
for _ in $(seq 1 180); do
  curl -fsS -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SPARK_PID" 2>/dev/null || { tail -40 "$LOG"; exit 1; }
  sleep 2
done
curl -fsS -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null
echo "server up — running BFCL-70"
"$BIN" benchmark run bfcl-subset \
  --url "http://127.0.0.1:$PORT" \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --param non_live_pct=4 --param live_pct=1 --param hallucination_pct=1 \
  --param subset_floor=2 --param max_new_tokens=1024 --param temperature=0 \
  --param min_overall=0 --param min_normalized=0 --param request_timeout_s=600 \
  2>&1 | tee "$BENCH_LOG"
echo "BFCL70 DONE logs=$LOG,$BENCH_LOG fingerprint=$FINGERPRINT"
