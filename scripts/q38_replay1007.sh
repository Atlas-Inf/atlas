#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# MLPerf edge-agentic 20-trajectory / 1,007-turn performance replay.
# Local: CARGO_TARGET_DIR=target-rocm10 ATLAS_ROCM_HOME=/path/to/rocm bash scripts/q38_replay1007.sh
# External endpoint: set START_SERVER=0, ENDPOINT_URL, REMOTE_ATLAS_COMMIT,
# REMOTE_ATLAS_DIRTY, and REMOTE_BINARY_SHA256.
set -euo pipefail
ROOT="${ATLAS_REPO:-$HOME/atlas-inf-pr8}"
HARNESS_DIR="${HARNESS_DIR:-$HOME/endpoints}"
BASE_CONFIG="${BASE_CONFIG:-$HARNESS_DIR/results/atlas_final_perf_20260720_143021/config.yaml}"
DATASET="$HARNESS_DIR/examples/10_Edge_Agentic_Example/agentic_coding_2.5h.jsonl"
ROCM_HOME="${ATLAS_ROCM_HOME:-/opt/rocm}"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
BIN="${ATLAS_BIN:-$ROOT/$TARGET_DIR/release/spark}"
MODEL=unsloth/Qwen3.8-27B-NVFP4
SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
PORT="${ATLAS_PORT:-8081}"
ENDPOINT_URL="${ENDPOINT_URL:-http://127.0.0.1:$PORT}"
START_SERVER="${START_SERVER:-1}"
TAG="${TAG:-q38_agentic_2_5h}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
SSM_SLOTS="${SSM_SLOTS:-16}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RD="results/${TAG}_${TS}"
CFG=$(mktemp -t q38_agentic_XXXX.yaml)
SERVE_LOG="$HOME/${TAG}_${TS}-serve.log"
SPARK_PID=
[[ "$ENDPOINT_URL" == http://* || "$ENDPOINT_URL" == https://* ]] || { echo "ENDPOINT_URL must be http(s)" >&2; exit 2; }
[[ "$ENDPOINT_URL" != *@* ]] || { echo "ENDPOINT_URL must not contain credentials" >&2; exit 2; }
[[ -r "$BASE_CONFIG" ]] || { echo "base config not readable: $BASE_CONFIG" >&2; exit 2; }
[[ -r "$DATASET" ]] || { echo "dataset not readable: $DATASET" >&2; exit 2; }
[[ -r "$SNAP/config.json" ]] || { echo "model snapshot not readable: $SNAP" >&2; exit 2; }
if [[ "$START_SERVER" == 1 ]]; then
  ATLAS_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
  ATLAS_DIRTY=$(test -n "$(git -C "$ROOT" status --porcelain)" && echo yes || echo no)
  BINARY_SHA256=$(sha256sum "$BIN" | awk '{print $1}')
else
  ATLAS_COMMIT="${REMOTE_ATLAS_COMMIT:?required when START_SERVER=0}"
  ATLAS_DIRTY="${REMOTE_ATLAS_DIRTY:?required when START_SERVER=0}"
  BINARY_SHA256="${REMOTE_BINARY_SHA256:?required when START_SERVER=0}"
fi
cleanup() {
  rm -f "$CFG"
  if [[ -n "${SPARK_PID:-}" ]]; then
    kill "$SPARK_PID" 2>/dev/null || true
    wait "$SPARK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "$HARNESS_DIR"
python3 - "$RD" "$CFG" "$BASE_CONFIG" "$ENDPOINT_URL" "$MODEL" "$SNAP" <<'PY'
import sys, yaml
rd, cfg, base, endpoint, model, tokenizer = sys.argv[1:]
c = yaml.safe_load(open(base))
c["report_dir"] = rd
c.setdefault("endpoint_config", {})["endpoints"] = [endpoint]
c["model_params"]["name"] = model
c["model_params"]["tokenizer_name"] = tokenizer
runtime = c.setdefault("settings", {}).setdefault("runtime", {})
runtime["min_duration_ms"] = 600_000
runtime["max_duration_ms"] = 14_400_000
runtime["scheduler_random_seed"] = 16159082839903944936
runtime["dataloader_random_seed"] = 2747215439041700203
agentic = c["datasets"][0].setdefault("agentic_inference", {})
agentic["num_trajectories_to_issue"] = 20
agentic["use_dataset_history"] = True
agentic["turn_timeout_s"] = 600.0
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
print("agentic config ->", cfg)
print("report dir ->", rd)
PY

if [[ "$START_SERVER" == 1 ]]; then
  cd "$ROOT"
  if pgrep -x spark >/dev/null 2>&1; then
    echo "spark is already running" >&2
    exit 1
  fi
  SHIM=$(ls -dt "$TARGET_DIR"/release/build/atlas-kernels-*/out | head -1)
  export PATH="$ROCM_HOME/bin:$HOME/.cargo/bin:$PATH"
  export LD_LIBRARY_PATH="$SHIM:$ROCM_HOME/lib:${LD_LIBRARY_PATH:-}"
  export ATLAS_W4A16_VARIANT=v1 ATLAS_W4A16_DP4A=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0
  export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
  "$BIN" serve "$SNAP" \
    --model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --max-seq-len "$MAX_SEQ_LEN" --max-prefill-tokens 2048 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype bf16 --lm-head-dtype bf16 --max-batch-size 1 \
    --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 \
    --enable-prefix-caching --ssm-cache-slots "$SSM_SLOTS" --ssm-checkpoint-interval 16 \
    --disable-tool-grammar true --disable-thinking \
    --dangerously-allow-unresolved-kernel-lookups \
    >"$SERVE_LOG" 2>&1 &
  SPARK_PID=$!
  for _ in $(seq 1 180); do
    curl -fsS -m2 "$ENDPOINT_URL/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SPARK_PID" 2>/dev/null || { tail -40 "$SERVE_LOG"; exit 1; }
    sleep 2
  done
fi
curl -fsS -m5 "$ENDPOINT_URL/v1/models" >/dev/null

mkdir -p "$HARNESS_DIR/$RD"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "atlas_commit=$ATLAS_COMMIT"
  echo "atlas_dirty=$ATLAS_DIRTY"
  echo "binary_sha256=$BINARY_SHA256"
  echo "endpoint=$ENDPOINT_URL"
  echo "start_server=$START_SERVER"
  echo "model=$MODEL"
  echo "checkpoint_revision=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
  echo "harness_commit=$(git -C "$HARNESS_DIR" rev-parse HEAD)"
  echo "harness_dirty=$(test -n "$(git -C "$HARNESS_DIR" status --porcelain)" && echo yes || echo no)"
  echo "base_config_sha256=$(sha256sum "$BASE_CONFIG" | awk '{print $1}')"
  echo "generated_config_sha256=$(sha256sum "$CFG" | awk '{print $1}')"
  echo "dataset_sha256=$(sha256sum "$DATASET" | awk '{print $1}')"
  echo "model_config_sha256=$(sha256sum "$SNAP/config.json" | awk '{print $1}')"
  echo "dataset=examples/10_Edge_Agentic_Example/agentic_coding_2.5h.jsonl trajectories=20 turns=1007"
  echo "runtime=min_duration_ms:600000 max_duration_ms:14400000 scheduler_seed:16159082839903944936 dataloader_seed:2747215439041700203"
  if [[ "$START_SERVER" == 1 ]]; then
    "$ROCM_HOME/bin/hipcc" --version | head -3
    amd-smi version
    amd-smi static --asic --driver | head -40
  fi
} | tee "$HARNESS_DIR/$RD/launch-fingerprint.txt"

cd "$HARNESS_DIR"
./.venv/bin/inference-endpoint benchmark from-config -c "$CFG" --mode perf -v
echo "REPLAY_DONE rd=$HARNESS_DIR/$RD serve_log=$SERVE_LOG"
