#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
ROOT="${ATLAS_REPO:-$HOME/atlas-inf-pr8}"
cd "$ROOT"
PORT="${ATLAS_PORT:-8092}"
MODEL=unsloth/Qwen3.8-27B-NVFP4
ROCM_HOME="${ATLAS_ROCM_HOME:-/opt/rocm}"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
BIN="${ATLAS_BIN:-$TARGET_DIR/release/spark}"
SNAP="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$HOME/q38-mtp-ab-$STAMP"
mkdir -p "$OUT"
SHIM=$(ls -dt "$TARGET_DIR"/release/build/atlas-kernels-*/out | head -1)
export PATH="$ROCM_HOME/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$SHIM:$ROCM_HOME/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_VARIANT=v1 ATLAS_W4A16_DP4A=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
export ATLAS_MTP_TIMING=1 ATLAS_MTP_ACCEPT_DEBUG=1 ATLAS_MTP_GATE_FORCE=1
SPARK_PID=
cleanup() {
  if [[ -n "${SPARK_PID:-}" ]]; then
    kill "$SPARK_PID" 2>/dev/null || true
    wait "$SPARK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if curl -fsS -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "port $PORT already has a server" >&2
  exit 1
fi
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "commit=$(git rev-parse HEAD)"
  echo "dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
  echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "rocm_home=$ROCM_HOME"
  echo "target_dir=$TARGET_DIR"
  echo "shim=$SHIM"
  echo "checkpoint=$MODEL"
  echo "checkpoint_revision=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
  echo "harness=q38_mtp_perf_ab MinHeap code prompt temperature=0 seed=0 max_tokens=1024 reasoning_effort=none warmup=1 runs=3"
  echo "serve_common=--max-seq-len 4096 --max-prefill-tokens 2048 --gpu-memory-utilization 0.88 --kv-cache-dtype bf16 --lm-head-dtype bf16 --max-batch-size 1 --disable-tool-grammar true --ssm-cache-slots 0 --ssm-checkpoint-interval 16 --disable-thinking"
  echo "serial_delta=none"
  echo "mtp_delta=--speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000"
  uname -a
  "$ROCM_HOME/bin/hipcc" --version | head -3
  amd-smi version
  amd-smi static --asic --driver | head -40
} | tee "$OUT/fingerprint.txt"
gpu_temp() {
  amd-smi metric --temperature 2>/dev/null | sed -n 's/.*EDGE: *//p' | head -1
}
serve() {
  local tag=$1
  shift
  local log="$OUT/serve-$tag.log"
  "$BIN" serve "$SNAP" \
    --model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --max-seq-len 4096 --max-prefill-tokens 2048 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype bf16 --lm-head-dtype bf16 \
    --max-batch-size 1 --disable-tool-grammar true \
    --ssm-cache-slots 0 --ssm-checkpoint-interval 16 \
    --disable-thinking --dangerously-allow-unresolved-kernel-lookups \
    "$@" >"$log" 2>&1 &
  SPARK_PID=$!
  for _ in $(seq 1 180); do
    if curl -fsS -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "$tag server ready; temperature=$(gpu_temp)" | tee -a "$OUT/results.txt"
      return
    fi
    kill -0 "$SPARK_PID" 2>/dev/null || { tail -40 "$log"; return 1; }
    sleep 2
  done
  echo "$tag server did not become ready" >&2
  return 1
}
run_leg() {
  local tag=$1
  python3 - "$PORT" "$MODEL" "$tag" "$OUT/results.jsonl" <<'PY'
import json, statistics, sys, time, urllib.request
port, model, tag, path = sys.argv[1:]
prompt = """Implement a complete, production-quality MinHeap class in Python. Include the methods insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, delete, __len__, and a validation method that checks the heap invariant. Include type hints, docstrings, clear error handling, complexity notes, and a compact executable test suite covering empty, singleton, duplicate, negative, and randomized inputs."""
def request(max_tokens):
    body = {"model": model, "stream": False, "temperature": 0.0, "seed": 0,
            "max_tokens": max_tokens, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as response:
        out = json.load(response)
    usage = out.get("usage", {})
    details = usage.get("completion_tokens_details") or {}
    return {"leg": tag, "wall_s": time.monotonic() - start,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "server_tps": usage.get("response_token/s"),
            "server_ttft_ms": usage.get("time_to_first_token_ms"),
            "accepted_prediction_tokens": details.get("accepted_prediction_tokens"),
            "finish_reason": out.get("choices", [{}])[0].get("finish_reason")}
request(16)
rows = []
for i in range(3):
    row = request(1024)
    row["run"] = i + 1
    rows.append(row)
    print(json.dumps(row), flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
tps = [r["server_tps"] for r in rows if isinstance(r["server_tps"], (int, float))]
print(json.dumps({"leg": tag, "median_server_tps": statistics.median(tps) if tps else None,
                  "range_server_tps": [min(tps), max(tps)] if tps else None}), flush=True)
PY
}
serve serial
run_leg serial | tee -a "$OUT/results.txt"
cleanup
SPARK_PID=
for _ in $(seq 1 120); do
  temp=$(gpu_temp)
  value=${temp%% *}
  [[ $value =~ ^[0-9]+$ ]] && (( value <= 45 )) && break
  sleep 5
done
echo "cooled; temperature=$(gpu_temp)" | tee -a "$OUT/results.txt"
serve mtp --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000
run_leg mtp | tee -a "$OUT/results.txt"
grep -a "MTP verify timing" "$OUT/serve-mtp.log" | tail -8 | sed 's/\x1b\[[0-9;]*m//g' | tee "$OUT/mtp-timing-tail.txt"
grep -aE "mtp_accept_debug|K2 summary" "$OUT/serve-mtp.log" | tail -8 | sed 's/\x1b\[[0-9;]*m//g' | tee "$OUT/mtp-accept-tail.txt"
echo "results=$OUT"
