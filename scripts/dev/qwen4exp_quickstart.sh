#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 on a GB10, from nothing to a validated server.
#
#   ./scripts/dev/qwen4exp_quickstart.sh /path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4
#
# Serves the model, waits for health, runs the three validation legs
# (coherence / tool-calling / perf at concurrency 1 and 4) and prints the
# numbers. Leaves the server running; Ctrl-C or `pkill -x spark` stops it.
#
# Docker equivalent (same flags, see docker/gb10/qwen3.8-flash-next/nvfp4/):
#   docker build -f docker/gb10/qwen3.8-flash-next/nvfp4/Dockerfile -t atlas-q4e .
#   docker run --gpus all --ipc=host -p 8889:8889 -v /path/to/ckpt:/ckpt:ro \
#     atlas-q4e serve --model-from-path /ckpt --model-name qwen4exp \
#       --kernel-target qwen3.8-flash-next --bind 0.0.0.0 --port 8889 \
#       --max-seq-len 8192 --kv-cache-dtype bf16 \
#       --gpu-memory-utilization 0.90 --fast-load-prefetch-shards
#
# WHY EACH NON-OBVIOUS FLAG IS THERE — all four were learned the hard way on
# the first real run, and three of them fail SILENTLY if omitted:
#
#   --kv-cache-dtype bf16      the checkpoint ships no k_scale/v_scale, so the
#                              FP8 default clips BF16 into E4M3 [-448, 448]
#   --gpu-memory-utilization   BF16 KV needs 0.90; at 0.80 the model loads and
#                              then dies with "No memory left for KV cache"
#   ATLAS_PLE_MAX_TOKENS       defaults to 2048; a 2191-token prompt exceeds it
#   --kernel-target            selects the 177 kernels this architecture needs;
#                              the startup audit fails closed without it
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

CKPT="${1:?usage: $0 /path/to/checkpoint}"
PORT="${PORT:-8889}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
[[ -f "$CKPT/config.json" ]] || { echo "no config.json under $CKPT" >&2; exit 2; }

command -v nvidia-smi >/dev/null || { echo "no nvidia-smi — this needs a GPU" >&2; exit 2; }
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | sed 's/^/gpu: /'
# A GB10 is coherent unified memory; `memory.total [N/A]` + ATS is the tell,
# and it is why the 51 GB n-gram table can be demand-paged rather than resident.
nvidia-smi -q 2>/dev/null | grep -i "addressing mode" | sed 's/^ */  /'

echo "building (release)..."
cargo build --release -p spark-server --bin spark --no-default-features --features cuda \
  || { echo "build failed" >&2; exit 1; }

export ATLAS_PLE_MAX_TOKENS="${ATLAS_PLE_MAX_TOKENS:-$MAX_SEQ_LEN}"
(( MAX_SEQ_LEN > 8192 )) && export ATLAS_PLE_MAX_TOKENS=9500

echo "serving on 127.0.0.1:$PORT ..."
target/release/spark serve \
  --model-from-path "$CKPT" \
  --model-name qwen4exp \
  --kernel-target qwen3.8-flash-next \
  --bind 127.0.0.1 --port "$PORT" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --kv-cache-dtype bf16 \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-batch-size "${MAX_BATCH_SIZE:-4}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --fast-load-prefetch-shards \
  --enable-prefix-caching \
  > qwen4exp-serve.log 2>&1 &
SERVER=$!
trap 'echo; echo "server still running as pid $SERVER (pkill -x spark to stop)"' EXIT

echo "waiting for health (load is ~100 s; 206 shards)..."
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited — see qwen4exp-serve.log" >&2; tail -20 qwen4exp-serve.log >&2; exit 1; }
  sleep 5
done

python3 scripts/dev/qwen4exp_smoke.py --base "http://127.0.0.1:$PORT" --wait 2
rc=$?
echo
echo "server log: qwen4exp-serve.log"
echo "expected on a GB10: coherence 7/7, tools 6/6, ~14.5 tok/s single, ~28 tok/s at 4"
exit $rc
