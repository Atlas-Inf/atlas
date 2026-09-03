#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Extended perf pass for the AMD-community numbers: prefill scaling
# (isl 512/2048/4096) and decode sustain (osl 1024) on the accuracy recipe
# (the stable fingerprint). Preservation dtypes on.
set -uo pipefail
cd ~/atlas-inf-pr8
SNAP=/home/azeez/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1
export ATLAS_KV_EXTERNAL_RESERVE_GB=0
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1

pkill -f '[s]park serve' 2>/dev/null && sleep 8 || true
target/release/spark serve "$SNAP" \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port 8081 \
  --max-seq-len 4096 --max-prefill-tokens 2048 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype bf16 --lm-head-dtype bf16 \
  --max-batch-size 1 \
  --disable-tool-grammar true \
  --ssm-cache-slots 0 --ssm-checkpoint-interval 16 \
  --disable-thinking \
  --dangerously-allow-unresolved-kernel-lookups \
  >"$HOME/q38-perf2-serve.log" 2>&1 &
SPARK_PID=$!
for i in $(seq 1 150); do
  curl -s -m2 http://127.0.0.1:8081/v1/models >/dev/null 2>&1 && break
  kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED:"; tail -5 "$HOME/q38-perf2-serve.log"; exit 1; }
  sleep 2
done
echo "server up; temp $(rocm-smi --showtemp 2>/dev/null | LC_ALL=C grep -oE '[0-9.]+ C' | head -1)"

for ISL in 512 2048 4096; do
  echo "=== isl=$ISL osl=512 runs=5 ==="
  ./target/release/spark benchmark run quick-speed-bench \
    --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 \
    --param isl=$ISL --param osl=512 --param runs=5 2>&1 \
    | LC_ALL=C sed "s/\x1b\[[0-9;]*m//g" | LC_ALL=C grep -aE "Decode tok/s \(server\)|TTFT \(server prefill\)|TPOT|Output tok|measured|recorded"
done
echo "=== decode sustain: isl=512 osl=1024 runs=3 ==="
./target/release/spark benchmark run quick-speed-bench \
  --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 \
  --param isl=512 --param osl=1024 --param runs=3 2>&1 \
  | LC_ALL=C sed "s/\x1b\[[0-9;]*m//g" | LC_ALL=C grep -aE "Decode tok/s \(server\)|TTFT \(server prefill\)|TPOT|Output tok|measured|recorded"
pkill -f '[s]park serve' 2>/dev/null
echo "EXTENDED PERF DONE"
