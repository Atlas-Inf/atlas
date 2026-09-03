#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Clean perf pass for Qwen3.8-27B-NVFP4 (unsloth) on AzeezStrix, after a
# cooldown: quick-speed-bench (isl 512, osl 512, n=5) under BOTH recipes —
# (1) the frozen accuracy recipe (serial decode, no MTP) and (2) the
# submission profile (MTP K=2, prefix caching, slots 16). Preservation
# dtypes on in both.
set -uo pipefail
cd ~/atlas-inf-pr8
SNAP=/home/azeez/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1
export ATLAS_KV_EXTERNAL_RESERVE_GB=0
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1

serve() {
  pkill -f '[s]park serve' 2>/dev/null && sleep 8 || true
  target/release/spark serve "$SNAP" \
    --model-name unsloth/Qwen3.8-27B-NVFP4 \
    --host 127.0.0.1 --port 8081 \
    --max-seq-len "$1" --max-prefill-tokens 2048 \
    --gpu-memory-utilization "$2" \
    --kv-cache-dtype bf16 --lm-head-dtype bf16 \
    --max-batch-size 1 \
    --disable-tool-grammar true \
    --ssm-cache-slots "$3" --ssm-checkpoint-interval 16 \
    $4 \
    --disable-thinking \
    --dangerously-allow-unresolved-kernel-lookups \
    >"$HOME/q38-perf-serve-$5.log" 2>&1 &
  SPARK_PID=$!
  for i in $(seq 1 150); do
    curl -s -m2 http://127.0.0.1:8081/v1/models >/dev/null 2>&1 && return 0
    kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED ($5):"; tail -5 "$HOME/q38-perf-serve-$5.log"; return 1; }
    sleep 2
  done
  echo "SERVER NOT UP ($5)"; return 1
}

echo "=== PASS 1: accuracy recipe (0.88/4096/no-spec/slots-0) — $(date -u +%H:%M) ===" | tee -a "$HOME/q38-perf-pass.log"
serve 4096 0.88 0 "" accuracy || exit 1
echo "server up; temp: $(rocm-smi --showtemp 2>/dev/null | LC_ALL=C grep -oE '[0-9.]+ C' | head -1)" | tee -a "$HOME/q38-perf-pass.log"
SHIM="$SHIM" ./target/release/spark benchmark run quick-speed-bench \
  --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 \
  --param isl=512 --param osl=512 --param runs=5 2>&1 \
  | LC_ALL=C sed "s/\x1b\[[0-9;]*m//g" | LC_ALL=C grep -aE "Decode tok/s|TTFT \(server|TPOT|Output tok|run [0-9]/|measured|recorded" \
  | tee -a "$HOME/q38-perf-pass.log"

echo "=== PASS 2: submission profile (0.92/64K/MTP K=2/prefix/slots-16) — $(date -u +%H:%M) ===" | tee -a "$HOME/q38-perf-pass.log"
serve 65536 0.92 16 "--speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 --enable-prefix-caching" submission || exit 1
echo "server up; temp: $(rocm-smi --showtemp 2>/dev/null | LC_ALL=C grep -oE '[0-9.]+ C' | head -1)" | tee -a "$HOME/q38-perf-pass.log"
SHIM="$SHIM" ./target/release/spark benchmark run quick-speed-bench \
  --url http://127.0.0.1:8081 --model unsloth/Qwen3.8-27B-NVFP4 \
  --param isl=512 --param osl=512 --param runs=5 2>&1 \
  | LC_ALL=C sed "s/\x1b\[[0-9;]*m//g" | LC_ALL=C grep -aE "Decode tok/s|TTFT \(server|TPOT|Output tok|run [0-9]/|measured|recorded|mtp_accept" \
  | tee -a "$HOME/q38-perf-pass.log"
LC_ALL=C grep -a "mtp_accept_debug" "$HOME/q38-perf-serve-submission.log" | LC_ALL=C tail -3 | LC_ALL=C sed "s/\x1b\[[0-9;]*m//g" | LC_ALL=C cut -c1-190 | tee -a "$HOME/q38-perf-pass.log"

pkill -f '[s]park serve' 2>/dev/null
echo "PERF PASS DONE — $(date -u +%H:%M)" | tee -a "$HOME/q38-perf-pass.log"
