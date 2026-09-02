#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# The 20-trajectory / 1,007-turn MLPerf-edge agentic performance replay on
# AzeezStrix — run under the PREVIOUS WORKING MLPerf v6.1 submission serve
# configuration (serve2_slots32.sh: util 0.92, seq 64K, MTP K=2, prefix
# caching, ssm slots 32 / interval 16, ATLAS_FORCE_GLOBAL_GDN +
# SSM_TAIL_MIDCHUNK + KV_OVERCOMMIT), mapped to the Qwen3.8-27B-NVFP4
# (unsloth) AMD dtype scheme: per-row FP8 attention/GDN/lm_head/final-8-FFN
# kept BF16 (preservation), everything else NVFP4, KV bf16, MTP bf16.
# Harness: ~/endpoints (inference_endpoint @ 0bc51d0); dataset
# agentic_coding_2.5h.jsonl, 20 trajectories -> 1,007 client turns.
set -euo pipefail
cd ~/endpoints
PORT=8081
TS=$(date +%Y%m%d_%H%M%S)
RD="results_strix_qwen38_unsloth_replay_${TS}"
CFG="$(mktemp -t q38_replay_XXXX.yaml)"
LOG="$HOME/q38-replay-1007.log"

python3 - "$RD" "$CFG" <<'PY'
import sys, yaml
rd, cfg = sys.argv[1:3]
c = yaml.safe_load(open("/home/azeez/endpoints/results/atlas_final_perf_20260720_143021/config.yaml"))
c["report_dir"] = rd
c.setdefault("endpoint_config", {})["endpoints"] = ["http://localhost:8081"]
c["model_params"]["name"] = "unsloth/Qwen3.8-27B-NVFP4"
runtime = c.setdefault("settings", {}).setdefault("runtime", {})
runtime["min_duration_ms"] = 600_000
runtime["max_duration_ms"] = 14_400_000
runtime["scheduler_random_seed"] = 16159082839903944936
runtime["dataloader_random_seed"] = 2747215439041700203
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
print("replay config ->", cfg)
PY

# Serve: the previous working MLPerf submission profile, mapped to Qwen3.8.
cd ~/atlas-inf-pr8
SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1
export ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_KV_OVERCOMMIT=1
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
pkill -f 'spark serve' 2>/dev/null && sleep 8 || true
target/release/spark serve ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108 \
  --model-name unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port "$PORT" \
  --max-seq-len 65536 --max-prefill-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype bf16 --lm-head-dtype bf16 \
  --max-batch-size 1 \
  --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots 32 --ssm-checkpoint-interval 16 \
  --disable-thinking \
  --dangerously-allow-unresolved-kernel-lookups \
  >"$LOG" 2>&1 &
SPARK_PID=$!
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SPARK_PID" 2>/dev/null || { echo "SERVER DIED:"; tail -8 "$LOG"; exit 1; }
  sleep 2
done
echo "server up (submission profile) — running the 1,007-turn replay"

cd ~/endpoints
./.venv/bin/inference-endpoint benchmark from-config -c "$CFG" --mode perf -v 2>&1 | tail -40
echo "REPLAY_DONE rd=/home/azeez/endpoints/${RD}"
kill "$SPARK_PID" 2>/dev/null || true
