#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Cross-library drift check: Atlas (unsloth/Qwen3.6-27B-NVFP4, the certified
# checkpoint) vs llama.cpp on the SAME architecture — Q8_0 as the near-lossless
# reference and Q4_K_M as the matched-4bit reference. Greedy, token-match +
# mean top-logprob KL via scripts/mlperf-edge/kl_coherence_gate.py.
#
# Run AFTER the ST-995 gate frees the GPU. ~20 min total.
set -uo pipefail
cd ~/atlas-inf-pr8
SHIM=$(ls -dt target/release/build/atlas-kernels-*/out | head -1)
export LD_LIBRARY_PATH="$SHIM:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1
export ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_KV_OVERCOMMIT=1
OUT="$HOME/q38-llamacpp-crosscheck.log"
: > "$OUT"
echo "# Atlas vs llama.cpp drift cross-check — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"

Q36=$(ls -d ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/*/ | head -1)
Q8=/home/azeez/models/qwen36-27b/Qwen3.6-27B-Q8_0.gguf
Q4=/home/azeez/models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf
LLAMA=~/llama.cpp/build/bin
[ -x "$LLAMA/llama-server" ] || { echo "llama-server missing" >> "$OUT"; exit 1; }
[ -f "$Q8" ] || { echo "Q8_0 gguf missing" >> "$OUT"; exit 1; }

pkill -f 'spark serve' 2>/dev/null; pkill -f 'llama-server' 2>/dev/null; sleep 6

# ── leg 1: llama.cpp Q8_0 (near-lossless reference) on 8093 ──
"$LLAMA/llama-server" -m "$Q8" --host 127.0.0.1 --port 8093 -c 8192 -ngl 99 \
  --threads 8 >"$HOME/q38-llama-q80.log" 2>&1 &
LLAMA_PID=$!
for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8093/health >/dev/null 2>&1 && break; sleep 2; done
curl -s -m2 http://127.0.0.1:8093/health >/dev/null 2>&1 || { echo "llama Q8_0 NOT UP" >> "$OUT"; kill $LLAMA_PID; exit 1; }
echo "llama.cpp Q8_0 reference up on 8093" >> "$OUT"

# ── leg 2: Atlas, certified checkpoint + certified recipe, on 8094 ──
export ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1 ATLAS_FP8_DEQUANT_FFN_TO_BF16=1 ATLAS_GDN_BF16_WEIGHTS=1
target/release/spark serve "$Q36" \
  --model-name unsloth/Qwen3.6-27B-NVFP4 \
  --host 127.0.0.1 --port 8094 \
  --max-seq-len 65536 --max-prefill-tokens 2048 \
  --gpu-memory-utilization 0.60 \
  --kv-cache-dtype bf16 --lm-head-dtype bf16 \
  --max-batch-size 1 \
  --speculative --num-drafts 1 --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots 32 --ssm-checkpoint-interval 16 \
  --disable-thinking \
  --dangerously-allow-unresolved-kernel-lookups \
  >"$HOME/q38-atlas-q36.log" 2>&1 &
ATLAS_PID=$!
for i in $(seq 1 150); do curl -s -m2 http://127.0.0.1:8094/v1/models >/dev/null 2>&1 && break; kill -0 $ATLAS_PID 2>/dev/null || { echo "ATLAS DIED:" >> "$OUT"; tail -5 "$HOME/q38-atlas-q36.log" >> "$OUT"; kill $LLAMA_PID; exit 1; }; sleep 2; done
curl -s -m2 http://127.0.0.1:8094/v1/models >/dev/null 2>&1 || { echo "Atlas NOT UP" >> "$OUT"; kill $LLAMA_PID $ATLAS_PID; exit 1; }
echo "Atlas (3.6 NVFP4, certified recipe) up on 8094" >> "$OUT"

# ── the gate: baseline = llama.cpp Q8_0, candidate = Atlas ──
echo "## Atlas vs llama.cpp Q8_0 (near-lossless reference)" >> "$OUT"
python3 ~/atlas-inf-pr8/scripts/mlperf-edge/kl_coherence_gate.py 8093 8094 "$HOME/q38-kl-q80.json" >> "$OUT" 2>&1
echo "  [exit $?]" >> "$OUT"

pkill -f 'llama-server' 2>/dev/null; sleep 4
# ── leg 3: llama.cpp Q4_K_M (matched 4-bit) on 8093 ──
"$LLAMA/llama-server" -m "$Q4" --host 127.0.0.1 --port 8093 -c 8192 -ngl 99 \
  --threads 8 >"$HOME/q38-llama-q4.log" 2>&1 &
LLAMA_PID=$!
for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:8093/health >/dev/null 2>&1 && break; sleep 2; done
curl -s -m2 http://127.0.0.1:8093/health >/dev/null 2>&1 || { echo "llama Q4_K_M NOT UP" >> "$OUT"; kill $LLAMA_PID $ATLAS_PID; exit 1; }
echo "## Atlas vs llama.cpp Q4_K_M (matched 4-bit)" >> "$OUT"
python3 ~/atlas-inf-pr8/scripts/mlperf-edge/kl_coherence_gate.py 8093 8094 "$HOME/q38-kl-q4.json" >> "$OUT" 2>&1
echo "  [exit $?]" >> "$OUT"

kill $LLAMA_PID $ATLAS_PID 2>/dev/null
echo "CROSSCHECK DONE" >> "$OUT"
tail -30 "$OUT"
