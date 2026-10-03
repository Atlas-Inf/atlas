# Recipe: Qwen3.8-Flash-Next EXL3 on DGX Spark (GB10) — Full 262K Context

This recipe documents the exact build instructions and runtime serving configurations for Qwen3.8-Flash-Next (`qwen4_exp`) EXL3 checkpoints (`flashnext-exl3-3.05bpw`, `flashnext-exl3-4.05bpw`, `CYBER-FROST-3.8-EXL3-SAGE-3.87bpw`) on NVIDIA DGX Spark (GB10, sm_121) at full native **262,144-token (262K) maximum context**.

---

## 1. Context & Memory Architecture (262K Tokens)

Qwen3.8-Flash-Next has a native maximum position embedding length of **262,144 tokens** (`max_position_embeddings: 262144`).

Unlike traditional full-attention transformers where 262K context requires hundreds of gigabytes of KV cache, Qwen3.8-Flash-Next uses a hybrid architecture:
- **36 Gated Delta Net (GDN) Linear Attention Layers**: Constant $O(1)$ recurrent state memory ($48 \times 128$ matrix) that never grows with context.
- **12 Full Attention Layers**: Only 12 layers store KV cache (2 KV heads, head dimension 256, BF16).
  $$\text{KV footprint per token} = 12 \times 2 \times 2 \times 256 \times 2\text{ bytes} = 24.576\text{ KB/token}$$
  $$\text{Full 262,144-token KV cache} = 262,144 \times 24.576\text{ KB} \approx \mathbf{6.29\text{ GiB per sequence}}$$

On a 119.7 GiB GB10 GPU:
- Model weights (4.05 bpw): ~61.7 GiB
- 262K KV cache (1 sequence): ~6.3 GiB
- Total resident memory: **~68.0 GiB** (leaving >51 GiB of VRAM headroom).
- Even with 4 concurrent 262K sequences, total KV memory is ~25.2 GiB (total ~86.9 GiB), fitting comfortably within the 119.7 GiB ceiling.

---

## 2. Build From Source (DGX Spark GB10)

### Prerequisites
- NVIDIA Driver >= 580 (CUDA 13.0+)
- `rustc` / `cargo` (1.80+)
- `clang` / `llvm-18` (for bindgen)

### Compilation Commands
```bash
# Set compiler and target variables
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda

# Pin compilation targets to GB10 + Qwen3.8 Flash-Next EXL3
export ATLAS_TARGET_HW=gb10
export ATLAS_TARGET_MODEL=qwen3.8-flash-next
export ATLAS_TARGET_QUANT=exl3

# Ensure libclang.so is discoverable
mkdir -p "$HOME/.local/lib"
ln -sfn /usr/lib/llvm-18/lib/libclang.so.1 "$HOME/.local/lib/libclang.so"
export LIBCLANG_PATH="$HOME/.local/lib"
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

# Build release binary (single GPU target)
cargo build --release -p spark-server --bin spark --no-default-features --features cuda
```
The binary lands at `./target/release/spark`.

---

## 3. Serving Recipes (Full 262K Context)

### Recipe A — Full 262K Context Single-Stream Serve (Zero Environment Variables)
Runs native CoopMK MoE decode (`coopmk_selected() = true`, `coopmk_requested() = true`) and device-resident QSA selection by default with full 262,144 context depth:

```bash
./target/release/spark serve \
  --model-from-path /home/cruzspark/models/flashnext-exl3-4.05bpw \
  --model-name qwen4exp \
  --kernel-target qwen3.8-flash-next \
  --bind 127.0.0.1 \
  --port 8888 \
  --max-seq-len 262144 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --gpu-memory-utilization 0.90 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching
```

### Recipe B — Full 262K Context Multi-Sequence Concurrent Serve
Supports up to 4 concurrent sequences up to 262,144 tokens:

```bash
./target/release/spark serve \
  --model-from-path /home/cruzspark/models/flashnext-exl3-4.05bpw \
  --model-name qwen4exp \
  --kernel-target qwen3.8-flash-next \
  --bind 127.0.0.1 \
  --port 8888 \
  --max-seq-len 262144 \
  --max-num-seqs 4 \
  --max-batch-size 4 \
  --gpu-memory-utilization 0.90 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching
```

### Recipe C — Peak Speculative Benchmark Serve (MTP K=1, Full 262K Context)
Enables packed M-row dense GEMV, grouped prefill, device QSA, and MTP fast-masked verify at full context depth:

```bash
ATLAS_EXL3_COOPMK=1 \
ATLAS_EXL3_DENSE_NATIVE=1 \
ATLAS_EXL3_PREFILL_GROUPED=1 \
ATLAS_EXL3_COOPMK_ROWS=1 \
ATLAS_EXL3_DENSE_ROWS=1 \
ATLAS_MTP_FAST_MASKED=1 \
ATLAS_MTP_DRAFT_CONFIDENCE=0.6 \
./target/release/spark serve \
  --model-from-path /home/cruzspark/models/flashnext-exl3-4.05bpw \
  --model-name qwen4exp \
  --kernel-target qwen3.8-flash-next \
  --bind 127.0.0.1 \
  --port 8888 \
  --max-seq-len 262144 \
  --max-num-seqs 4 \
  --max-batch-size 4 \
  --gpu-memory-utilization 0.90 \
  --speculative \
  --num-drafts 1 \
  --fast-load-prefetch-shards \
  --enable-prefix-caching
```

---

## 4. Querying the Endpoint (OpenAI API Compatible)

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "What is the capital of Spain?"}],
    "temperature": 0.0,
    "max_tokens": 64
  }'
```
