# Qwen3.8-Flash-Next EXL3 on one GB10 (262K context)

This is the minimal native packed-EXL3 MoE serving recipe for PR #160.
It keeps routed and shared MoE experts as packed trellis weights and uses
CoopMK for decode. Dense linears continue to materialize to BF16 at load.
The shared attention/QSA, MTP, and scheduler implementations come from
current Atlas main; no separate QSA/device-dense/grouped-prefill experiment
is required.

## Checkpoint for the merge gate

- Hugging Face: `turboderp/Qwen3.8-Flash-Next-exl3`
- Revision previously tested on GB10: `3.05bpw_h5_ng5`
- Also supported by the native loader: 4.05 bpw checkpoints with the
  same packed-EXL3 layout

For a reproducible gate, resolve the HF revision to an immutable commit
SHA and record it alongside the model directory. Earlier test numbers
pre-date this integration with main and are **not** new benchmark claims.

## Build on GB10 (SM121)

Use a CUDA toolchain supporting GB10 and Atlas's current Rust toolchain.

```bash
export ATLAS_TARGET_HW=gb10
export ATLAS_TARGET_MODEL=qwen3.8-flash-next
export ATLAS_TARGET_QUANT=exl3

cargo build --locked --release -p spark-server --bin spark \
  --no-default-features --features cuda
```

Do not set `ATLAS_SKIP_BUILD=1` for a real GB10 build; that skips GPU
kernel compilation. `ATLAS_NO_RDMA=1` can disable the optional RDMA
shim on systems without libibverbs headers.

## Single-sequence decode / prefill

```bash
./target/release/spark serve \
  --model-from-path /path/to/flashnext-exl3-3.05bpw \
  --model-name qwen4exp \
  --kernel-target qwen3.8-flash-next \
  --bind 127.0.0.1 \
  --port 8888 \
  --max-seq-len 262144 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --gpu-memory-utilization 0.90
```

`ATLAS_EXL3_COOPMK=1` is the default; setting it to `0` selects the
packed EXL3 reconstruct fallback for diagnosis. Neither path requantizes
MoE experts into a resident NVFP4 copy. The max sequence length is a
configuration upper bound, **not** a guarantee that a particular workload
or concurrent set of requests fits in memory.

## MTP smoke gate

On the same checkpoint, add `--speculative --num-drafts 1` to the
serve command above. The MTP block detects trellis experts and loads
them through the same packed EXL3 MoE path; all other draft functionality
uses current main. Compare greedy output with the non-speculative serve
and run the maintainer's existing NVFP4/agentic/BFCL gates.

## Greedy request

```bash
curl -sS http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen4exp",
    "messages": [{"role": "user", "content": "What is the capital of Spain?"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```
