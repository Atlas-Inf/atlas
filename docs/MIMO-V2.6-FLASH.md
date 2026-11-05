# MiMo-V2.6-Flash on 2× GB10 (EP=2) — phase-1 bring-up

Target: `XiaomiMiMo/MiMo-V2.6-Flash-RL` (`model_type = "mimo_v2"`), text
backbone, official checkpoint served as-is (~173 GB → ~89 GB per rank).

## What is implemented

| Piece | Where |
|---|---|
| Config parser (`mimo_v2`): full/SWA layer pattern, per-type KV heads (4 / 8), Q/K 192 vs V 128, partial RoPE (64 dims), SWA theta, `attention_value_scale`, dense layer 0, sigmoid + `noaux_tc` routing | `crates/atlas-core/src/config/parsers/mimo_v2.rs` |
| Weight loader: chunked FP8 `qkv_proj` (4 TP chunks, per-chunk scale grids) → BF16; BF16 `o_proj`; SWA `attention_sink_bias`; dense FFN → NVFP4; native MXFP4 experts (`weight_scale` E8M0) transcode-free; EP expert filtering; per-layer KV dims | `crates/spark-model/src/weight_loader/mimo_v2.rs` |
| Attention path (decode, multi-seq decode, chunked prefill, chunk-0 cache-skip prefill) | `crates/spark-model/src/layers/qwen3_attention/mimo.rs` |
| CUDA kernels: QKV split + partial RoPE + paged cache write; paged decode (GQA, sink, sliding window, split-K) + reduce; causal prefill over paged history | `kernels/gb10/mimo-v2.6-flash/nvfp4/mimo_attn.cu` |
| Kernel target (reuses deepseek-v4-flash sources for the E8M0 MoE GEMMs) | `kernels/gb10/mimo-v2.6-flash/MODEL.toml` |
| Dual-node launcher | `scripts/start-mimo-ep2.sh` |

## Phase-1 limits (by design)

- `--kv-cache-dtype bf16` only; the loader rejects other KV dtypes.
- No MTP / speculative decoding (3 MTP modules in `model_mtp.safetensors` are skipped).
- No vision / audio (towers are dead namespaces at load).
- Batched multi-stream prefill is disabled for MiMo (per-stream prefill only).
- Sliding-window layers keep their full paged history (no block eviction yet);
  attention only reads the last 128 tokens.

## Verified off-hardware

- `cargo test -p atlas-core --lib mimo_v2` against the published `config.json`.
- `cargo build -p atlas-kernels` with `ATLAS_TARGET_MODEL=mimo-v2.6-flash`
  (nvcc 13.0, sm_121): all kernels compile; `mimo_attn` kernels have no
  register spills.
- fmt / clippy / license headers.

Nothing has run on a GPU yet. Everything below is for the local tester.

## Local test plan (2× GB10)

### 0. Build

```bash
gh pr checkout <PR> && git log -1
docker build -f docker/gb10/Dockerfile -t atlas-gb10:mimo-ep2 .
# copy the image to the worker (or build there too)
docker save atlas-gb10:mimo-ep2 | ssh $WORKER_IP 'sudo docker load'
huggingface-cli download XiaomiMiMo/MiMo-V2.6-Flash-RL   # on BOTH nodes
```

### 1. Launch

```bash
HEAD_IP=<head> WORKER_IP=<worker> scripts/start-mimo-ep2.sh
sudo docker logs -f atlas-mimo-ep0
```

Expected in the logs: `mimo_v2: loading 48 layers, local experts [0, 128)`
on rank 0 (`[128, 256)` on rank 1), then `mimo_v2: built 48 layers`.

### 2. Smoke tests (greedy)

```bash
curl -s http://$HEAD_IP:8888/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"mimo","temperature":0,"max_tokens":64,
  "messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}]}'
```

Then, in order:
1. Short factual prompt (above) — expect `Paris`.
2. Count 1..50 — checks decode stability past the 128-token window.
3. A ~2k-token prompt (paste a long article, ask for a summary) — exercises
   chunked prefill across sliding/full layers and split-K decode.
4. Same prompt twice — second run hits the prefix cache (chunk-0 skip path).
5. `MAX_BATCH=4` relaunch + 4 concurrent requests — multi-seq decode.

### 3. What to send back

For any failure, include:
- `git log -1` of the tested commit, the launch command/env.
- Full rank-0 and rank-1 logs from start to first error
  (`docker logs atlas-mimo-ep0 > rank0.log 2>&1`, same for ep1).
- Which smoke step failed and the exact response text (garbage text vs.
  crash vs. hang matters).
- If there is a CUDA error, rerun once with `CUDA_LAUNCH_BLOCKING=1`
  added to both containers' env and send that log too.
- `nvidia-smi` memory after load on both nodes.

## Known risk areas to watch

1. Missing-kernel errors at startup (kernel audit / `gpu.kernel` lookup) —
   the target reuses deepseek-v4-flash sources via `kernel_source`.
2. Norm convention: MiMo norms are plain RMSNorm (`mimo_v2` is on the
   vanilla-norm list). Gibberish from token 1 points here or at the QKV chunk
   split.
3. MoE without a shared expert on the E8M0 path (DeepSeek-V4 always has one).
4. FP8 `qkv_proj` scale layout: assumed `[4 × ceil(chunk_rows/128), 32]`,
   chunk-major; the loader validates the shape and fails loudly otherwise.
