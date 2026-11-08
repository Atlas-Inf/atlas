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

## Native DFlash drafter (`dflash/`)

The checkpoint ships Xiaomi's DFlash drafter: 5 non-causal SWA-1024 layers,
block size 8 (7 drafts per step), taps on target layers `[0,11,23,35,47]`,
sink bias, V scale 0.612 (folded into `v_proj` at load), RoPE on the first 64
of 128 head dims. Its trained mask-token vector (`mask_embedding.pt`) replaces
the target's zero `mask_token_id` (151675) row, as SGLang does.

```bash
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --include 'dflash/*'   # both nodes
DFLASH=1 bash scripts/start-mimo-ep2.sh   # adds --dflash --draft-model <snapshot>/dflash --dflash-gamma 8 --dflash-window-size 1024
```

`--dflash` and `--speculative` (MTP) are mutually exclusive. Report the
acceptance rate (`DFlash` lines in the rank-0 log) and tok/s against serial
and MTP K=1 on the same prompts.

## Row-exact verify (`MOE_ROWS=1`)

By default an 8-row speculative verify (DFlash γ=8) runs MiMo's MoE through
the pairwise batched path (a batched router, then per-row experts), each pair
with its own EP all-reduce, and layer 0's dense FFN through the batched
`w4a16_gemv_batchm` kernels. Both differ numerically from serial decode.
`MOE_ROWS=1` (`ATLAS_MOE_ROWS_DECODE=1`, both ranks) instead runs every
verify row through the serial FFN code: MoE rows via
`MoeLayer::forward_rows` (serial per-row arithmetic, one `[n, h]`
all-reduce per layer) and the dense layer per row.

Two batch-invariance fixes apply always (not gated by the flag):

- Full-attention split-K partitions are chosen per row from the row's own
  length (`mimo_row_splits` in `mimo_attn.cu`). Before, they came from the
  batch's longest block table, so a verify row split differently from the
  serial step of the same token.
- The LM head for 2–8 rows uses `dense_gemv_batchm`, which is bit-identical
  per row to serial `dense_gemv_bf16`. It used to be a tiled GEMM.

With `MOE_ROWS=1`, greedy DFlash output should match serial decode
byte-for-byte. If it doesn't, compare the first diverging token's logits
between the two runs.

## Sliding-window KV eviction (`SWA_EVICT=1`, experimental)

**Experimental:** startup, 524K capacity and short-prompt greedy output are
verified on 2× GB10. On the fork (`62ef5e27`) the 67K and 131K needle tests
pass, and prefix caching with pinned SWA tails passes a 67K warm follow-up
(99.9% cache hit, 0.87s TTFT vs 383s cold). The 262K / 524K needle tests are
still open.

The 39 sliding-window layers read only their last 128 tokens. With
`ATLAS_SWA_EVICT=1` their K/V pools hold a per-sequence ring of slots
(`kv_cache/swa_slots.rs`): a block is lent a slot when allocated and gives it
back once it lags `window + max_batch_tokens + 64` rows behind the sequence
end. The block pool then pays only for the 9 full-attention layers, so
`--max-seq-len` can go well past 65536.

```bash
SWA_EVICT=1 MAX_SEQ_LEN=131072 bash scripts/start-mimo-ep2.sh
```

Prefix caching stays on under eviction (`kv_cache/swa_pins.rs`). When a
prompt finishes prefill, its last `ceil((128 + 64) / 16) = 12` whole blocks
keep their sliding-window slots (~38 MB on MiMo) in an LRU of
`ATLAS_SWA_PREFIX_TAILS` tails (launcher: `SWA_PREFIX_TAILS`, default 32).
Pins come from the sequence's block table alone, so both EP ranks pin the
same prompt ends even when their radix caches disagree. After the ranks
agree on a match (F83 EP min-sync), it is capped to the deepest block whose
128-row window is still resident, so the next turn resumes at the previous
prompt's end (or up to 64 rows short of it) and prefills only the reply and
the new message. Without a resident tail it falls back to a full prefill. A
sequence releases only the slots it lent; blocks taken from the cache keep
theirs. `SWA_PREFIX_TAILS=0` turns prefix caching off.

A `sliding-window KV slots exhausted` error means a step wrote more rows than the lag
covers (prefill chunk > `--max-batch-tokens`).

## FP8 block QKV projections (`FP8_QKV=1`, default)

`qkv_proj` stays in the checkpoint's native FP8 E4M3 + F32 128×128 block-scale
form (`ATLAS_MIMO_FP8_QKV=1`) instead of being dequantized to BF16 at load.
Decode/verify (≤64 rows) runs the native `fp8b` GEMV tiers; longer prefills
run the W8A16 GEMM (`fp8b_prefill_gemm`). Full-attention TP chunks (3392
rows) are padded to the 128-row block grid (3456); the pad rows are never
read. `o_proj` stays BF16. The kernel target carries `mxfp8_gemv.cu` (same
source as glm-5.3-flash). `FP8_QKV=0` restores the BF16 path for A/B.

## MXFP4 tensor-core expert gate/up (`ATLAS_MOE_MXFP4_TC=1`, opt-in)

Decode gate+up over routed MXFP4 (E8M0) experts can run a tensor-core
(`mma.sync` BF16) GEMV, `mxfp4_gate_up_tc` (`nvfp4/mxfp4_gemv.cu`), instead of
the scalar `moe_expert_gate_up_shared_t_e8m0`. It keeps the decode grid (16
weight rows per CTA, K split across 8 warps) and stages the transposed
`[K/2, N]` FP4 weights through smem; FP4 nibbles dequantize through a 16-entry
BF16 LUT and the E8M0 scales decode exactly as in the MXFP8 tiers. A second
opt-in, `ATLAS_MOE_MXFP4_TC_SD=1`, swaps the SiLU+down half
(`mxfp4_silu_down_tc`), which computes `silu(gate) * up` on the fly with one
BF16 rounding per element (the scalar kernel keeps FP32 intermediates —
the only intended numeric deviation). Either flag alone leaves the other
half scalar. Both default off pending hardware validation; require
`N % 16 == 0` and `K % 64 == 0` (MiMo: gate/up 2048/4096, silu/down
4096/2048). Other quant targets never resolve the kernels (`try_kernel`
gives 0), so the scalar path is untouched everywhere else.

## Tensor-core paged prefill (`mimo_prefill_tc`)

Chunked (paged) prefill runs a tensor-core (`mma.sync` BF16) flash-attention
kernel, `mimo_prefill_tc`, forked from the shared `prefill_paged_compute.cuh`
for MiMo's geometry: Q/K head dim 192, V head dim 128 (output 128), SWA
`attention_sink` bias (nullable; full-attention layers pass null) and the
`SWA_EVICT=1` `swa_map` logical→ring-slot indirection. The chunk-0 contiguous
prefill and any stale kernel build fall back to the scalar `mimo_prefill_attn`
in `mimo_attn.cu`. The fork lives in this target only
(`prefill_paged_compute_mimo.cuh`); the shared header is untouched. Decode and
the DFlash drafter are unaffected. Off-hardware checks: `nvcc -arch=sm_121a`
compile, fmt/clippy/license/typos; correctness and TTFT are validated on the
fork before this is relied on.

## Known risk areas to watch

1. Missing-kernel errors at startup (kernel audit / `gpu.kernel` lookup) —
   the target reuses deepseek-v4-flash sources via `kernel_source`.
2. Norm convention: MiMo norms are plain RMSNorm (`mimo_v2` is on the
   vanilla-norm list). Gibberish from token 1 points here or at the QKV chunk
   split.
3. MoE without a shared expert on the E8M0 path (DeepSeek-V4 always has one).
4. FP8 `qkv_proj` scale layout: assumed `[4 × ceil(chunk_rows/128), 32]`,
   chunk-major; the loader validates the shape and fails loudly otherwise.
