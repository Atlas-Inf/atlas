# Hy4 gated MLA and coarse IndexPool

These blocks are also used by the dedicated eager Hy4 model. They follow
the pinned reference in the parent README, Flash `attention.py`,
`compressed_indexer.py`, `indexpool_ops.py` and `indexpool_prepare.py`.
No vLLM source is copied into the engine.

## MLA

`Hy4Attention::load` takes a post-topology ModelConfig: the ordinary DSA
head count is already TP-local, as in GLM-5.3. SWA metadata still contains
the global head count. TP=1/2 works for both layer kinds. A projections,
latent KV and indexer are replicated; B/gate/sink are head-local; output
projection is column-sliced, reduced via CommBackend, then post-normalized.
The loader checks shapes/dtypes before dispatch and creates lossless BF16
transposed/sliced weights for local heads.

Query: q_a -> RMS -> q_b -> NoPE absorption + rotary64.
KV: kv_a -> RMS(latent512) + rotary64. Attention reads a BF16 latent576
cache, applies qk256^-0.5, and starts online softmax with the learned FP32
per-head sink as a zero-value entry. Value extraction precedes elementwise
sigmoid gating, output projection, TP reduction and post_proj_norm.
The caller supplies the decoder input RMSNorm output after iHC collapse.

The DeepSeek `mla_absorbed.cu` per-head GEMV kernels are included unchanged
for query absorption and value extraction. DeepSeek attention/compression
cache formats and fixed sink/sparse ABIs are not interchangeable with this
latent576 cache; Hy4 has a distinct paged sink loop.

DSA and SWA select separate default RoPE theta values from metadata. Both
ordinary text positions (three identical axes) and three-axis interleaved
MRoPE are explicit. Main attention stores adjacent pairs; the indexer
rearranges its rotated suffix into split halves before FWHT.

## IndexPool

Only DSA loads `self_attn.compressed_indexer.*`. Token keys are BF16
projected and normalized. Four committed keys feed FP32 learned value/gate
projections, per-channel softmax with intra-group APE, weighted pooling and
RMS, then a BF16 boundary. Rotate at the group's **first token's** position.
FWHT128 is normalized; E4M3 quantization uses power-of-two scales and 448.
Query projection/rotation/quantization and head weights are replicated.
Scores sum weighted ReLU dot products over heads with the normalization
and query/key scales. Select 511 historical groups, sort their logical
IDs, expand to original token IDs, append the causal current tail, pad -1.
Ties choose the lower group ID; backend-specific reference tie parity is
not established.

## Request state and cache ABI

`AttentionCache` owns one request's pages and scratch for one layer.
The latent table is request-local, with BF16 `[page,slot,576]` rows.
Pool pages have 64 groups: 8192 E4M3 bytes followed by 64 FP32 scales
(8448 bytes/page). Pool and latent tables are independent. Four BF16
128-wide tail rows preserve incomplete groups; first-group coordinates
are retained in the host metadata. No attention content is compressed.

SWA attends only `[max(0,p-511),p]`. Its circular table reserves
`ceil((512-1)/page_size)+1` pages, including the extra boundary page needed
when the window spans partial pages. Rows enqueue sequentially so prefill
cannot overwrite keys needed by an earlier query. This same path serves
chunked prefill and decode, including sinks; there is no dense-prefill
fallback that omits sinks.

Request caches never alias mutable pages. `fork` deep-copies resident
pages and partial carry on the bound stream; fork before speculative
work and discard rejected state rather than rewinding lengths over
overwritten data. Empty batches enqueue nothing. Metadata/capacity and
communicator errors are checked before dispatch. Runtime dispatch errors
poison the cache; do not resume it. One stream is bound to each cache.
Drain it before explicit release or Drop. Caller buffers must cover
`[rows,H]` and may not overlap each other or internal state.

This is an eager, correctness-first implementation, not a throughput
claim: projections/scoring/top-k are not optimized, per-row launches are
serial, and prefix forks copy rather than COW-share pages. The eager model
owns request caches and Engram lookback; Marconi prefix reuse, graph capture
and DFly reuse are not provided.

## Verification limits

CPU tests cover masking, sinks, gates, rotary ordering/theta, FWHT/FP8
scales, pooling and selection, TP head plans, window wrap, chunk/carry
and speculative state isolation. Mock tests exercise real loading,
head slices/transposition, allocation cleanup, dispatch ABI, reduction
placement and prefix forks, not CUDA numerics. GB10 PTX compiles on
robert2; no Spark or GPU execution is performed.

Sparse MLA absorption/BF16 boundaries, FP8/DeepGEMM parity and tied top-k
behavior, real TP collectives, quality, residency/boot peaks and speed
remain GPU-unverified. Retain the full-model factory guard until those
checks and the remaining architecture are complete.
