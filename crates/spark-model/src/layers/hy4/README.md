# Hy4 eager text model blocks

The factory uses the dedicated eager model in `model/hy4`, including Engram.
See [`docs/HY4_TWO_SPARK.md`](../../../../../docs/HY4_TWO_SPARK.md) for the
unexecuted TP2/EP2 launch and validation limits.

The standalone gated MLA/SWA/IndexPool path is documented in
[`attention/README.md`](attention/README.md).

## Numeric contract

Pinned equation reference: vLLM PR #68296,
`af127016aa1e14f1e597b546e14f5fdfbd05db11`, inherited HYV4 `hc.py`/`moe.py`
and Flash `moe.py`. Implemented independently; no vLLM code is vendored.
The scalar CPU oracles live in `atlas-core::hy4_reference`.

The highway is **BF16**, matching the reference's `.to(x.dtype)` after
collapse/scatter. Gate projection, flattened RMS and sigmoid are FP32.
Pre gates are `sigmoid(dot * scale[0] + base) + hc_eps`; post gates are
`hc_magnitude * sigmoid(dot * scale[1] + base) + hc_eps`.
Post is independent residual addition, not Sinkhorn or a comb matrix.
Head uses four pre gates and the same flattened RMS. `rms_norm_eps` is the
flattened RMS epsilon; `hc_eps` is only the additive gate epsilon.
FP32 checkpoint gates are borrowed; BF16 gates (including a BF16 head) are
explicitly promoted with the existing FP32 dense loader and owned separately.

The existing DeepSeek/Qwen mHC dispatch uses an FP32 highway and different
gate layouts/ABIs; substituting those kernels would lose the reference BF16
rounding. Their code and dispatch are left unchanged. Hy4 uses the same
GpuBackend, KernelLaunch, WeightStore and kernel-registry infrastructure.

The router reads the **full H** input, with BF16 checkpoint weights and
FP32 dot products/logits. Correction bias changes selection only.
Mixture weights come from unbiased sigmoid scores, optionally normalized,
then multiplied by routed_scaling_factor. Zero score sums are guarded by
`1e-20`. Corrected-score ties choose the lower expert ID deterministically;
tie parity with vLLM backend-specific top-k is not established.

Only the routed branch projects H -> latent -> routed experts -> latent -> H.
Gate is upper-clamped before SiLU (negative gates are not lower-clamped);
up is symmetrically clamped. Shared experts and dense layer 0 are unclamped.
Combine shared+routed before combine_output_norm; dense layer 0 applies
post_fc2_norm. RMS weights are plain multipliers, not offset-by-one norms.
Both are before iHC scatter. The caller applies the decoder block's input
RMSNorm after iHC collapse, before calling the FFN.

## Ownership and execution

Hy4Ffn::load uses the existing ModelOpt `weight_map::quantized` loader
after checking packed U8 weights, E4M3 per-16 scales and FP32 scalar
global/input scales. W4A16 consumes BF16 activations, so input_scale is
validated but not used to quantize activations. No dequant/requant copy.

TP/EP are coincident groups of size 1 or 2. Whole contiguous experts are
owned by one rank; nonlocal descriptors are null and their tensors are not
requested. Router, latent projections, shared/dense MLP and norms are
replicated. Reduce only routed full-H outputs **after latent up**, then
compute/add replicated shared once per rank, then normalize. This is not
expert tensor-slicing and not a Cartesian four-rank mesh. A matching
CommBackend is mandatory for EP2 sparse execution.

Checkpoint tensors are borrowed for the layer's lifetime. The descriptor
table and per-call scratch are explicitly released or reclaimed on Drop.
Scratch is capacity/shape checked, and must not be shared across concurrent
requests or streams. Caller-provided buffers must cover the documented
tensor extents and may not overlap inputs/scratch. Drain the stream before
releasing any source or scratch. Empty batches launch nothing.

The CUDA FFN uses warp-cooperative W4A16 GEMVs over the original ModelOpt
layout, not the generic grouped/tensor-core MoE (which assumes routed width
equals H and different clamp/combine boundaries). Down projections compute
selected experts in parallel, round each result to BF16, then combine in
selection order. Routing is block-parallel with deterministic lower-ID ties.
It supports prefill/decode rows without host routing or tensor readbacks.
`scripts/dev/hy4_fast_kernels_test.cu` checks the CUDA arithmetic and tails.

## Contiguous token rows

`model/hy4` processes prefill chunks layer-major, with at most 128 contiguous
rows per internal chunk and the scheduler's `--max-prefill-tokens` budget
unchanged. BF16/FP8 projections reuse weights across eight-row tiles.
NVFP4 routing compacts rows by expert on device, reuses weights across the
group, and restores the original top-k combination order and BF16 boundaries.
Engram hashes use each row's actual two preceding tokens; its projections
preserve the original accumulation order. iHC and reductions cover all rows.

Intermediate prefill chunks omit the head; the final chunk evaluates only
its last row. `Model::decode_verify` evaluates every row and exposes contiguous
`[N,vocab]` FP32 logits through `logits_buffer_ptr`, as well as returning
per-row argmax IDs. It commits all N tokens and supports one sequence,
not a packed batch of independent sequences. Draft proposals, acceptance,
cache rollback and DFly execution are not implemented by this primitive.
Single-token decode retains the existing GEMV path.

For a benchmark-only deployment, `ATLAS_HY4_VERIFY_BENCH=1` benchmarks
N=1/2/4/6 on the first prefill request, with a prefix of up to 57 tokens.
Inputs are six greedy tokens obtained with single-row decode; per-row argmax
is compared with that reference. Each size has one warmup and three timed
forwards, including all-row logits
and synchronization but excluding argmax. Request caches are recreated
outside timing and restored empty before the real prefill. The environment
variable is startup-latched and off by default; disable it for normal serving.

## Verification boundary

CPU tests cover arithmetic, extreme/tied routing, clamp asymmetry, shape
errors, ownership, collective placement, buffer ABI and cleanup. Mock GPU
tests prove dispatch/ownership, not CUDA numerical correctness.
`scripts/dev/hy4_batch_test.cu` compares batched projection/expert/Engram
arithmetic with the single-row CUDA kernels, including strided rows, tails,
remote experts, empty groups and TP2 Q/KV joins. Full-model continuation
parity, quality and throughput require the real TP2/EP2 deployment; mock
tests cannot establish them.
