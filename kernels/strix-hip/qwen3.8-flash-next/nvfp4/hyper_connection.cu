// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next multi-hyperconnection (mHC) — the LOW-RANK mixer.
//
// Same four entry points and the same `[T, hc, H]` FP32 highway as
// DeepSeek-V4's `hyper_connection.cu`, and a DIFFERENT mixer. DeepSeek mixes
// with a Sinkhorn-normalized matrix over `hc_fn` / `hc_scale` / `hc_base`;
// Qwen mixes through a low-rank pair of rank `hc_lowrank` (320). The layouts
// coincide, the math does not — running DeepSeek's kernel against these
// weights produces fluent, confident, wrong output, which is why this file
// exists rather than a symlink.
//
// Transcribed from `Qwen4ExpTextGatedResidual.forward` (see
// `bench/qwen4_exp/ARCHITECTURE.md` §1):
//
//     normed = hc_norm(hyper_input)              # GROUPED RMSNorm, group=H
//     w = silu(down(normed) / hc)                # [hc*H] -> [R]
//     w = sigmoid(up(w))                         # [R] -> [hc*H]
//     mixed = (w.unflatten * normed.unflatten).mean(dim=-2)     # -> [H]
//     inj   = 2 * sigmoid(block_inject(normed) / hc)            # -> [hc]
//
// and the block output is injected back by `hc_post`:
//
//     residual[t, s*H + d] = hyper_input[t, s*H + d] + hidden[t, d] * inj[t, s]
//
// TWO THINGS THAT DO NOT FAIL LOUDLY IF GOT WRONG, both load-bearing:
//
//   1. `hc_norm` is GROUPED with `group_size = hidden_size`: the `hc` streams
//      normalize INDEPENDENTLY inside the `hc*H` vector. One RMS across all
//      `hc*H` is a different function that still produces plausible numbers.
//   2. The reduction over streams is a MEAN, not a sum. With hc = 4 a sum is
//      4x the intended magnitude — survivable-looking, and wrong.
//
// `normed` is recomputed on the fly from the per-stream RMS rather than
// staged: at hc*H = 10240 floats per token it would be 40 KB of shared (over
// budget) or ~84 MB of global traffic at T=2048. Only the `hc` reciprocals
// and the rank-R vector are kept resident.
//
// Grid: (T,1,1)   Block: (256,1,1)

#include <cuda_bf16.h>

#define QHC_BLOCK 256
#define QHC_MAX_MULT 8
#define QHC_MAX_RANK 512

__device__ __forceinline__ float qhc_silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float qhc_sigmoid(float v) {
    return 1.0f / (1.0f + __expf(-v));
}

// Per-stream RMS reciprocals for one token: rms_inv[s] over x[s*H .. s*H+H).
// Leaves the result in `smem_rms`, block-wide visible after __syncthreads().
__device__ __forceinline__ void qhc_stream_rms(
    const float* __restrict__ x,
    unsigned int H,
    unsigned int hc,
    float eps,
    float* __restrict__ smem_rms,   // [hc]
    float* __restrict__ smem_red    // [QHC_BLOCK / 32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = QHC_BLOCK / 32;

    for (unsigned int s = 0; s < hc; ++s) {
        const float* xs = x + (size_t)s * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += QHC_BLOCK) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w = 0; w < warps; ++w) tot += smem_red[w];
            smem_rms[s] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }
}

// ── hc_expand ──
// Broadcast a single hidden state into `hc` identical streams. Identical in
// behaviour to the DeepSeek twin; duplicated because a model shadow overrides
// a whole FILE, not individual entry points.
extern "C" __global__ void hc_expand(
    const __nv_bfloat16* __restrict__ hidden, // [T, H]
    float* __restrict__ streams,              // [T, hc, H] FP32 highway
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = hidden + (size_t)t * H;
    float* s = streams + (size_t)t * hc_mult * H;
    for (unsigned int d = tid; d < H; d += QHC_BLOCK) {
        float v = (float)x[d];
        for (unsigned int i = 0; i < hc_mult; ++i) s[i * H + d] = v;
    }
}

// Shared core for `hc_pre` and `hc_head`: both run the identical low-rank
// collapse; `hc_head` is the model-level mixer built with `use_combine=False`,
// so it simply has no `block_inject_weight` and emits no injection vector.
// Passing `inject_w == nullptr` selects that form.
//
// PERFORMANCE SHAPE (this core was the entire decode budget — 4.5 ms per
// call, x96 calls/token ~= 435 ms of a 455 ms token). Three rules:
//
//  1. The normed vector is staged ONCE in shared memory (hc*H floats = 40 KB
//     at 4x2560). The first cut recomputed `x * rms * (1 + w)` — three loads
//     and two multiplies — at every one of its ~6.6M uses.
//  2. The down projection runs one WARP per rank row: lanes stride the
//     10240-wide row (coalesced), then warp-reduce. The first cut gave each
//     THREAD a serial row: uncoalesced and 32x less parallel.
//  3. The up projection gives each THREAD one output element's rank-320 loop
//     per stream, reading `up_w` in its TRANSPOSED `[rank, hc*H]` layout so
//     that adjacent threads (adjacent `d`) read adjacent bf16 — coalesced by
//     construction. See `hc_pre_finish` below for why the layout, and not the
//     loop, is what had to change.
//
// The launcher passes block=1024 (32 warps). Grid stays [num_tokens]: at
// prefill that is thousands of independent blocks; at decode it is one block,
// which rule 2 finally keeps busy.
//
// The `1.0f +` in the norm is NOT optional — see the offset-from-1 note in
// the header. The parity probe (`hyper_connection_lowrank_tests.rs`) holds
// this core to the reference at every entry point.
#define QHC_WBLOCK 1024
#define QHC_SMEM_NORMED (QHC_MAX_MULT * 2560)

__device__ __forceinline__ void qhc_collapse(
    const float* __restrict__ streams,
    const float* __restrict__ hc_norm_w,
    const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ up_w,
    const __nv_bfloat16* __restrict__ inject_w,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ inj_out,
    unsigned int H,
    unsigned int hc,
    unsigned int rank,
    float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;
    const unsigned int hc_dim = hc * H;
    const float* x = streams + (size_t)t * hc_dim;

    extern __shared__ float smem[];
    float* smem_normed = smem;                 // [hc*H]
    float* smem_low = smem + hc_dim;           // [rank]
    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_WBLOCK / 32];

    // ── per-stream RMS ──
    for (unsigned int s2 = 0; s2 < hc; ++s2) {
        const float* xs = x + (size_t)s2 * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += blockDim.x) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w2 = 0; w2 < warps; ++w2) tot += smem_red[w2];
            smem_rms[s2] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }

    // ── stage normed = x * rms * (1 + w) once ──
    for (unsigned int i = tid; i < hc_dim; i += blockDim.x) {
        smem_normed[i] = x[i] * smem_rms[i / H] * (1.0f + hc_norm_w[i]);
    }
    __syncthreads();

    // ── down: warp per rank row, lanes stride the row ──
    const float inv_hc = 1.0f / (float)hc;
    for (unsigned int r = warp; r < rank; r += warps) {
        const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
        float acc = 0.0f;
        for (unsigned int i = lane; i < hc_dim; i += 32) {
            acc += (float)row[i] * smem_normed[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_low[r] = qhc_silu(acc * inv_hc);
    }
    __syncthreads();

    // ── up + gate + mean over streams: lane owns one output element ──
    __nv_bfloat16* y = y_out + (size_t)t * H;
    for (unsigned int d = tid; d < H; d += blockDim.x) {
        float mixed = 0.0f;
        for (unsigned int s2 = 0; s2 < hc; ++s2) {
            const unsigned int i = s2 * H + d;
            float acc = 0.0f;
            for (unsigned int r = 0; r < rank; ++r) {
                acc += (float)up_w[(size_t)r * hc_dim + i] * smem_low[r];
            }
            mixed += qhc_sigmoid(acc) * smem_normed[i];
        }
        y[d] = __float2bfloat16(mixed * inv_hc);
    }

    // ── injection weights: warp per stream ──
    if (inject_w != nullptr) {
        __syncthreads();
        for (unsigned int s2 = warp; s2 < hc; s2 += warps) {
            const __nv_bfloat16* row = inject_w + (size_t)s2 * hc_dim;
            float acc = 0.0f;
            for (unsigned int i = lane; i < hc_dim; i += 32) {
                acc += (float)row[i] * smem_normed[i];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            }
            if (lane == 0) {
                inj_out[(size_t)t * hc + s2] = 2.0f * qhc_sigmoid(acc * inv_hc);
            }
        }
    }
}

// ── hc_pre ──
// streams [T, hc, H] -> y_out [T, H] collapsed, inj_out [T, hc].
extern "C" __global__ void hc_pre(
    const float* __restrict__ streams,
    const float* __restrict__ hc_norm_w,  // [hc*H]
    const __nv_bfloat16* __restrict__ down_w,     // [rank, hc*H]
    const __nv_bfloat16* __restrict__ up_w,       // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,   // [hc, hc*H]
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ inj_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int rank,
    const float norm_eps
) {
    qhc_collapse(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out,
                 hidden_size, hc_mult, rank, norm_eps);
}

// ── hc_head ──
// The model-level `hyper_connection_mixer` (`use_combine=False`): the same
// collapse with no injection. This IS the model's final normalization — the
// checkpoint ships no `model.norm.weight` because `hc_norm` here plays that
// role.
extern "C" __global__ void hc_head(
    const float* __restrict__ streams,
    const float* __restrict__ hc_norm_w,
    const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ up_w,
    __nv_bfloat16* __restrict__ y_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int rank,
    const float norm_eps
) {
    qhc_collapse(streams, hc_norm_w, down_w, up_w, nullptr, y_out, nullptr,
                 hidden_size, hc_mult, rank, norm_eps);
}

// ── hc_post ──
// residual[t, s*H + d] = hyper_input[t, s*H + d] + block_out[t, d] * inj[t, s]
//
// `hyper_input` is the PRE-NORM highway, not the normalized one — the
// reference keeps the raw residual and adds to it.
extern "C" __global__ void hc_post(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* __restrict__ residual,          // [T, hc, H]
    const float* __restrict__ inj,               // [T, hc]
    float* __restrict__ out,                     // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* w = inj + (size_t)t * hc;
    float* o = out + (size_t)t * hc * H;

    float wv[QHC_MAX_MULT];
    for (unsigned int s = 0; s < hc; ++s) wv[s] = w[s];

    for (unsigned int d = tid; d < H; d += QHC_BLOCK) {
        float xd = (float)x[d];
        for (unsigned int s = 0; s < hc; ++s) {
            o[s * H + d] = res[s * H + d] + xd * wv[s];
        }
    }
}

// ── Split collapse, for SMALL T (decode) ─────────────────────────────────
// grid=[1] starves the fused kernel at decode: one block, one SM, ~13 MB of
// weights per call (measured 2.0 ms). These three launches spread the same
// math across the whole GPU; the Rust dispatcher picks them when
// `num_tokens` is small and keeps the fused kernel for prefill.

// Stage 1: normed = x * rms * (1 + w) -> global scratch [T, hc*H].
extern "C" __global__ void hc_pre_stage(
    const float* __restrict__ streams,
    const float* __restrict__ hc_norm_w,
    float* __restrict__ normed_out,            // [T, hc*H]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* x = streams + (size_t)t * hc_dim;
    float* out = normed_out + (size_t)t * hc_dim;

    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_WBLOCK / 32];
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;

    for (unsigned int s2 = 0; s2 < hc; ++s2) {
        const float* xs = x + (size_t)s2 * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += blockDim.x) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w2 = 0; w2 < warps; ++w2) tot += smem_red[w2];
            smem_rms[s2] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }
    for (unsigned int i = tid; i < hc_dim; i += blockDim.x) {
        out[i] = x[i] * smem_rms[i / H] * (1.0f + hc_norm_w[i]);
    }
}

// Stage 2: low[r] = silu(down[r] . normed / hc), rank rows split over
// blockIdx.y. Warp per row, coalesced lane strides.
//
// STRIX-LOCAL COPY — this file is a real file under strix-hip, not the
// gb10 symlink, because of THIS kernel. On gfx1151 (2026-09-20 GPU-timed
// winbox trace, serial decode) hc_pre_down ran 150 us x 97 launches =
// ~14.6 ms of an ~89 ms GPU step: all 320 rank rows re-read the same
// `normed[t]` vector (40 KB FP32) from L2/DRAM alongside each row's 20 KB
// of `down_w` — ~13 MB of `normed` re-reads vs 6.5 MB of weights per
// launch, on a 10-block grid that pays full latency per re-read. Staging
// `normed[t]` into LDS once per block removes the re-reads; RDNA3.5's LDS
// latency profile is why this pays here and is not assumed to pay on
// gb10, which keeps its own copy.
//
// BIT-EXACT BY CONSTRUCTION: the per-lane FMA sequence
// (`acc += (float)row[i] * nx[i]`, i = lane, lane+32, …) and the shfl
// reduction are unchanged — `nx` is only re-sourced from shared memory.
// The call site passes hc_dim*4 bytes of DYNAMIC shared (40 KB at
// hc=4, H=2560): static `__shared__ smem_nx[10240]` plus the dynamic
// request would double-count against the 64 KB LDS cap — that wedged
// this exact launch on the main-merged call site (decode hang, 2026-09-26).
extern "C" __global__ void hc_pre_down(
    const float* __restrict__ normed,          // [T, hc*H]
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    float* __restrict__ low_out,               // [T, rank]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank,
    const unsigned int num_tokens
) {
    const unsigned int t = blockIdx.x;
    if (t >= num_tokens) return;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = blockDim.x >> 5;
    const unsigned int hc_dim = hc * hidden_size;
    const float inv_hc = 1.0f / (float)hc;

    // DYNAMIC shared (the call site passes hc_dim*4 bytes, 40 KB): the LDS
    // staging is the strix-local part — static smem_nx PLUS a dynamic
    // request double-counted against the 64 KB LDS cap and wedged the
    // decode launch. Same staging logic as before, dynamic storage.
    extern __shared__ float smem_nx[];
    for (unsigned int i = threadIdx.x; i < hc_dim; i += blockDim.x) {
        smem_nx[i] = normed[(size_t)t * hc_dim + i];
    }
    __syncthreads();

    // Rows split first across grid.y, then across warps in the block.
    const unsigned int rows_per_split = (rank + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_split;
    const unsigned int r1 = min(r0 + rows_per_split, rank);
    for (unsigned int r = r0 + warp; r < r1; r += warps) {
        const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
        float acc = 0.0f;
        for (unsigned int i = lane; i < hc_dim; i += 32) {
            acc += (float)row[i] * smem_nx[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) low_out[(size_t)t * rank + r] = qhc_silu(acc * inv_hc);
    }
}

// Stage 3: y[d] = mean_s sigmoid(up[s*H+d] . low) * normed[s*H+d], the
// d-range split over blockIdx.y; block y==0 also emits the injection vector.
//
// WHY `up_w` IS STORED TRANSPOSED. One thread owns one output dim `d` and
// contracts over `rank` sequentially. In the checkpoint's `[hc*H, rank]`
// layout that thread walks a contiguous rank-320 row, so consecutive threads
// touch rows 640 B apart and every lane of a warp lands on its own sector:
// nsys measured this kernel at a flat ~173 us regardless of T, ~38 GB/s
// against the part's ~273, and 23% of ALL decode GPU time (11.86 s of
// 51.47 s) — the largest single kernel in the profile.
//
// Two kernel-side fixes were built and measured before the layout one:
//
//   * warp-per-output-dim with lane-strided `r` + a shfl reduction: +17.8%
//     end-to-end, but it REASSOCIATES the FP32 contraction, so the logits
//     move. On a speculative-decoding model that is not a free trade — and
//     on this checkpoint decode-path reassociation was measured flipping
//     tool-calling behaviour on BFCL, not just the last mantissa bits.
//   * staging `up_w` tiles through shared memory, which keeps each thread's
//     sequential `r` accumulation and so IS bit-exact: 44% SLOWER (9.06 vs
//     16.10 tok/s). Re-staging the whole rank in one pass (8 syncs instead of
//     40) measured 9.06 vs 9.40, so the barriers were not the cost — a useful
//     tile is ~42 KB, which caps occupancy near one block per SM.
//
// AND COALESCING ALONE WAS NOT ENOUGH — measured, 19.79 vs 20.01 tok/s, i.e.
// nothing. This kernel was never bandwidth-limited. Thread-per-`d` caps it at
// H = 2560 threads, which at block 256 is TEN blocks: ten of the part's 48 SMs
// participate, ~2 warps each, every thread walking one strictly dependent
// 320-step FP32 chain. There are nowhere near enough loads in flight to cover
// DRAM latency, so a perfectly coalesced access pattern buys nothing on its
// own. The launcher therefore also shrinks the block (more blocks over the
// same 2560 threads => more SMs), and the `d` loop below interleaves the `hc`
// streams so each thread carries `hc` independent chains. Neither touches the
// summation order.
//
// Storing `up_w` as `[rank, hc*H]` instead gets both properties for free:
// thread `d` reads `up_w[r*hc_dim + i]`, consecutive threads read consecutive
// bf16, and the per-thread `for r` order is IDENTICAL to the row-major
// version's — so the output is bitwise unchanged. The transpose happens once
// at load (`weight_loader::qwen4_exp::hc`); the prefill GEMM, whose NT tensor-
// core kernel wants the checkpoint layout, transposes a staging copy back.
extern "C" __global__ void hc_pre_finish(
    const float* __restrict__ normed,          // [T, hc*H]
    const float* __restrict__ low,             // [T, rank]
    const __nv_bfloat16* __restrict__ up_w,    // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    float* __restrict__ inj_out,               // [T, hc] (unused if null inject)
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* nx = normed + (size_t)t * hc_dim;
    const float inv_hc = 1.0f / (float)hc;

    extern __shared__ float smem_lo[];         // [rank]
    for (unsigned int r = tid; r < rank; r += blockDim.x) {
        smem_lo[r] = low[(size_t)t * rank + r];
    }
    __syncthreads();

    const unsigned int d_per_split = (H + gridDim.y - 1) / gridDim.y;
    const unsigned int d0 = blockIdx.y * d_per_split;
    const unsigned int d1 = min(d0 + d_per_split, H);
    __nv_bfloat16* y = y_out + (size_t)t * H;
    for (unsigned int d = d0 + tid; d < d1; d += blockDim.x) {
        // The `hc` streams are INTERLEAVED rather than run one after another:
        // each keeps its own accumulator and they advance together over `r`.
        // Every accumulator still sums r = 0,1,...,rank-1 into one FP32
        // register in that exact order, and `mixed` still folds the streams in
        // s = 0,1,2,3 order, so the result is bit-for-bit what the sequential
        // version produced — but the thread now has `hc` independent load+FMA
        // chains in flight instead of one, which is what a latency-bound
        // kernel is short of.
        //
        // SPELLED OUT for hc == 4 (this checkpoint's only value) instead of
        // looping an `acc[]` array: `hc` is a runtime argument, so a
        // `for (s2 < hc)` loop over an array cannot be unrolled, the indices
        // stay dynamic, and nvcc puts the accumulators in LOCAL memory —
        // which would cost far more than the interleaving wins. The generic
        // fallback keeps the original one-chain-at-a-time shape.
        float mixed = 0.0f;
        if (hc == 4) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            // LOADS IN FLIGHT. nsys (2026-09-06, C=1, 3.9K ctx): this kernel
            // was 121 us x 97 launches = 11.7 ms of a 69 ms token, ~54 GB/s
            // on 6.5 MB of `up_w` — latency-bound, as the note above says.
            // Each `r` step below is one dependent load+FMA per stream and
            // nvcc does not pipeline a runtime-bound loop, so a warp sat on
            // one DRAM round trip per `r`. Unrolling `r` by 8 with the 32
            // loads hoisted ahead of the 32 accumulations puts eight `r`
            // steps' worth of loads in flight per stream. The accumulation
            // itself is UNCHANGED: every `a_s` still sums r = 0,1,2,... in
            // that exact order into one FP32 register, so the result is
            // bit-for-bit the same as the one-`r`-at-a-time loop — this is
            // pure scheduling, not reassociation (which the note above
            // measured moving logits and was rejected).
            const __nv_bfloat16* ub = up_w + d;
            unsigned int r = 0;
            for (; r + 8 <= rank; r += 8) {
                float l[8];
                __nv_bfloat16 u0[8], u1[8], u2[8], u3[8];
                #pragma unroll
                for (unsigned int k = 0; k < 8; ++k) {
                    const __nv_bfloat16* u = ub + (size_t)(r + k) * hc_dim;
                    l[k] = smem_lo[r + k];
                    u0[k] = u[0];
                    u1[k] = u[H];
                    u2[k] = u[2 * H];
                    u3[k] = u[3 * H];
                }
                #pragma unroll
                for (unsigned int k = 0; k < 8; ++k) {
                    a0 += (float)u0[k] * l[k];
                    a1 += (float)u1[k] * l[k];
                    a2 += (float)u2[k] * l[k];
                    a3 += (float)u3[k] * l[k];
                }
            }
            for (; r < rank; ++r) {
                const float lo = smem_lo[r];
                const __nv_bfloat16* u = ub + (size_t)r * hc_dim;
                a0 += (float)u[0] * lo;
                a1 += (float)u[H] * lo;
                a2 += (float)u[2 * H] * lo;
                a3 += (float)u[3 * H] * lo;
            }
            mixed += qhc_sigmoid(a0) * nx[d];
            mixed += qhc_sigmoid(a1) * nx[H + d];
            mixed += qhc_sigmoid(a2) * nx[2 * H + d];
            mixed += qhc_sigmoid(a3) * nx[3 * H + d];
        } else {
            for (unsigned int s2 = 0; s2 < hc; ++s2) {
                const unsigned int i = s2 * H + d;
                float acc = 0.0f;
                for (unsigned int r = 0; r < rank; ++r) {
                    acc += (float)up_w[(size_t)r * hc_dim + i] * smem_lo[r];
                }
                mixed += qhc_sigmoid(acc) * nx[i];
            }
        }
        y[d] = __float2bfloat16(mixed * inv_hc);
    }

    if (inject_w != nullptr && blockIdx.y == 0) {
        const unsigned int lane = tid & 31u;
        const unsigned int warp = tid >> 5;
        const unsigned int warps = blockDim.x >> 5;
        for (unsigned int s2 = warp; s2 < hc; s2 += warps) {
            const __nv_bfloat16* row = inject_w + (size_t)s2 * hc_dim;
            float acc = 0.0f;
            for (unsigned int i = lane; i < hc_dim; i += 32) {
                acc += (float)row[i] * nx[i];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            }
            if (lane == 0) {
                inj_out[(size_t)t * hc + s2] = 2.0f * qhc_sigmoid(acc * inv_hc);
            }
        }
    }
}

// Stage 3, FOUR-STREAM LAYOUT: one warp per stream, lanes over 32 consecutive
// output dims. `hc_pre_finish` gives every thread all `hc` streams of one `d`,
// which caps the kernel at H = 2560 threads per token: 20 blocks of 128 on a
// 48-SM part, each thread walking four dependent rank-320 chains. nsys
// (2026-09-06) measured it at 121 us x 97 launches = 11.7 ms of a 69 ms
// token. This variant puts the same work on 4x the threads: warp `s` of a
// block owns stream `s` for 32 dims, so a token spans 80 blocks x 4 warps.
//
// BIT-EXACT BY CONSTRUCTION. Each (d, s) accumulator still sums
// r = 0,1,...,rank-1 into one FP32 register in that order (--fmad=false, so
// multiply then add, exactly as before); the per-stream products
// sigmoid(a_s) * normed[s*H+d] are the same floats; and `mixed` still folds
// them in s = 0,1,2,3 order in one thread. Nothing is reassociated — the
// warp-per-dim shuffle rewrite this file's note rejected split ONE
// accumulator across lanes; this splits the four INDEPENDENT accumulators
// across warps.
extern "C" __global__ void hc_pre_finish_x4(
    const float* __restrict__ normed,          // [T, hc*H]
    const float* __restrict__ low,             // [T, rank]
    const __nv_bfloat16* __restrict__ up_w,    // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    float* __restrict__ inj_out,               // [T, hc]
    const unsigned int hidden_size,
    const unsigned int hc,                     // must be 4 (host checks)
    const unsigned int rank
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;        // == stream s
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* nx = normed + (size_t)t * hc_dim;
    const float inv_hc = 1.0f / (float)hc;

    extern __shared__ float smem_lo[];         // [rank]
    __shared__ float part[4][32];
    for (unsigned int r = tid; r < rank; r += blockDim.x) {
        smem_lo[r] = low[(size_t)t * rank + r];
    }
    __syncthreads();

    const unsigned int d = blockIdx.y * 32u + lane;
    if (d < H) {
        const unsigned int i = warp * H + d;
        const __nv_bfloat16* ub = up_w + i;
        float acc = 0.0f;
        unsigned int r = 0;
        for (; r + 8 <= rank; r += 8) {
            float l[8];
            __nv_bfloat16 u[8];
            #pragma unroll
            for (unsigned int k = 0; k < 8; ++k) {
                l[k] = smem_lo[r + k];
                u[k] = ub[(size_t)(r + k) * hc_dim];
            }
            #pragma unroll
            for (unsigned int k = 0; k < 8; ++k) {
                acc += (float)u[k] * l[k];
            }
        }
        for (; r < rank; ++r) {
            acc += (float)ub[(size_t)r * hc_dim] * smem_lo[r];
        }
        part[warp][lane] = qhc_sigmoid(acc) * nx[i];
    }
    __syncthreads();
    if (warp == 0 && d < H) {
        float mixed = 0.0f;
        mixed += part[0][lane];
        mixed += part[1][lane];
        mixed += part[2][lane];
        mixed += part[3][lane];
        y_out[(size_t)t * H + d] = __float2bfloat16(mixed * inv_hc);
    }

    // Injection vector: same warp-per-stream contraction as `hc_pre_finish`
    // (there `s2 = warp; s2 < hc; s2 += warps` with 4 warps is this mapping).
    if (inject_w != nullptr && blockIdx.y == 0) {
        const __nv_bfloat16* row = inject_w + (size_t)warp * hc_dim;
        float acc = 0.0f;
        for (unsigned int j = lane; j < hc_dim; j += 32) {
            acc += (float)row[j] * nx[j];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) {
            inj_out[(size_t)t * hc + warp] = 2.0f * qhc_sigmoid(acc * inv_hc);
        }
    }
}

// ───────────────────────── GEMM-path collapse (large T) ─────────────────────
//
// PERFORMANCE SHAPE: at prefill the fused kernel measured ~45 ms per call —
// 47% of the whole prefill (two calls per layer x 48 layers). Its down/up
// projections are GEMM-shaped ([T,hc*H]x[hc*H,rank] and back), but ran as
// hand-rolled FP32 warp loops at ~4% of the machine. For T > 64 the collapse
// instead stages `normed` in BF16 and hands both projections to
// `dense_gemm_bf16_pipelined` (tensor cores), keeping only the cheap
// elementwise seams as custom kernels:
//
//   hc_pre_stage_bf16   grid=[T]    rms + (1+w) scale -> normed  [T, hc*H] BF16
//   dense_gemm          low_pre  = normed x down_w^T             [T, rank]
//   hc_silu_scale       low      = silu(low_pre / hc)            in place
//   hc_transpose_bf16   up_wt    = up_w^T   [hc*H, rank]  (staging copy;
//                                 up_w is stored [rank, hc*H] for decode)
//   dense_gemm          up_pre   = low x up_wt^T                 [T, hc*H]
//   dense_gemm          inj_pre  = normed x inject_w^T           [T, hc]
//   hc_pre_mix          grid=[T]    y = mean_s sigmoid(up_pre)*normed;
//                                   inj = 2*sigmoid(inj_pre / hc)
//
// Numerics: normed is rounded to BF16 before the GEMMs (the fused kernel kept
// it FP32 in smem). The checkpoint's hyper-connection weights are BF16 and the
// reference module computes in BF16, so this is parity-gated the same way as
// every other collapse variant (probe cosine vs the FP32 fused path).

extern "C" __global__ void hc_pre_stage_bf16(
    const float* __restrict__ streams,
    const float* __restrict__ hc_norm_w,
    __nv_bfloat16* __restrict__ normed_out,    // [T, hc*H] BF16
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* x = streams + (size_t)t * hc_dim;
    __nv_bfloat16* out = normed_out + (size_t)t * hc_dim;

    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_WBLOCK / 32];
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;

    for (unsigned int s2 = 0; s2 < hc; ++s2) {
        const float* xs = x + (size_t)s2 * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += blockDim.x) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w2 = 0; w2 < warps; ++w2) tot += smem_red[w2];
            smem_rms[s2] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }
    for (unsigned int i = tid; i < hc_dim; i += blockDim.x) {
        out[i] = __float2bfloat16(
            x[i] * smem_rms[i / H] * (1.0f + hc_norm_w[i]));
    }
}

// low = silu(low_pre * inv_hc), elementwise in place over n = T*rank.
extern "C" __global__ void hc_silu_scale(
    __nv_bfloat16* __restrict__ low,
    const unsigned int n,
    const float inv_hc
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = (float)low[i] * inv_hc;
        low[i] = __float2bfloat16(qhc_silu(v));
    }
}

// y[d] = mean_s sigmoid(up_pre[s*H+d]) * normed[s*H+d];
// inj[s] = 2*sigmoid(inj_pre[s] * inv_hc) (skipped when inj_pre is null).
extern "C" __global__ void hc_pre_mix(
    const __nv_bfloat16* __restrict__ normed,  // [T, hc*H]
    const __nv_bfloat16* __restrict__ up_pre,  // [T, hc*H]
    const __nv_bfloat16* __restrict__ inj_pre, // [T, hc] or null
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    float* __restrict__ inj_out,               // [T, hc]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float inv_hc
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const __nv_bfloat16* nx = normed + (size_t)t * hc_dim;
    const __nv_bfloat16* ux = up_pre + (size_t)t * hc_dim;
    __nv_bfloat16* y = y_out + (size_t)t * H;

    for (unsigned int d = tid; d < H; d += blockDim.x) {
        float mixed = 0.0f;
        for (unsigned int s2 = 0; s2 < hc; ++s2) {
            const unsigned int i = s2 * H + d;
            mixed += qhc_sigmoid((float)ux[i]) * (float)nx[i];
        }
        y[d] = __float2bfloat16(mixed * inv_hc);
    }
    if (inj_pre != nullptr && tid < hc) {
        inj_out[(size_t)t * hc + tid] =
            2.0f * qhc_sigmoid((float)inj_pre[(size_t)t * hc + tid] * inv_hc);
    }
}

// Transpose a BF16 matrix `[R, C] -> [C, R]`. Exists for one caller: the
// prefill GEMM path needs `up_w` in the checkpoint's `[hc*H, rank]` layout
// (its tensor-core kernel is NT, `C[m,n] = A[m,k] . B[n,k]`), while every
// decode kernel needs the transposed `[rank, hc*H]` that `hc_pre_finish`
// documents. Storing both would cost 6.55 MB x 97 sites = 635 MB on a box
// that already loads at 113 of 119.6 GB, so prefill pays ~50 us per call to
// stage a transposed copy instead — against a ~45 ms collapse, and only for
// T > 64.
//
// Classic 32x32 tile with a padded stride: both the read and the write are
// coalesced, and the +1 keeps the smem access bank-conflict free.
extern "C" __global__ void hc_transpose_bf16(
    const __nv_bfloat16* __restrict__ src,     // [R, C]
    __nv_bfloat16* __restrict__ dst,           // [C, R]
    const unsigned int R,
    const unsigned int C
) {
    __shared__ __nv_bfloat16 tile[32][33];
    const unsigned int x = blockIdx.x * 32u + threadIdx.x;   // col of src
    const unsigned int y = blockIdx.y * 32u + threadIdx.y;   // row of src
    if (x < C && y < R) {
        tile[threadIdx.y][threadIdx.x] = src[(size_t)y * C + x];
    }
    __syncthreads();
    // Swap which of the two block indices supplies the fast axis, so the
    // store is contiguous in `dst` too.
    const unsigned int xt = blockIdx.y * 32u + threadIdx.x;  // col of dst
    const unsigned int yt = blockIdx.x * 32u + threadIdx.y;  // row of dst
    if (xt < R && yt < C) {
        dst[(size_t)yt * R + xt] = tile[threadIdx.x][threadIdx.y];
    }
}

// Prefill-shaped sibling of `hc_pre_down`, tiled over BOTH tokens and hc_dim.
//
// `hc_pre_down` stages the whole `normed` row (hc_dim floats, 40 KB) in shared
// and makes ONE pass. That is right at T=1: a decode call needs a single pass
// and pays no barriers. It is wrong at prefill widths, where it re-reads the
// 6.55 MB `down_w` once per token -- 393 MB at T=60, measured 859 GB/s, ~85% of
// L2 peak, i.e. L2-bandwidth-bound.
//
// This tiles tokens (HC_TT per block) so each weight row is amortised, and
// chunks hc_dim (HC_CH) so the staged `nx` footprint stays HC_TT x HC_CH x 4 B
// instead of HC_TT x 40 KB. Tiling tokens ALONE was measured and gave only -6%:
// TT x 40 KB overflows L1, so `nx` starts missing to L2 and cancels the weight
// saving. Both dimensions have to move together.
//
// Measured on an 87-token prefill (nsys): prefill `hc_pre_down` time
// 89.2 ms -> 33.0 ms, and the whole prefill window 491.5 -> 438.3 ms.
// At T=1 it is 2.3x SLOWER than the single-pass version (28.8 -> 65.0 ms over
// 679 decode calls), because hc_dim=10240 becomes 20 chunks = 40 barriers for
// work that needs one pass. Hence two kernels and a dispatch on T, not one.
//
// BITWISE IDENTICAL to `hc_pre_down`: for every (t, r) a lane still walks
// i = lane, lane+32, ... in increasing order followed by the same shfl
// reduction. Chunks are contiguous, processed in order, and HC_CH is a multiple
// of 32, so chunking cannot reorder a lane's walk. Only the order in which
// independent (t, r) pairs are visited changed.
#ifndef HC_TT
#define HC_TT 8u
#endif
#ifndef HC_CH
#define HC_CH 512u
#endif

extern "C" __global__ void hc_pre_down_tiled(
    const float* __restrict__ normed,          // [T, hc*H]
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    float* __restrict__ low_out,               // [T, rank]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank,
    const unsigned int num_tokens
) {
    __shared__ float s_nx[HC_TT][HC_CH];

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = blockDim.x >> 5;
    const unsigned int hc_dim = hc * hidden_size;
    const float inv_hc = 1.0f / (float)hc;

    const unsigned int t0 = blockIdx.x * HC_TT;
    if (t0 >= num_tokens) return;
    const unsigned int tn = min(HC_TT, num_tokens - t0);

    const unsigned int rows_per_split = (rank + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_split;
    const unsigned int r1 = min(r0 + rows_per_split, rank);

    for (unsigned int rbase = r0; rbase < r1; rbase += warps) {
        const unsigned int r = rbase + warp;
        float acc[HC_TT];
        #pragma unroll
        for (unsigned int t = 0; t < HC_TT; ++t) acc[t] = 0.0f;

        for (unsigned int c0 = 0; c0 < hc_dim; c0 += HC_CH) {
            const unsigned int cn = min(HC_CH, hc_dim - c0);
            for (unsigned int idx = threadIdx.x; idx < tn * cn; idx += blockDim.x) {
                const unsigned int t = idx / cn;
                const unsigned int i = idx - t * cn;
                s_nx[t][i] = normed[(size_t)(t0 + t) * hc_dim + c0 + i];
            }
            __syncthreads();
            if (r < r1) {
                const __nv_bfloat16* row = down_w + (size_t)r * hc_dim + c0;
                for (unsigned int i = lane; i < cn; i += 32u) {
                    const float w = (float)row[i];
                    #pragma unroll
                    for (unsigned int t = 0; t < HC_TT; ++t) {
                        if (t < tn) acc[t] += w * s_nx[t][i];
                    }
                }
            }
            __syncthreads();
        }

        if (r < r1) {
            #pragma unroll
            for (unsigned int t = 0; t < HC_TT; ++t) {
                if (t >= tn) continue;
                float a = acc[t];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    a += __shfl_down_sync(0xFFFFFFFFu, a, off);
                }
                if (lane == 0) {
                    low_out[(size_t)(t0 + t) * rank + r] = qhc_silu(a * inv_hc);
                }
            }
        }
    }
}


// ═══ Ported verbatim from kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu lines 1460-end (2026-10-29): the vectorized decode collapse
// (ATLAS_QWEN4EXP_HC_FAST). This strix-local copy lacked it, so the env silently fell back to
// the scalar kernels on gfx1151 (f84: HC = 26.5% of Flash-Next decode). HIP edits: the PTX ld.global.nc
// loaders are plain vector loads (same bits), and the cluster-fused hc_post+stage kernel is compiled out
// (gfx1151 has no thread-block clusters; the host falls back to hc_post_vec + hc_pre_stage_vec).
#include "atlas_pdl.cuh"
// ── Vectorized decode collapse (ATLAS_QWEN4EXP_HC_FAST, T <= HC_V_MAX) ────
//
// nsys, TP2 C1 decode, 96 mHC sites per token (2026-10-05):
//
//     hc_pre_finish_x4  61.8 us   hc_pre_down  45.4 us
//     hc_pre_stage       9.4 us   hc_post       6.3 us    = 11.8 of 43.8 ms
//
// Every site streams 13.1 MB of BF16 from DRAM (down_w and up_w, 6.55 MB
// each, plus 80 KB of inject_w) — ~52 us at the part's ~250 GB/s against the
// 107 us down + finish take. Neither kernel is short of threads in total; both
// are short of BYTES IN FLIGHT:
//
//   * every lane loads 2 bytes per instruction (one bf16 of a lane-strided
//     walk), so a warp request is 64 B and a lane can only keep a handful of
//     rows' worth of loads outstanding;
//   * `hc_pre_down` at T=1 runs on grid (1, 10) of 1024 threads with 40 KB of
//     shared each — ten SMs of 48 — and every block first stages the 40 KB
//     `normed` row behind a barrier;
//   * `hc_pre_finish_x4` block 0 runs the four 10240-long injection dot
//     products (2-byte loads, one warp per stream) AFTER its own share of the
//     up projection: a serial tail on the critical path of every launch.
//
// Bytes in flight per launch are (k-steps unrolled ahead) x (one k-step of
// every row = 20 KB), whatever the vector width — so the unroll depth, not
// the vector width, is the lever, and the width only buys registers back.
//
// THE CONSTRAINT IS THE ACCUMULATION ORDER, NOT THE LOAD SHAPE. Every output
// below is the same sequence of IEEE operations as the kernel it replaces
// (this target builds with --fmad=false, so `acc += w * x` is a rounded
// multiply then a rounded add, never an FMA):
//
//   down / inject  per (t, row): 32 lane chains, chain c summing
//                  w[32k + c] * nx[32k + c] for k = 0, 1, ... in order, then
//                  the shfl_down 16/8/4/2/1 tree. Here a thread owns CPT
//                  consecutive chains c = CPT*p .. CPT*p + CPT-1 (one vector
//                  load per k covers them), 32/CPT lanes cover a row, and the
//                  tree is replayed exactly: a step of offset >= CPT is a
//                  shuffle from lane p + off/CPT (same register j), a step of
//                  offset < CPT pairs registers j and j + off inside lane
//                  p = 0. Same operands, same pairing, same order — fadd is
//                  commutative, so which lane holds the left operand is
//                  immaterial.
//   finish         per (t, s, d): sum over r = 0..rank-1 of up[r][s*H+d] *
//                  low[r] in order, then sigmoid(acc) * normed, folded over
//                  s = 0..3 into `mixed = 0.0f` in that order. Here a thread
//                  owns DPT consecutive d of one stream (one vector load per
//                  r), the four streams of a d sit in adjacent lanes, and
//                  lane s = 0 folds them in s order after three shuffles.
//   stage          the 1024-thread RMS of `hc_pre_stage`, unchanged per
//                  thread (d = tid, tid+1024, ...) but with the hc streams
//                  walked in one pass and one barrier, replicated in every
//                  block of a (T, S) grid so S blocks each write 1/S of
//                  `normed` instead of one block writing all of it.
//   post           elementwise; four consecutive d per thread.
//
// The injection rows are folded into the down launch as rows rank..rank+hc-1
// (they ARE the down contraction against a different matrix, with a
// different epilogue), which removes the finish kernel's serial tail.
//
// `scripts/dev/qwen4exp_hc_decode_bench.cu` compares every output byte of
// these against the kernels they replace, sweeps CPT / DPT / unroll / block,
// and times both; the HC_V_* defaults below are its pick. Measured there
// (ennspark03 GB10, weights cycled through 8 copies so every launch streams
// from DRAM; us per launch incl. the ~2.2 us back-to-back launch gap):
//
//                T=1: old -> vec      T=4: old -> vec      T=8: old -> vec8
//     stage       8.3 ->  6.0          8.4 ->  6.1          8.9 ->  8.3
//     down       40.7 -> 32.3         40.2 -> 34.4         52.7 -> 41.7
//     finish     60.6 -> 33.3         60.3 -> 38.9         58.9 -> 50.8
//     post        8.1 ->  4.3          8.2 ->  4.2          8.3 ->  4.2
//     site      121.4 -> 75.5        119.6 -> 83.1        131.4 -> 105.2
//
// against 28.5 us for a bare 6.55 MB streaming read in the same harness.
// (T = 5..8 is the multi-sequence batch, the `_vec8` twins below: at 8
// tokens the finish thread's 32 chains make each ring step long enough that
// its 16-row lookahead no longer covers DRAM latency; it stays the slower
// half.)

#ifndef HC_V_MAX
#define HC_V_MAX 8u
#endif
#ifndef HC_V_DOWN_CPT
#define HC_V_DOWN_CPT 2u
#endif
#ifndef HC_V_DOWN_UNROLL
#define HC_V_DOWN_UNROLL 32u
#endif
#ifndef HC_V_FIN_DPT
#define HC_V_FIN_DPT 4u
#endif
#ifndef HC_V_FIN_UNROLL
#define HC_V_FIN_UNROLL 32u
#endif

// N streamed bf16 (N = 2, 4, 8) as N/2 packed words: read-only path, no L1
// allocation (each weight byte is used once per launch), 256 B L2 prefetch.
template <unsigned int N>
__device__ __forceinline__ void qhc_ld_bf(const __nv_bfloat16* p, unsigned int (&w)[N / 2]);
template <>
__device__ __forceinline__ void qhc_ld_bf<2>(const __nv_bfloat16* p, unsigned int (&w)[1]) {
    w[0] = *reinterpret_cast<const unsigned int*>(p);  // HIP: plain load (same bits as the PTX ld.global.nc)
}
template <>
__device__ __forceinline__ void qhc_ld_bf<4>(const __nv_bfloat16* p, unsigned int (&w)[2]) {
    const uint2 v = *reinterpret_cast<const uint2*>(p);  // HIP: plain vector load
    w[0] = v.x; w[1] = v.y;
}
template <>
__device__ __forceinline__ void qhc_ld_bf<8>(const __nv_bfloat16* p, unsigned int (&w)[4]) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);  // HIP: plain vector load
    w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
}

// One bf16 lane of a packed word -> float, through the same
// `(float)__nv_bfloat16` conversion the scalar kernels use. Element 0 of a
// packed pair is the low half.
__device__ __forceinline__ float qhc_bf_lo(unsigned int w) {
    return (float)__ushort_as_bfloat16((unsigned short)(w & 0xFFFFu));
}
__device__ __forceinline__ float qhc_bf_hi(unsigned int w) {
    return (float)__ushort_as_bfloat16((unsigned short)(w >> 16));
}

template <unsigned int N>
__device__ __forceinline__ void qhc_unpack(const unsigned int (&w)[N / 2], float (&f)[N]) {
    #pragma unroll
    for (unsigned int k = 0; k < N / 2; ++k) {
        f[2 * k] = qhc_bf_lo(w[k]);
        f[2 * k + 1] = qhc_bf_hi(w[k]);
    }
}

// N consecutive floats (N = 2, 4, 8); 8-byte aligned for N = 2, else 16.
template <unsigned int N>
__device__ __forceinline__ void qhc_ld_f(const float* p, float (&f)[N]) {
    if constexpr (N == 2) {
        const float2 v = *reinterpret_cast<const float2*>(p);
        f[0] = v.x; f[1] = v.y;
    } else {
        #pragma unroll
        for (unsigned int k = 0; k < N / 4; ++k) {
            const float4 v = reinterpret_cast<const float4*>(p)[k];
            f[4 * k] = v.x; f[4 * k + 1] = v.y; f[4 * k + 2] = v.z; f[4 * k + 3] = v.w;
        }
    }
}

// Down + injection rows. Thread g: row g / (32/CPT), chains CPT*p ..
// CPT*p + CPT-1 with p = g % (32/CPT).
//
// The weight loads run as a ROLLING ring of U k-steps: consuming step k
// immediately issues step k + U into the same registers, so U steps stay in
// flight for the whole walk rather than draining to zero between load-U /
// use-U batches (~25% against DRAM at T = 1). `normed` rides a second,
// shorter ring DN steps ahead: at T > 1 the tokens' 40 KB rows no longer fit
// L1 together and every step would otherwise wait on an L2 hit (T = 4 went
// 41.8 -> 34.5 us with a 16-step ring, against 32.5 at T = 1).
//
// Every ring load is UNCONDITIONAL — the last U steps are peeled rather than
// predicated — because ptxas counts outstanding loads per scoreboard and a
// runtime-predicated load makes it wait for zero, which serializes the ring
// (measured 4x slower). Host checks (hc_dim / 32) % U == 0.
template <unsigned int CPT, unsigned int U, unsigned int DN, unsigned int NT>
__device__ __forceinline__ void qhc_down_vec(
    const float* __restrict__ normed,
    const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ inject_w,
    float* __restrict__ low_out,
    float* __restrict__ inj_out,
    const unsigned int hc_dim,
    const unsigned int hc,
    const unsigned int rank,
    // Tokens actually written: `_wide` runs a padded NT over scratch rows
    // past the batch and drops their outputs. NT everywhere else.
    const unsigned int nt_live = NT
) {
    static_assert(DN >= 1 && DN <= U && U % DN == 0, "normed ring must tile the weight ring");
    constexpr unsigned int LPR = 32u / CPT;    // lanes per row
    const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = g / LPR;
    const unsigned int p = g % LPR;
    const unsigned int rows = rank + (inject_w != nullptr ? hc : 0u);
    const bool live = row < rows;
    const float inv_hc = 1.0f / (float)hc;
    // A dead thread (the grid's tail) walks nothing but its prologue, in row 0.
    const __nv_bfloat16* w = (!live ? down_w
        : row < rank ? down_w + (size_t)row * hc_dim
                     : inject_w + (size_t)(row - rank) * hc_dim) + CPT * p;
    const float* nx = normed + CPT * p;
    const unsigned int nk = live ? hc_dim / 32u : 0u;

    float acc[NT][CPT];
    #pragma unroll
    for (unsigned int t = 0; t < NT; ++t) {
        #pragma unroll
        for (unsigned int j = 0; j < CPT; ++j) acc[t][j] = 0.0f;
    }

    unsigned int wr[U][CPT / 2];
    float nr[DN][NT][CPT];
    #pragma unroll
    for (unsigned int u = 0; u < U; ++u) qhc_ld_bf<CPT>(w + (size_t)u * 32u, wr[u]);
    #pragma unroll
    for (unsigned int d = 0; d < DN; ++d) {
        #pragma unroll
        for (unsigned int t = 0; t < NT; ++t) {
            qhc_ld_f<CPT>(nx + (size_t)t * hc_dim + (size_t)d * 32u, nr[d][t]);
        }
    }
    // `reload_w` / `reload_n` are compile-time constants once unrolled.
    auto step = [&](const unsigned int u, const unsigned int k, const bool reload_w,
                    const bool reload_n) {
        float wf[CPT];
        qhc_unpack<CPT>(wr[u], wf);
        if (reload_w) qhc_ld_bf<CPT>(w + (size_t)(k + U) * 32u, wr[u]);
        #pragma unroll
        for (unsigned int t = 0; t < NT; ++t) {
            float n[CPT];
            #pragma unroll
            for (unsigned int j = 0; j < CPT; ++j) n[j] = nr[u % DN][t][j];
            if (reload_n) {
                qhc_ld_f<CPT>(nx + (size_t)t * hc_dim + (size_t)(k + DN) * 32u, nr[u % DN][t]);
            }
            #pragma unroll
            for (unsigned int j = 0; j < CPT; ++j) acc[t][j] += wf[j] * n[j];
        }
    };
    unsigned int k0 = 0;
    for (; k0 + U < nk; k0 += U) {
        #pragma unroll
        for (unsigned int u = 0; u < U; ++u) step(u, k0 + u, true, true);
    }
    if (nk != 0) {
        #pragma unroll
        for (unsigned int u = 0; u < U; ++u) step(u, k0 + u, false, u + DN < U);
    }

    // The scalar kernels' shfl_down tree, replayed on chain c = CPT*p + j.
    #pragma unroll
    for (unsigned int t = 0; t < NT; ++t) {
        #pragma unroll
        for (unsigned int off = 16; off > 0; off >>= 1) {
            if (off >= CPT) {
                #pragma unroll
                for (unsigned int j = 0; j < CPT; ++j) {
                    acc[t][j] += __shfl_down_sync(0xFFFFFFFFu, acc[t][j], off / CPT);
                }
            } else {
                #pragma unroll
                for (unsigned int j = 0; j < off; ++j) acc[t][j] = acc[t][j] + acc[t][j + off];
            }
        }
        if (live && p == 0 && t < nt_live) {
            const float v = acc[t][0];
            if (row < rank) {
                low_out[(size_t)t * rank + row] = qhc_silu(v * inv_hc);
            } else {
                inj_out[(size_t)t * hc + (row - rank)] = 2.0f * qhc_sigmoid(v * inv_hc);
            }
        }
    }
}

#ifndef HC_V_DOWN_DN
#define HC_V_DOWN_DN 16u
#endif
template <unsigned int CPT, unsigned int U, unsigned int DN>
__device__ __forceinline__ void qhc_down_vec_t(
    const float* normed, const __nv_bfloat16* down_w, const __nv_bfloat16* inject_w,
    float* low_out, float* inj_out, unsigned int hidden_size, unsigned int hc,
    unsigned int rank, unsigned int num_tokens
) {
    const unsigned int hc_dim = hc * hidden_size;
    switch (num_tokens) {
    case 1: qhc_down_vec<CPT, U, DN, 1>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    case 2: qhc_down_vec<CPT, U, DN, 2>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    case 3: qhc_down_vec<CPT, U, DN, 3>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    default: qhc_down_vec<CPT, U, DN, 4>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    }
}

// T = 5..8 (multi-sequence batches, ATLAS_QWEN4EXP_BATCH_FAST): the same
// walk for 5..8 tokens, so every weight element is still read once for all
// of them. Separate entry points, so the T <= 4 kernels keep their own
// register allocation, and a shorter `normed` ring (the 8 tokens' slots would
// otherwise cost 256 registers) -- pure scheduling, as above.
template <unsigned int CPT, unsigned int U, unsigned int DN>
__device__ __forceinline__ void qhc_down_vec_t8(
    const float* normed, const __nv_bfloat16* down_w, const __nv_bfloat16* inject_w,
    float* low_out, float* inj_out, unsigned int hidden_size, unsigned int hc,
    unsigned int rank, unsigned int num_tokens
) {
    const unsigned int hc_dim = hc * hidden_size;
    switch (num_tokens) {
    case 5: qhc_down_vec<CPT, U, DN, 5>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    case 6: qhc_down_vec<CPT, U, DN, 6>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    case 7: qhc_down_vec<CPT, U, DN, 7>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    default: qhc_down_vec<CPT, U, DN, 8>(normed, down_w, inject_w, low_out, inj_out, hc_dim, hc, rank); break;
    }
}

// Grid: (ceil((rank + hc) * 32 / HC_V_DOWN_CPT / block), 1, 1), block a
// multiple of 32. `inject_w == nullptr` (the model-level head) drops the
// injection rows.
extern "C" __global__ void hc_pre_down_vec(
    const float* __restrict__ normed,          // [T, hc*H]
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    float* __restrict__ low_out,               // [T, rank]
    float* __restrict__ inj_out,               // [T, hc]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank,
    const unsigned int num_tokens              // 1..4 (host checks; 5..8: the _vec8 twin)
) {
    atlas_pdl_enter();
    qhc_down_vec_t<HC_V_DOWN_CPT, HC_V_DOWN_UNROLL, HC_V_DOWN_DN>(
        normed, down_w, inject_w, low_out, inj_out, hidden_size, hc, rank, num_tokens);
}

#ifndef HC_V_DOWN_DN8
#define HC_V_DOWN_DN8 4u
#endif

// `hc_pre_down_vec` for num_tokens 5..HC_V_MAX (host checks); same grid.
extern "C" __global__ void hc_pre_down_vec8(
    const float* __restrict__ normed,          // [T, hc*H]
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    float* __restrict__ low_out,               // [T, rank]
    float* __restrict__ inj_out,               // [T, hc]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank,
    const unsigned int num_tokens              // 5..HC_V_MAX (host checks)
) {
    atlas_pdl_enter();
    qhc_down_vec_t8<HC_V_DOWN_CPT, HC_V_DOWN_UNROLL, HC_V_DOWN_DN8>(
        normed, down_w, inject_w, low_out, inj_out, hidden_size, hc, rank, num_tokens);
}

// Up + gate + stream mean, hc == 4. Thread g: stream g % 4, d = DPT * (g / 4)
// .. +DPT-1. Dynamic shared: NT * rank floats. Same rolling ring of U rows of
// `up_w` as the down kernel.
template <unsigned int DPT, unsigned int U, unsigned int NT>
__device__ __forceinline__ void qhc_finish_vec(
    const float* __restrict__ normed,
    const float* __restrict__ low,
    const __nv_bfloat16* __restrict__ up_w,
    __nv_bfloat16* __restrict__ y_out,
    const unsigned int H,
    const unsigned int rank
) {
    const unsigned int hc_dim = 4u * H;
    const float inv_hc = 1.0f / 4.0f;
    const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int s = g & 3u;
    const unsigned int dg = g >> 2;
    const bool live = dg < H / DPT;
    const unsigned int i0 = live ? s * H + DPT * dg : 0u;
    const __nv_bfloat16* ub = up_w + i0;
    const unsigned int nr = live ? rank : 0u;

    // The first U rows are issued before the `low` staging barrier: they do
    // not depend on it. Host checks rank % U == 0; the ring is unconditional
    // for the reason given in `qhc_down_vec`.
    unsigned int ur[U][DPT / 2];
    #pragma unroll
    for (unsigned int u = 0; u < U; ++u) qhc_ld_bf<DPT>(ub + (size_t)u * hc_dim, ur[u]);

    extern __shared__ float s_lo[];            // [NT, rank]
    for (unsigned int j = threadIdx.x; j < NT * rank; j += blockDim.x) s_lo[j] = low[j];
    __syncthreads();

    float acc[NT][DPT];
    #pragma unroll
    for (unsigned int t = 0; t < NT; ++t) {
        #pragma unroll
        for (unsigned int j = 0; j < DPT; ++j) acc[t][j] = 0.0f;
    }

    auto step = [&](const unsigned int u, const unsigned int r, const bool reload) {
        float uf[DPT];
        qhc_unpack<DPT>(ur[u], uf);
        if (reload) qhc_ld_bf<DPT>(ub + (size_t)(r + U) * hc_dim, ur[u]);
        #pragma unroll
        for (unsigned int t = 0; t < NT; ++t) {
            const float l = s_lo[t * rank + r];
            #pragma unroll
            for (unsigned int j = 0; j < DPT; ++j) acc[t][j] += uf[j] * l;
        }
    };
    unsigned int r0 = 0;
    for (; r0 + U < nr; r0 += U) {
        #pragma unroll
        for (unsigned int u = 0; u < U; ++u) step(u, r0 + u, true);
    }
    if (nr != 0) {
        #pragma unroll
        for (unsigned int u = 0; u < U; ++u) step(u, r0 + u, false);
    }

    #pragma unroll
    for (unsigned int t = 0; t < NT; ++t) {
        float p[DPT];
        if (live) {
            float n[DPT];
            qhc_ld_f<DPT>(normed + (size_t)t * hc_dim + i0, n);
            #pragma unroll
            for (unsigned int j = 0; j < DPT; ++j) p[j] = qhc_sigmoid(acc[t][j]) * n[j];
        } else {
            #pragma unroll
            for (unsigned int j = 0; j < DPT; ++j) p[j] = 0.0f;
        }
        unsigned int packed[DPT / 2];
        #pragma unroll
        for (unsigned int j = 0; j < DPT; ++j) {
            const float p1 = __shfl_down_sync(0xFFFFFFFFu, p[j], 1);
            const float p2 = __shfl_down_sync(0xFFFFFFFFu, p[j], 2);
            const float p3 = __shfl_down_sync(0xFFFFFFFFu, p[j], 3);
            float mixed = 0.0f;
            mixed += p[j];
            mixed += p1;
            mixed += p2;
            mixed += p3;
            const unsigned int b = __bfloat16_as_ushort(__float2bfloat16(mixed * inv_hc));
            if (j & 1u) packed[j / 2] |= b << 16;
            else packed[j / 2] = b;
        }
        if (live && s == 0) {
            __nv_bfloat16* y = y_out + (size_t)t * H + DPT * dg;
            if constexpr (DPT == 2) {
                *reinterpret_cast<unsigned int*>(y) = packed[0];
            } else if constexpr (DPT == 4) {
                *reinterpret_cast<uint2*>(y) = make_uint2(packed[0], packed[1]);
            } else {
                *reinterpret_cast<uint4*>(y) =
                    make_uint4(packed[0], packed[1], packed[2], packed[3]);
            }
        }
    }
}

template <unsigned int DPT, unsigned int U>
__device__ __forceinline__ void qhc_finish_vec_t(
    const float* normed, const float* low, const __nv_bfloat16* up_w,
    __nv_bfloat16* y_out, unsigned int hidden_size, unsigned int rank,
    unsigned int num_tokens
) {
    switch (num_tokens) {
    case 1: qhc_finish_vec<DPT, U, 1>(normed, low, up_w, y_out, hidden_size, rank); break;
    case 2: qhc_finish_vec<DPT, U, 2>(normed, low, up_w, y_out, hidden_size, rank); break;
    case 3: qhc_finish_vec<DPT, U, 3>(normed, low, up_w, y_out, hidden_size, rank); break;
    default: qhc_finish_vec<DPT, U, 4>(normed, low, up_w, y_out, hidden_size, rank); break;
    }
}

// T = 5..8: as `qhc_down_vec_t8`. (Splitting a (stream, d group)'s tokens
// over 2 or 4 lanes, which share the up_w loads, measured 5-25% slower at
// T = 5..8: twice the load instructions for the same bytes.)
template <unsigned int DPT, unsigned int U>
__device__ __forceinline__ void qhc_finish_vec_t8(
    const float* normed, const float* low, const __nv_bfloat16* up_w,
    __nv_bfloat16* y_out, unsigned int hidden_size, unsigned int rank,
    unsigned int num_tokens
) {
    switch (num_tokens) {
    case 5: qhc_finish_vec<DPT, U, 5>(normed, low, up_w, y_out, hidden_size, rank); break;
    case 6: qhc_finish_vec<DPT, U, 6>(normed, low, up_w, y_out, hidden_size, rank); break;
    case 7: qhc_finish_vec<DPT, U, 7>(normed, low, up_w, y_out, hidden_size, rank); break;
    default: qhc_finish_vec<DPT, U, 8>(normed, low, up_w, y_out, hidden_size, rank); break;
    }
}

// Grid: (ceil(4 * H / HC_V_FIN_DPT / block), 1, 1), block a multiple of 32.
// Dynamic shared: num_tokens * rank floats.
extern "C" __global__ void hc_pre_finish_vec(
    const float* __restrict__ normed,          // [T, 4*H]
    const float* __restrict__ low,             // [T, rank]
    const __nv_bfloat16* __restrict__ up_w,    // [rank, 4*H]
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    const unsigned int hidden_size,
    const unsigned int rank,
    const unsigned int num_tokens              // 1..4 (host checks; 5..8: the _vec8 twin)
) {
    atlas_pdl_enter();
    qhc_finish_vec_t<HC_V_FIN_DPT, HC_V_FIN_UNROLL>(
        normed, low, up_w, y_out, hidden_size, rank, num_tokens);
}

// A 16-row ring at T = 5..8 (the bench's pick: 48.5 vs 52 us at T = 8).
#ifndef HC_V_FIN_UNROLL8
#define HC_V_FIN_UNROLL8 16u
#endif

// `hc_pre_finish_vec` for num_tokens 5..HC_V_MAX (host checks); same grid
// and dynamic shared (num_tokens * rank floats).
extern "C" __global__ void hc_pre_finish_vec8(
    const float* __restrict__ normed,          // [T, 4*H]
    const float* __restrict__ low,             // [T, rank]
    const __nv_bfloat16* __restrict__ up_w,    // [rank, 4*H]
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    const unsigned int hidden_size,
    const unsigned int rank,
    const unsigned int num_tokens              // 5..HC_V_MAX (host checks)
) {
    atlas_pdl_enter();
    qhc_finish_vec_t8<HC_V_FIN_DPT, HC_V_FIN_UNROLL8>(
        normed, low, up_w, y_out, hidden_size, rank, num_tokens);
}

// ── T = 9..HC_V_ROWS_MAX in one launch (ATLAS_QWEN4EXP_BATCH_FAST) ─────────
//
// A batched verify of C4..C8 sequences x K=4 rows ran the collapse in 8-token
// launches: 2..4 back-to-back passes over the site's 13.1 MB. These run them
// as ONE launch whose blockIdx.y is a group of HC_V_MAX tokens: the CTAs of
// every group are co-resident and walk the same weight rows together, so the
// weights stream from DRAM about once. Each group IS the `_vec`/`_vec8`
// body on its own (up to 8) tokens -- the same per-token chains, the same
// ring scheduling as the T = 5..8 twin -- so every token's bytes are the
// T = 1 kernel's (scripts/dev/qwen4exp_hc_rows_bench.cu checks T = 1..32).
#ifndef HC_V_ROWS_MAX
#define HC_V_ROWS_MAX 32u
#endif

// Grid: (as `hc_pre_down_vec`, ceil(num_tokens / HC_V_MAX), 1).
extern "C" __global__ void hc_pre_down_vec_rows(
    const float* __restrict__ normed,          // [T, hc*H]
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    float* __restrict__ low_out,               // [T, rank]
    float* __restrict__ inj_out,               // [T, hc] (unused without inject_w)
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank,
    const unsigned int num_tokens              // 1..HC_V_ROWS_MAX (host checks)
) {
    atlas_pdl_enter();
    const unsigned int t0 = blockIdx.y * HC_V_MAX;
    const unsigned int nt = min(HC_V_MAX, num_tokens - t0);
    const unsigned int hc_dim = hc * hidden_size;
    const float* nx = normed + (size_t)t0 * hc_dim;
    float* lo = low_out + (size_t)t0 * rank;
    float* inj = inject_w != nullptr ? inj_out + (size_t)t0 * hc : inj_out;
#define QHC_DOWN_ROWS(T_) \
    qhc_down_vec<HC_V_DOWN_CPT, HC_V_DOWN_UNROLL, HC_V_DOWN_DN8, T_>( \
        nx, down_w, inject_w, lo, inj, hc_dim, hc, rank)
    switch (nt) {
    case 1: QHC_DOWN_ROWS(1); break;
    case 2: QHC_DOWN_ROWS(2); break;
    case 3: QHC_DOWN_ROWS(3); break;
    case 4: QHC_DOWN_ROWS(4); break;
    case 5: QHC_DOWN_ROWS(5); break;
    case 6: QHC_DOWN_ROWS(6); break;
    case 7: QHC_DOWN_ROWS(7); break;
    default: QHC_DOWN_ROWS(8); break;
    }
#undef QHC_DOWN_ROWS
}

// Grid: (as `hc_pre_finish_vec`, ceil(num_tokens / HC_V_MAX), 1).
// Dynamic shared: HC_V_MAX * rank floats (one group's `low`).
extern "C" __global__ void hc_pre_finish_vec_rows(
    const float* __restrict__ normed,          // [T, 4*H]
    const float* __restrict__ low,             // [T, rank]
    const __nv_bfloat16* __restrict__ up_w,    // [rank, 4*H]
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    const unsigned int hidden_size,
    const unsigned int rank,
    const unsigned int num_tokens              // 1..HC_V_ROWS_MAX (host checks)
) {
    atlas_pdl_enter();
    const unsigned int t0 = blockIdx.y * HC_V_MAX;
    const unsigned int nt = min(HC_V_MAX, num_tokens - t0);
    const float* nx = normed + (size_t)t0 * 4u * hidden_size;
    const float* lo = low + (size_t)t0 * rank;
    __nv_bfloat16* y = y_out + (size_t)t0 * hidden_size;
#define QHC_FIN_ROWS(T_) \
    qhc_finish_vec<HC_V_FIN_DPT, HC_V_FIN_UNROLL8, T_>(nx, lo, up_w, y, hidden_size, rank)
    switch (nt) {
    case 1: QHC_FIN_ROWS(1); break;
    case 2: QHC_FIN_ROWS(2); break;
    case 3: QHC_FIN_ROWS(3); break;
    case 4: QHC_FIN_ROWS(4); break;
    case 5: QHC_FIN_ROWS(5); break;
    case 6: QHC_FIN_ROWS(6); break;
    case 7: QHC_FIN_ROWS(7); break;
    default: QHC_FIN_ROWS(8); break;
    }
#undef QHC_FIN_ROWS
}

// ── T = 25..HC_V_ROWS_MAX down walk, re-tiled (ATLAS_QWEN4EXP_HC_WIDE) ─────
//
// `hc_pre_down_vec_rows` at 32 tokens is four 8-token groups of 41 CTAs at
// 150 registers: more CTAs than one wave holds, so the last group runs alone
// (~105 us against ~65 at 25..28 tokens). This is the same walk with FOUR
// chains per thread (the tree replay covers any CPT), which halves the CTAs
// a group needs and the normed loads per FLOP, so every group fits one wave.
// Groups of HC_V_WIDE_G tokens on blockIdx.y; the last group is padded, its
// padding tokens computing on scratch rows past the batch (host checks the
// scratch holds 64 rows) that are never written (`nt_live`). Every written
// token's chains are the T = 1 kernel's, in the same order, so the bytes are
// too (scripts/dev/qwen4exp_hc_wide_bench.cu checks T = 9..32). The finish
// kernel stays `hc_pre_finish_vec_rows`: no finish tile measured faster.
#ifndef HC_V_WIDE_G
#define HC_V_WIDE_G 8u
#endif
#ifndef HC_V_WIDE_DOWN_CPT
#define HC_V_WIDE_DOWN_CPT 4u
#endif
#ifndef HC_V_WIDE_DOWN_UNROLL
#define HC_V_WIDE_DOWN_UNROLL 16u
#endif
#ifndef HC_V_WIDE_DOWN_DN
#define HC_V_WIDE_DOWN_DN 2u
#endif
// Grid: (ceil((rank + hc) * 32 / HC_V_WIDE_DOWN_CPT / block),
//        ceil(num_tokens / HC_V_WIDE_G), 1).
extern "C" __global__ void hc_pre_down_vec_wide(
    const float* __restrict__ normed,          // [64, hc*H] scratch, T live rows
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    float* __restrict__ low_out,               // [T, rank]
    float* __restrict__ inj_out,               // [T, hc]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank,
    const unsigned int num_tokens              // 9..HC_V_ROWS_MAX (host checks)
) {
    atlas_pdl_enter();
    const unsigned int t0 = blockIdx.y * HC_V_WIDE_G;
    const unsigned int hc_dim = hc * hidden_size;
    qhc_down_vec<HC_V_WIDE_DOWN_CPT, HC_V_WIDE_DOWN_UNROLL, HC_V_WIDE_DOWN_DN, HC_V_WIDE_G>(
        normed + (size_t)t0 * hc_dim, down_w, inject_w, low_out + (size_t)t0 * rank,
        inject_w != nullptr ? inj_out + (size_t)t0 * hc : inj_out, hc_dim, hc, rank,
        min(HC_V_WIDE_G, num_tokens - t0));
}

// Stage 1 over a (T, S) grid, block 1024 (host checks: the RMS below is
// `hc_pre_stage`'s 1024-thread reduction and is only bit-identical at that
// width). Every block recomputes the token's hc RMS values and writes
// columns [y * hc_dim / S, (y + 1) * hc_dim / S) of `normed`.
extern "C" __global__ void hc_pre_stage_vec(
    const float* __restrict__ streams,
    const float* __restrict__ hc_norm_w,
    float* __restrict__ normed_out,            // [T, hc*H]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    atlas_pdl_enter();
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* x = streams + (size_t)t * hc_dim;
    float* out = normed_out + (size_t)t * hc_dim;

    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_MAX_MULT][QHC_WBLOCK / 32];

    // Per stream, thread `tid` sums d = tid, tid + 1024, ... in order — the
    // streams are only interleaved, which no accumulator can see.
    float acc[QHC_MAX_MULT];
    #pragma unroll
    for (unsigned int s2 = 0; s2 < QHC_MAX_MULT; ++s2) acc[s2] = 0.0f;
    for (unsigned int d = tid; d < H; d += QHC_WBLOCK) {
        #pragma unroll
        for (unsigned int s2 = 0; s2 < QHC_MAX_MULT; ++s2) {
            if (s2 < hc) {
                const float v = x[(size_t)s2 * H + d];
                acc[s2] += v * v;
            }
        }
    }
    #pragma unroll
    for (unsigned int s2 = 0; s2 < QHC_MAX_MULT; ++s2) {
        if (s2 < hc) {
            float a = acc[s2];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFFu, a, off);
            }
            if (lane == 0) smem_red[s2][warp] = a;
        }
    }
    __syncthreads();
    if (tid < hc) {
        float tot = 0.0f;
        for (unsigned int w2 = 0; w2 < QHC_WBLOCK / 32; ++w2) tot += smem_red[tid][w2];
        smem_rms[tid] = rsqrtf(tot / (float)H + eps);
    }
    __syncthreads();

    // Four columns per thread (H % 4 == 0 and hc_dim % (4 * S) == 0, host
    // checks, so a group never straddles two streams or two blocks).
    const unsigned int span = hc_dim / gridDim.y;
    const unsigned int c0 = blockIdx.y * span;
    const unsigned int c1 = blockIdx.y + 1 == gridDim.y ? hc_dim : c0 + span;
    for (unsigned int i = c0 + 4u * tid; i < c1; i += 4u * QHC_WBLOCK) {
        const float4 xv = *reinterpret_cast<const float4*>(x + i);
        const float4 wv = *reinterpret_cast<const float4*>(hc_norm_w + i);
        const float rms = smem_rms[i / H];
        float4 o;
        o.x = xv.x * rms * (1.0f + wv.x);
        o.y = xv.y * rms * (1.0f + wv.y);
        o.z = xv.z * rms * (1.0f + wv.z);
        o.w = xv.w * rms * (1.0f + wv.w);
        *reinterpret_cast<float4*>(out + i) = o;
    }
}

// `hc_post` over a (T, ceil(H / 4 / block)) grid, four consecutive d per
// thread. Same per-element arithmetic; `out` may alias `residual` (each
// element is read, then written, by the same thread).
extern "C" __global__ void hc_post_vec(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* residual,                       // [T, hc, H]
    const float* __restrict__ inj,               // [T, hc]
    float* out,                                  // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    atlas_pdl_enter();
    const unsigned int t = blockIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int d = 4u * (blockIdx.y * blockDim.x + threadIdx.x);
    if (d >= H) return;

    const uint2 xv = *reinterpret_cast<const uint2*>(block_out + (size_t)t * H + d);
    const float x0 = qhc_bf_lo(xv.x);
    const float x1 = qhc_bf_hi(xv.x);
    const float x2 = qhc_bf_lo(xv.y);
    const float x3 = qhc_bf_hi(xv.y);
    const float* res = residual + (size_t)t * hc * H;
    float* o = out + (size_t)t * hc * H;
    for (unsigned int s = 0; s < hc; ++s) {
        const float wv = inj[(size_t)t * hc + s];
        const float4 r = *reinterpret_cast<const float4*>(res + (size_t)s * H + d);
        float4 v;
        v.x = r.x + x0 * wv;
        v.y = r.y + x1 * wv;
        v.z = r.z + x2 * wv;
        v.w = r.w + x3 * wv;
        *reinterpret_cast<float4*>(o + (size_t)s * H + d) = v;
    }
}

// ── hc_post_vec + hc_pre_stage_vec in one launch (ATLAS_QWEN4EXP_DECODE_FUSE)
//
// Inside a decode layer the mixer's `hc_post` is followed at once by the MoE
// site's `hc_pre_stage` on the same token row: the post writes the highway
// and the stage reads it back for its per-stream RMS and `normed`. This
// kernel does both. Each block of a token's (1, S) cluster recomputes the
// post value of every element (`r + x * inj`, `hc_post_vec`'s expression) in
// `hc_pre_stage_vec`'s d order for the RMS, then, after a cluster barrier
// (no block of the token still reads the old highway), writes its 1/S slice
// of both the highway and `normed` from the same recomputed values. Every
// float is the one the two kernels produce: bit-identical (--fmad=false).
//
// In place only (the decode highway is post's `residual` and `out`).
// Grid: (T, HC_V_STAGE_SPLIT), clusters of (1, HC_V_STAGE_SPLIT), block 1024.
// Host checks: hc <= QHC_MAX_MULT, H % 4 == 0, hc*H % (4*S) == 0, 16-byte
// aligned highway, normed and norm weight, 8-byte aligned block_out.
#ifndef HC_V_STAGE_SPLIT
#define HC_V_STAGE_SPLIT 8u
#endif

#if !defined(__HIP_PLATFORM_AMD__)
extern "C" __global__ void __cluster_dims__(1, HC_V_STAGE_SPLIT, 1) __launch_bounds__(QHC_WBLOCK, 1)
hc_post_stage_vec(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    float* streams,                              // [T, hc, H]: residual in, post out
    const float* __restrict__ inj,               // [T, hc]
    const float* __restrict__ hc_norm_w, // [hc*H]
    float* __restrict__ normed_out,              // [T, hc*H]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    atlas_pdl_enter();
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const __nv_bfloat16* xb = block_out + (size_t)t * H;
    float* x = streams + (size_t)t * hc_dim;
    const float* w = inj + (size_t)t * hc;
    float* out = normed_out + (size_t)t * hc_dim;

    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_MAX_MULT][QHC_WBLOCK / 32];

    float wv[QHC_MAX_MULT];
    #pragma unroll
    for (unsigned int s2 = 0; s2 < QHC_MAX_MULT; ++s2) wv[s2] = s2 < hc ? w[s2] : 0.0f;

    float acc[QHC_MAX_MULT];
    #pragma unroll
    for (unsigned int s2 = 0; s2 < QHC_MAX_MULT; ++s2) acc[s2] = 0.0f;
    for (unsigned int d = tid; d < H; d += QHC_WBLOCK) {
        const float xd = (float)xb[d];
        #pragma unroll
        for (unsigned int s2 = 0; s2 < QHC_MAX_MULT; ++s2) {
            if (s2 < hc) {
                const float v = x[(size_t)s2 * H + d] + xd * wv[s2];
                acc[s2] += v * v;
            }
        }
    }
    #pragma unroll
    for (unsigned int s2 = 0; s2 < QHC_MAX_MULT; ++s2) {
        if (s2 < hc) {
            float a = acc[s2];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFFu, a, off);
            }
            if (lane == 0) smem_red[s2][warp] = a;
        }
    }
    __syncthreads();
    if (tid < hc) {
        float tot = 0.0f;
        for (unsigned int w2 = 0; w2 < QHC_WBLOCK / 32; ++w2) tot += smem_red[tid][w2];
        smem_rms[tid] = rsqrtf(tot / (float)H + eps);
    }
    // The old highway row has been read by every block of this token.
    cooperative_groups::this_cluster().sync();

    const unsigned int span = hc_dim / gridDim.y;
    const unsigned int c0 = blockIdx.y * span;
    const unsigned int c1 = blockIdx.y + 1 == gridDim.y ? hc_dim : c0 + span;
    for (unsigned int i = c0 + 4u * tid; i < c1; i += 4u * QHC_WBLOCK) {
        const unsigned int s = i / H;
        const unsigned int d = i - s * H;
        const float4 r = *reinterpret_cast<const float4*>(x + i);
        const uint2 xv = *reinterpret_cast<const uint2*>(xb + d);
        const float ws = w[s];
        float4 v;
        v.x = r.x + qhc_bf_lo(xv.x) * ws;
        v.y = r.y + qhc_bf_hi(xv.x) * ws;
        v.z = r.z + qhc_bf_lo(xv.y) * ws;
        v.w = r.w + qhc_bf_hi(xv.y) * ws;
        *reinterpret_cast<float4*>(x + i) = v;
        const float4 nw = *reinterpret_cast<const float4*>(hc_norm_w + i);
        const float rms = smem_rms[s];
        float4 o;
        o.x = v.x * rms * (1.0f + nw.x);
        o.y = v.y * rms * (1.0f + nw.y);
        o.z = v.z * rms * (1.0f + nw.z);
        o.w = v.w * rms * (1.0f + nw.w);
        *reinterpret_cast<float4*>(out + i) = o;
    }
}
#endif  // !__HIP_PLATFORM_AMD__ (thread-block clusters)
