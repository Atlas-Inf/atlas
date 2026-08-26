// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Spark — Gemma-4 E2B Vision Tower CUDA Kernels (module "gemma_vision")
//
// Wave 3 of the E2B multimodal bring-up: the six gemma-specific entry points
// the Rust `GemmaVisionEncoder` declares via
// `try_kernel(gpu, "gemma_vision", "<name>")` (crates/spark-model/src/layers/
// gemma_vision_encoder/). Shared kernels (dense GEMM, RMSNorm, GeGLU, BF16
// add) are REUSED from this target's existing tree — nothing here redefines
// them. Every launch contract below mirrors its Rust launch site EXACTLY
// (arg order, grid, block): a mismatch is a runtime crash, not a compile
// error.
//
// All ops use BF16 storage; computation uses f32 accumulators. The tower runs
// once per image prefill (P ≤ 2520 patches, 16 layers), so simplicity wins
// over peak occupancy — same posture as the Qwen3-VL encoder.
//
// Launch-site index (Rust → kernel):
//   gemma_vision_patch_scale — host-side stand-in today (enc_impl/patch_embed.rs)
//   gemma_vision_qk_norm     — enc_impl/qk_norm.rs (drop-in for norm::rms_norm)
//   gemma_vision_rope_rotate — enc_impl/attention.rs attention_stage
//   gemma_vision_attention   — enc_impl/attention.rs attention_stage
//   gemma_vision_clamp       — enc_impl/mlp.rs clipped_gemm (pre/post GEMM)
//   gemma_vision_pool        — enc_impl/pool.rs pool_stage

#include <cuda_bf16.h>

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// ── 1. Patch scale: pixels → BF16 with the embedder's 2×(x−0.5) ────────────
// Fused replacement for the Wave-2 host-side scale+convert loop
// (enc_impl/patch_embed.rs `patch_embed_batched`): elementwise and exact in
// f32, so the upload bytes are identical either way. Not yet dispatched by
// the Rust (the host loop is the stand-in); kept in the module so the tower
// is complete and the Rust can point a handle at it.
//
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void gemma_vision_patch_scale(
    const float* __restrict__ pixels,   // [n] f32 image pixels, row-major
    __nv_bfloat16* __restrict__ out,    // [n] scaled BF16 pixels
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_to_bf16(2.0f * (pixels[i] - 0.5f));
}

// ── 2. QK-Norm: per-row RMSNorm over head_dim (absolute formula) ───────────
// Per-head QK-Norm for q and k: RMSNorm over head_dim 64 applied AFTER the
// Q/K projections and BEFORE the rotary + attention (Gemma's attention scale
// is 1.0 — QK-norm replaces 1/√head_dim). Consumes the interleaved
// [p, heads×head_dim] layout directly, one block per (token × head) row.
//
// The Rust currently resolves `k_qk_norm` to the generic `norm::rms_norm`
// (enc_impl/qk_norm.rs `qk_norm_inplace`, launched via ops::rms_norm with the
// shared (input, weight, output, dim, eps) convention and rows as the grid);
// this kernel is the shape-compatible drop-in with identical math.
//
// Grid: (rows, 1, 1)  Block: (min(dim, 1024), 1, 1)
extern "C" __global__ void gemma_vision_qk_norm(
    const __nv_bfloat16* __restrict__ input,   // [rows, dim]
    const __nv_bfloat16* __restrict__ weight,  // [dim]
    __nv_bfloat16* __restrict__ output,        // [rows, dim]
    unsigned int dim,
    float eps
) {
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + (unsigned long long)row * dim;
    __nv_bfloat16* out = output + (unsigned long long)row * dim;

    // Sum of squares, 2-wide BF16 loads.
    float sum_sq = 0.0f;
    const unsigned int half = dim / 2;
    const unsigned int* x32 = (const unsigned int*)x;
    for (unsigned int i = tid; i < half; i += blockDim.x) {
        float v0 = bf16_to_f32(__ushort_as_bfloat16((unsigned short)(x32[i] & 0xFFFF)));
        float v1 = bf16_to_f32(__ushort_as_bfloat16((unsigned short)(x32[i] >> 16)));
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((dim & 1) && tid == 0) {
        float v = bf16_to_f32(x[dim - 1]);
        sum_sq += v * v;
    }

    // Warp + block reduction.
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, off);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    // Absolute formula: out = x * rsqrt(mean(x²) + eps) * w (no 1+offset).
    float rms = rsqrtf(warp_sums[0] / (float)dim + eps);
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    for (unsigned int i = tid; i < half; i += blockDim.x) {
        float xv0 = bf16_to_f32(__ushort_as_bfloat16((unsigned short)(x32[i] & 0xFFFF)));
        float xv1 = bf16_to_f32(__ushort_as_bfloat16((unsigned short)(x32[i] >> 16)));
        float wv0 = bf16_to_f32(__ushort_as_bfloat16((unsigned short)(w32[i] & 0xFFFF)));
        float wv1 = bf16_to_f32(__ushort_as_bfloat16((unsigned short)(w32[i] >> 16)));
        out32[i] = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(xv0 * rms * wv0)) |
                   ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(xv1 * rms * wv1)) << 16);
    }
    if ((dim & 1) && tid == 0) {
        float w = bf16_to_f32(weight[dim - 1]);
        out[dim - 1] = f32_to_bf16(bf16_to_f32(x[dim - 1]) * rms * w);
    }
}

// ── 3. RoPE rotate: in-place 2D rotary on q and k ───────────────────────────
// Applies the per-token 2D rotary to the q and k planes in place. q/k layout
// is the interleaved [seq, heads×head_dim]; cos/sin are the host-precomputed
// [seq, head_dim] BF16 tables from enc_impl/pos.rs `build_rope_cossin_into`
// (row = [x_freq; y_freq; x_freq; y_freq], second half duplicating the first,
// θ=100, head_dim 64 — layout pinned by the numerical oracle). The rotation
// is the standard rotate_half form, same sign pattern the Qwen3-VL
// `vision_attention_rope` uses:
//   d <  D/2:  x' = x[d]·cos[d] − x[d+D/2]·sin[d]
//   d ≥ D/2:   x' = x[d]·cos[d] + x[d−D/2]·sin[d]
// Read-phase / write-phase are separated by __syncthreads so the in-place
// partner read (thread d reads element d±D/2, owned by another thread) never
// races with a write.
//
// Rust launch (enc_impl/attention.rs `attention_stage`, lines 41-57):
//   grid (seq, 1, 1), block (head_dim, 1, 1),
//   args (q, k, cos, sin, seq, heads, head_dim)
extern "C" __global__ void gemma_vision_rope_rotate(
    __nv_bfloat16* __restrict__ q,      // [seq, heads*head_dim] in-place
    __nv_bfloat16* __restrict__ k,      // [seq, heads*head_dim] in-place
    const __nv_bfloat16* __restrict__ cos,  // [seq, head_dim]
    const __nv_bfloat16* __restrict__ sin,  // [seq, head_dim]
    unsigned int seq, unsigned int heads, unsigned int head_dim
) {
    unsigned int qi = blockIdx.x;
    unsigned int d = threadIdx.x;
    if (qi >= seq || d >= head_dim) return;

    // HF apply_multidimensional_rope: head_dim = 64 is split into two
    // 32-channel spatial segments (x, y); each segment is rotated
    // independently with rotate_half on half=16 WITHIN the segment. So the
    // partner index is d ± 16 (not d ± 32): channel d pairs with d±16 in
    // the same [0..32) or [32..64) segment. The cos/sin row is
    // [x(32); y(32)] built by build_rope_cossin_into.
    unsigned int seg = d / (head_dim / 2);        // 0 = x, 1 = y
    unsigned int dseg = d % (head_dim / 2);       // 0..31 within the segment
    unsigned int half = (head_dim / 2) / 2;       // 16
    unsigned int partner = seg * (head_dim / 2) + ((dseg < half) ? dseg + half : dseg - half);
    float sign = (dseg < half) ? -1.0f : 1.0f;
    float c = bf16_to_f32(cos[(unsigned long long)qi * head_dim + d]);
    float s = bf16_to_f32(sin[(unsigned long long)qi * head_dim + d]);

    unsigned long long row = (unsigned long long)qi * heads * head_dim;
    for (unsigned int h = 0; h < heads; ++h) {
        unsigned long long off = row + (unsigned long long)h * head_dim;
        // Read phase: own element + partner element from BOTH planes.
        float xq = bf16_to_f32(q[off + d]);
        float pq = bf16_to_f32(q[off + partner]);
        float xk = bf16_to_f32(k[off + d]);
        float pk = bf16_to_f32(k[off + partner]);
        __syncthreads();
        // Write phase: rotated values (all reads for this head done).
        q[off + d] = f32_to_bf16(xq * c + sign * pq * s);
        k[off + d] = f32_to_bf16(xk * c + sign * pk * s);
        __syncthreads();
    }
}

// ── 4. Causal MHA on pre-rotated q/k ────────────────────────────────────────
// One warp per (query token, head). q/k arrive ALREADY RoPE-rotated by
// `gemma_vision_rope_rotate`; the cos/sin args are accepted for ABI
// compatibility with the Rust launch (the pre-rotation made them unused).
// Causal: key rows kj > qi are masked to −∞. Attention scale is 1.0 — the
// per-head QK-Norm replaces 1/√head_dim, so no scaling is applied here.
//
// Rust launch (enc_impl/attention.rs `attention_stage`, lines 58-70):
//   grid (seq, heads, 1), block (32, 1, 1),
//   args (q, k, v, o, cos, sin, seq, heads, head_dim)
extern "C" __global__ void gemma_vision_attention(
    const __nv_bfloat16* __restrict__ q,  // [seq, heads*head_dim] (rotated)
    const __nv_bfloat16* __restrict__ k,  // [seq, heads*head_dim] (rotated)
    const __nv_bfloat16* __restrict__ v,  // [seq, heads*head_dim]
    __nv_bfloat16* __restrict__ o,        // [seq, heads*head_dim]
    const __nv_bfloat16* __restrict__ cos, // ABI-compat — unused (rope pre-applied)
    const __nv_bfloat16* __restrict__ sin, // ABI-compat — unused (rope pre-applied)
    unsigned int seq, unsigned int heads, unsigned int head_dim
) {
    // Static smem caps: shipped tower is seq ≤ 2520 (max_patches), head_dim
    // 64. The scores buffer must stay under the static shared-memory limit
    // (16 KB on sm_121): 2520×4 B + 64×4 B = 10336 B. Guard explicitly so a
    // future geometry change fails loudly, not out-of-bounds.
    const unsigned int MAX_SEQ = 2520;
    const unsigned int MAX_HD = 64;

    unsigned int qi = blockIdx.x;
    unsigned int h = blockIdx.y;
    unsigned int t = threadIdx.x;
    if (qi >= seq || h >= heads || seq > MAX_SEQ || head_dim > MAX_HD) return;

    const float scale = 1.0f;  // QK-norm replaces 1/sqrt(head_dim)
    unsigned long long h_off = (unsigned long long)h * head_dim;
    unsigned long long q_off = (unsigned long long)qi * heads * head_dim + h_off;
    unsigned long long stride = (unsigned long long)heads * head_dim;

    __shared__ float scores[MAX_SEQ];
    __shared__ float q_row[MAX_HD];

    // Cache this query's head slice (each thread owns dims {t, t+32, ...}).
    for (unsigned int d = t; d < head_dim; d += 32) q_row[d] = bf16_to_f32(q[q_off + d]);
    for (unsigned int j = t; j < seq; j += 32) scores[j] = -INFINITY;
    __syncthreads();

    // Scores: each thread computes the FULL dot over all head_dim dims for
    // its own set of key rows (kj = t, t+32, ...), writing every scores[j].
    // The per-iteration __syncthreads() keeps softmax (thread 0) from racing
    // the writes; OOB lanes (kj ≥ seq when seq%32 != 0) are skipped.
    for (unsigned int kj = t; kj < seq; kj += 32) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d) {
            dot += q_row[d] * bf16_to_f32(k[(unsigned long long)kj * stride + h_off + d]);
        }
        scores[kj] = dot * scale;
        __syncthreads();
    }

    // Softmax over the (causal) scores, by thread 0.
    if (t == 0) {
        float max_s = scores[0];
        for (unsigned int j = 1; j < seq; ++j) max_s = fmaxf(max_s, scores[j]);
        float sum_exp = 0.0f;
        for (unsigned int j = 0; j < seq; ++j) {
            scores[j] = expf(scores[j] - max_s);
            sum_exp += scores[j];
        }
        float inv = 1.0f / sum_exp;  // diagonal term is finite → sum > 0
        for (unsigned int j = 0; j < seq; ++j) scores[j] *= inv;
    }
    __syncthreads();

    // Weighted sum of values → output row.
    for (unsigned int d = t; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < seq; ++j) {
            acc += scores[j] * bf16_to_f32(v[(unsigned long long)j * stride + h_off + d]);
        }
        o[q_off + d] = f32_to_bf16(acc);
    }
}

// ── 5. ClippableLinear clamp (in place) ─────────────────────────────────────
// Clamps a BF16 buffer to [lo, hi] in place — the ClippableLinear
// activation clamp applied pre-GEMM (on the input) and post-GEMM (on the
// output). torch.clamp semantics: min(max(x, lo), hi).
//
// Rust launch (enc_impl/mlp.rs `clipped_gemm`, lines 36-43 and 56-63):
//   grid (ceil(n/256), 1, 1), block (256, 1, 1),
//   args (data, lo, hi, n)
extern "C" __global__ void gemma_vision_clamp(
    __nv_bfloat16* __restrict__ data,  // [n] in-place
    float lo, float hi,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bf16_to_f32(data[i]);
    data[i] = f32_to_bf16(fminf(fmaxf(v, lo), hi));
}

// ── 6. Average pool: pks×pks patch groups → soft tokens, ×√hidden ──────────
// src is one image's [seq, hidden] patch states in row-major grid order
// (grid_h × grid_w patches); dst is [soft, hidden]. Soft token s covers the
// patch band (s_row·pks .. +pks) × (s_col·pks .. +pks) and stores
// (mean of the pks² patch rows) × scale, where scale = √hidden_size (f32).
// Padding-strip: source patch rows with index ≥ seq are skipped and excluded
// from the mean (the preprocessor's grids always divide, so in practice all
// pks² rows are valid).
//
// Rust launch (enc_impl/pool.rs `pool_stage`, lines 36-47):
//   grid (soft, 1, 1), block (min(hidden, 256), 1, 1),
//   args (src, dst, grid_w, pks, hidden, seq, soft, scale)
extern "C" __global__ void gemma_vision_pool(
    const __nv_bfloat16* __restrict__ src,  // [seq, hidden]
    __nv_bfloat16* __restrict__ dst,        // [soft, hidden]
    unsigned int grid_w,
    unsigned int pks,
    unsigned int hidden,
    unsigned int seq,
    unsigned int soft,
    float scale
) {
    unsigned int s = blockIdx.x;
    unsigned int d = threadIdx.x;
    if (s >= soft || d >= hidden || pks == 0) return;

    unsigned int out_w = grid_w / pks;   // soft tokens per grid band
    unsigned int s_row = s / out_w;
    unsigned int s_col = s % out_w;
    unsigned int pks2 = pks * pks;

    float acc = 0.0f;
    unsigned int n = 0;
    for (unsigned int i = 0; i < pks; ++i) {
        for (unsigned int j = 0; j < pks; ++j) {
            unsigned int patch = (s_row * pks + i) * grid_w + (s_col * pks + j);
            if (patch >= seq) continue;  // padding strip
            acc += bf16_to_f32(src[(unsigned long long)patch * hidden + d]);
            ++n;
        }
    }
    float mean = (n > 0) ? acc / (float)pks2 : 0.0f;
    dst[(unsigned long long)s * hidden + d] = f32_to_bf16(mean * scale);
}
