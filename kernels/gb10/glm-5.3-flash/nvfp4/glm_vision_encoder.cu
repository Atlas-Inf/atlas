// SPDX-License-Identifier: AGPL-3.0-only

// Native GLM-5.3 vision primitives.  These deliberately live in a separate
// module from the Qwen vision kernels: GLM uses RMSNorm, clamped SwiGLU,
// partial 2-D RoPE, and a convolutional merger.

#include <cuda_bf16.h>
#include <math.h>

__device__ inline float glm_bf16(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline __nv_bfloat16 glm_bf16_from_f32(float x) {
    return __float2bfloat16(x);
}

extern "C" __global__ void glm_vision_f32_to_bf16(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, unsigned int n) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = glm_bf16_from_f32(src[i]);
}

// Block-wide sum for the one-row-per-block norms (blockDim.x a multiple of 32,
// at most 1024). Every thread receives the total.
__device__ inline float glm_block_sum(float v) {
    __shared__ float partial[32];
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();  // `partial` may still be read by a previous call
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    v = lane < (blockDim.x >> 5) ? partial[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// One block per row.
extern "C" __global__ void glm_vision_rms_norm(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    unsigned int rows, unsigned int hidden, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x + (size_t)row * hidden;
    __nv_bfloat16* yr = y + (size_t)row * hidden;
    float sum = 0.0f;
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float v = glm_bf16(xr[d]);
        sum += v * v;
    }
    float inv = rsqrtf(glm_block_sum(sum) / hidden + eps);
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x)
        yr[d] = glm_bf16_from_f32(glm_bf16(xr[d]) * inv * glm_bf16(w[d]));
}

// Row-broadcast bias for the tensor-core GEMM output: c[m, n] += bias[n].
extern "C" __global__ void glm_vision_add_bias(
    __nv_bfloat16* __restrict__ c,
    const __nv_bfloat16* __restrict__ bias,
    unsigned int m, unsigned int n) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < m * n) c[i] = glm_bf16_from_f32(glm_bf16(c[i]) + glm_bf16(bias[i % n]));
}

extern "C" __global__ void glm_vision_add(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    unsigned int n) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = glm_bf16_from_f32(glm_bf16(dst[i]) + glm_bf16(src[i]));
}

// One block per row, in place.
extern "C" __global__ void glm_vision_layer_norm(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b,
    unsigned int rows, unsigned int hidden, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    __nv_bfloat16* xr = x + (size_t)row * hidden;
    float sum = 0.0f;
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) sum += glm_bf16(xr[d]);
    float mean = glm_block_sum(sum) / hidden;
    float var = 0.0f;
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float delta = glm_bf16(xr[d]) - mean;
        var += delta * delta;
    }
    float inv = rsqrtf(glm_block_sum(var) / hidden + eps);
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float v = (glm_bf16(xr[d]) - mean) * inv;
        xr[d] = glm_bf16_from_f32(v * glm_bf16(w[d]) + glm_bf16(b[d]));
    }
}

extern "C" __global__ void glm_vision_gelu(
    __nv_bfloat16* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = glm_bf16(x[i]);
    x[i] = glm_bf16_from_f32(0.5f * v * (1.0f + erff(v * 0.7071067811865475f)));
}

extern "C" __global__ void glm_vision_swiglu_clamp(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    unsigned int rows, unsigned int hidden, float limit) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    unsigned int n = rows * hidden;
    if (i >= n) return;
    unsigned int row = i / hidden;
    unsigned int col = i % hidden;
    // vLLM's SiluAndMulWithClamp clamps the gate only on the upper side;
    // the up branch is clamped symmetrically below.
    float gate = fminf(glm_bf16(x[row * (2 * hidden) + col]), limit);
    float up = fminf(fmaxf(glm_bf16(x[row * (2 * hidden) + hidden + col]), -limit), limit);
    y[i] = glm_bf16_from_f32((gate / (1.0f + expf(-gate))) * up);
}


// q/k RMSNorm followed by GLM's partial 2-D RoPE, written back in place over
// the Q and K thirds of the fused [rows, 3 * heads * head_dim] QKV tensor so
// attention reads ready BF16 operands. One block of `head_dim` threads per
// (row, head, q|k); blockIdx.z selects Q (0) or K (1).
extern "C" __global__ void glm_vision_qk_norm_rope(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    unsigned int rows, unsigned int heads, unsigned int head_dim) {
    __shared__ float normed[128];
    unsigned int row = blockIdx.x, head = blockIdx.y, dim = threadIdx.x;
    if (row >= rows || head >= heads || dim >= head_dim) return;
    unsigned int hidden = heads * head_dim;
    size_t base = (size_t)row * 3 * hidden + blockIdx.z * hidden + head * head_dim;
    const __nv_bfloat16* norm = blockIdx.z ? k_norm : q_norm;
    float raw = glm_bf16(qkv[base + dim]);
    float inv = rsqrtf(glm_block_sum(raw * raw) / head_dim + 1.0e-5f);
    // fused_q_kv_rmsnorm returns BF16; keep that boundary before the rotary op.
    normed[dim] = glm_bf16(glm_bf16_from_f32(raw * inv * glm_bf16(norm[dim])));
    __syncthreads();
    // ApplyRotaryEmb is Neox-style over the full head. GLM's partial rotary
    // factor makes the cache row `head_dim / 2` wide; its first half carries
    // H frequencies and its second half W frequencies, while the two halves
    // of the head are paired by the rotary transform.
    unsigned int rotary_dim = head_dim / 2;
    unsigned int partner = dim < rotary_dim ? dim + rotary_dim : dim - rotary_dim;
    float c = glm_bf16(cos_table[row * rotary_dim + dim % rotary_dim]);
    float s = glm_bf16(sin_table[row * rotary_dim + dim % rotary_dim]);
    float x = normed[dim], p = normed[partner];
    qkv[base + dim] = glm_bf16_from_f32(dim < rotary_dim ? x * c - p * s : x * c + p * s);
}

// Bidirectional flash attention over one image sequence, reading the rotated
// Q/K and raw V straight out of the fused QKV rows and writing the
// [rows, heads * 64] output. Four warps own 16 queries each; K/V tiles of 64
// keys are staged in shared memory (V transposed so P.V uses the same
// K-contiguous B-fragment layout as Q.K^T). Scores and the online softmax stay
// in FP32; P is rounded to BF16 for the P.V MMA as in FlashAttention-2.
#define GLM_FA_D 64
#define GLM_FA_BM 64
#define GLM_FA_BN 64
#define GLM_FA_STRIDE (GLM_FA_D + 8)

__device__ __forceinline__ void glm_mma_bf16(float c[4], const unsigned int a[4],
                                             unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int glm_pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned int*>(&v);
}

extern "C" __global__ void __launch_bounds__(128) glm_vision_flash_attention(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ out,
    unsigned int seq, unsigned int heads) {
    __shared__ __align__(16) __nv_bfloat16 s_k[GLM_FA_BN][GLM_FA_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 s_vt[GLM_FA_D][GLM_FA_STRIDE];
    const unsigned int head = blockIdx.y;
    const unsigned int hidden = heads * GLM_FA_D;
    const size_t row_stride = 3 * (size_t)hidden;
    const __nv_bfloat16* q_base = qkv + head * GLM_FA_D;
    const __nv_bfloat16* k_base = q_base + hidden;
    const __nv_bfloat16* v_base = q_base + 2 * hidden;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned int group = lane >> 2, quad = lane & 3;
    const unsigned int r0 = blockIdx.x * GLM_FA_BM + warp * 16 + group, r1 = r0 + 8;

    // Q A-fragments for the four 16-wide K steps of the head dimension.
    unsigned int qa[4][4];
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        unsigned int c0 = ks * 16 + quad * 2, c1 = c0 + 8;
        qa[ks][0] = r0 < seq ? *(const unsigned int*)(q_base + r0 * row_stride + c0) : 0u;
        qa[ks][1] = r1 < seq ? *(const unsigned int*)(q_base + r1 * row_stride + c0) : 0u;
        qa[ks][2] = r0 < seq ? *(const unsigned int*)(q_base + r0 * row_stride + c1) : 0u;
        qa[ks][3] = r1 < seq ? *(const unsigned int*)(q_base + r1 * row_stride + c1) : 0u;
    }

    const float scale_log2 = 0.125f * 1.4426950408889634f;  // head_dim^-0.5 * log2(e)
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    float o[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

    for (unsigned int kv0 = 0; kv0 < seq; kv0 += GLM_FA_BN) {
        __syncthreads();  // the previous tile is no longer being read
        for (unsigned int c = threadIdx.x; c < GLM_FA_BN * GLM_FA_D / 8; c += 128) {
            unsigned int key = c / (GLM_FA_D / 8), d = (c % (GLM_FA_D / 8)) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kv0 + key < seq) {
                kv = *(const uint4*)(k_base + (kv0 + key) * row_stride + d);
                vv = *(const uint4*)(v_base + (kv0 + key) * row_stride + d);
            }
            *(uint4*)&s_k[key][d] = kv;
            const __nv_bfloat16* v8 = reinterpret_cast<const __nv_bfloat16*>(&vv);
            #pragma unroll
            for (int e = 0; e < 8; ++e) s_vt[d + e][key] = v8[e];
        }
        __syncthreads();

        float s[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const __nv_bfloat16* krow = s_k[nt * 8 + group];
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks)
                glm_mma_bf16(s[nt], qa[ks], *(const unsigned int*)(krow + ks * 16 + quad * 2),
                             *(const unsigned int*)(krow + ks * 16 + 8 + quad * 2));
        }

        float t0 = m0, t1 = m1;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            unsigned int key = kv0 + nt * 8 + quad * 2;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                bool valid = key + j < seq;
                s[nt][j] = valid ? s[nt][j] * scale_log2 : -INFINITY;
                s[nt][2 + j] = valid ? s[nt][2 + j] * scale_log2 : -INFINITY;
                t0 = fmaxf(t0, s[nt][j]);
                t1 = fmaxf(t1, s[nt][2 + j]);
            }
        }
        #pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1) {
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, offset));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, offset));
        }
        // Every tile holds at least one valid key, so the new maxima are finite.
        float alpha0 = exp2f(m0 - t0), alpha1 = exp2f(m1 - t1);
        m0 = t0;
        m1 = t1;
        l0 *= alpha0;
        l1 *= alpha1;
        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            o[dt][0] *= alpha0; o[dt][1] *= alpha0;
            o[dt][2] *= alpha1; o[dt][3] *= alpha1;
        }
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - m0); s[nt][1] = exp2f(s[nt][1] - m0);
            s[nt][2] = exp2f(s[nt][2] - m1); s[nt][3] = exp2f(s[nt][3] - m1);
            l0 += s[nt][0] + s[nt][1];
            l1 += s[nt][2] + s[nt][3];
        }

        // The score accumulator layout of key tiles (2j, 2j+1) is exactly the
        // A-fragment layout of the j-th 16-key step of P.
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int pa[4] = {
                glm_pack_bf16(s[2 * j][0], s[2 * j][1]),
                glm_pack_bf16(s[2 * j][2], s[2 * j][3]),
                glm_pack_bf16(s[2 * j + 1][0], s[2 * j + 1][1]),
                glm_pack_bf16(s[2 * j + 1][2], s[2 * j + 1][3]),
            };
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                const __nv_bfloat16* vrow = s_vt[dt * 8 + group];
                glm_mma_bf16(o[dt], pa, *(const unsigned int*)(vrow + j * 16 + quad * 2),
                             *(const unsigned int*)(vrow + j * 16 + 8 + quad * 2));
            }
        }
    }

    #pragma unroll
    for (int offset = 1; offset < 4; offset <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, offset);
        l1 += __shfl_xor_sync(0xffffffffu, l1, offset);
    }
    float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        unsigned int col = head * GLM_FA_D + dt * 8 + quad * 2;
        if (r0 < seq)
            *(__nv_bfloat162*)(out + (size_t)r0 * hidden + col) =
                __floats2bfloat162_rn(o[dt][0] * inv0, o[dt][1] * inv0);
        if (r1 < seq)
            *(__nv_bfloat162*)(out + (size_t)r1 * hidden + col) =
                __floats2bfloat162_rn(o[dt][2] * inv1, o[dt][3] * inv1);
    }
}

// Lay each 2x2 merge block out in Conv2d weight order so the merger's
// downsample becomes one GEMM against the [out, hidden, 2, 2] weight viewed as
// [out, hidden * 4]. The processor emits the four members of a block as
// consecutive rows (`bh`, `bw`, `ih`, `iw`), so row `token * 4 + q` of `src`
// holds member q = ih * 2 + iw: dst[token][c * 4 + q] = src[token * 4 + q][c].
extern "C" __global__ void glm_vision_merge_reorder(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned int tokens, unsigned int hidden) {
    unsigned int i = blockIdx.x * 256 + threadIdx.x;
    unsigned int width = 4 * hidden;
    if (i >= tokens * width) return;
    unsigned int token = i / width, c = (i % width) / 4, q = i % 4;
    dst[i] = src[((size_t)token * 4 + q) * hidden + c];
}
