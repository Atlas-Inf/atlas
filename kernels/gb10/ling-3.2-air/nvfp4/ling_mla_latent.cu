// SPDX-License-Identifier: AGPL-3.0-only
//
// Ling MLA over the compressed latent cache.
//
// Each cached token is one 576-wide BF16 row: the normalized 512-channel
// latent (kv_a_layernorm output) followed by the rotated 64-channel RoPE key
// shared by all 32 heads. The V side of the paged pool aliases the K side.
//
// Decode, batched decode and the MTP verify rows attend in the absorbed form:
//   q_abs[h]   = [W_kb[h]^T q_nope[h] | q_rope[h]]            (576)
//   score[h,t] = q_abs[h] . cache[t] * scale
//   lat[h]     = sum_t softmax(score)[h,t] cache[t][0:512]
//   out[h]     = W_vb[h] lat[h]                                  (128)
// where kv_b_proj rows h*256 + [0,128) are W_kb[h] and h*256 + [128,256) are
// W_vb[h]. Every row's arithmetic depends only on that row and its sequence
// length, never on how many rows share a launch, so a two-row verify
// reproduces serial decode byte for byte.
//
// Prompts rebuild full per-head keys and values from the latent in bounded
// tiles (ling_latent_gather + kv_b_proj + ling_latent_expand), attend each
// tile with ling_mla_prefill, and merge tiles by their softmax state.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#define LING_HEADS 32u
#define LING_LATENT 512u
#define LING_ROPE 64u
#define LING_ROW 576u        // LING_LATENT + LING_ROPE
#define LING_QK 192u         // per-head query width in q_full: nope 128 | rope 64
#define LING_NOPE 128u
#define LING_V 128u
#define LING_KVB 256u        // kv_b_proj rows per head: nope 128 | v 128

// ---- Cache write ------------------------------------------------------------

// One row per block: [kv_norm (512) | k_rope (64)] at the row's slot.
extern "C" __global__ void ling_latent_cache_write(
    const __nv_bfloat16* __restrict__ kv_norm,   // [rows][512]
    const __nv_bfloat16* __restrict__ k_rope,    // [rows][64]
    __nv_bfloat16* __restrict__ pool,            // [slots][576]
    const long long* __restrict__ slots) {
    const unsigned row = blockIdx.x;
    const long long slot = slots[row];
    if (slot < 0) return;
    __nv_bfloat16* dst = pool + static_cast<unsigned long long>(slot) * LING_ROW;
    for (unsigned i = threadIdx.x; i < LING_ROW; i += blockDim.x)
        dst[i] = i < LING_LATENT ? kv_norm[row * LING_LATENT + i]
                                 : k_rope[row * LING_ROPE + i - LING_LATENT];
}

// ---- Query absorption -------------------------------------------------------

__device__ __forceinline__ float2 bf2(unsigned w) {
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

#define ABSORB_ROWS 8u

// kv_b key rows h*256 + d -> kb_t [h*512 + c][d], once at load, so each
// absorbed output row is one contiguous 256-byte read. grid (32 * 512 / 256),
// block 256.
extern "C" __global__ void ling_kb_transpose(const __nv_bfloat16* __restrict__ kv_b,
                                             __nv_bfloat16* __restrict__ out) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;  // h * 512 + c
    const unsigned h = row / LING_LATENT, c = row % LING_LATENT;
    for (unsigned d = 0; d < LING_NOPE; ++d)
        out[row * LING_NOPE + d] = kv_b[(h * LING_KVB + d) * LING_LATENT + c];
}

// grid (32 * 512 / 64, ceil(rows / 8)), block 256: warp w of the block owns
// latent output rows base + w * 8 .. +7 of one head; lane l reads query
// channels 4l .. 4l + 3 of each row (coalesced) and the warp reduces in a
// fixed shuffle order.
extern "C" __global__ void __launch_bounds__(256) ling_q_absorb(
    const __nv_bfloat16* __restrict__ q_full,    // [rows][32][192], rope rotated
    const __nv_bfloat16* __restrict__ kb_t,      // [32 * 512][128]
    __nv_bfloat16* __restrict__ q_abs,           // [rows][32][576]
    const unsigned rows) {
    const unsigned base = blockIdx.x * 64, h = base / LING_LATENT, r0 = blockIdx.y * ABSORB_ROWS;
    const unsigned n = min(ABSORB_ROWS, rows - r0);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float q[ABSORB_ROWS][4];
#pragma unroll
    for (unsigned r = 0; r < ABSORB_ROWS; ++r) {
        const uint2 v = r < n ? *reinterpret_cast<const uint2*>(
                                    q_full + ((r0 + r) * LING_HEADS + h) * LING_QK + 4 * lane)
                              : make_uint2(0, 0);
        const float2 a = bf2(v.x), b = bf2(v.y);
        q[r][0] = a.x, q[r][1] = a.y, q[r][2] = b.x, q[r][3] = b.y;
    }
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = base + warp * 8 + i, c = row % LING_LATENT;
        const uint2 wv = reinterpret_cast<const uint2*>(kb_t + static_cast<size_t>(row) * LING_NOPE)[lane];
        const float2 w0 = bf2(wv.x), w1 = bf2(wv.y);
#pragma unroll
        for (unsigned r = 0; r < ABSORB_ROWS; ++r) {
            if (r >= n) break;
            float acc = q[r][0] * w0.x;
            acc += q[r][1] * w0.y;
            acc += q[r][2] * w1.x;
            acc += q[r][3] * w1.y;
            acc += __shfl_xor_sync(0xffffffffu, acc, 16);
            acc += __shfl_xor_sync(0xffffffffu, acc, 8);
            acc += __shfl_xor_sync(0xffffffffu, acc, 4);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            if (lane == 0) q_abs[((r0 + r) * LING_HEADS + h) * LING_ROW + c] = __float2bfloat16(acc);
        }
    }
    if (base % LING_LATENT == 0)
        for (unsigned i = threadIdx.x; i < n * LING_ROPE; i += blockDim.x) {
            const unsigned r = i / LING_ROPE, j = i % LING_ROPE;
            q_abs[((r0 + r) * LING_HEADS + h) * LING_ROW + LING_LATENT + j] =
                q_full[((r0 + r) * LING_HEADS + h) * LING_QK + LING_NOPE + j];
        }
}

// ---- NVFP4 absorption (Spark decode precision) -------------------------------
//
// Under NVFP4 decode precision the two halves of kv_b_proj are also kept as
// NVFP4 (`quantize_bf16_to_nvfp4`: groups of 16 along the reduction, E4M3
// group scales, one FP32 scale per matrix, low nibble first): the key half as
// kb_t [32 * 512][128], the value half [32 * 128][512]. The kernels mirror the
// BF16 ones (one warp per output row, coalesced, a fixed shuffle order); the
// E2M1 table sits in shared memory because per-thread nibbles would serialize
// constant memory.

__device__ __constant__ float LING_E2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                               -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float ling_e4m3(unsigned char b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3));
}

// grid (32 * 512 / 128, ceil(rows / 8)), block 128: thread t owns latent
// output row h * 512 + c and reduces its 128 query channels in order.
extern "C" __global__ void __launch_bounds__(128) ling_q_absorb_nvfp4(
    const __nv_bfloat16* __restrict__ q_full,    // [rows][32][192], rope rotated
    const unsigned char* __restrict__ packed,    // [32 * 512][64]
    const unsigned char* __restrict__ scales,    // [32 * 512][8]
    const float scale2,
    __nv_bfloat16* __restrict__ q_abs,           // [rows][32][576]
    const unsigned rows) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;  // h * 512 + c
    const unsigned h = row / LING_LATENT, c = row % LING_LATENT, r0 = blockIdx.y * ABSORB_ROWS;
    const unsigned n = min(ABSORB_ROWS, rows - r0);
    __shared__ float q[ABSORB_ROWS][LING_NOPE];
    __shared__ float lut[16];  // per-thread nibbles: constant memory would serialize
    if (threadIdx.x < 16) lut[threadIdx.x] = LING_E2M1[threadIdx.x];
    for (unsigned i = threadIdx.x; i < n * LING_NOPE; i += blockDim.x)
        q[i / LING_NOPE][i % LING_NOPE] =
            __bfloat162float(q_full[((r0 + i / LING_NOPE) * LING_HEADS + h) * LING_QK + i % LING_NOPE]);
    const uint4* src = reinterpret_cast<const uint4*>(packed + static_cast<size_t>(row) * 64);
    const uint4 w4[4] = {src[0], src[1], src[2], src[3]};
    const uint2 sc = *reinterpret_cast<const uint2*>(scales + static_cast<size_t>(row) * 8);
    __syncthreads();
    float w[LING_NOPE];
#pragma unroll
    for (unsigned g = 0; g < 8; ++g) {
        const float s = ling_e4m3(static_cast<unsigned char>((g < 4 ? sc.x : sc.y) >> (8 * (g % 4)))) * scale2;
        const unsigned words[2] = {g % 2 ? w4[g / 2].z : w4[g / 2].x, g % 2 ? w4[g / 2].w : w4[g / 2].y};
#pragma unroll
        for (unsigned j = 0; j < 16; ++j) w[g * 16 + j] = lut[(words[j / 8] >> (4 * (j % 8))) & 0xf] * s;
    }
    for (unsigned r = 0; r < n; ++r) {
        float acc = 0.f;
#pragma unroll
        for (unsigned d = 0; d < LING_NOPE; ++d) acc += q[r][d] * w[d];
        __nv_bfloat16* out = q_abs + ((r0 + r) * LING_HEADS + h) * LING_ROW;
        out[c] = __float2bfloat16(acc);
        if (c < LING_ROPE) out[LING_LATENT + c] = q_full[((r0 + r) * LING_HEADS + h) * LING_QK + LING_NOPE + c];
    }
}

// grid (32 * 128 / 16, ceil(rows / 8)), block 256: as ling_v_up; lane l
// reads the eight bytes holding latent channels 16l .. 16l + 15 (group l).
extern "C" __global__ void __launch_bounds__(256) ling_v_up_nvfp4(
    const __nv_bfloat16* __restrict__ latent,    // [rows][32][512]
    const unsigned char* __restrict__ packed,    // [32 * 128][256]
    const unsigned char* __restrict__ scales,    // [32 * 128][32]
    const float scale2,
    __nv_bfloat16* __restrict__ out,             // [rows][32 * 128]
    const unsigned rows) {
    const unsigned base = blockIdx.x * 16, h = base / LING_V, r0 = blockIdx.y * ABSORB_ROWS;
    const unsigned n = min(ABSORB_ROWS, rows - r0);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float lat[ABSORB_ROWS][LING_LATENT];
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = LING_E2M1[threadIdx.x];
    for (unsigned i = threadIdx.x; i < n * LING_LATENT; i += blockDim.x)
        lat[i / LING_LATENT][i % LING_LATENT] =
            __bfloat162float(latent[((r0 + i / LING_LATENT) * LING_HEADS + h) * LING_LATENT + i % LING_LATENT]);
    __syncthreads();
    for (unsigned i = 0; i < 2; ++i) {
        const unsigned row = base + warp * 2 + i, v = row % LING_V;
        const uint2 word = reinterpret_cast<const uint2*>(packed + static_cast<size_t>(row) * 256)[lane];
        const float s = ling_e4m3(scales[static_cast<size_t>(row) * 32 + lane]) * scale2;
        float w[16];
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            w[j] = lut[(word.x >> (4 * j)) & 0xf] * s;
            w[8 + j] = lut[(word.y >> (4 * j)) & 0xf] * s;
        }
#pragma unroll
        for (unsigned r = 0; r < ABSORB_ROWS; ++r) {
            if (r >= n) break;
            float acc = 0.f;
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) acc += w[j] * lat[r][16 * lane + j];
            acc += __shfl_xor_sync(0xffffffffu, acc, 16);
            acc += __shfl_xor_sync(0xffffffffu, acc, 8);
            acc += __shfl_xor_sync(0xffffffffu, acc, 4);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            if (lane == 0) out[(r0 + r) * LING_HEADS * LING_V + h * LING_V + v] = __float2bfloat16(acc);
        }
    }
}

// ---- Absorbed paged decode ----------------------------------------------------

#define DEC_SPLITS 64u
#define DEC_MIN_CHUNK 64u
#define DEC_TILE 32u
// Shared row stride: 576 values + 8 of padding (1,168 bytes, an odd number of
// 16-byte units) so ldmatrix rows fall in different banks.
#define DEC_STRIDE (LING_ROW + 8u)
#define DEC_P_STRIDE (DEC_TILE + 8u)
// Dynamic shared memory: two key tiles, two partial score halves, P, and
// per-head running max, sum and correction.
#define LING_DECODE_SMEM (2u * DEC_TILE * DEC_STRIDE * 2u + 2u * LING_HEADS * DEC_TILE * 4u + \
                          LING_HEADS * DEC_P_STRIDE * 2u + 3u * LING_HEADS * 4u)

// Keys per split, from the sequence length alone.
__device__ __forceinline__ unsigned ling_decode_chunk(unsigned len) {
    unsigned chunk = (len + DEC_SPLITS - 1) / DEC_SPLITS;
    chunk = chunk < DEC_MIN_CHUNK ? DEC_MIN_CHUNK : chunk;
    return (chunk + DEC_TILE - 1) / DEC_TILE * DEC_TILE;
}


__device__ __forceinline__ void ling_cp16(void* dst, const void* src, bool valid) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ling_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// grid (DEC_SPLITS, rows), block 256, LING_DECODE_SMEM dynamic shared bytes.
// Tensor-core absorbed attention over one split of one row's latent cache:
//   scores [32 heads x 32 keys] = q_abs [32 x 576] . keys^T: warp w owns
//     heads (w & 1) * 16.., keys ((w >> 1) & 1) * 16.., and half (w >> 2) of
//     the 576 channels, its query fragments held in registers;
//   one pass of the online softmax per head over the tile (fixed order);
//   latent [32 x 512] += P . keys[:, 0:512]: warp w owns heads (w & 1) * 16..
//     and channels (w >> 1) * 128...
// Two tiles are in flight: the next one loads while this one computes.
extern "C" __global__ void __launch_bounds__(256) ling_latent_decode(
    const __nv_bfloat16* __restrict__ q_abs,     // [rows][32][576]
    const __nv_bfloat16* __restrict__ pool,      // [blocks][block_size][576]
    const int* __restrict__ block_table,         // [rows][bt_stride] or shared
    const int* __restrict__ seq_lens,            // [rows]
    const unsigned bt_stride,                    // 0: every row shares row 0's table
    const unsigned block_size,
    const float scale,
    float* __restrict__ part_o,                  // [rows][SPLITS][32][512]
    float* __restrict__ part_ml) {               // [rows][SPLITS][32][2]
    const unsigned split = blockIdx.x, row = blockIdx.y;
    const unsigned len = static_cast<unsigned>(seq_lens[row]);
    const unsigned chunk = ling_decode_chunk(len);
    const unsigned begin = split * chunk;
    if (begin >= len) return;
    const unsigned end = min(begin + chunk, len);
    const unsigned tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned g = lane / 4, tig = lane % 4;
    const int* table = block_table + row * bt_stride;

    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* tiles = reinterpret_cast<__nv_bfloat16*>(smem);
    float* part = reinterpret_cast<float*>(smem + 2u * DEC_TILE * DEC_STRIDE * 2u);  // [2][32][32]
    __nv_bfloat16* P = reinterpret_cast<__nv_bfloat16*>(part + 2u * LING_HEADS * DEC_TILE);
    float* run_m = reinterpret_cast<float*>(P + LING_HEADS * DEC_P_STRIDE);
    float* run_l = run_m + LING_HEADS;
    float* corr = run_l + LING_HEADS;
    if (tid < LING_HEADS) {
        run_m[tid] = -CUDART_INF_F;
        run_l[tid] = 0.f;
    }

    // Query fragments: heads m0..m0+15, channel k-steps kh*18 .. kh*18+17.
    const unsigned m0 = (warp & 1u) * 16u, n0 = ((warp >> 1) & 1u) * 16u, kh = warp >> 2;
    unsigned qa[18][4];
    {
        const __nv_bfloat16* q = q_abs + static_cast<size_t>(row) * LING_HEADS * LING_ROW;
#pragma unroll
        for (unsigned ks = 0; ks < 18; ++ks) {
            const unsigned k = (kh * 18u + ks) * 16u + tig * 2u;
            const __nv_bfloat16* r0 = q + (m0 + g) * LING_ROW + k;
            const __nv_bfloat16* r1 = q + (m0 + g + 8u) * LING_ROW + k;
            qa[ks][0] = *reinterpret_cast<const unsigned*>(r0);
            qa[ks][1] = *reinterpret_cast<const unsigned*>(r1);
            qa[ks][2] = *reinterpret_cast<const unsigned*>(r0 + 8);
            qa[ks][3] = *reinterpret_cast<const unsigned*>(r1 + 8);
        }
    }
    float acc[16][4];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    auto load = [&](unsigned buf, unsigned t0) {
        __nv_bfloat16* dst = tiles + buf * DEC_TILE * DEC_STRIDE;
        for (unsigned i = tid; i < DEC_TILE * (LING_ROW / 8u); i += blockDim.x) {
            const unsigned k = i / (LING_ROW / 8u), c = (i % (LING_ROW / 8u)) * 8u, t = t0 + k;
            const bool valid = t < end;
            const unsigned tt = valid ? t : begin;
            const unsigned long long slot =
                static_cast<unsigned long long>(table[tt / block_size]) * block_size + tt % block_size;
            ling_cp16(dst + k * DEC_STRIDE + c, pool + slot * LING_ROW + c, valid);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    load(0, begin);
    unsigned buf = 0;
    for (unsigned t0 = begin; t0 < end; t0 += DEC_TILE, buf ^= 1u) {
        if (t0 + DEC_TILE < end) {
            load(buf ^ 1u, t0 + DEC_TILE);
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
        const __nv_bfloat16* K = tiles + buf * DEC_TILE * DEC_STRIDE;

        // Scores: two key tiles of 8 per warp, half the channels.
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (unsigned ks = 0; ks < 18; ++ks) {
            const unsigned kb = (kh * 18u + ks) * 16u;
            const unsigned mi = lane / 8u, r = lane % 8u;
            const unsigned key = n0 + (mi >> 1) * 8u + r;
            const unsigned addr = static_cast<unsigned>(
                __cvta_generic_to_shared(K + key * DEC_STRIDE + kb + (mi & 1u) * 8u));
            unsigned b[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                         : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
            ling_mma(s[0], qa[ks], b[0], b[1]);
            ling_mma(s[1], qa[ks], b[2], b[3]);
        }
#pragma unroll
        for (unsigned nt = 0; nt < 2; ++nt) {
            float* dst = part + (kh * LING_HEADS) * DEC_TILE;
            const unsigned col = n0 + nt * 8u + tig * 2u;
            dst[(m0 + g) * DEC_TILE + col] = s[nt][0];
            dst[(m0 + g) * DEC_TILE + col + 1] = s[nt][1];
            dst[(m0 + g + 8u) * DEC_TILE + col] = s[nt][2];
            dst[(m0 + g + 8u) * DEC_TILE + col + 1] = s[nt][3];
        }
        __syncthreads();

        // Online softmax: eight threads per head, four keys each.
        {
            const unsigned head = tid / 8u, k0 = (tid % 8u) * 4u;
            float v[4], top = -CUDART_INF_F;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned k = k0 + j;
                v[j] = t0 + k < end
                    ? (part[head * DEC_TILE + k] + part[(LING_HEADS + head) * DEC_TILE + k]) * scale
                    : -CUDART_INF_F;
                top = fmaxf(top, v[j]);
            }
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 4));
            const float old = run_m[head], next = fmaxf(old, top);
            float sum = 0.f;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const float p = expf(v[j] - next);
                const __nv_bfloat16 pb = __float2bfloat16(p);
                P[head * DEC_P_STRIDE + k0 + j] = pb;
                sum += __bfloat162float(pb);
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            __syncwarp();
            if (tid % 8u == 0) {
                const float c = expf(old - next);
                corr[head] = c;
                run_l[head] = run_l[head] * c + sum;
                run_m[head] = next;
            }
        }
        __syncthreads();

        // latent += P . V for heads m0.., channels (warp >> 1) * 128...
        {
            const float c0 = corr[m0 + g], c1 = corr[m0 + g + 8u];
#pragma unroll
            for (unsigned nt = 0; nt < 16; ++nt) {
                acc[nt][0] *= c0;
                acc[nt][1] *= c0;
                acc[nt][2] *= c1;
                acc[nt][3] *= c1;
            }
            const unsigned d0 = (warp >> 1) * 128u;
#pragma unroll
            for (unsigned ks = 0; ks < DEC_TILE / 16u; ++ks) {
                const unsigned mi = lane / 8u, r = lane % 8u;
                unsigned a[4];
                const unsigned pa = static_cast<unsigned>(__cvta_generic_to_shared(
                    P + (m0 + (mi & 1u) * 8u + r) * DEC_P_STRIDE + ks * 16u + (mi >> 1) * 8u));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(pa));
#pragma unroll
                for (unsigned np = 0; np < 8; ++np) {
                    const unsigned key = ks * 16u + (mi & 1u) * 8u + r;
                    const unsigned dim = d0 + (np * 2u + (mi >> 1)) * 8u;
                    const unsigned va = static_cast<unsigned>(__cvta_generic_to_shared(K + key * DEC_STRIDE + dim));
                    unsigned b[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
                                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(va));
                    ling_mma(acc[np * 2u], a, b[0], b[1]);
                    ling_mma(acc[np * 2u + 1u], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }

    const unsigned base = (row * DEC_SPLITS + split) * LING_HEADS;
    const unsigned d0 = (warp >> 1) * 128u;
#pragma unroll
    for (unsigned nt = 0; nt < 16; ++nt) {
        const unsigned col = d0 + nt * 8u + tig * 2u;
        *reinterpret_cast<float2*>(part_o + (base + m0 + g) * LING_LATENT + col) = make_float2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<float2*>(part_o + (base + m0 + g + 8u) * LING_LATENT + col) = make_float2(acc[nt][2], acc[nt][3]);
    }
    if (tid < LING_HEADS) {
        part_ml[(base + tid) * 2] = run_m[tid];
        part_ml[(base + tid) * 2 + 1] = run_l[tid];
    }
}

// grid (heads, rows), block 128: thread i owns latent channels 4i..4i+3.
extern "C" __global__ void __launch_bounds__(128) ling_latent_decode_merge(
    const float* __restrict__ part_o,
    const float* __restrict__ part_ml,
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ latent) {        // [rows][32][512]
    const unsigned head = blockIdx.x, row = blockIdx.y;
    const unsigned len = static_cast<unsigned>(seq_lens[row]);
    const unsigned chunk = ling_decode_chunk(len);
    const unsigned splits = (len + chunk - 1) / chunk;
    float top = -CUDART_INF_F;
    for (unsigned s = 0; s < splits; ++s)
        top = fmaxf(top, part_ml[((row * DEC_SPLITS + s) * LING_HEADS + head) * 2]);
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    float total = 0.f;
    for (unsigned s = 0; s < splits; ++s) {
        const unsigned base = (row * DEC_SPLITS + s) * LING_HEADS + head;
        const float w = expf(part_ml[base * 2] - top);
        total += part_ml[base * 2 + 1] * w;
        const float4 v = reinterpret_cast<const float4*>(part_o + base * LING_LATENT)[threadIdx.x];
        o.x += v.x * w;
        o.y += v.y * w;
        o.z += v.z * w;
        o.w += v.w * w;
    }
    const float inv = total > 0.f ? 1.f / total : 0.f;
    __nv_bfloat16* out = latent + (row * LING_HEADS + head) * LING_LATENT + 4 * threadIdx.x;
    out[0] = __float2bfloat16(o.x * inv);
    out[1] = __float2bfloat16(o.y * inv);
    out[2] = __float2bfloat16(o.z * inv);
    out[3] = __float2bfloat16(o.w * inv);
}

// ---- Value up-projection ------------------------------------------------------

// grid (32 * 128 / 16, ceil(rows / 8)), block 256: warp w owns value output
// rows base + w * 2 .. +1 of one head; lane l reads latent channels
// 16l .. 16l + 15 of each row (coalesced) and the warp reduces in a fixed
// shuffle order.
extern "C" __global__ void __launch_bounds__(256) ling_v_up(
    const __nv_bfloat16* __restrict__ latent,    // [rows][32][512]
    const __nv_bfloat16* __restrict__ kv_b,      // [32*256][512]
    __nv_bfloat16* __restrict__ out,             // [rows][32*128]
    const unsigned rows) {
    const unsigned base = blockIdx.x * 16, h = base / LING_V, r0 = blockIdx.y * ABSORB_ROWS;
    const unsigned n = min(ABSORB_ROWS, rows - r0);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float lat[ABSORB_ROWS][LING_LATENT];
    for (unsigned i = threadIdx.x; i < n * LING_LATENT; i += blockDim.x)
        lat[i / LING_LATENT][i % LING_LATENT] =
            __bfloat162float(latent[((r0 + i / LING_LATENT) * LING_HEADS + h) * LING_LATENT + i % LING_LATENT]);
    __syncthreads();
    for (unsigned i = 0; i < 2; ++i) {
        const unsigned row = base + warp * 2 + i, v = row % LING_V;
        const uint4* w = reinterpret_cast<const uint4*>(
            kv_b + (static_cast<size_t>(h) * LING_KVB + LING_NOPE + v) * LING_LATENT + 16 * lane);
        const uint4 w0 = w[0], w1 = w[1];
        const unsigned words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (unsigned r = 0; r < ABSORB_ROWS; ++r) {
            if (r >= n) break;
            float acc = 0.f;
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const float2 wv = bf2(words[j]);
                acc += wv.x * lat[r][16 * lane + 2 * j];
                acc += wv.y * lat[r][16 * lane + 2 * j + 1];
            }
            acc += __shfl_xor_sync(0xffffffffu, acc, 16);
            acc += __shfl_xor_sync(0xffffffffu, acc, 8);
            acc += __shfl_xor_sync(0xffffffffu, acc, 4);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            if (lane == 0) out[(r0 + r) * LING_HEADS * LING_V + h * LING_V + v] = __float2bfloat16(acc);
        }
    }
}

// ---- Prompt expansion -----------------------------------------------------------

// Rows [first, first + n) of one sequence's cache into a contiguous latent
// [n][512] and RoPE key [n][64]. grid n, block 64.
extern "C" __global__ void ling_latent_gather(
    const __nv_bfloat16* __restrict__ pool,
    const int* __restrict__ block_table,
    const unsigned block_size,
    const unsigned first,
    __nv_bfloat16* __restrict__ latent,
    __nv_bfloat16* __restrict__ rope) {
    const unsigned k = blockIdx.x, t = first + k;
    const unsigned long long slot =
        static_cast<unsigned long long>(block_table[t / block_size]) * block_size + t % block_size;
    const uint4* src = reinterpret_cast<const uint4*>(pool + slot * LING_ROW);
    // 72 sixteen-byte words per row: 64 latent, 8 RoPE.
    for (unsigned i = threadIdx.x; i < LING_ROW / 8; i += blockDim.x) {
        const uint4 v = src[i];
        if (i < LING_LATENT / 8)
            reinterpret_cast<uint4*>(latent + k * LING_LATENT)[i] = v;
        else
            reinterpret_cast<uint4*>(rope + k * LING_ROPE)[i - LING_LATENT / 8] = v;
    }
}

// Expanded rows in the layout ling_mla_prefill reads: K = 32 x 128 NoPE
// channels then the shared 64-channel RoPE tail, V = 32 x 128 channels (its
// row keeps K's width; the tail is never read).
// grid n, block 256; row k lands at scratch row `first + k`.
extern "C" __global__ void ling_latent_expand(
    const __nv_bfloat16* __restrict__ kv_b,      // [n][32*256]
    const __nv_bfloat16* __restrict__ rope,      // [n][64]
    __nv_bfloat16* __restrict__ k_rows,
    __nv_bfloat16* __restrict__ v_rows,
    const unsigned first) {
    const unsigned k = blockIdx.x;
    constexpr unsigned wide = LING_HEADS * LING_NOPE, stride = wide + LING_ROPE;
    const unsigned long long dst = static_cast<unsigned long long>(first + k) * stride;
    const __nv_bfloat16* src = kv_b + static_cast<unsigned long long>(k) * LING_HEADS * LING_KVB;
    for (unsigned i = threadIdx.x; i < wide; i += blockDim.x) {
        const unsigned h = i / LING_NOPE, d = i % LING_NOPE;
        k_rows[dst + i] = src[h * LING_KVB + d];
        v_rows[dst + i] = src[h * LING_KVB + LING_NOPE + d];
    }
    if (threadIdx.x < LING_ROPE) k_rows[dst + wide + threadIdx.x] = rope[k * LING_ROPE + threadIdx.x];
}

// Fold one key tile's normalized output and (max, sum) into the running
// result. The last tile writes the BF16 [rows][32][128] attention output.
// grid (heads, rows), block 64: thread i owns channels 2i, 2i + 1 of 128.
extern "C" __global__ void ling_prompt_tile_merge(
    const float* __restrict__ tile_o,            // [rows][32][128]
    const float* __restrict__ tile_ml,           // [rows][32][2]
    float* __restrict__ acc_o,
    float* __restrict__ acc_ml,
    __nv_bfloat16* __restrict__ out,             // [rows][32][128]
    const unsigned first,
    const unsigned last) {
    const unsigned head = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
    const unsigned at = row * LING_HEADS + head;
    const float2 t = reinterpret_cast<const float2*>(tile_o + at * LING_V)[i];
    float2 o = t;
    float m = tile_ml[at * 2], l = tile_ml[at * 2 + 1];
    if (!first) {
        const float2 a = reinterpret_cast<const float2*>(acc_o + at * LING_V)[i];
        const float am = acc_ml[at * 2], al = acc_ml[at * 2 + 1];
        const float top = fmaxf(am, m);
        const float wa = al * expf(am - top), wt = l * expf(m - top);
        const float inv = 1.f / (wa + wt);
        o = make_float2((a.x * wa + t.x * wt) * inv, (a.y * wa + t.y * wt) * inv);
        m = top;
        l = wa + wt;
    }
    if (last) {
        __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(out + at * LING_V);
        dst[i] = __floats2bfloat162_rn(o.x, o.y);
        return;
    }
    reinterpret_cast<float2*>(acc_o + at * LING_V)[i] = o;
    __syncthreads();
    if (i == 0) {
        acc_ml[at * 2] = m;
        acc_ml[at * 2 + 1] = l;
    }
}
