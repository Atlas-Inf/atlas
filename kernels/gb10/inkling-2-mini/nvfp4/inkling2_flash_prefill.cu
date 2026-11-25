// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 prefill attention, one sequence's rows over its paged BF16 KV cache, GQA
// sharing each K/V tile: a CTA takes one KV head and 64 query tokens, its 12 warps the
// group's 3 query heads x 4 tiles of 16 rows, so every K/V tile read from the cache serves
// 192 query rows (the per-head 32-row kernel re-read the whole context for every 32 rows
// of every head). Same numerics as inkling2_prefill_attn (common/prefill_paged_compute.cuh
// with INKLING2_PREFILL_BIAS / INKLING2_ROW_SINKS): score = q.k / sqrt(d) + bias for
// q - k < rel_extent, causal and sliding-window masks, online softmax in fp32 with P
// rounded to BF16 for the PV MMA, one sink logit per (row, head) in the normaliser. Key
// tiles wholly outside a tile's window or causal range are skipped. The bias is
// BF16(sum_j rel[row, head, j] * rel_proj[j, q - k]) computed where it is added, in
// inkling2_rel_bias_table's order (the per-head kernel's table is
// q_len x heads x rel_extent BF16, written and read back every visit).
//
// grid (num_kv_heads, ceil(q_len / 64)), block 384. HDIM 128, group 3.

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// AMD (strix-hip) port of the flash body below onto wave32 bf16 WMMA
// (gfx1151). Same tile geometry and semantics: CTA = (KV head, 64 query
// rows), 12 warps = 3 query heads x 4 row-tiles of 16. K/V smem tiles are
// shared by all three query heads of the group — one read serves 192 rows.
// Register online softmax with the fragment layout's half-warp shfl
// reductions (score element e of lane l is row 2e+(l>>4), col l&15), the
// same fp32 sum-j + BF16 round bias, causal/SWA masks, per-token sinks, and
// fp16 P for the PV WMMA — the numerics of the per-head body
// (common/prefill_paged_compute.cuh) at 3x fewer K/V reads.
#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define FP_BQ 64
#define FP_BC 32
#define FP_GROUP 3
#define FP_HD 128
#define FP_STRIDE (FP_HD + 8)
#define FP_WARPS (FP_GROUP * (FP_BQ / 16))
#define FP_THREADS (FP_WARPS * 32)
#define FP_D_REL 16

typedef __bf16 fp_v16bf __attribute__((ext_vector_type(16)));
typedef __fp16 fp_v16h __attribute__((ext_vector_type(16)));
typedef float fp_v8f __attribute__((ext_vector_type(8)));

// hipcc's __syncwarp lowers to a bare wave barrier with no memory ordering, so
// LDS writes are not guaranteed visible to the wave's other lanes (the pattern
// inkling2_moe_decode.cu documents). The P staging below needs the fenced form.
__device__ __forceinline__ void fp_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

extern "C" __global__ void __launch_bounds__(FP_THREADS, 1) inkling2_flash_prefill(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const __nv_bfloat16* __restrict__ rel,
    const float* __restrict__ rel_proj_t,
    const __nv_bfloat16* __restrict__ sinks,
    const unsigned int q_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const unsigned int sliding_window,
    const unsigned int rel_extent,
    const float inv_sqrt_d
) {
    atlas_pdl_enter();
    __shared__ __align__(16) __nv_bfloat16 Ks[FP_BC][FP_STRIDE];
    // V staged fp16 once at tile load — the same bf16->fp16 conversion the
    // per-head kernel does in registers per MMA; keeps conversion temps out
    // of the PV loop's live set (hipcc 7.16 spilled otherwise).
    __shared__ __align__(16) __fp16 Vs[FP_BC][FP_STRIDE];
    __shared__ __align__(16) float Rs[FP_GROUP * FP_BQ][FP_D_REL];
    __shared__ __align__(16) __fp16 Ps[FP_WARPS][16][FP_BC + 8];
    const unsigned int kvh = blockIdx.x, q0 = blockIdx.y * FP_BQ;
    const unsigned int tid = threadIdx.x, warp = tid >> 5;
    const unsigned int lane = tid & 31, lane_lo = lane & 15, lane_hi = lane >> 4;
    const unsigned int head = kvh * FP_GROUP + warp / (FP_BQ / 16);
    const unsigned int r_base = q0 + (warp % (FP_BQ / 16)) * 16;
    const unsigned int kv_len = q_offset + q_len;

    const unsigned int q_last = q_offset + min(q0 + FP_BQ, q_len) - 1;
    const unsigned int k_end = q_last + 1;
    unsigned int k_begin = 0;
    if (sliding_window > 0 && q_offset + q0 + 1 > sliding_window)
        k_begin = q_offset + q0 + 1 - sliding_window;
    k_begin = (k_begin / FP_BC) * FP_BC;
    const unsigned int tiles = (k_end - k_begin + FP_BC - 1) / FP_BC;

    // this lane's Q row (clamped like the PTX kernel); A frags read 16
    // contiguous bf16 per k16 step straight from global — the tile is 4 KB
    // and stays L1/L2-hot across the KV loop.
    const unsigned long long qs = (unsigned long long)num_q_heads * FP_HD;
    const __nv_bfloat16* q_row = Q + (unsigned long long)min(r_base + lane_lo, q_len - 1) * qs + head * FP_HD;

    const unsigned long long page = (unsigned long long)block_size * num_kv_heads * FP_HD;
    const unsigned long long rowp = (unsigned long long)num_kv_heads * FP_HD;
    auto issue = [&](unsigned int kt) {
        for (unsigned int i = tid; i < FP_BC * FP_HD / 8; i += FP_THREADS) {
            const unsigned int row = i / (FP_HD / 8), col = (i % (FP_HD / 8)) * 8;
            const unsigned int pos = kt + row;
            const bool ok = pos < kv_len;
            const unsigned long long off =
                ok ? (unsigned long long)block_table[pos / block_size] * page +
                     (unsigned long long)(pos % block_size) * rowp + kvh * FP_HD + col
                   : 0ull;
            *(uint4*)&Ks[row][col] = ok ? *(const uint4*)(K_cache + off) : make_uint4(0, 0, 0, 0);
            const uint4 v = ok ? *(const uint4*)(V_cache + off) : make_uint4(0, 0, 0, 0);
            const __nv_bfloat16* vb = (const __nv_bfloat16*)&v;
            __fp16* vh = (__fp16*)&Vs[row][col];
#pragma unroll
            for (int j = 0; j < 8; j++) vh[j] = (__fp16)__bfloat162float(vb[j]);
        }
    };

    // this CTA's rows' rel features (Rs[head_local*64 + row][j], fp32)
    for (unsigned int i = tid; i < FP_GROUP * FP_BQ * FP_D_REL; i += FP_THREADS) {
        const unsigned int hr = i / FP_D_REL, j = i % FP_D_REL;
        const unsigned int r = min(q0 + hr % FP_BQ, q_len - 1), hh = kvh * FP_GROUP + hr / FP_BQ;
        Rs[hr][j] = __bfloat162float(rel[((unsigned long long)r * num_q_heads + hh) * FP_D_REL + j]);
    }
    const unsigned int rs_base = (warp / (FP_BQ / 16)) * FP_BQ + (r_base - q0);

    fp_v8f acc_o[FP_HD / 16];
    float m_st[8], l_st[8];
#pragma unroll
    for (int i = 0; i < FP_HD / 16; i++)
        acc_o[i] = fp_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int e = 0; e < 8; e++) { m_st[e] = -1e30f; l_st[e] = 0.0f; }

    issue(k_begin);
    __syncthreads();
    for (unsigned int t = 0; t < tiles; t++) {
        const unsigned int kt = k_begin + t * FP_BC;

        // S = Q K^T over the tile: acc_s[nt][e] -> row 2e+lane_hi, key nt*16+lane_lo
        fp_v8f acc_s[FP_BC / 16];
#pragma unroll
        for (int n = 0; n < FP_BC / 16; n++)
            acc_s[n] = fp_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (unsigned int ks = 0; ks < FP_HD / 16; ks++) {
            fp_v16bf a;
            const uint4* qp = (const uint4*)(q_row + ks * 16);
            uint4 lo = qp[0], hi = qp[1];
            __builtin_memcpy(&a, &lo, 16);
            __builtin_memcpy((char*)&a + 16, &hi, 16);
#pragma unroll
            for (int nt = 0; nt < FP_BC / 16; nt++) {
                fp_v16bf bb;
                const __nv_bfloat16* kr = &Ks[nt * 16 + lane_lo][ks * 16];
#pragma unroll
                for (int k = 0; k < 16; k++) bb[k] = (__bf16)(float)kr[k];
                acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
            }
        }

        // scale, bias, masks — same conditions and order as the PTX kernel
        const unsigned int wq = q_offset + r_base;
        const bool plain = r_base + 15 < q_len && kt + FP_BC <= wq && wq - (kt + FP_BC - 1) >= rel_extent
                         && (sliding_window == 0 || wq + 15 - kt < sliding_window);
        if (plain) {
#pragma unroll
            for (int n = 0; n < FP_BC / 16; n++)
#pragma unroll
                for (int e = 0; e < 8; e++) acc_s[n][e] *= inv_sqrt_d;
        } else {
#pragma unroll
            for (int n = 0; n < FP_BC / 16; n++) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    const unsigned int row = 2 * e + lane_hi;
                    const unsigned int key = kt + n * 16 + lane_lo;
                    const unsigned int qa = wq + row;
                    float v = acc_s[n][e] * inv_sqrt_d;
                    const bool live = r_base + row < q_len && key <= qa && key < kv_len
                                    && (sliding_window == 0 || qa - key < sliding_window);
                    if (live && qa - key < rel_extent) {
                        const float4* rl = (const float4*)&Rs[rs_base + row][0];
                        const float4* pt = (const float4*)(rel_proj_t + (unsigned long long)(qa - key) * FP_D_REL);
                        float b = 0.0f;
#pragma unroll
                        for (int i = 0; i < FP_D_REL / 4; i++) {
                            const float4 w = __ldg(pt + i), x = rl[i];
                            b = __fadd_rn(b, __fmul_rn(x.x, w.x));
                            b = __fadd_rn(b, __fmul_rn(x.y, w.y));
                            b = __fadd_rn(b, __fmul_rn(x.z, w.z));
                            b = __fadd_rn(b, __fmul_rn(x.w, w.w));
                        }
                        v += __bfloat162float(__float2bfloat16(b));
                    }
                    acc_s[n][e] = live ? v : -1e30f;
                }
            }
        }

        // Per row (element e): half-warp shfl_xor max over the row's 16 keys
        // (rows sit in lane_hi halves), the online-softmax update, and the P
        // store — Ps[warp][2e+lane_hi][n*16+lane_lo] transposes the fragment
        // layout's [key=lane_lo] into the A operand's [row][key] rows. Keeping
        // tm/pv out of arrays saves ~24 live VGPRs; hipcc 7.16 spilled otherwise.
        for (int e = 0; e < 8; e++) {
            float x = fmaxf(acc_s[0][e], acc_s[FP_BC / 16 - 1][e]);
            for (int o = 1; o < 16; o <<= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffull, x, o));
            const float nm = fmaxf(m_st[e], x);
            const float sc = __expf(m_st[e] - nm);
            m_st[e] = nm;
            float rs = 0.0f;
#pragma unroll
            for (int n = 0; n < FP_BC / 16; n++) {
                const float s = acc_s[n][e];
                const float p = s <= -1e29f ? 0.0f : __expf(s - nm);
                Ps[warp][2 * e + lane_hi][n * 16 + lane_lo] = (__fp16)p;
                rs += p;
            }
            for (int o = 1; o < 16; o <<= 1) rs += __shfl_xor_sync(0xffffffffull, rs, o);
            l_st[e] = l_st[e] * sc + rs;
#pragma unroll
            for (int d = 0; d < FP_HD / 16; d++) acc_o[d][e] *= sc;
        }
        fp_wave_sync();
#pragma unroll
        for (int nt = 0; nt < FP_BC / 16; nt++) {
            fp_v16h pa;
            const __fp16* pr = &Ps[warp][lane_lo][nt * 16];
#pragma unroll
            for (int i = 0; i < 16; i++) pa[i] = pr[i];
#pragma unroll
            for (int d = 0; d < FP_HD / 16; d++) {
                fp_v16h bb;
#pragma unroll
                for (int k = 0; k < 16; k++)
                    bb[k] = Vs[nt * 16 + k][d * 16 + lane_lo];
                acc_o[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bb, acc_o[d]);
            }
        }
        __syncthreads();
        if (t + 1 < tiles) {
            issue(kt + FP_BC);
            __syncthreads();
        }
    }

    // one sink logit per (row, head) joins the normaliser
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const unsigned int row = 2 * e + lane_hi;
        const float sg = __bfloat162float(
            sinks[(unsigned long long)min(r_base + row, q_len - 1) * num_q_heads + head]);
        const float n = fmaxf(m_st[e], sg);
        const float sc = __expf(m_st[e] - n);
        l_st[e] = l_st[e] * sc + __expf(sg - n);
#pragma unroll
        for (int d = 0; d < FP_HD / 16; d++) acc_o[d][e] *= sc;
    }
    const unsigned long long os = (unsigned long long)num_q_heads * FP_HD;
#pragma unroll
    for (int d = 0; d < FP_HD / 16; d++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int row = 2 * e + lane_hi;
            if (r_base + row < q_len)
                O[(unsigned long long)(r_base + row) * os + head * FP_HD + d * 16 + lane_lo] =
                    __float2bfloat16(acc_o[d][e] / l_st[e]);
        }
    }
}
#else
#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define FP_BQ 64        // query tokens a CTA
#define FP_BC 32        // keys a tile
#define FP_GROUP 3      // query heads per KV head
#define FP_HD 128
#define FP_STRIDE (FP_HD + 8)   // BF16 row pitch in shared memory (conflict-free fragments)
#define FP_WARPS (FP_GROUP * (FP_BQ / 16))
#define FP_THREADS (FP_WARPS * 32)
#define FP_D_REL 16

__device__ __forceinline__ void fp_cp16(void* smem, const void* gmem, bool pred) {
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void fp_mma(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int fp_pack(float lo, float hi) {
    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(lo)) |
           ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}

extern "C" __global__ void __launch_bounds__(FP_THREADS, 1) inkling2_flash_prefill(
    const __nv_bfloat16* __restrict__ Q,         // [q_len, num_q_heads * 128]
    const __nv_bfloat16* __restrict__ K_cache,   // [blocks, block_size, num_kv_heads, 128]
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,               // [q_len, num_q_heads * 128]
    const int* __restrict__ block_table,         // the sequence's table
    const __nv_bfloat16* __restrict__ rel,       // [q_len, num_q_heads, FP_D_REL]
    const float* __restrict__ rel_proj_t,        // [rel_extent, FP_D_REL]
    const __nv_bfloat16* __restrict__ sinks,     // [q_len, num_q_heads]
    const unsigned int q_len,
    const unsigned int q_offset,                 // absolute position of query row 0
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const unsigned int sliding_window,           // 0 = full attention
    const unsigned int rel_extent,
    const float inv_sqrt_d
) {
    atlas_pdl_enter();
    __shared__ __align__(16) __nv_bfloat16 Ks[2][FP_BC][FP_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 Vs[2][FP_BC][FP_STRIDE];
    __shared__ __align__(16) float Rs[FP_GROUP * FP_BQ][FP_D_REL];   // rel features (head, row)
    const unsigned int kvh = blockIdx.x, q0 = blockIdx.y * FP_BQ;
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned int g8 = lane / 4, t2 = (lane % 4) * 2;
    const unsigned int head = kvh * FP_GROUP + warp / (FP_BQ / 16);
    const unsigned int r_base = q0 + (warp % (FP_BQ / 16)) * 16;   // this warp's first row
    const unsigned int r0 = r_base + g8, r1 = r0 + 8;              // this thread's two rows
    const unsigned int qa0 = q_offset + r0, qa1 = q_offset + r1;    // their absolute positions
    const unsigned int kv_len = q_offset + q_len;

    // the tile's key range: causal end, window start
    const unsigned int q_last = q_offset + min(q0 + FP_BQ, q_len) - 1;
    const unsigned int k_end = q_last + 1;
    unsigned int k_begin = 0;
    if (sliding_window > 0 && q_offset + q0 + 1 > sliding_window) k_begin = q_offset + q0 + 1 - sliding_window;
    k_begin = (k_begin / FP_BC) * FP_BC;
    const unsigned int tiles = (k_end - k_begin + FP_BC - 1) / FP_BC;

    // Q fragments of this warp's 16 rows, 8 k16 steps of 4 registers
    unsigned int qf[FP_HD / 16][4];
    {
        const unsigned long long qs = (unsigned long long)num_q_heads * FP_HD;
        const __nv_bfloat16* p0 = Q + (unsigned long long)min(r0, q_len - 1) * qs + head * FP_HD;
        const __nv_bfloat16* p1 = Q + (unsigned long long)min(r1, q_len - 1) * qs + head * FP_HD;
#pragma unroll
        for (int k = 0; k < FP_HD / 16; k++) {
            qf[k][0] = *(const unsigned int*)(p0 + k * 16 + t2);
            qf[k][1] = *(const unsigned int*)(p1 + k * 16 + t2);
            qf[k][2] = *(const unsigned int*)(p0 + k * 16 + t2 + 8);
            qf[k][3] = *(const unsigned int*)(p1 + k * 16 + t2 + 8);
        }
    }

    const unsigned long long page = (unsigned long long)block_size * num_kv_heads * FP_HD;
    const unsigned long long rowp = (unsigned long long)num_kv_heads * FP_HD;
    auto issue = [&](int buf, unsigned int kt) {
        // 32 keys x 128 dims of K and of V: 512 chunks of 16 bytes each
        for (unsigned int i = tid; i < FP_BC * FP_HD / 8; i += FP_THREADS) {
            const unsigned int row = i / (FP_HD / 8), col = (i % (FP_HD / 8)) * 8;
            const unsigned int pos = kt + row;
            const bool ok = pos < kv_len;
            const unsigned long long off = ok ? (unsigned long long)block_table[pos / block_size] * page
                                                    + (unsigned long long)(pos % block_size) * rowp + kvh * FP_HD + col
                                              : 0ull;
            fp_cp16(&Ks[buf][row][col], K_cache + off, ok);
            fp_cp16(&Vs[buf][row][col], V_cache + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[FP_HD / 8][4];
#pragma unroll
    for (int i = 0; i < FP_HD / 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    // the CTA's rows' relative-position features (read before the first tile's barrier)
    for (unsigned int i = tid; i < FP_GROUP * FP_BQ * FP_D_REL; i += FP_THREADS) {
        const unsigned int hr = i / FP_D_REL, j = i % FP_D_REL;
        const unsigned int r = min(q0 + hr % FP_BQ, q_len - 1), hh = kvh * FP_GROUP + hr / FP_BQ;
        Rs[hr][j] = __bfloat162float(rel[((unsigned long long)r * num_q_heads + hh) * FP_D_REL + j]);
    }
    const float* rs0 = Rs[(warp / (FP_BQ / 16)) * FP_BQ + (r0 - q0)];
    const float* rs1 = rs0 + 8 * FP_D_REL;

    issue(0, k_begin);
    for (unsigned int t = 0; t < tiles; t++) {
        const int buf = t & 1;
        const unsigned int kt = k_begin + t * FP_BC;
        if (t + 1 < tiles) {
            issue(buf ^ 1, kt + FP_BC);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        // S = Q K^T over this tile: 4 n8 tiles of keys
        float s[FP_BC / 8][4];
#pragma unroll
        for (int n = 0; n < FP_BC / 8; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < FP_HD / 16; k++) {
                const __nv_bfloat16* kr = &Ks[buf][n * 8 + g8][k * 16 + t2];
                fp_mma(s[n], qf[k], *(const unsigned int*)kr, *(const unsigned int*)(kr + 8));
            }
        }
        // scale, bias, masks; a tile wholly before every row of this warp, inside its
        // window and beyond the bias's reach needs only the scale
        const unsigned int wq = q_offset + r_base;   // the warp's first absolute row
        const bool plain = r_base + 15 < q_len && kt + FP_BC <= wq && wq - (kt + FP_BC - 1) >= rel_extent
                         && (sliding_window == 0 || wq + 15 - kt < sliding_window);
        if (plain) {
#pragma unroll
            for (int n = 0; n < FP_BC / 8; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) s[n][j] *= inv_sqrt_d;
        } else {
#pragma unroll
            for (int n = 0; n < FP_BC / 8; n++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const unsigned int key = kt + n * 8 + t2 + (j & 1);
                    const unsigned int qa = j < 2 ? qa0 : qa1;
                    const unsigned int r = j < 2 ? r0 : r1;
                    float v = s[n][j] * inv_sqrt_d;
                    const bool live = r < q_len && key <= qa && key < kv_len
                                    && (sliding_window == 0 || qa - key < sliding_window);
                    if (live && qa - key < rel_extent) {
                        const float4* rl = (const float4*)(j < 2 ? rs0 : rs1);
                        const float4* pt = (const float4*)(rel_proj_t + (unsigned long long)(qa - key) * FP_D_REL);
                        float b = 0.0f;
#pragma unroll
                        for (int i = 0; i < FP_D_REL / 4; i++) {
                            const float4 w = __ldg(pt + i), x = rl[i];
                            b = __fadd_rn(b, __fmul_rn(x.x, w.x));
                            b = __fadd_rn(b, __fmul_rn(x.y, w.y));
                            b = __fadd_rn(b, __fmul_rn(x.z, w.z));
                            b = __fadd_rn(b, __fmul_rn(x.w, w.w));
                        }
                        v += __bfloat162float(__float2bfloat16(b));
                    }
                    s[n][j] = live ? v : -1e30f;
                }
            }
        }
        float tmax0 = -1e30f, tmax1 = -1e30f;
#pragma unroll
        for (int n = 0; n < FP_BC / 8; n++) {
            tmax0 = fmaxf(tmax0, fmaxf(s[n][0], s[n][1]));
            tmax1 = fmaxf(tmax1, fmaxf(s[n][2], s[n][3]));
        }
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffff, tmax0, 1));
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffff, tmax0, 2));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffff, tmax1, 1));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffff, tmax1, 2));
        const float nm0 = fmaxf(m0, tmax0), nm1 = fmaxf(m1, tmax1);
        const float e0 = __expf(m0 - nm0), e1 = __expf(m1 - nm1);
        m0 = nm0;
        m1 = nm1;
        float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
        for (int n = 0; n < FP_BC / 8; n++) {
            s[n][0] = s[n][0] <= -1e29f ? 0.0f : __expf(s[n][0] - m0);
            s[n][1] = s[n][1] <= -1e29f ? 0.0f : __expf(s[n][1] - m0);
            s[n][2] = s[n][2] <= -1e29f ? 0.0f : __expf(s[n][2] - m1);
            s[n][3] = s[n][3] <= -1e29f ? 0.0f : __expf(s[n][3] - m1);
            ps0 += s[n][0] + s[n][1];
            ps1 += s[n][2] + s[n][3];
        }
        ps0 += __shfl_xor_sync(0xffffffff, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffff, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffff, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffff, ps1, 2);
        l0 = l0 * e0 + ps0;
        l1 = l1 * e1 + ps1;
#pragma unroll
        for (int i = 0; i < FP_HD / 8; i++) {
            acc[i][0] *= e0;
            acc[i][1] *= e0;
            acc[i][2] *= e1;
            acc[i][3] *= e1;
        }
        // O += P V: P (BF16) from the score fragments, V^T fragments by ldmatrix.trans
#pragma unroll
        for (int kk = 0; kk < FP_BC / 16; kk++) {
            const unsigned int pa[4] = {fp_pack(s[2 * kk][0], s[2 * kk][1]), fp_pack(s[2 * kk][2], s[2 * kk][3]),
                                        fp_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                        fp_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int d = 0; d < FP_HD / 8; d++) {
                const unsigned int addr =
                    (unsigned int)__cvta_generic_to_shared(&Vs[buf][kk * 16 + (lane % 16)][d * 8]);
                unsigned int b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b0), "=r"(b1) : "r"(addr));
                fp_mma(acc[d], pa, b0, b1);
            }
        }
        __syncthreads();
    }

    // the sink: one extra logit with a zero value joins the normaliser
    const float sg0 = __bfloat162float(sinks[(unsigned long long)min(r0, q_len - 1) * num_q_heads + head]);
    const float sg1 = __bfloat162float(sinks[(unsigned long long)min(r1, q_len - 1) * num_q_heads + head]);
    {
        const float n0 = fmaxf(m0, sg0), n1 = fmaxf(m1, sg1);
        const float e0 = __expf(m0 - n0), e1 = __expf(m1 - n1);
        l0 = l0 * e0 + __expf(sg0 - n0);
        l1 = l1 * e1 + __expf(sg1 - n1);
#pragma unroll
        for (int i = 0; i < FP_HD / 8; i++) {
            acc[i][0] *= e0;
            acc[i][1] *= e0;
            acc[i][2] *= e1;
            acc[i][3] *= e1;
        }
    }
    const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    const unsigned long long os = (unsigned long long)num_q_heads * FP_HD;
#pragma unroll
    for (int d = 0; d < FP_HD / 8; d++) {
        const unsigned int col = head * FP_HD + d * 8 + t2;
        if (r0 < q_len)
            *(__nv_bfloat162*)&O[(unsigned long long)r0 * os + col] = __floats2bfloat162_rn(acc[d][0] * inv0, acc[d][1] * inv0);
        if (r1 < q_len)
            *(__nv_bfloat162*)&O[(unsigned long long)r1 * os + col] = __floats2bfloat162_rn(acc[d][2] * inv1, acc[d][3] * inv1);
    }
}
#endif  // AMD
