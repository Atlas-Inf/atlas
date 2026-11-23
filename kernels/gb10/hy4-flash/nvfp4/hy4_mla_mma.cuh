// SPDX-License-Identifier: AGPL-3.0-only
// Tensor-core hy4_mla_batch_partial: one CTA per (row, 128-key partition,
// 16-head tile). The scalar kernel re-reads every 576-dim KV row per local
// head; here the CTA gathers K into shared once per 16-head tile (two
// 64-key chunks to fit the ~99KB opt-in smem cap on sm_120) and runs both
// attention GEMMs on tensor cores for the tile's heads.
//
//   scores  = (K[128 x 576] @ Q[576 x heads]) * 1/16      bf16 -> fp32
//   partial = softmax(scores) @ K[128 x 512]              bf16 -> fp32
//
// Softmax is online across the two K chunks: a running max/denominator
// per head, and the GEMM2 accumulators rescale by exp(old-max - new-max)
// between chunks — same math as the scalar impl's single-pass softmax.
//
// Gate: ATLAS_HY4_MLA_MMA=1, heads <= 32 in 16-head tiles.

#include <cuda_bf16.h>

#define MLM_KEYS 128
#define MLM_KCH 64
#define MLM_DIMS 512
#define MLM_SCORE_DIMS 576
#define MLM_KP 584
#define MLM_HEADS 16
#define MLM_PP 72

// Dynamic shared carve: srows | sMaxD | sS | sPt | sQ | sK
//   sMaxD per head: [0]=running max, [1]=running denom, [2]=rescale
#define MLM_OFF_ROWS 0
#define MLM_OFF_MAXD (MLM_OFF_ROWS + MLM_KEYS * 4)
#define MLM_OFF_S (MLM_OFF_MAXD + MLM_HEADS * 3 * 4)
#define MLM_OFF_PT (MLM_OFF_S + MLM_KCH * MLM_HEADS * 4)
#define MLM_OFF_Q (MLM_OFF_PT + MLM_HEADS * MLM_PP * 2)
#define MLM_OFF_K (MLM_OFF_Q + MLM_HEADS * MLM_KP * 2)
#define MLM_SMEM_BYTES (MLM_OFF_K + MLM_KCH * MLM_KP * 2)

__device__ __forceinline__ void mlm_cp16(void* dst, const void* src, bool pred) {
    unsigned saddr = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(saddr), "l"(src), "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void mlm_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void mlm_wait0() {
    asm volatile("cp.async.wait_group 0;\n");
}

// bf16 m16n8k16
__device__ __forceinline__ void mlm_mma(float& c0, float& c1, float& c2, float& c3,
                                        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mlm_ldx4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                         const void* addr) {
    unsigned saddr = (unsigned)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr));
}

__device__ __forceinline__ void mlm_ldx2(unsigned& r0, unsigned& r1, const void* addr) {
    unsigned saddr = (unsigned)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(saddr));
}

__device__ __forceinline__ void mlm_ldx2t(unsigned& r0, unsigned& r1, const void* addr) {
    unsigned saddr = (unsigned)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(saddr));
}

extern "C" __global__ void hy4_mla_batch_partial_mma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ cache,
    const unsigned* __restrict__ table,
    const int* __restrict__ indices,
    float* __restrict__ partials,
    unsigned position,
    unsigned page_size,
    unsigned columns,
    unsigned window,
    unsigned count,
    unsigned splits,
    unsigned heads,
    unsigned index_stride) {
    extern __shared__ unsigned char smem[];
    int* srows = (int*)(smem + MLM_OFF_ROWS);
    float* sMaxD = (float*)(smem + MLM_OFF_MAXD);
    float* sS = (float*)(smem + MLM_OFF_S);                  // [key][head]
    __nv_bfloat16* sPt = (__nv_bfloat16*)(smem + MLM_OFF_PT); // [head][key]
    __nv_bfloat16* sQ = (__nv_bfloat16*)(smem + MLM_OFF_Q);   // [head][576]
    __nv_bfloat16* sK = (__nv_bfloat16*)(smem + MLM_OFF_K);   // [64][576]

    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31;
    const unsigned row = blockIdx.y;
    const unsigned part = blockIdx.x;
    const unsigned h0 = blockIdx.z * MLM_HEADS;
    const unsigned hn = min(MLM_HEADS, heads > h0 ? heads - h0 : 0);
    if (hn == 0) return;
    const unsigned first = part * MLM_KEYS;
    const unsigned gid = lane >> 2;
    const unsigned t4 = lane & 3;

    // ---- resolve logical->physical rows for this partition ----
    // Mirrors hy4_mla_partial_impl: valid = logical>=0 && <=position && >=begin
    unsigned pos = position + row;
    unsigned row_count = window
        ? min(pos + 1, window)
        : min(pos / 4, index_stride / 4 - 1) * 4 + pos % 4 + 1;
    unsigned begin = window && pos + 1 > window ? pos + 1 - window : 0;
    for (unsigned i = tid; i < MLM_KEYS; i += blockDim.x) {
        unsigned at = first + i;
        int phys = -1;
        if (at < row_count) {
            int logical = indices ? indices[row * index_stride + at]
                                  : (int)(begin + at);
            if (logical >= 0 && (unsigned)logical <= pos &&
                (unsigned)logical >= begin) {
                unsigned page = (unsigned)logical / page_size;
                unsigned pcol = window ? page % columns : page;
                phys = (int)(table[pcol] * page_size +
                             (unsigned)logical % page_size);
            }
        }
        srows[i] = phys;
    }
    __syncthreads();

    // ---- gather Q once (16 heads x 576 dims) ----
    const __nv_bfloat16* qbase = q + ((size_t)row * heads + h0) * 576;
    for (unsigned i = tid; i < MLM_HEADS * (MLM_SCORE_DIMS / 8);
         i += blockDim.x) {
        unsigned h = i / (MLM_SCORE_DIMS / 8);
        unsigned seg = i % (MLM_SCORE_DIMS / 8);
        mlm_cp16(sQ + h * MLM_KP + seg * 8, qbase + h * 576 + seg * 8,
                 h < hn);
    }
    mlm_commit();
    mlm_wait0();
    __syncthreads();

    // GEMM2 accumulators live across both chunks: warp w owns dims
    // n0 = w*64 .. +64 (8 n-tiles of 8), m = all 16 heads.
    float acc[8][4];
    memset(acc, 0, sizeof(acc));

    const unsigned mt = warp & 3;   // GEMM1: 4 key m-tiles of 16
    const unsigned ht = warp >> 2;  // GEMM1: 2 head n-tiles of 8
    const unsigned n0 = warp * 64;  // GEMM2 dim base
    const unsigned ar = (lane & 7) + ((lane >> 3) & 1) * 8;
    const unsigned ac4 = (lane >> 4) * 8;
    const unsigned br = lane & 7;
    const unsigned bc2 = ((lane >> 3) & 1) * 8;

    for (unsigned chunk = 0; chunk < 2; ++chunk) {
        // ---- gather this chunk's 64 keys x 576 dims ----
        for (unsigned i = tid; i < MLM_KCH * (MLM_SCORE_DIMS / 8);
             i += blockDim.x) {
            unsigned k = i / (MLM_SCORE_DIMS / 8);
            unsigned seg = i % (MLM_SCORE_DIMS / 8);
            int phys = srows[chunk * MLM_KCH + k];
            mlm_cp16(sK + k * MLM_KP + seg * 8,
                     phys >= 0 ? cache + (size_t)phys * 576 + seg * 8
                               : cache,
                     phys >= 0);
        }
        mlm_commit();
        mlm_wait0();
        __syncthreads();

        // ---- GEMM1 chunk: sS[key][head] = 1/16 * K.Q over 576 dims ----
        {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            for (unsigned ks = 0; ks < MLM_SCORE_DIMS; ks += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                mlm_ldx4(a0, a1, a2, a3,
                         sK + (mt * 16 + ar) * MLM_KP + ks + ac4);
                mlm_ldx2(b0, b1,
                         sQ + (ht * 8 + br) * MLM_KP + ks + bc2);
                mlm_mma(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
            }
            unsigned key_base = chunk * MLM_KCH + mt * 16;
            unsigned head_base = ht * 8;
            int k0 = srows[key_base + gid], k1 = srows[key_base + gid + 8];
            float neg = -INFINITY;
            sS[(mt * 16 + gid) * MLM_HEADS + head_base + t4 * 2] =
                (k0 >= 0 && head_base + t4 * 2 < hn) ? c0 * 0.0625f : neg;
            sS[(mt * 16 + gid) * MLM_HEADS + head_base + t4 * 2 + 1] =
                (k0 >= 0 && head_base + t4 * 2 + 1 < hn) ? c1 * 0.0625f : neg;
            sS[(mt * 16 + gid + 8) * MLM_HEADS + head_base + t4 * 2] =
                (k1 >= 0 && head_base + t4 * 2 < hn) ? c2 * 0.0625f : neg;
            sS[(mt * 16 + gid + 8) * MLM_HEADS + head_base + t4 * 2 + 1] =
                (k1 >= 0 && head_base + t4 * 2 + 1 < hn) ? c3 * 0.0625f : neg;
        }
        __syncthreads();

        // ---- online softmax for this chunk; one half-warp per head ----
        {
            unsigned h = warp * 2 + (lane >> 4);
            unsigned kc = lane & 15;
            float sc[4];
            float mx = -INFINITY;
            bool hv = h < hn;
            for (unsigned j = 0; j < 4; ++j) {
                sc[j] = hv ? sS[(kc * 4 + j) * MLM_HEADS + h] : -INFINITY;
                mx = fmaxf(mx, sc[j]);
            }
            for (unsigned s = 1; s < 16; s <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
            float mold = hv ? sMaxD[h * 3] : -INFINITY;
            // merge: M_new = max(M_old, mx)
            if (chunk == 0) mold = -INFINITY;
            float mnew = fmaxf(mold, mx);
            float den = 0.f;
            for (unsigned j = 0; j < 4; ++j) {
                float e = sc[j] == -INFINITY ? 0.f : __expf(sc[j] - mnew);
                sc[j] = e;
                den += e;
            }
            for (unsigned s = 1; s < 16; s <<= 1)
                den += __shfl_xor_sync(0xffffffffu, den, s);
            float scale = mold == -INFINITY ? 0.f : __expf(mold - mnew);
            if (hv) {
                for (unsigned j = 0; j < 4; ++j)
                    sPt[h * MLM_PP + kc * 4 + j] = __float2bfloat16_rn(sc[j]);
                if (kc == 0) {
                    sMaxD[h * 3] = mnew;
                    sMaxD[h * 3 + 1] =
                        (chunk == 0 ? 0.f : sMaxD[h * 3 + 1] * scale) + den;
                    sMaxD[h * 3 + 2] = scale;
                }
            }
        }
        __syncthreads();

        // ---- GEMM2 chunk: acc += P[64] . K[512], rescale first ----
        {
            float s0 = sMaxD[gid * 3 + 2];
            float s1 = sMaxD[(gid + 8) * 3 + 2];
            for (unsigned nt = 0; nt < 8; ++nt) {
                acc[nt][0] *= s0; acc[nt][1] *= s0;
                acc[nt][2] *= s1; acc[nt][3] *= s1;
            }
            for (unsigned ks = 0; ks < MLM_KCH; ks += 16) {
                unsigned a0, a1, a2, a3;
                mlm_ldx4(a0, a1, a2, a3,
                         sPt + ar * MLM_PP + ks + ac4);
                for (unsigned nt = 0; nt < 8; ++nt) {
                    unsigned b0, b1;
                    unsigned kr = ks + (lane & 15);
                    unsigned nc = n0 + nt * 8 + (lane >> 4) * 8;
                    mlm_ldx2t(b0, b1, sK + kr * MLM_KP + nc);
                    mlm_mma(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                            a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: write partials ----
    float* out = partials + (((size_t)row * heads + h0) * splits + part) * 514;
    for (unsigned nt = 0; nt < 8; ++nt) {
        unsigned ncol = n0 + nt * 8 + t4 * 2;
        unsigned m0 = gid, m1 = gid + 8;
        if (m0 < hn) {
            out[m0 * splits * 514 + ncol] = acc[nt][0];
            out[m0 * splits * 514 + ncol + 1] = acc[nt][1];
        }
        if (m1 < hn) {
            out[m1 * splits * 514 + ncol] = acc[nt][2];
            out[m1 * splits * 514 + ncol + 1] = acc[nt][3];
        }
    }
    if (tid < hn) {
        out[tid * splits * 514 + 512] = sMaxD[tid * 3];
        out[tid * splits * 514 + 513] = sMaxD[tid * 3 + 1];
    }
}
