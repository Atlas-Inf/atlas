// SPDX-License-Identifier: AGPL-3.0-only
// Opt-in (ATLAS_GLM_SPARSE_PREFILL_PIPE=1) pipelined fp8_g128 GLM sparse-MLA
// prefill. Included by glm_sparse_prefill_kv_reuse.cu, whose tile constants,
// glm_kvp_exp and cp.async helpers it shares.
//
// Bit-identical to glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad (same
// grid, one query token x 32 heads per CTA): every score, probability and
// output element is produced by the same BF16 mma.sync chain in the same
// k-order, the row max is exact, and the softmax denominator is summed in the
// kv_pad kernel's order ((s0+s1)+(s2+s3), s_g = ((q0+q1)+q2)+q3). What changes:
//   * QK runs on all 8 warps (warp w: rows 16*(w&1), keys 8*(w>>1)) instead of
//     warps 0-1, so QK no longer takes 4x the PV time;
//   * Q/K/P/V fragments come from ldmatrix(.trans) instead of 16-bit LDS;
//   * the selected IDs resolve once to cache slots in shared memory, and each
//     thread loads its share of the next tile's FP8 codes and scales into
//     registers under the current tile's QK and PV, then dequantizes them
//     straight into the K rows (no second shared buffer, no dependent loads
//     in the loop).
// Shared: Q 32x520 BF16 | K 32x520 BF16 | slots 2080 | P 32x40 BF16 |
// row max 32x4 | pair sums 32x4x4 | l 32. Requires 16-token cache blocks.

#define GLM_PIPE_SLOTS 2080u           // 65 tiles of 32 selected IDs
#define GLM_PIPE_NO_SLOT 0xFFFFFFFFu
#define GLM_PIPE_OFF_K    (BR_512 * GLM_KVP_STRIDE * 2)
#define GLM_PIPE_OFF_SLOT (GLM_PIPE_OFF_K + BC_512 * GLM_KVP_STRIDE * 2)
#define GLM_PIPE_OFF_P    (GLM_PIPE_OFF_SLOT + GLM_PIPE_SLOTS * 4)
#define GLM_PIPE_OFF_MAX  (GLM_PIPE_OFF_P + BR_512 * (BC_512 + PAD_P_512) * 2)
#define GLM_PIPE_OFF_SUM  (GLM_PIPE_OFF_MAX + BR_512 * 4 * 4)
#define GLM_PIPE_OFF_L    (GLM_PIPE_OFF_SUM + BR_512 * 16 * 4)
#define GLM_PIPE_SMEM     (GLM_PIPE_OFF_L + BR_512 * 4)
static_assert(GLM_PIPE_SMEM == 80128, "kernel_spec (glm_sparse_prefill_tc.rs) launches the pipe with 80128 bytes");

__device__ __forceinline__ void glm_pipe_ldsm_x4(unsigned int (&r)[4], const void* p) {
    const unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void glm_pipe_ldsm_x4_t(unsigned int (&r)[4], const void* p) {
    const unsigned a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void glm_pipe_mma(float (&d)[4], const unsigned int (&a)[4],
                                             unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1),
         "f"(d[0]),"f"(d[1]),"f"(d[2]),"f"(d[3]));
}

// One thread's four 16-code chunks of a selected tile (rows warp + 8j, codes
// 16 * lane), with their group scales; unselected rows read as zero codes and
// scale, which dequantize to +0. A slot is physical_block * 16 + offset.
struct GlmPipeChunks { uint4 q[4]; float s[4]; };

__device__ __forceinline__ GlmPipeChunks glm_pipe_load_tile(
    const void* cache, const unsigned int* slots, unsigned int kv_s, unsigned int tid) {
    GlmPipeChunks c;
    const unsigned int col = (tid % 32) * 16;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned int slot = slots[kv_s + tid / 32 + 8 * j];
        const bool ok = slot != GLM_PIPE_NO_SLOT;
        c.q[j] = ok ? *reinterpret_cast<const uint4*>(
                          glm_fp8g128_values(cache, slot >> 4, slot & 15u, 16u) + col)
                    : make_uint4(0, 0, 0, 0);
        c.s[j] = ok ? glm_fp8g128_scales(cache, slot >> 4, slot & 15u, 16u)[col / 128u] : 0.0f;
    }
    return c;
}

// Loaded chunks -> BF16 K rows, with glm_kvp_load_tile<true>'s arithmetic.
__device__ __forceinline__ void glm_pipe_store_tile(
    const GlmPipeChunks& c, __nv_bfloat16* smem_K, unsigned int tid) {
    const unsigned int col = (tid % 32) * 16;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        __nv_bfloat16* dst = &smem_K[(tid / 32 + 8 * j) * GLM_KVP_STRIDE + col];
        *reinterpret_cast<uint4*>(dst) = glm_fp8g128_dequant8(make_uint2(c.q[j].x, c.q[j].y), c.s[j]);
        *reinterpret_cast<uint4*>(dst + 8) =
            glm_fp8g128_dequant8(make_uint2(c.q[j].z, c.q[j].w), c.s[j]);
    }
}

// HEADS 32: the CTA's 32 heads as above. HEADS 16 (a TP4 rank, 16 local
// heads): every warp takes rows 0-15 instead of the half its parity names.
// QK runs on warps 0-3 (warp w: keys 8w), the same chains as the 32-head
// kernel's even warps; PV spreads the 512 latent columns over all 8 warps,
// 64 each, instead of 128 on half of them (the other half ran the masked
// rows 16-31); warp 0 alone keeps the row sums. Every output element is the
// same mma chain over the same probabilities and running maxima, so the
// output is the 32-head kernel's first 16 heads bit for bit.
template <int HEADS>
__device__ __forceinline__ void glm_pipe_body(GLM_KV_PAD_ARGS) {
    static_assert(HEADS == 16 || HEADS == 32, "the pipe serves 16 or 32 local heads");
    constexpr bool H16 = HEADS == 16;
    // PV n-tiles (8 latent columns) per warp.
    constexpr unsigned int PV_TILES = H16 ? N_TILES_PER_WARP_512 / 2 : N_TILES_PER_WARP_512;
    (void)V_cache;
    const unsigned int token_row = blockIdx.y;
    const unsigned int head_start = blockIdx.x * HEADS;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (token_row >= rows || head_start >= num_heads || head_dim != 512 || blockDim.x != 256
        || index_width > GLM_PIPE_SLOTS || cache_block_size != 16) return;
    const unsigned int q_len = min((unsigned int)HEADS, num_heads - head_start);
    const int* indices = token_indices + (unsigned long long)token_row * index_width;
    Q += ((unsigned long long)token_row * num_heads + head_start) * head_dim;
    O += ((unsigned long long)token_row * num_heads + head_start) * head_dim;

    extern __shared__ __align__(16) unsigned char smem_pipe[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_pipe);
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem_pipe + GLM_PIPE_OFF_K);
    unsigned int* slots = reinterpret_cast<unsigned int*>(smem_pipe + GLM_PIPE_OFF_SLOT);
    __nv_bfloat16* smem_P = reinterpret_cast<__nv_bfloat16*>(smem_pipe + GLM_PIPE_OFF_P);
    float* smem_max = reinterpret_cast<float*>(smem_pipe + GLM_PIPE_OFF_MAX);
    float* smem_sum = reinterpret_cast<float*>(smem_pipe + GLM_PIPE_OFF_SUM);
    float* smem_l = reinterpret_cast<float*>(smem_pipe + GLM_PIPE_OFF_L);
    const unsigned int p_stride = BC_512 + PAD_P_512;

    // Selected ID -> cache slot, once per CTA (unselected / past the width -> none).
    for (unsigned int i = tid; i < GLM_PIPE_SLOTS; i += 256) {
        const int token = i < index_width ? indices[i] : -1;
        slots[i] = token >= 0
            ? block_table[(unsigned int)token >> 4] * 16u + ((unsigned int)token & 15u)
            : GLM_PIPE_NO_SLOT;
    }
    for (unsigned int idx = tid; idx < TILE_CHUNKS_512; idx += 256) {
        const unsigned int row = idx / 64, col = (idx % 64) * 8;
        if (row < q_len) glm_kvp_cp16(&smem_Q[row * GLM_KVP_STRIDE + col], &Q[row * head_dim + col]);
        else *((uint4*)&smem_Q[row * GLM_KVP_STRIDE + col]) = make_uint4(0, 0, 0, 0);
    }
    glm_kvp_cp_commit();
    const unsigned int num_kv_blocks = (index_width + BC_512 - 1) / BC_512;
    __syncthreads();  // slots
    if (num_kv_blocks > 0) glm_pipe_store_tile(glm_pipe_load_tile(K_cache, slots, 0, tid), smem_K, tid);
    glm_kvp_cp_wait();
    __syncthreads();

    const unsigned int group_id = lane_id >> 2, tid_in_group = lane_id & 3;
    const unsigned int warp_m = H16 ? 0u : (warp_id & 1) * 16;   // QK and PV rows
    const unsigned int qk_nt = H16 ? warp_id : warp_id >> 1;     // QK: this warp's 8 keys
    const bool qk_warp = !H16 || warp_id < 4;
    // The warps that keep the row sums (one per 16 rows).
    const bool l_warp = H16 ? warp_id == 0 : warp_id < 2;
    const unsigned int pv_n_start = H16 ? warp_id * PV_TILES : (warp_id >> 1) * PV_TILES;
    const unsigned int row0 = warp_m + group_id, row1 = row0 + 8;
    // ldmatrix lane addresses: A rows (m16 x k16) and B/V rows.
    const unsigned int a_row = warp_m + (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
    const unsigned int a_col = (lane_id >> 4) * 8;
    const __nv_bfloat16* q_lane = smem_Q + a_row * GLM_KVP_STRIDE + a_col;
    const __nv_bfloat16* k_lane = smem_K + (qk_nt * 8 + (lane_id & 7)) * GLM_KVP_STRIDE + (lane_id >> 3) * 8;
    const __nv_bfloat16* p_lane = smem_P + a_row * p_stride + a_col;
    const __nv_bfloat16* v_lane = smem_K + ((lane_id & 7) + ((lane_id >> 3) & 1) * 8) * GLM_KVP_STRIDE
                                  + (pv_n_start + (lane_id >> 4)) * 8;

    float acc_o[PV_TILES][4];
    #pragma unroll
    for (int i = 0; i < (int)PV_TILES; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f; acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;   // kept by the l warps
    // HEADS 16: the QK warps hold their Q fragments (16 rows x 512) for the
    // whole token instead of reading them from shared memory every tile.
    unsigned int qa[H16 ? HDIM_512 / 16 : 1][4];
    if constexpr (H16) {
        if (qk_warp) {
            #pragma unroll
            for (unsigned int ks = 0; ks < HDIM_512 / 16; ks++) glm_pipe_ldsm_x4(qa[ks], q_lane + ks * 16);
        }
    }

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const unsigned int kv_start = kv_block * BC_512;
        // The next tile's loads fly under this tile's QK and PV.
        GlmPipeChunks next;
        if (kv_block + 1 < num_kv_blocks) next = glm_pipe_load_tile(K_cache, slots, kv_start + BC_512, tid);

        // === QK: this warp's 16 rows x 8 keys, full 512-deep chain ===
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        const unsigned int c0 = qk_nt * 8 + tid_in_group * 2, c1 = c0 + 1;
        bool valid0 = false, valid1 = false;
        if (qk_warp) {
            #pragma unroll
            for (unsigned int ks = 0; ks < HDIM_512 / 16; ks += 2) {
                unsigned int b[4], a[4];
                glm_pipe_ldsm_x4(b, k_lane + ks * 16);
                if constexpr (H16) {
                    glm_pipe_mma(s, qa[ks], b[0], b[1]);
                    glm_pipe_mma(s, qa[ks + 1], b[2], b[3]);
                } else {
                    glm_pipe_ldsm_x4(a, q_lane + ks * 16);
                    glm_pipe_mma(s, a, b[0], b[1]);
                    glm_pipe_ldsm_x4(a, q_lane + (ks + 1) * 16);
                    glm_pipe_mma(s, a, b[2], b[3]);
                }
            }
            valid0 = slots[kv_start + c0] != GLM_PIPE_NO_SLOT;
            valid1 = slots[kv_start + c1] != GLM_PIPE_NO_SLOT;
            s[0] *= inv_sqrt_d; s[1] *= inv_sqrt_d; s[2] *= inv_sqrt_d; s[3] *= inv_sqrt_d;
            if (!valid0) { s[0] = -1e30f; s[2] = -1e30f; }
            if (!valid1) { s[1] = -1e30f; s[3] = -1e30f; }
            if (row0 >= q_len) { s[0] = -1e30f; s[1] = -1e30f; }
            if (row1 >= q_len) { s[2] = -1e30f; s[3] = -1e30f; }
            float pm0 = fmaxf(s[0], s[1]), pm1 = fmaxf(s[2], s[3]);
            pm0 = fmaxf(pm0, __shfl_xor_sync(0xFFFFFFFF, pm0, 1));
            pm0 = fmaxf(pm0, __shfl_xor_sync(0xFFFFFFFF, pm0, 2));
            pm1 = fmaxf(pm1, __shfl_xor_sync(0xFFFFFFFF, pm1, 1));
            pm1 = fmaxf(pm1, __shfl_xor_sync(0xFFFFFFFF, pm1, 2));
            if (tid_in_group == 0) { smem_max[row0 * 4 + qk_nt] = pm0; smem_max[row1 * 4 + qk_nt] = pm1; }
        }
        __syncthreads();

        // === Online softmax: every warp derives the same running max ===
        const float4 x0 = *reinterpret_cast<const float4*>(&smem_max[row0 * 4]);
        const float4 x1 = *reinterpret_cast<const float4*>(&smem_max[row1 * 4]);
        const float mn0 = fmaxf(m_r0, fmaxf(fmaxf(fmaxf(x0.x, x0.y), x0.z), x0.w));
        const float mn1 = fmaxf(m_r1, fmaxf(fmaxf(fmaxf(x1.x, x1.y), x1.z), x1.w));
        if (mn0 != m_r0) {
            const float eo0 = glm_kvp_exp(m_r0 - mn0); l_r0 *= eo0;
            #pragma unroll
            for (int i = 0; i < (int)PV_TILES; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
            m_r0 = mn0;
        }
        if (mn1 != m_r1) {
            const float eo1 = glm_kvp_exp(m_r1 - mn1); l_r1 *= eo1;
            #pragma unroll
            for (int i = 0; i < (int)PV_TILES; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
            m_r1 = mn1;
        }
        if (qk_warp) {
            const float p00 = valid0 ? glm_kvp_exp(s[0] - m_r0) : 0.0f;
            const float p01 = valid1 ? glm_kvp_exp(s[1] - m_r0) : 0.0f;
            const float p10 = valid0 ? glm_kvp_exp(s[2] - m_r1) : 0.0f;
            const float p11 = valid1 ? glm_kvp_exp(s[3] - m_r1) : 0.0f;
            smem_P[row0 * p_stride + c0] = __float2bfloat16(p00);
            smem_P[row0 * p_stride + c1] = __float2bfloat16(p01);
            smem_P[row1 * p_stride + c0] = __float2bfloat16(p10);
            smem_P[row1 * p_stride + c1] = __float2bfloat16(p11);
            smem_sum[(row0 * 4 + qk_nt) * 4 + tid_in_group] = p00 + p01;
            smem_sum[(row1 * 4 + qk_nt) * 4 + tid_in_group] = p10 + p11;
        }
        __syncthreads();

        if (l_warp) {
            // kv_pad's order: per lane g, s_g = ((q0+q1)+q2)+q3 over the key
            // n-tiles; then the xor-1/xor-2 butterfly (s0+s1)+(s2+s3).
            float t0[4], t1[4];
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const float* a0 = &smem_sum[row0 * 16 + g];
                const float* a1 = &smem_sum[row1 * 16 + g];
                t0[g] = ((a0[0] + a0[4]) + a0[8]) + a0[12];
                t1[g] = ((a1[0] + a1[4]) + a1[8]) + a1[12];
            }
            l_r0 += (t0[0] + t0[1]) + (t0[2] + t0[3]);
            l_r1 += (t1[0] + t1[1]) + (t1[2] + t1[3]);
        }

        // === PV: this warp's 16 rows x 128 latent columns ===
        #pragma unroll
        for (unsigned int ks = 0; ks < 2; ks++) {
            unsigned int a[4];
            glm_pipe_ldsm_x4(a, p_lane + ks * 16);
            #pragma unroll
            for (int nt = 0; nt < (int)PV_TILES; nt += 2) {
                unsigned int b[4];
                glm_pipe_ldsm_x4_t(b, v_lane + ks * 16 * GLM_KVP_STRIDE + nt * 8);
                glm_pipe_mma(acc_o[nt], a, b[0], b[1]);
                glm_pipe_mma(acc_o[nt + 1], a, b[2], b[3]);
            }
        }

        if (kv_block + 1 < num_kv_blocks) {
            __syncthreads();  // PV done with K and P
            glm_pipe_store_tile(next, smem_K, tid);
            __syncthreads();  // next K ready
        }
    }

    if (l_warp && tid_in_group == 0) { smem_l[row0] = l_r0; smem_l[row1] = l_r1; }
    __syncthreads();
    const float lv0 = smem_l[row0], lv1 = smem_l[row1];
    const float il0 = (lv0 > 0) ? (1.f / lv0) : 0, il1 = (lv1 > 0) ? (1.f / lv1) : 0;
    #pragma unroll
    for (int nt = 0; nt < (int)PV_TILES; nt++) {
        const unsigned int c = (pv_n_start + nt) * 8 + tid_in_group * 2;
        if (row0 < q_len) {
            const unsigned int lo = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * il0));
            const unsigned int hi = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * il0));
            *(unsigned int*)&O[row0 * head_dim + c] = lo | (hi << 16);
        }
        if (row1 < q_len) {
            const unsigned int lo = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * il1));
            const unsigned int hi = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * il1));
            *(unsigned int*)&O[row1 * head_dim + c] = lo | (hi << 16);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 1)
glm_sparse_mla_prefill_fp8g128_head32_tc_pipe(GLM_KV_PAD_ARGS) {
    glm_pipe_body<32>(Q, K_cache, V_cache, token_indices, O, block_table, rows, num_heads,
                      head_dim, index_width, cache_block_size, inv_sqrt_d);
}

// TP4's 16 local heads (glm_pipe_body).
extern "C" __global__ void __launch_bounds__(256, 1)
glm_sparse_mla_prefill_fp8g128_head16_tc_pipe(GLM_KV_PAD_ARGS) {
    glm_pipe_body<16>(Q, K_cache, V_cache, token_indices, O, block_table, rows, num_heads,
                      head_dim, index_width, cache_block_size, inv_sqrt_d);
}

// ATLAS_GLM_SPARSE_PIPE_WG2 (TP4's 16 local heads, opt-in, not bit-exact):
// two warpgroups of glm_pipe_body<16> in one 512-thread CTA, each over half
// of the token's key tiles with its own K tile, running max, row sum and
// accumulators, merged at the end as flash-decoding merges splits. The pipe
// is a serial chain of QK, softmax, PV and the next tile's dequant per
// 32-key tile, and its 80 KB of shared memory leave one CTA, 8 warps, per
// SM; a second warpgroup's chain fills the first one's gaps. The halves sum
// in a different order from the whole (FP32 throughout).
// Shared: Q 16x520 BF16 | slots 2080 | per warpgroup: K 32x520 BF16 |
// P 16x40 BF16 | row max 16x4 | pair sums 16x4x4 | l 16.
#define GLM_WG2_OFF_SLOT (16 * GLM_KVP_STRIDE * 2)
#define GLM_WG2_OFF_WG   (GLM_WG2_OFF_SLOT + GLM_PIPE_SLOTS * 4)
#define GLM_WG2_K        0
#define GLM_WG2_P        (BC_512 * GLM_KVP_STRIDE * 2)
#define GLM_WG2_MAX      (GLM_WG2_P + 16 * (BC_512 + PAD_P_512) * 2)
#define GLM_WG2_SUM      (GLM_WG2_MAX + 16 * 4 * 4)
#define GLM_WG2_L        (GLM_WG2_SUM + 16 * 16 * 4)
#define GLM_WG2_WG_BYTES (GLM_WG2_L + 16 * 4)
#define GLM_WG2_SMEM     (GLM_WG2_OFF_WG + 2 * GLM_WG2_WG_BYTES)
static_assert(GLM_WG2_SMEM == 96768, "kernel_spec (glm_sparse_prefill_tc.rs) launches the wg2 pipe with 96768 bytes");

__device__ __forceinline__ void glm_wg2_bar(unsigned int wg) {
    asm volatile("bar.sync %0, 256;" :: "r"(wg + 1) : "memory");
}

extern "C" __global__ void __launch_bounds__(512, 1)
glm_sparse_mla_prefill_fp8g128_head16_tc_pipe_wg2(GLM_KV_PAD_ARGS) {
    constexpr unsigned int PV_TILES = N_TILES_PER_WARP_512 / 2;   // 64 latent columns a warp
    (void)V_cache;
    const unsigned int token_row = blockIdx.y;
    const unsigned int head_start = blockIdx.x * 16;
    const unsigned int wg = threadIdx.x / 256, tid = threadIdx.x % 256;
    const unsigned int warp_id = tid / 32, lane_id = tid % 32;
    if (token_row >= rows || head_start >= num_heads || head_dim != 512 || blockDim.x != 512
        || index_width > GLM_PIPE_SLOTS || cache_block_size != 16) return;
    const unsigned int q_len = min(16u, num_heads - head_start);
    const int* indices = token_indices + (unsigned long long)token_row * index_width;
    Q += ((unsigned long long)token_row * num_heads + head_start) * head_dim;
    O += ((unsigned long long)token_row * num_heads + head_start) * head_dim;

    extern __shared__ __align__(16) unsigned char smem_wg2[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_wg2);
    unsigned int* slots = reinterpret_cast<unsigned int*>(smem_wg2 + GLM_WG2_OFF_SLOT);
    unsigned char* mine = smem_wg2 + GLM_WG2_OFF_WG + wg * GLM_WG2_WG_BYTES;
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(mine + GLM_WG2_K);
    __nv_bfloat16* smem_P = reinterpret_cast<__nv_bfloat16*>(mine + GLM_WG2_P);
    float* smem_max = reinterpret_cast<float*>(mine + GLM_WG2_MAX);
    float* smem_sum = reinterpret_cast<float*>(mine + GLM_WG2_SUM);
    float* smem_l = reinterpret_cast<float*>(mine + GLM_WG2_L);
    const unsigned int p_stride = BC_512 + PAD_P_512;

    for (unsigned int i = threadIdx.x; i < GLM_PIPE_SLOTS; i += 512) {
        const int token = i < index_width ? indices[i] : -1;
        slots[i] = token >= 0
            ? block_table[(unsigned int)token >> 4] * 16u + ((unsigned int)token & 15u)
            : GLM_PIPE_NO_SLOT;
    }
    for (unsigned int idx = threadIdx.x; idx < 16 * (HDIM_512 / 8); idx += 512) {
        const unsigned int row = idx / 64, col = (idx % 64) * 8;
        if (row < q_len) glm_kvp_cp16(&smem_Q[row * GLM_KVP_STRIDE + col], &Q[row * head_dim + col]);
        else *((uint4*)&smem_Q[row * GLM_KVP_STRIDE + col]) = make_uint4(0, 0, 0, 0);
    }
    glm_kvp_cp_commit();
    // This warpgroup's key tiles: the first or second half.
    const unsigned int all_blocks = (index_width + BC_512 - 1) / BC_512;
    const unsigned int first = wg == 0 ? 0 : (all_blocks + 1) / 2;
    const unsigned int last = wg == 0 ? (all_blocks + 1) / 2 : all_blocks;
    __syncthreads();  // slots
    if (first < last) glm_pipe_store_tile(glm_pipe_load_tile(K_cache, slots, first * BC_512, tid), smem_K, tid);
    glm_kvp_cp_wait();
    __syncthreads();

    const unsigned int group_id = lane_id >> 2, tid_in_group = lane_id & 3;
    const unsigned int qk_nt = warp_id;                 // QK on warps 0-3: 8 keys each
    const bool qk_warp = warp_id < 4;
    const unsigned int pv_n_start = warp_id * PV_TILES;
    const unsigned int row0 = group_id, row1 = row0 + 8;
    const unsigned int a_row = (lane_id & 7) + ((lane_id >> 3) & 1) * 8;
    const unsigned int a_col = (lane_id >> 4) * 8;
    const __nv_bfloat16* q_lane = smem_Q + a_row * GLM_KVP_STRIDE + a_col;
    const __nv_bfloat16* k_lane = smem_K + (qk_nt * 8 + (lane_id & 7)) * GLM_KVP_STRIDE + (lane_id >> 3) * 8;
    const __nv_bfloat16* p_lane = smem_P + a_row * p_stride + a_col;
    const __nv_bfloat16* v_lane = smem_K + ((lane_id & 7) + ((lane_id >> 3) & 1) * 8) * GLM_KVP_STRIDE
                                  + (pv_n_start + (lane_id >> 4)) * 8;

    float acc_o[PV_TILES][4];
    #pragma unroll
    for (int i = 0; i < (int)PV_TILES; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f; acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;   // kept by warp 0 of the warpgroup
    const unsigned int c0 = qk_nt * 8 + tid_in_group * 2, c1 = c0 + 1;

    for (unsigned int kv_block = first; kv_block < last; kv_block++) {
        const unsigned int kv_start = kv_block * BC_512;
        GlmPipeChunks next;
        if (kv_block + 1 < last) next = glm_pipe_load_tile(K_cache, slots, kv_start + BC_512, tid);

        float s[4] = {0.f, 0.f, 0.f, 0.f};
        bool valid0 = false, valid1 = false;
        if (qk_warp) {
            #pragma unroll
            for (unsigned int ks = 0; ks < HDIM_512 / 16; ks += 2) {
                unsigned int b[4], a[4];
                glm_pipe_ldsm_x4(b, k_lane + ks * 16);
                glm_pipe_ldsm_x4(a, q_lane + ks * 16);
                glm_pipe_mma(s, a, b[0], b[1]);
                glm_pipe_ldsm_x4(a, q_lane + (ks + 1) * 16);
                glm_pipe_mma(s, a, b[2], b[3]);
            }
            valid0 = slots[kv_start + c0] != GLM_PIPE_NO_SLOT;
            valid1 = slots[kv_start + c1] != GLM_PIPE_NO_SLOT;
            s[0] *= inv_sqrt_d; s[1] *= inv_sqrt_d; s[2] *= inv_sqrt_d; s[3] *= inv_sqrt_d;
            if (!valid0) { s[0] = -1e30f; s[2] = -1e30f; }
            if (!valid1) { s[1] = -1e30f; s[3] = -1e30f; }
            if (row0 >= q_len) { s[0] = -1e30f; s[1] = -1e30f; }
            if (row1 >= q_len) { s[2] = -1e30f; s[3] = -1e30f; }
            float pm0 = fmaxf(s[0], s[1]), pm1 = fmaxf(s[2], s[3]);
            pm0 = fmaxf(pm0, __shfl_xor_sync(0xFFFFFFFF, pm0, 1));
            pm0 = fmaxf(pm0, __shfl_xor_sync(0xFFFFFFFF, pm0, 2));
            pm1 = fmaxf(pm1, __shfl_xor_sync(0xFFFFFFFF, pm1, 1));
            pm1 = fmaxf(pm1, __shfl_xor_sync(0xFFFFFFFF, pm1, 2));
            if (tid_in_group == 0) { smem_max[row0 * 4 + qk_nt] = pm0; smem_max[row1 * 4 + qk_nt] = pm1; }
        }
        glm_wg2_bar(wg);

        const float4 x0 = *reinterpret_cast<const float4*>(&smem_max[row0 * 4]);
        const float4 x1 = *reinterpret_cast<const float4*>(&smem_max[row1 * 4]);
        const float mn0 = fmaxf(m_r0, fmaxf(fmaxf(fmaxf(x0.x, x0.y), x0.z), x0.w));
        const float mn1 = fmaxf(m_r1, fmaxf(fmaxf(fmaxf(x1.x, x1.y), x1.z), x1.w));
        if (mn0 != m_r0) {
            const float eo0 = glm_kvp_exp(m_r0 - mn0); l_r0 *= eo0;
            #pragma unroll
            for (int i = 0; i < (int)PV_TILES; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
            m_r0 = mn0;
        }
        if (mn1 != m_r1) {
            const float eo1 = glm_kvp_exp(m_r1 - mn1); l_r1 *= eo1;
            #pragma unroll
            for (int i = 0; i < (int)PV_TILES; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
            m_r1 = mn1;
        }
        if (qk_warp) {
            const float p00 = valid0 ? glm_kvp_exp(s[0] - m_r0) : 0.0f;
            const float p01 = valid1 ? glm_kvp_exp(s[1] - m_r0) : 0.0f;
            const float p10 = valid0 ? glm_kvp_exp(s[2] - m_r1) : 0.0f;
            const float p11 = valid1 ? glm_kvp_exp(s[3] - m_r1) : 0.0f;
            smem_P[row0 * p_stride + c0] = __float2bfloat16(p00);
            smem_P[row0 * p_stride + c1] = __float2bfloat16(p01);
            smem_P[row1 * p_stride + c0] = __float2bfloat16(p10);
            smem_P[row1 * p_stride + c1] = __float2bfloat16(p11);
            smem_sum[(row0 * 4 + qk_nt) * 4 + tid_in_group] = p00 + p01;
            smem_sum[(row1 * 4 + qk_nt) * 4 + tid_in_group] = p10 + p11;
        }
        glm_wg2_bar(wg);

        if (warp_id == 0) {
            float t0[4], t1[4];
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const float* a0 = &smem_sum[row0 * 16 + g];
                const float* a1 = &smem_sum[row1 * 16 + g];
                t0[g] = ((a0[0] + a0[4]) + a0[8]) + a0[12];
                t1[g] = ((a1[0] + a1[4]) + a1[8]) + a1[12];
            }
            l_r0 += (t0[0] + t0[1]) + (t0[2] + t0[3]);
            l_r1 += (t1[0] + t1[1]) + (t1[2] + t1[3]);
        }

        #pragma unroll
        for (unsigned int ks = 0; ks < 2; ks++) {
            unsigned int a[4];
            glm_pipe_ldsm_x4(a, p_lane + ks * 16);
            #pragma unroll
            for (int nt = 0; nt < (int)PV_TILES; nt += 2) {
                unsigned int b[4];
                glm_pipe_ldsm_x4_t(b, v_lane + ks * 16 * GLM_KVP_STRIDE + nt * 8);
                glm_pipe_mma(acc_o[nt], a, b[0], b[1]);
                glm_pipe_mma(acc_o[nt + 1], a, b[2], b[3]);
            }
        }

        if (kv_block + 1 < last) {
            glm_wg2_bar(wg);  // PV done with K and P
            glm_pipe_store_tile(next, smem_K, tid);
            glm_wg2_bar(wg);  // next K ready
        }
    }

    // Merge: warpgroup 1 leaves its running max, row sum and accumulators in
    // its K tile; warpgroup 0 rescales both to the common max and writes O.
    if (warp_id == 0 && tid_in_group == 0) { smem_l[row0] = l_r0; smem_l[row1] = l_r1; }
    glm_wg2_bar(wg);  // every warp of the warpgroup is done with its K tile
    float* merge = reinterpret_cast<float*>(smem_wg2 + GLM_WG2_OFF_WG + GLM_WG2_WG_BYTES + GLM_WG2_K);
    float* merge_m = merge + 256 * PV_TILES * 4;
    if (wg == 1) {
        #pragma unroll
        for (int i = 0; i < (int)PV_TILES; i++)
            *reinterpret_cast<float4*>(&merge[(i * 256 + tid) * 4]) =
                make_float4(acc_o[i][0], acc_o[i][1], acc_o[i][2], acc_o[i][3]);
        if (warp_id == 0 && tid_in_group == 0) { merge_m[row0] = m_r0; merge_m[row1] = m_r1; }
    }
    __syncthreads();
    if (wg == 1) return;
    const float* l1 = reinterpret_cast<const float*>(smem_wg2 + GLM_WG2_OFF_WG + GLM_WG2_WG_BYTES + GLM_WG2_L);
    const float m1_0 = merge_m[row0], m1_1 = merge_m[row1];
    const float mm0 = fmaxf(m_r0, m1_0), mm1 = fmaxf(m_r1, m1_1);
    const float a0 = glm_kvp_exp(m_r0 - mm0), b0 = glm_kvp_exp(m1_0 - mm0);
    const float a1 = glm_kvp_exp(m_r1 - mm1), b1 = glm_kvp_exp(m1_1 - mm1);
    const float lv0 = smem_l[row0] * a0 + l1[row0] * b0, lv1 = smem_l[row1] * a1 + l1[row1] * b1;
    const float il0 = (lv0 > 0) ? (1.f / lv0) : 0, il1 = (lv1 > 0) ? (1.f / lv1) : 0;
    #pragma unroll
    for (int nt = 0; nt < (int)PV_TILES; nt++) {
        const float4 o1 = *reinterpret_cast<const float4*>(&merge[(nt * 256 + tid) * 4]);
        const unsigned int c = (pv_n_start + nt) * 8 + tid_in_group * 2;
        if (row0 < q_len) {
            const unsigned int lo = __bfloat16_as_ushort(__float2bfloat16((acc_o[nt][0] * a0 + o1.x * b0) * il0));
            const unsigned int hi = __bfloat16_as_ushort(__float2bfloat16((acc_o[nt][1] * a0 + o1.y * b0) * il0));
            *(unsigned int*)&O[row0 * head_dim + c] = lo | (hi << 16);
        }
        if (row1 < q_len) {
            const unsigned int lo = __bfloat16_as_ushort(__float2bfloat16((acc_o[nt][2] * a1 + o1.z * b1) * il1));
            const unsigned int hi = __bfloat16_as_ushort(__float2bfloat16((acc_o[nt][3] * a1 + o1.w * b1) * il1));
            *(unsigned int*)&O[row1 * head_dim + c] = lo | (hi << 16);
        }
    }
}
