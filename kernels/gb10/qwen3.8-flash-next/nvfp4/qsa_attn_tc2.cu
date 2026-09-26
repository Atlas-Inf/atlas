// SPDX-License-Identifier: AGPL-3.0-only

// QSA prefill attention on TENSOR CORES -- BOTH kv heads per CTA.
//
// v2 of `qsa_attn_tc.cu`. That kernel puts one kv head's `gqa`=12 q-heads in
// a BR=32 M tile and wastes 2.67x. Here rows 0..11 are kv head 0's heads and
// rows 16..27 kv head 1's -- exactly the split `(warp_id & 1) * 16` already
// makes -- so the waste falls to 32/24 = 1.33x. Both halves are the SAME row,
// so they share one block list and the two gathers differ only by a constant
// `kvh * head_dim` inside each cached token.
//
// BC drops 32 -> 16 to pay for the second K/V buffer: K is [2][2][16][264] and
// V [2][16][264], which keeps shared at ~69 KB rather than the 118 KB a second
// buffer at BC=32 would need (that would cost a CTA per SM).
//
// DERIVED, deliberately, from `kernels/gb10/common/inferspark_prefill.cu` (the
// BR=32 arm) rather than written fresh: every mma fragment index, the online
// softmax, the cp.async staging and the SM121 `ldmatrix` workaround below are
// reused verbatim from a kernel that is already correct on this hardware. Only
// the four addressing sites differ.
//
// WHY. TTFT_GAP.md 27a measured `inferspark_prefill_64` at **30.1 TFLOP/s** and
// `qsa_prefill_attn_g` at **3.1** — same box, same head_dim=256, same model,
// same request. Four experiments (22, 23) bracketed the scalar kernel and found
// a genuine local optimum; what they never tried was tensor cores, which is
// where the 9.7x lives.
//
// THE MAPPING, and why it is not the obvious one. QSA selects blocks PER ROW, so
// the natural tensor-core shape — tile query ROWS and attend the union of their
// selections — needs neighbouring rows to agree. Measured, they do not:
// `ATLAS_QSA_UNION_DIAG` puts the union over 16 adjacent rows at 4.7-5.3x topk
// (27b), so a row-tiled kernel computes ~5x the necessary work and the 10x
// evaporates.
//
// So hold the ROW fixed and tile the HEADS. All `nq/nkv` q-heads of one row
// attend the IDENTICAL selected set — the property `QSA_PA_G` already exploits
// to share K/V loads — so:
//
//     S = Q[BR, hd] @ K^T[hd, n_tok]        rows of Q are HEADS, not tokens
//     O = P[BR, n_tok] @ V[n_tok, hd]
//
// No union, no extra blocks, no semantic change. One CTA per (row, kv head);
// `gqa = nq/nkv` = 12 heads occupy rows 0..11 of a BR=32 tile, so the M waste is
// 32/12 = 2.67x. That is the price of reusing this file's warp mapping intact
// (`(warp_id & 1) * 16` splits BR=32 into two 16-row halves) and it is still far
// ahead of scalar: 18.3 TFLOP x 2.67 at tensor-core rates against 5.9 s today.
// Packing both kv heads into one tile would cut the waste to 1.33x and is the
// obvious follow-up, but it needs two K/V buffers and is not what v1 risks.
//
// NO CAUSAL MASK. Every token in the selected list is at or below this row's
// position by construction, so `causal` is hard-wired off and the whole
// mask/sliding-window path is gone.
//
// NOT BIT-IDENTICAL: a different summation tree. The drift lands in the
// attention OUTPUT, not in a chaotic top-k selection (unlike 16a's GEMM
// scorer), but it still needs `scripts/ppl.py`. Dispatch is behind
// `ATLAS_QSA_ATTN_TC=1` and defaults OFF.

#include <cuda_bf16.h>

#define BR 32
// Overridable so a target can instantiate this kernel at another shape without
// forking 500 lines. Gemma-4's global layers are HDIM=512, which at BC=32 needs
// 132.8 KB of shared memory (cap is 101,376 B); BC=16 fits in 84,992 B. BR must
// stay 32 — the warp mapping (`(warp_id & 1) * 16`, `warp_id < 2`) splits a
// 32-row tile across two warp pairs.
#ifndef BC
#define BC 16
#endif
#ifndef HDIM
#ifndef HDIM
#define HDIM 256
#endif
#endif
#define PAD_KV 8           // 16-byte row alignment: (256+8)*2 = 528 bytes
#define HDIM_PAD (HDIM + PAD_KV)  // 264
#define PAD_P 8            // P stride: BC + PAD_P = 40, 40*2=80 bytes (16-byte aligned)
#define N_TILES_PER_WARP ((HDIM / 8) / 2)  // total n-tiles / 2 warp-pairs

// Number of 16-byte (8-element) chunks per tile: 32 rows * (256/8) = 32*32 = 1024
#define TILE_CHUNKS (BR * (HDIM / 8))
// K and V tiles are BC rows, not BR. Identical while BR == BC (the only shape
// this file has ever been built at), so the three K/V load loops below used the
// BR-sized bound and were never wrong. At BC != BR that bound walks rows
// BC..BR-1 of a BC-row tile and writes OUT OF BOUNDS in shared memory —
// confirmed with compute-sanitizer at BR=32/BC=16.
#define TILE_CHUNKS_KV_TILE (BC * (HDIM / 8))

// SCALE/gfx1151: RDNA3.5 hard 64 KB/workgroup LDS cap. This file is
// COMPILE-ONLY on AMD (non-paged contiguous prefill — not dispatched for
// FP8 chunked serving). Single-buffer smem_K/smem_K64 (+ BR64=32 below)
// only need to fit LDS so the binary builds. NVIDIA #else verbatim.
#if defined(__SCALE__)
#define ATLAS_KBUFN 1
#define ATLAS_KB(x) 0u
#else
#define ATLAS_KBUFN 2
#define ATLAS_KB(x) (x)
#endif

// Entry name is overridable alongside the shape, so the two instantiations do
// not collide when both are compiled into one module.
#ifndef ATLAS_PREFILL_ENTRY
#define ATLAS_PREFILL_ENTRY qsa_prefill_attn_tc2
#endif
// Physical element offset of selected key `t` for one row -- the same address
// arithmetic `qsa_prefill_attn_g` uses, duplicated here so this file stands
// alone. Returns an offset into k_cache/v_cache that already includes kvh*hd.
__device__ __forceinline__ unsigned long long qsa_tc2_key_off(
    unsigned int t, unsigned int topk_ratio, unsigned int ratio, bool ratio_p2,
    unsigned int ratio_sh, unsigned int ratio_mask, unsigned int complete,
    unsigned int block_size, bool bs_p2, unsigned int bs_sh, unsigned int bs_mask,
    const int* __restrict__ my_list, const int* __restrict__ block_table,
    unsigned long long page_stride, unsigned int row_elems,
    unsigned int kvh, unsigned int hd)
{
    unsigned int tok;
    if (t < topk_ratio) {
        const unsigned int li = ratio_p2 ? (t >> ratio_sh) : (t / ratio);
        const unsigned int lo = ratio_p2 ? (t & ratio_mask) : (t % ratio);
        tok = (unsigned int)my_list[li] * ratio + lo;
    } else {
        tok = complete * ratio + (t - topk_ratio);
    }
    const unsigned int pg = bs_p2 ? (tok >> bs_sh) : (tok / block_size);
    const unsigned int inb = bs_p2 ? (tok & bs_mask) : (tok % block_size);
    return (unsigned long long)(unsigned int)block_table[pg] * page_stride
         + (unsigned long long)inb * row_elems
         + (unsigned long long)kvh * hd;
}

extern "C" __global__ void ATLAS_PREFILL_ENTRY(
    const __nv_bfloat16* __restrict__ Q,          // [rows, nq, hd]
    const __nv_bfloat16* __restrict__ K,          // paged NHD
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,                // [rows, nq, hd]
    const int* __restrict__ block_table,
    const int* __restrict__ lists,                // [rows, topk]
    const unsigned int first_pos,
    const unsigned int topk,
    const unsigned int ratio,
    const unsigned int block_size,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d
) {
    const unsigned int r = blockIdx.y;           // one CTA per ROW; both kv heads
    const unsigned int gqa = num_q_heads / num_kv_heads;

    // Selected-key count for this row: topk whole blocks plus the partial tail.
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int seq_len = topk * ratio + tail;   // == n_tok; bounds K/V

    const int* my_list = lists + (size_t)r * topk;
    const unsigned int row_elems = num_kv_heads * head_dim;
    const unsigned long long page_stride =
        (unsigned long long)block_size * row_elems;
    const bool ratio_p2 = ratio != 0u && (ratio & (ratio - 1u)) == 0u;
    const bool bs_p2 = block_size != 0u && (block_size & (block_size - 1u)) == 0u;
    const unsigned int ratio_sh = ratio_p2 ? (unsigned int)(__ffs((int)ratio) - 1) : 0u;
    const unsigned int ratio_mask = ratio_p2 ? (ratio - 1u) : 0u;
    const unsigned int bs_sh = bs_p2 ? (unsigned int)(__ffs((int)block_size) - 1) : 0u;
    const unsigned int bs_mask = bs_p2 ? (block_size - 1u) : 0u;
    const unsigned int topk_ratio = topk * ratio;
    // Both halves of the M tile are the SAME row, so they share one block
    // list; the two kv heads differ only by a constant `kvh * head_dim` inside
    // each cached token. That is what makes the double-fill nearly free.
    #define QSA_TC2_KEY_OFF(t, kvh_) qsa_tc2_key_off((t), topk_ratio, ratio, ratio_p2, \
        ratio_sh, ratio_mask, complete, block_size, bs_p2, bs_sh, bs_mask, \
        my_list, block_table, page_stride, row_elems, (kvh_), head_dim)

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    // Rows 0..15 are kv head 0's heads, rows 16..31 kv head 1's -- which is
    // exactly the split this file's warp mapping already makes at 16.
    if (num_kv_heads != 2u || gqa > 16u) return;

    // Rows of the M tile are this row's q-heads: 0..gqa-1 live, the rest zero.

    const unsigned int q_seq_stride = num_q_heads * head_dim;

    // Q and O are indexed by (row, head); K/V are the paged pools themselves,
    // addressed through the selection list rather than by a sequence offset.
    const __nv_bfloat16* Q_batch = Q + (size_t)r * q_seq_stride;
    const __nv_bfloat16* K_batch = K;
    const __nv_bfloat16* V_batch = V;
    __nv_bfloat16* O_batch = O + (size_t)r * q_seq_stride;

    // Shared memory — double-buffered K + separate V for full async overlap
    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[ATLAS_KBUFN][2][BC][HDIM_PAD];  // [buf][kv head]
    __shared__ __nv_bfloat16 smem_V[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
    __shared__ float smem_ml[BR][2]; // [row][0]=m, [row][1]=l

    // MMA lane mapping
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    // Warp role mapping
    const unsigned int qk_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_warp_m = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP;

    // Output accumulators
    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    // Per-row online softmax state (each thread owns 2 rows)
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride = BC + PAD_P;

    // === KV block count (computed early for merged load) ===
    // No causal trim: seq_len here is the SELECTED key COUNT, and every
    // selected key is at or below this row's position by construction.
    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    unsigned int kv_block_lo = 0;

    // === Merged Q + K[kv_block_lo] load (single cp.async commit group) ===
    // Saves one commit/wait/sync vs separate Q and K loads.
    {
        const unsigned int chunks_per_row = HDIM / 8;  // 32

        // Q tile
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if ((row & 15u) < gqa) {
                const unsigned int gh = (row >> 4) * gqa + (row & 15u);
                const void* gmem = (const void*)&Q_batch[gh * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // K[kv_block_lo] tile (same commit group — no extra sync)
        if (num_kv_blocks > 0) {
            for (unsigned int idx = tid; idx < 2u * TILE_CHUNKS_KV_TILE; idx += 128) {
                unsigned int kvh = idx / TILE_CHUNKS_KV_TILE;
                unsigned int j = idx - kvh * TILE_CHUNKS_KV_TILE;
                unsigned int row = j / chunks_per_row;
                unsigned int chunk = j % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int k_row = kv_block_lo * BC + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[0][kvh][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[QSA_TC2_KEY_OFF(k_row, kvh) + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K[0][kvh][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_lo; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = (kv_block - kv_block_lo) & 1;

        // === Start async V load into smem_V (overlaps with QK^T below) ===
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < 2u * TILE_CHUNKS_KV_TILE; idx += 128) {
                unsigned int kvh = idx / TILE_CHUNKS_KV_TILE;
                unsigned int j = idx - kvh * TILE_CHUNKS_KV_TILE;
                unsigned int row = j / chunks_per_row;
                unsigned int chunk = j % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_V[kvh][row][col]);

                if (v_row < seq_len) {
                    const void* gmem = (const void*)&V_batch[QSA_TC2_KEY_OFF(v_row, kvh) + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_V[kvh][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");  // V tile loading in background
        }

        // K[kv_block] already in smem_K[ATLAS_KB(buf)] (preloaded or from prev iteration)

        // === QK^T (warps 0-1, register-based) ===
        // BC/8 n-tiles: each `mma.sync.m16n8k16` yields 8 score columns. `4` was
        // BC/8 at BC=32; at BC=16 the GEMM produced 32 columns into a
        // (BC + PAD_P)=24-wide smem_P row.
        float acc_s[BC / 8][4];  // [n_tile][{row0_c0, row0_c1, row1_c0, row1_c1}]
        if (warp_id < 2) {
            // BC/8, not 4: `acc_s` is [BC/8][4], and the literal wrote
            // past the end of a 2-entry array at BC=16.
            #pragma unroll
            for (int i = 0; i < (int)(BC / 8); i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_base = ks * 16;

                // SM121 workaround: manual Q register loading
                // (ldmatrix.x4 produces incorrect results on GB10)
                const unsigned short* sQ_u16 = (const unsigned short*)smem_Q;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_base + tid_in_group * 2;
                unsigned int ac1 = k_base + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ_u16[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ_u16[ar1 * HDIM_PAD + ac1];

                // B fragments: iterate over 4 N-tiles of K^T
                // SM121 workaround: manual B-operand register loading
                // (ldmatrix.trans produces incorrect results on GB10)
                const unsigned short* sK_u16 = (const unsigned short*)smem_K[ATLAS_KB(buf)][warp_id & 1];
                #pragma unroll
                for (int nt = 0; nt < (int)(BC / 8); nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2;
                    unsigned int k1 = k_base + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK_u16[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK_u16[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // === Register-based softmax ===
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;

            // Scale + boundary checks (in registers). No causal mask: every
            // selected key is visible to this row by construction.
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if ((row0 & 15u) >= gqa) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if ((row1 & 15u) >= gqa) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            // Row max: local max then warp shuffle across tid_in_group (4 threads)
            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            // Online softmax update: rescale previous accumulators
            float m_new0 = fmaxf(m_r0, rmax0);
            float exp_old0 = __expf(m_r0 - m_new0);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            float m_new1 = fmaxf(m_r1, rmax1);
            float exp_old1 = __expf(m_r1 - m_new1);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            // Compute P = exp(s - m) in registers, write to smem_P
            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < (int)(BC / 8); nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0);
                float p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1);
                float p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }

            // Row sum reduction via warp shuffle
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            // Share m/l with warps 2-3
            if (tid_in_group == 0) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        // Wait for V tile load (group 1 complete) — was loading during QK^T+softmax
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 2-3: rescale accumulators to match current m
        if (warp_id >= 2) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml[row0][0];
            float cur_m1 = smem_ml[row1][0];
            float exp_r0 = __expf(m_r0 - cur_m0);
            float exp_r1 = __expf(m_r1 - cur_m1);
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // === Preload K[i+1] into smem_K[1-buf] (overlaps with PV below) ===
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < 2u * TILE_CHUNKS_KV_TILE; idx += 128) {
                unsigned int kvh2 = idx / TILE_CHUNKS_KV_TILE;
                unsigned int j2 = idx - kvh2 * TILE_CHUNKS_KV_TILE;
                unsigned int row = j2 / chunks_per_row_k;
                unsigned int chunk = j2 % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[ATLAS_KB(1 - buf)][kvh2][row][col]);

                if (k_row < seq_len) {
                    const void* gmem = (const void*)&K_batch[QSA_TC2_KEY_OFF(k_row, kvh2) + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K[ATLAS_KB(1 - buf)][kvh2][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");  // K[i+1] loading in background
        }

        // === PV MMA (all 4 warps, 16 n-tiles each, V from smem_V) ===
        {
            #pragma unroll
            // BC/16 k-steps: each `mma.sync.m16n8k16` contracts 16 of the BC
            // kv rows. `2` was BC/16 at BC=32; at BC=16 it read a second,
            // nonexistent 16-row half of smem_V.
            for (unsigned int ks = 0; ks < (BC / 16); ks++) {
                unsigned int k_off = ks * 16;

                // SM121 workaround: manual P register loading
                // (ldmatrix.x4 produces incorrect results on GB10)
                const unsigned short* sP_u16 = (const unsigned short*)smem_P;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP_u16[ar0 * p_smem_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP_u16[ar1 * p_smem_stride + ac1];

                // SM121 workaround: manual V register loading
                // (ldmatrix.trans produces incorrect results on GB10)
                const unsigned short* sV_u16 = (const unsigned short*)smem_V[warp_id & 1];

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;
                    unsigned int b0 = ((unsigned int)sV_u16[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV_u16[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV_u16[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        // Wait for K[i+1] prefetch to complete before next iteration
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 2) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml[row0][1];
            float l1 = smem_ml[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;

            __nv_bfloat16* o_base = O_batch;

            // Rows of this tile are HEADS of row r, so the store strides by
            // head_dim within the row rather than by q_seq_stride.
            if ((row0 & 15u) < gqa && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[((row0 >> 4) * gqa + (row0 & 15u)) * head_dim + col0] = lo | (hi << 16);
            }
            if ((row1 & 15u) < gqa && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[((row1 >> 4) * gqa + (row1 & 15u)) * head_dim + col0] = lo | (hi << 16);
            }
        }
    }
}
