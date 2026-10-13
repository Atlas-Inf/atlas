// SPDX-License-Identifier: AGPL-3.0-only

// GLM verify-decode twins of the K128W kernels (ATLAS_GLM_MOE_DECODE_M16).
// Included by moe_w4a16_grouped_gemm.cu after the K128W pieces it reuses.
//
// Top-k takes an expert at most once per row, so a verify-decode batch of at
// most 16 rows (a DFlash verify block or owner batch of 2-8 rows, 3-row owner
// batches of 6-12) leaves every expert at most 16 sorted rows: one m16 MMA
// slab. The M64 kernels decode otherwise runs (compact k64 gate/up,
// silu_mul_quant_nvfp4 + a D2D copy, dense K128 down) pad each tile to 64
// rows, three MMAs in four on padding. They already read the routed experts'
// weights at 96-100% of a read-only pass over the same bytes at 5-8 rows
// (moe_decode_bench), so what is left to take is the padding, the separate
// SiLU pass and copy, and the short-launch latency at 1-4 rows.
//
// One tile here is 16 rows x 256 B columns (gate/up: 128 gate columns and the
// same 128 up columns, SiLU·mul + NVFP4 quantization in the epilogue; down:
// 256 columns). Its 8 warps each own 32 of the B columns (gate/up: 16 gate +
// 16 up), so every MMA is a live one, and PQD_STAGES K128 stages are in
// flight. Tiles, loads and fragments are the K128W ones (pqw_*): each output
// element accumulates the same m16n8k64 MMAs, same operands and scales, k64
// slices in K order, and the epilogues are pqw_epilogue's arithmetic, so the
// bytes match the K128W kernels, which match the k64 / K128 ones.
//
// Grid (N/128 gate/up or N/256 down, bound, 1): CTA row y takes the y-th
// routed local expert from `worklist`, the scratch moe_build_tile_worklist
// fills with one N tile per M64 row tile ([0] the item count, items from
// word 4 as (expert, m_tile << 6 | n_tile)), with bound >= that count.
// Requires K % 128 == 0, N % 256 == 0 (gate/up: N % 128 == 0) and at most
// PQD_M rows per expert. The host only selects these kernels for <= 16 rows;
// should an expert hold more anyway, its down CTAs store NaN over its rows
// (and gate/up stores nothing), so a broken host guard cannot pass as stale
// bytes. PQD_STAGES 4 is about 80 KB of static shared memory per CTA, above
// the classic 48 KB: a module-load or launch failure on another driver would
// point here.
#pragma once

#define PQD_M 16
#ifndef PQD_STAGES
#define PQD_STAGES 4
#endif
#define PQD_ZSKIP_KP 1024   // packed K bytes a ZSKIP tile can mask (K <= 2048)

typedef unsigned char PqdA[PQD_M][PQ2_AP];
typedef unsigned char PqdAs[PQD_M][PQ2_KS / GROUP_SIZE];
typedef float PqdAcc[4][4];

// Warp w's nb-th 16-byte column chunk of the tile (gate/up: gate chunk w,
// then the up chunk of the same columns).
template<bool GATE_UP>
__device__ __forceinline__ unsigned int pqd_warp_chunk(unsigned int warp_id, int nb) {
    return GATE_UP ? nb * (PQW_NT / 32) + warp_id : warp_id * 2 + nb;
}

// Thread t's cp.async loads of K stage kb: pqw_issue over a 16-row A tile.
template<bool GATE_UP>
__device__ __forceinline__ void pqd_issue(
    unsigned int t, PqdA& sA, PqdAs& sAs, PqwB& sB, PqwS& sS, const int* sTok,
    const unsigned char* __restrict__ A_packed, const unsigned char* __restrict__ A_scale,
    const unsigned char* B_expert, const unsigned char* S_expert,
    const unsigned char* U_expert, const unsigned char* US_expert,
    unsigned int M_eff, unsigned int cta_n, unsigned int N, unsigned int K, unsigned int kb,
    const unsigned char* live
) {
    if (t < PQD_M * 4) {
        const unsigned int row = t >> 2, col = (t & 3) << 4;
        moe_cp_async_pred_16(&sA[row][col],
            &A_packed[(unsigned long long)(unsigned int)sTok[row] * (K / 2) + kb / 2 + col], row < M_eff);
    }
    if (t < PQD_M)
        moe_cp_async_pred_8(&sAs[t][0],
            &A_scale[(unsigned long long)(unsigned int)sTok[t] * (K / GROUP_SIZE) + kb / GROUP_SIZE], t < M_eff);
    #pragma unroll
    for (int r = 0; r < PQW_NT / 64; ++r) {
        const unsigned int j = t + r * 256;
        const unsigned int kp = j / (PQW_NT / 16), c = j % (PQW_NT / 16);
        moe_cp_async_pred_16(&sB[kp][(c ^ (kp & 7)) << 4],
            &(pqw_chunk_up<GATE_UP>(c) ? U_expert : B_expert)[(unsigned long long)(kb / 2 + kp) * N + pqw_chunk_col<GATE_UP>(cta_n, c)],
            !live || live[kb / 2 + kp]);
    }
    if (t < PQW_NT / 2) {
        const unsigned int g = t / (PQW_NT / 16), c = t % (PQW_NT / 16);
        moe_cp_async_pred_16(&sS[g][(c ^ g) << 4],
            &(pqw_chunk_up<GATE_UP>(c) ? US_expert : S_expert)[(unsigned long long)(kb / GROUP_SIZE + g) * N + pqw_chunk_col<GATE_UP>(cta_n, c)], true);
    }
}

// Warp warp_id's MMAs of one K128 stage: pqw_mma_stage's fragments for the
// one row slab and this warp's two column chunks.
template<bool GATE_UP>
__device__ __forceinline__ void pqd_mma_stage(
    PqdAcc& acc, const PqdA& sA, const PqdAs& sAs, const PqwB& sB, const PqwS& sS,
    unsigned int warp_id, unsigned int lane_id
) {
    const unsigned int kp_l = lane_id, g_l = lane_id & 7;
    const unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2);
    const unsigned short tid_a = 0, bid_a = 0, bid_b = 0;
    unsigned int sfb[4];
    #pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
        const unsigned int c = pqd_warp_chunk<GATE_UP>(warp_id, nb);
        unsigned int d[2];
        pqw_ldsm_t1(d, __cvta_generic_to_shared(&sS[g_l][(c ^ g_l) << 4]));
        sfb[nb * 2] = d[0];
        sfb[nb * 2 + 1] = d[1];
    }
    #pragma unroll
    for (int sl = 0; sl < 2; ++sl) {
        const unsigned int ko = sl * 32;
        const unsigned short tid_b = sl;
        unsigned int a[4];
        {
            const unsigned int j = lane_id >> 3, r = lane_id & 7;
            const unsigned int addr = __cvta_generic_to_shared(&sA[r + (j & 1) * 8][ko + (j >> 1) * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
        }
        const unsigned int sfa = *(const unsigned int*)&sAs[sfa_m][sl * 4];
        #pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            const unsigned int c = pqd_warp_chunk<GATE_UP>(warp_id, nb);
            const unsigned int kp = ko + kp_l;
            unsigned int b[4];
            pqw_ldsm_t2(b, __cvta_generic_to_shared(&sB[kp][(c ^ (kp & 7)) << 4]));
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nt = nb * 2 + h;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                    "{%14},{%15,%16},{%17},{%18,%19};"
                    :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[h]),"r"(b[2 + h]),
                     "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]),
                     "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb[nt]),"h"(bid_b),"h"(tid_b));
            }
        }
    }
}

// Warp warp_id's stores of a finished tile, as pqw_epilogue: BF16 C, or
// (GATE_UP) the packed NVFP4 SiLU·mul of its 16 gate/up columns.
template<bool GATE_UP>
__device__ __forceinline__ void pqd_epilogue(
    const PqdAcc& acc, unsigned int warp_id, unsigned int lane_id,
    unsigned int expert_id, float scale2, unsigned int cta_m, int M_expert,
    unsigned int cta_n, unsigned int N, __nv_bfloat16* __restrict__ C, const PqwGateUp& up
) {
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;
    if constexpr (GATE_UP) {
        const float scale2_up = up.scale2_vals[expert_id];
        const unsigned int col = cta_n + warp_id * 16;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const unsigned int row = cta_m + group_id + half * 8;
            const bool valid = (int)(group_id + half * 8) < M_expert;
            float v[4], group_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int j = i >> 1, e = half * 2 + (i & 1);
                v[i] = silu_nvfp4_act(
                    __bfloat162float(__float2bfloat16(acc[j][e] * scale2)),
                    __bfloat162float(__float2bfloat16(acc[j + 2][e] * scale2_up)));
                group_max = fmaxf(group_max, fabsf(v[i]));
            }
            group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 1));
            group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 2));
            float inv;
            const unsigned char sc = silu_nvfp4_group_scale(group_max, &inv);
            if (!valid) continue;
            if (tid == 0) up.out_scale[(unsigned long long)row * (N / 16) + col / 16] = sc;
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const unsigned int q0 = silu_nvfp4_quantize_e2m1(v[hh * 2] * inv);
                const unsigned int q1 = silu_nvfp4_quantize_e2m1(v[hh * 2 + 1] * inv);
                up.out_packed[(unsigned long long)row * (N / 2) + (col + hh * 8) / 2 + tid] =
                    (unsigned char)((q1 << 4) | q0);
            }
        }
    } else {
        const bool r0v = (int)group_id < M_expert, r1v = (int)(group_id + 8) < M_expert;
        const unsigned int r0 = cta_m + group_id, r1 = r0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const unsigned int c0 = cta_n + warp_id * 32 + nt * 8 + tid * 2;
            if (r0v)
                *(__nv_bfloat162*)&C[r0 * N + c0] = __floats2bfloat162_rn(
                    acc[nt][0] * scale2, acc[nt][1] * scale2);
            if (r1v)
                *(__nv_bfloat162*)&C[r1 * N + c0] = __floats2bfloat162_rn(
                    acc[nt][2] * scale2, acc[nt][3] * scale2);
        }
    }
}

template<bool GATE_UP, bool ZSKIP>
__device__ __forceinline__ void pqd_impl(
    PQ2_ARGS,
    const unsigned int* __restrict__ worklist,
    const PqwGateUp up
) {
    atlas_pdl_enter();
    if ((int)blockIdx.y >= (int)worklist[0]) return;
    const unsigned int expert_id = worklist[4 + blockIdx.y * 2];
    // One row tile per expert: a second M64 tile means more than 64 rows.
    if (expert_id >= num_experts || worklist[5 + blockIdx.y * 2] != 0) return;
    const unsigned int cta_m = expert_offsets[expert_id];
    const int M_expert = expert_offsets[expert_id + 1] - (int)cta_m;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (M_expert <= 0 || B_expert == 0) return;
    if (M_expert > PQD_M) {
        // More rows than the one slab the host selects these kernels for.
        if constexpr (!GATE_UP) {
            const __nv_bfloat16 nan = __float2bfloat16(__int_as_float(0x7fc00000));
            for (int r = 0; r < M_expert; ++r)
                C[(unsigned long long)(cta_m + r) * N + blockIdx.x * PQW_NT + threadIdx.x] = nan;
        }
        return;
    }
    const float scale2 = scale2_vals[expert_id];
    const unsigned char* U_expert = GATE_UP ? (const unsigned char*)up.packed_ptrs[expert_id] : nullptr;
    const unsigned char* US_expert = GATE_UP ? (const unsigned char*)up.scale_ptrs[expert_id] : nullptr;
    const unsigned int cta_n = blockIdx.x * (GATE_UP ? PQW_NT / 2 : PQW_NT);

    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) PqdA sA[PQD_STAGES];
    __shared__ __align__(16) PqdAs sAs[PQD_STAGES];
    __shared__ __align__(16) PqwB sB[PQD_STAGES];
    __shared__ __align__(16) PqwS sS[PQD_STAGES];
    __shared__ int sTok[PQD_M];

    if (t < PQD_M)
        sTok[t] = (sorted_token_ids && (int)t < M_expert) ? sorted_token_ids[cta_m + t] : (int)(cta_m + t);
    __syncthreads();

    // ZSKIP: weight rows [kp] whose two activations are E2M1 zeros (codes 0
    // and 8, +0 and -0) in every row of the tile are not loaded; their shared
    // bytes are zero-filled. Each such term of an MMA is 0 * w before and
    // 0 * 0 after, an exact zero either way, and the accumulators start at
    // +0, so they are unchanged (moe_decode_bench's signed-zero stress).
    __shared__ unsigned char sLive[ZSKIP ? PQD_ZSKIP_KP : 1];
    const bool zskip = ZSKIP && K / 2 <= PQD_ZSKIP_KP;
    if (zskip) {
        for (unsigned int kp = t; kp < K / 2; kp += blockDim.x) {
            unsigned int any = 0;
            for (int r = 0; r < M_expert; ++r)
                any |= A_packed[(unsigned long long)(unsigned int)sTok[r] * (K / 2) + kp] & 0x77u;
            sLive[kp] = any != 0;
        }
        __syncthreads();
    }

    auto issue = [&](int buf, unsigned int kb) {
        pqd_issue<GATE_UP>(t, sA[buf], sAs[buf], sB[buf], sS[buf], sTok, A_packed, A_scale,
            B_expert, S_expert, U_expert, US_expert, (unsigned int)M_expert, cta_n, N, K, kb,
            zskip ? sLive : nullptr);
    };

    PqdAcc acc;
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;

    const unsigned int stages = K / PQ2_KS;
    #pragma unroll
    for (int s0 = 0; s0 < PQD_STAGES - 1; ++s0) {
        if (s0 < stages) issue(s0, s0 * PQ2_KS);
        moe_cp_async_commit();
    }
    for (unsigned int st = 0; st < stages; ++st) {
        const int buf = st % PQD_STAGES;
        asm volatile("cp.async.wait_group %0;" :: "n"(PQD_STAGES - 2));
        __syncthreads();   // stage `st` landed; every warp finished stage st-1
        const unsigned int nxt = st + PQD_STAGES - 1;
        if (nxt < stages) issue(nxt % PQD_STAGES, nxt * PQ2_KS);
        moe_cp_async_commit();
        pqd_mma_stage<GATE_UP>(acc, sA[buf], sAs[buf], sB[buf], sS[buf], warp_id, lane_id);
    }
    pqd_epilogue<GATE_UP>(acc, warp_id, lane_id, expert_id, scale2, cta_m, M_expert, cta_n, N, C, up);
}

// Down (or any one projection): arguments as
// moe_w4a4_grouped_gemm_prequant_t_k128w_compact, the worklist for its prefix.
extern "C" __global__ void __launch_bounds__(256) glm_moe_decode_m16_k128w(
    PQ2_ARGS,
    const unsigned int* __restrict__ worklist
) {
    pqd_impl<false, false>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, worklist, PqwGateUp{});
}

// The same projection, not loading weight rows that only multiply E2M1
// zeros (ATLAS_GLM_MOE_DOWN_ZSKIP): for the down projection, whose NVFP4
// SiLU·mul input has many. Same bytes.
extern "C" __global__ void __launch_bounds__(256) glm_moe_decode_m16_k128w_zskip(
    PQ2_ARGS,
    const unsigned int* __restrict__ worklist
) {
    pqd_impl<false, true>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, worklist, PqwGateUp{});
}

// Gate + up + SiLU·mul NVFP4: arguments as
// moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w, the worklist for its prefix.
extern "C" __global__ void __launch_bounds__(256) glm_moe_decode_m16_gate_up_silu_k128w(
    PQ2_ARGS,
    const unsigned int* __restrict__ worklist,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ out_packed,
    unsigned char* __restrict__ out_scale
) {
    pqd_impl<true, false>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, worklist,
        PqwGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, out_packed, out_scale});
}
