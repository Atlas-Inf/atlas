// SPDX-License-Identifier: AGPL-3.0-only

// Ling-3.2-air (bailing_moe_v3_vl text) GB10 decode-path MoE kernels:
// the 512-expert top-8 sigmoid+bias group-limited router (fp32 logits), the
// W4A4 NVFP4 routed-expert tiles, the BF16 shared expert, and the fused
// unpermute. Decode-path contract per MoE layer on layers 2..=42 (the MTP
// layer 42 included — the same kernels run per draft/verify step with M > 1
// rows):
//
//   ling_moe_route            one launch: fp32 router GEMV over [M, K] rows,
//                             NVFP4 quantization of the MoE input, then the
//                             last CTA to finish runs the group top-8
//                             selection, the expert sort and the worklist
//   ling_moe_gate_up_silu*    fused gate+up+swiglu+NVFP4 requant per expert
//                             (grid (N/128, items))
//   ling_moe_down*            down projection to BF16 (grid (N/256, items))
//   ling_moe_shared_gate_up*  BF16 shared-expert gate+up+limited swiglu
//   ling_moe_shared_down*     BF16 shared-expert down
//   ling_moe_unpermute        weighted reduce + shared output, one block/row
//
// Dimensions are runtime parameters (the handbook rule: no hardcoded GLM
// dims); the host asserts the Ling shapes hidden 2560, N (intermediate) 768,
// 512 experts, n_group 8, topk_group 4, top_k 8, one shared expert.
// Every launch is PDL-entered; the route kernel's `done` word must be zero
// before each launch (the kernel resets it).
//
// Buffer contracts (loader/integration allocates):
//   logits            fp32 [M, E] scratch, overwritten by routing each call
//   a_packed/a_scale  [M, K/2] / [M, K/16] NVFP4 of the MoE input (rows are
//                     token-major: gate/up gathers them by sorted token id)
//   gu_packed/scale   [M * top_k, N/2] / [.., N/16] sorted-row-major
//   expert_out        bf16 [M * top_k, K] sorted-row-major (the down C)
//   shared scratch    bf16 [M, I] (gate/up) and [M, K] (shared down)
//   worklist          u32 [1 + 2 * max_items]; word 0 the item count, items
//                     from word 1 as (expert, mt) pairs, mt a row tile of
//                     m_tile rows (16 or 32 to match the expert kernels)
//
// Scale contract (the GLM fp4l/input_scales precedent, see
// ling_moe_quant.cuh): a_scale2 is the dequant global of the packed MoE
// input (the RECIPROCAL of the checkpoint's gate/up input_global_scale —
// the compressed-tensors convention stores the reciprocal); each
// projection's scale2_vals[e] = (1/input-side global) * (1/
// weight_global_scale[e]) folds both reciprocals, and
// a2_gscale[e] = input_global_scale(down) (the re-quantizer's g in
// sf = e4m3(amax * g / 6), whose dequant global is 1/g = 1/ig).
//
// Layout: routed expert weights must be loader-repacked to the GLM native
// NVFP4 layout — packed bytes [K/2, N] and group scales [K/16, N]
// (ling_moe_tile.cuh) — as the GLM target does at load time.

#include <cuda_bf16.h>
#include "../../common/atlas_pdl.cuh"
#include "ling_moe_quant.cuh"
#include "ling_moe_route.cuh"
#include "ling_moe_tile.cuh"
#include "ling_moe_shared.cuh"

#define LMOE_ROUTE_MAX_E 512
#define LMOE_ROUTE_WARPS 8
#define LMOE_MAX_TOPK 8

// ── Route: one launch for router GEMV + input quant + select + sort ────────
//
// Grid (router_ctas + quant_ctas, 1, 1), block (256, 1, 1):
//   * router CTAs [0, ceil(E/8)): one warp per expert row, every row of the
//     fp32 router matvec (BF16 activations widened to fp32 — the reference
//     casts both operands to float32; the warp-sliced fp32 accumulation
//     order is ours, the reference's is a cublas GEMV's);
//   * quant CTAs: NVFP4 groups (16 values) of the MoE input, 256 per CTA;
//   * every CTA counts into `done`; the one that sees gridDim.x - 1 (all
//     others' writes fenced) runs the selection (8 warps, a row each --
//     ling_route_topk_row), the counting sort into expert_offsets /
//     sorted_token_ids / token_to_perm and the worklist build, then resets
//     `done` to 0 for the next launch.
extern "C" __global__ void __launch_bounds__(256) ling_moe_route(
    const __nv_bfloat16* __restrict__ A,     // [M, K]
    const float* __restrict__ gate_w,        // [E, K] (mlp.gate.weight, fp32)
    const float* __restrict__ bias,          // [E] (mlp.gate.expert_bias, fp32)
    float a_scale2, float a_scale2_inv,      // gate/up input_global_scale
    float* __restrict__ logits,              // [M, E] fp32 scratch
    unsigned char* __restrict__ a_packed,    // [M, K/2]
    unsigned char* __restrict__ a_scale,     // [M, K/16]
    unsigned int* __restrict__ topk_ids,      // [M, top_k]
    float* __restrict__ topk_weights,         // [M, top_k]
    int* __restrict__ sorted_token_ids,      // [M, top_k]
    int* __restrict__ token_to_perm,         // [M, top_k]
    int* __restrict__ expert_offsets,        // [E + 1]
    unsigned int* __restrict__ worklist,     // [1 + 2 * max_items]
    unsigned int max_items, unsigned int m_tile,
    unsigned int M, unsigned int E, unsigned int K,
    unsigned int top_k, unsigned int n_group, unsigned int topk_group,
    float routed_scaling_factor,
    const unsigned int* __restrict__ row_live,  // [M], or null (all live)
    unsigned int* __restrict__ done              // zeroed between launches
) {
    atlas_pdl_enter();
    if (E > LMOE_ROUTE_MAX_E || E % n_group != 0u || n_group > 16u || top_k > 16u
        || K % LMOE_KS != 0u || M == 0u || (E / n_group) % 32u != 0u)
        return;
    const unsigned int router_ctas = (E + 7u) / 8u;
    const unsigned int groups = M * (K / 16u);
    const unsigned int quant_ctas = (groups + 255u) / 256u;
    const unsigned int t = threadIdx.x;

    if (blockIdx.x < router_ctas) {
        const unsigned int warp = t / 32u, lane = t % 32u;
        const unsigned int e = blockIdx.x * 8u + warp;
        if (e < E) {
            const float* __restrict__ w = gate_w + (unsigned long long)e * K;
            for (unsigned int r = 0; r < M; ++r) {
                const __nv_bfloat16* __restrict__ x = A + (unsigned long long)r * K;
                float acc = 0.0f;
                for (unsigned int k0 = lane * 4u; k0 < K; k0 += 128u) {
                    const float4 wv = *(const float4*)(w + k0);
                    const uint2 xv = *(const uint2*)(x + k0);
                    const __nv_bfloat16* xb = (const __nv_bfloat16*)&xv;
                    acc += __bfloat162float(xb[0]) * wv.x
                         + __bfloat162float(xb[1]) * wv.y
                         + __bfloat162float(xb[2]) * wv.z
                         + __bfloat162float(xb[3]) * wv.w;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
                if (lane == 0u)
                    logits[(unsigned long long)r * E + e] = acc;
            }
        }
    } else if (blockIdx.x < router_ctas + quant_ctas) {
        const unsigned int q = (blockIdx.x - router_ctas) * 256u + t;
        if (q < groups) {
            const unsigned int kg = K / 16u;
            const unsigned int row = q / kg, g = q % kg;
            ling_quantize_group(A + (unsigned long long)row * K,
                                a_packed + (unsigned long long)row * (K / 2u),
                                a_scale + (unsigned long long)row * kg, g,
                                a_scale2, a_scale2_inv);
        }
    }

    // Every CTA fences and counts; the last one to finish runs selection.
    __shared__ unsigned int s_last;
    __threadfence();
    __syncthreads();
    if (t == 0u) s_last = (atomicAdd(done, 1u) == gridDim.x - 1u) ? 1u : 0u;
    __syncthreads();
    if (s_last == 0u) return;
    __threadfence();

    // ── Selection: 8 warps, a row each ────────────────────────────────────
    __shared__ float s_sig[LMOE_ROUTE_WARPS][LMOE_ROUTE_MAX_E];
    __shared__ float s_sel[LMOE_ROUTE_WARPS][LMOE_ROUTE_MAX_E];
    __shared__ float s_bias[LMOE_ROUTE_MAX_E];
    for (unsigned int i = t; i < E; i += 256u) s_bias[i] = __ldcg(bias + i);
    __syncthreads();
    const unsigned int warp = t / 32u;
    for (unsigned int row = warp; row < M; row += LMOE_ROUTE_WARPS)
        ling_route_topk_row(logits + (unsigned long long)row * E, s_bias,
                            s_sig[warp], s_sel[warp], topk_ids, topk_weights,
                            row_live, row, E, top_k, n_group, topk_group,
                            routed_scaling_factor);
    __syncthreads();

    // ── Sort: counting sort over the (row, slot) assignments ──────────────
    __shared__ unsigned int s_cnt[LMOE_ROUTE_MAX_E];
    __shared__ unsigned int s_cur[LMOE_ROUTE_MAX_E];
    for (unsigned int e = t; e < E; e += 256u) s_cnt[e] = 0u;
    __syncthreads();
    const unsigned int slots = M * top_k;
    for (unsigned int i = t; i < slots; i += 256u)
        if (topk_ids[i] < E) atomicAdd(&s_cnt[topk_ids[i]], 1u);
    __syncthreads();

    if (t == 0u) {
        // Exclusive prefix over the live rows; dead slots (ids E) tail.
        unsigned int run = 0u, dead = 0u;
        for (unsigned int e = 0; e < E; ++e) {
            expert_offsets[e] = (int)run;
            s_cur[e] = run;
            run += s_cnt[e];
        }
        expert_offsets[E] = (int)run;
        for (unsigned int s = 0; s < slots; ++s) {
            const unsigned int e = topk_ids[s];
            if (e < E) {
                const unsigned int p = s_cur[e]++;
                sorted_token_ids[p] = (int)(s / top_k);
                token_to_perm[s] = (int)p;
            } else {
                token_to_perm[s] = (int)(run + dead++);
            }
        }
        // Worklist: (expert, mt) per m_tile rows; grid.x of the expert
        // kernels covers the column tiles.
        unsigned int items = 0u;
        for (unsigned int e = 0; e < E; ++e) {
            for (unsigned int mt = 0; mt * m_tile < s_cnt[e]; ++mt) {
                if (items >= max_items) goto full;
                worklist[1u + items * 2u] = e;
                worklist[2u + items * 2u] = mt;
                ++items;
            }
        }
    full:
        worklist[0] = items;
        *done = 0u;
    }
}

// ── Routed experts: gate/up fused swiglu, and down ─────────────────────────
//
// Grid (N / 128 gate/up — N / 256 down, bound >= worklist[0]). The gate/up
// kernel reads the token-major quantized activation through
// sorted_token_ids; the down kernel passes sorted_token_ids null and reads
// its NVFP4 intermediate by sorted row. `swiglu_limit` is the routed
// expert limit of the layer (0 = none: layers 2..34; 4 on 35..41 — the
// host reads expert_swiglu_limit_list).

extern "C" __global__ void __launch_bounds__(256, 2) ling_moe_gate_up_silu(
    const unsigned char* __restrict__ a_packed, const unsigned char* __restrict__ a_scale,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ C,               // unused (gate/up writes packed)
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K,
    const unsigned int* __restrict__ worklist, float swiglu_limit,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ out_packed, unsigned char* __restrict__ out_scale,
    const float* __restrict__ a2_gscale
) {
    lm_impl<true, 1>(a_packed, a_scale, gate_packed_ptrs, gate_scale_ptrs,
                     gate_scale2_vals, expert_offsets, sorted_token_ids, num_experts,
                     N, K, worklist, swiglu_limit,
                     LmGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, a2_gscale,
                              out_packed, out_scale},
                     C);
}

extern "C" __global__ void __launch_bounds__(256, 2) ling_moe_gate_up_silu_m32(
    const unsigned char* __restrict__ a_packed, const unsigned char* __restrict__ a_scale,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K,
    const unsigned int* __restrict__ worklist, float swiglu_limit,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ out_packed, unsigned char* __restrict__ out_scale,
    const float* __restrict__ a2_gscale
) {
    lm_impl<true, 2>(a_packed, a_scale, gate_packed_ptrs, gate_scale_ptrs,
                     gate_scale2_vals, expert_offsets, sorted_token_ids, num_experts,
                     N, K, worklist, swiglu_limit,
                     LmGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, a2_gscale,
                              out_packed, out_scale},
                     C);
}

extern "C" __global__ void __launch_bounds__(256, 2) ling_moe_down(
    const unsigned char* __restrict__ a_packed, const unsigned char* __restrict__ a_scale,
    const unsigned long long* __restrict__ down_packed_ptrs,
    const unsigned long long* __restrict__ down_scale_ptrs,
    const float* __restrict__ down_scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K,
    const unsigned int* __restrict__ worklist
) {
    lm_impl<false, 1>(a_packed, a_scale, down_packed_ptrs, down_scale_ptrs,
                      down_scale2_vals, expert_offsets, sorted_token_ids, num_experts,
                      N, K, worklist, 0.0f, LmGateUp{}, C);
}

extern "C" __global__ void __launch_bounds__(256, 2) ling_moe_down_m32(
    const unsigned char* __restrict__ a_packed, const unsigned char* __restrict__ a_scale,
    const unsigned long long* __restrict__ down_packed_ptrs,
    const unsigned long long* __restrict__ down_scale_ptrs,
    const float* __restrict__ down_scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K,
    const unsigned int* __restrict__ worklist
) {
    lm_impl<false, 2>(a_packed, a_scale, down_packed_ptrs, down_scale_ptrs,
                      down_scale2_vals, expert_offsets, sorted_token_ids, num_experts,
                      N, K, worklist, 0.0f, LmGateUp{}, C);
}

// ── Shared expert (BF16, one per layer, alongside the routed output) ───────
// Row-bucket exports; `limit` is the layer's share_expert_swiglu_limit_list
// value (0 = none). Grid (I/8) gate/up, (K/8) down.

#define LMOE_SHARED_EXPORT(MR)                                                          \
    extern "C" __global__ void __launch_bounds__(256) ling_moe_shared_gate_up_r##MR(    \
        const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ gate_w,  \
        const __nv_bfloat16* __restrict__ up_w, __nv_bfloat16* __restrict__ H,           \
        unsigned int M, unsigned int I, unsigned int K, float limit                      \
    ) { lm_shared_gate_up_impl<MR>(X, gate_w, up_w, H, M, I, K, limit); }                \
    extern "C" __global__ void __launch_bounds__(256) ling_moe_shared_down_r##MR(       \
        const __nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ w,        \
        __nv_bfloat16* __restrict__ Y, unsigned int M, unsigned int N,                  \
        unsigned int K                                                                   \
    ) { lm_shared_down_impl<MR>(H, w, Y, M, N, K); }

LMOE_SHARED_EXPORT(1)
LMOE_SHARED_EXPORT(4)
LMOE_SHARED_EXPORT(8)
LMOE_SHARED_EXPORT(16)
LMOE_SHARED_EXPORT(32)
#undef LMOE_SHARED_EXPORT

// ── Unpermute + shared add: one block per token row ───────────────────────
//
// y[r] = bf16(bf16(sum_k w_k * down[perm_k]) + shared[r]) — exactly the
// reference's chain (moe_infer's fp32 mul/sum typed to BF16, then the BF16
// add of the shared output, modeling :549-556). Dead verify rows (row_live)
// are zeroed, never reduced. top_k <= 8.
extern "C" __global__ void __launch_bounds__(256) ling_moe_unpermute(
    const __nv_bfloat16* __restrict__ expert_out,   // [M * top_k, D] (down C)
    const __nv_bfloat16* __restrict__ shared_out,   // [M, D]
    const int* __restrict__ token_to_perm,           // [M, top_k]
    const float* __restrict__ topk_weights,          // [M, top_k]
    const unsigned int* __restrict__ row_live,       // [M], or null
    __nv_bfloat16* __restrict__ Y,                   // [M, D]
    unsigned int M, unsigned int D, unsigned int top_k
) {
    atlas_pdl_enter();
    if (top_k > LMOE_MAX_TOPK || D % 8u != 0u) return;
    const unsigned int r = blockIdx.x;
    if (r >= M) return;
    __nv_bfloat16* yrow = Y + (unsigned long long)r * D;
    if (row_live != nullptr && row_live[r] == 0u) {
        for (unsigned int c = threadIdx.x * 8u; c < D; c += blockDim.x * 8u)
            *(uint4*)&yrow[c] = make_uint4(0, 0, 0, 0);
        return;
    }
    int rows[LMOE_MAX_TOPK] = {0, 0, 0, 0, 0, 0, 0, 0};
    float w[LMOE_MAX_TOPK] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int k = 0; k < top_k; ++k) {
        rows[k] = token_to_perm[r * top_k + k];
        w[k] = topk_weights[r * top_k + k];
    }
    for (unsigned int c = threadIdx.x * 8u; c < D; c += blockDim.x * 8u) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < top_k; ++k) {
            const uint4 v = *(const uint4*)&expert_out[(unsigned long long)rows[k] * D + c];
            const __nv_bfloat16* e = (const __nv_bfloat16*)&v;
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] += w[k] * __bfloat162float(e[j]);
        }
        const uint4 s = *(const uint4*)&shared_out[(unsigned long long)r * D + c];
        const __nv_bfloat16* sb = (const __nv_bfloat16*)&s;
        uint4 out;
        __nv_bfloat16* o = (__nv_bfloat16*)&out;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float routed = __bfloat162float(__float2bfloat16(acc[j]));
            o[j] = __float2bfloat16(routed + __bfloat162float(sb[j]));
        }
        *(uint4*)&yrow[c] = out;
    }
}
