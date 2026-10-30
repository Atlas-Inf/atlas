// SPDX-License-Identifier: AGPL-3.0-only

// GLM routed-MoE routing chain in two launches (ATLAS_GLM_MOE_ROUTE_FUSE=1).
//
// A canonical verify block (1..32 rows) reaches its routed gate/up through
// five dependent launches per MoE layer: the router GEMV
// (dense_gemv_bf16_tc{8,16,32}), moe_topk_sigmoid_batched,
// moe_sort_by_expert_scan, moe_build_tile_worklist_scan and
// quantize_bf16_to_nvfp4. Each is a few microseconds of work on one to
// eighteen CTAs, so the chain is launch latency, and the router reads its
// 2.4 MB weight from DRAM on eighteen CTAs. Here:
//
// * glm_moe_router_quant: the router CTAs ask their weight rows
//   into L2 while the launch waits on its PDL predecessor (the shared expert;
//   atlas_pdl_touch.cuh), then run the tier's own body
//   (dense_gemv_bf16_tc_impl<NT>), so every logit is the same BF16 word.
//   The CTAs past them quantize the MoE input to NVFP4 with
//   quantize_bf16_to_nvfp4's per-group body: groups are independent, so the
//   same bytes.
// * glm_moe_route_select: one CTA runs the sigmoid top-k of every row (a
//   warp per row; see below), then moe_sort_by_expert_scan's and
//   moe_build_tile_worklist_scan's bodies on its result.
//
// Top-k: moe_topk_sigmoid_batched takes, eight times, the largest selection
// score with the lowest index among equal ones (strict `>` in each thread's
// ascending scan, lower index on a tie in every reduction), marks it taken,
// and sums the chosen pre-bias scores in that order on one thread. That
// winner is a function of the scores alone, not of how the scan is split
// over threads (a NaN never wins either way; a thread holding nothing above
// the -1e30 sentinel offers index 0 in both), so one warp scanning a row
// picks the same experts in the same order, and lane 0 repeats the weight
// arithmetic. Same index and weight words.
//
// Output contract: what the five launches write (logits excepted: the
// select overwrites them with the sort's outputs, as the sort did), with the
// sort's freedom of row order within an expert group.

#include "dense_gemv_bf16_batchm.cu"
#include "moe_permute.cu"
#include "../../common/quantize_bf16_to_nvfp4.cu"
#include "atlas_pdl_touch.cuh"

#define ROUTE_MAX_EXPERTS 512   // moe_topk_sigmoid.cu MAX_EXPERTS
#define ROUTE_MAX_TOP_K 32      // moe_topk_sigmoid.cu MAX_TOP_K
#define ROUTE_WARPS 8

// Router CTAs `[0, router_ctas)`: rows of `gate_w` (`[E, K]` BF16) 16 per
// CTA into `logits` (`[M, E]` BF16). The rest: NVFP4 groups of `A`
// (`[M, K]`), 256 per CTA, into `a_packed` (`[M, K/2]`) and `a_scale`
// (`[M, K/16]`) with scale2 = `a_scale2`: 1 (gemm_fp4.rs quantize_bf16_to_nvfp4), or
// the gate/up input scale under ATLAS_GLM_FP4L_INPUT_SCALES (input_scales.rs).
template <int NT>
__device__ __forceinline__ void glm_moe_router_quant_impl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ gate_w,
    __nv_bfloat16* __restrict__ logits, unsigned char* __restrict__ a_packed,
    unsigned char* __restrict__ a_scale, unsigned int M, unsigned int E, unsigned int K,
    unsigned int router_ctas, float a_scale2
) {
    if (blockIdx.x < router_ctas) {
        dense_gemv_bf16_tc_impl<NT>(A, reinterpret_cast<const __nv_bfloat16*>(gate_w), logits,
                                    M, E, K, E, K);
        return;
    }
    const unsigned int groups = K / GROUP_SIZE;
    const unsigned int q = (blockIdx.x - router_ctas) * blockDim.x + threadIdx.x;
    if (q >= M * groups) return;
    const unsigned int row = q / groups;
    nvfp4_quantize_group(A + (unsigned long long)row * K,
                         a_packed + (unsigned long long)row * (K / 2),
                         a_scale + (unsigned long long)row * groups, q % groups, a_scale2,
                         1.0f / a_scale2);
}

// `touch_rows` (E, or 0 for none) router weight rows and the correction
// bias, immutable weights, are asked into L2 by the router CTAs before the
// wait. Grid (router_ctas + quant CTAs, 1, 1), block (256, 1, 1); the row
// count picks the canonical tier's activation tiles (dense_tier: NT = 1 up to
// 8 rows, 2 up to 16, 4 up to 32).
extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE) glm_moe_router_quant(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ bias, __nv_bfloat16* __restrict__ logits,
    unsigned char* __restrict__ a_packed, unsigned char* __restrict__ a_scale, unsigned int M,
    unsigned int E, unsigned int K, unsigned int router_ctas, unsigned int touch_rows,
    float a_scale2
) {
    atlas_pdl_enter_touch({gate_w, K * 2u, K * 2ull}, {bias, 4u, 4ull}, touch_rows, blockIdx.x,
                          router_ctas);
    if (M <= 8u)
        glm_moe_router_quant_impl<1>(A, gate_w, logits, a_packed, a_scale, M, E, K, router_ctas,
                                        a_scale2);
    else if (M <= 16u)
        glm_moe_router_quant_impl<2>(A, gate_w, logits, a_packed, a_scale, M, E, K, router_ctas,
                                        a_scale2);
    else
        glm_moe_router_quant_impl<4>(A, gate_w, logits, a_packed, a_scale, M, E, K, router_ctas,
                                        a_scale2);
}

// The sigmoid top-k of row `row` on one warp (moe_topk_sigmoid_batched's
// arithmetic; see the file comment), into `ids` / `wts` at row * top_k.
__device__ __forceinline__ void glm_moe_route_topk_row(
    const __nv_bfloat16* __restrict__ logits, const float* __restrict__ bias,
    unsigned int* __restrict__ ids, float* __restrict__ wts, float* s_sig, float* s_sel,
    unsigned int row, unsigned int E, unsigned int top_k, unsigned int normalize,
    float scaling_factor
) {
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = E < ROUTE_MAX_EXPERTS ? E : ROUTE_MAX_EXPERTS;
    const unsigned int k = top_k < ROUTE_MAX_TOP_K ? top_k : ROUTE_MAX_TOP_K;
    const __nv_bfloat16* my = logits + (unsigned long long)row * E;
    for (unsigned int i = lane; i < n; i += WARP_SIZE) {
        const float logit = __bfloat162float(my[i]);
        const float sig = 1.0f / (1.0f + __expf(-logit));
        s_sig[i] = sig;
        s_sel[i] = sig + bias[i];
    }
    __syncwarp();
    unsigned int top[ROUTE_MAX_TOP_K];
    for (unsigned int t = 0; t < k && t < n; t++) {
        float best = -1e30f;
        unsigned int at = 0;
        for (unsigned int i = lane; i < n; i += WARP_SIZE) {
            const float v = s_sel[i];
            if (v > best) {
                best = v;
                at = i;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, offset);
            const unsigned int other_at = __shfl_xor_sync(0xFFFFFFFF, at, offset);
            if (other > best || (other == best && other_at < at)) {
                best = other;
                at = other_at;
            }
        }
        top[t] = at;
        if (lane == 0) s_sel[at] = -1e30f;
        __syncwarp();
    }
    if (lane == 0) {
        float vals[ROUTE_MAX_TOP_K];
        float topk_sum = 0.0f;
        for (unsigned int t = 0; t < k && t < n; t++) {
            vals[t] = s_sig[top[t]];
            topk_sum += vals[t];
        }
        if (normalize && topk_sum > 1e-20f) {
            for (unsigned int t = 0; t < k && t < n; t++) vals[t] /= topk_sum;
        }
        for (unsigned int t = 0; t < k && t < n; t++) {
            ids[(unsigned long long)row * top_k + t] = top[t];
            wts[(unsigned long long)row * top_k + t] = vals[t] * scaling_factor;
        }
    }
    __syncwarp();
}

// Grid (1, 1, 1), block (256, 1, 1). `logits` is the router's output; the
// sort's `sorted_token_ids` / `sorted_expert_ids` / `expert_offsets` /
// `token_to_perm` may overlay it (forward_prefill.rs places them in the
// gate-logit buffer): the top-k reads every logit before the sort writes.
// These aliased arguments deliberately have no restrict promise: the barrier
// orders their access phases but does not make the allocations disjoint.
// `worklist` / `total_tiles` / `n_tiles` / `m_tile` as
// moe_build_tile_worklist_scan.
extern "C" __global__ void __launch_bounds__(ROUTE_WARPS * WARP_SIZE) glm_moe_route_select(
    const __nv_bfloat16* logits, const float* __restrict__ bias,
    unsigned int* __restrict__ topk_ids, float* __restrict__ topk_weights,
    int* sorted_token_ids, int* sorted_expert_ids,
    int* expert_offsets, int* token_to_perm,
    const unsigned long long* __restrict__ B_weight_ptrs, unsigned int* __restrict__ worklist,
    int* __restrict__ total_tiles, unsigned int M, unsigned int E, unsigned int top_k,
    unsigned int normalize, float scaling_factor, unsigned int n_tiles, unsigned int m_tile
) {
    atlas_pdl_enter();
    __shared__ float s_sig[ROUTE_WARPS][ROUTE_MAX_EXPERTS];
    __shared__ float s_sel[ROUTE_WARPS][ROUTE_MAX_EXPERTS];
    if (blockDim.x != ROUTE_WARPS * WARP_SIZE) return;
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    for (unsigned int row = warp; row < M; row += ROUTE_WARPS)
        glm_moe_route_topk_row(logits, bias, topk_ids, topk_weights, s_sig[warp], s_sel[warp], row,
                               E, top_k, normalize, scaling_factor);
    // The block's global writes (top-k ids, then expert offsets) are visible
    // to the whole block after each barrier.
    __syncthreads();
    moe_sort_by_expert_scan_body(topk_ids, sorted_token_ids, sorted_expert_ids, expert_offsets,
                                 token_to_perm, M * top_k, E, top_k);
    __syncthreads();
    moe_build_tile_worklist_scan_body(expert_offsets, B_weight_ptrs, worklist, total_tiles, E,
                                      n_tiles, m_tile);
}
