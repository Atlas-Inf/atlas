// SPDX-License-Identifier: AGPL-3.0-only

// GLM routed-MoE routing chain in two launches (ATLAS_GLM_MOE_ROUTE_FUSE=1).
//
// A canonical verify block (1..32 rows) reaches its routed gate/up through
// five dependent launches per MoE layer: the router GEMV
// (dense_gemv_bf16_tc{8,16,32}_f32), moe_topk_sigmoid_batched_f32,
// moe_sort_by_expert_scan, moe_build_tile_worklist_scan and
// quantize_bf16_to_nvfp4. Each is a few microseconds of work on one to
// eighteen CTAs, so the chain is launch latency, and the router reads its
// 2.4 MB weight from DRAM on eighteen CTAs. Here:
//
// * glm_moe_router_quant: the router CTAs ask their weight rows
//   into L2 while the launch waits on its PDL predecessor (the shared expert;
//   atlas_pdl_touch.cuh), then run the tier's own body
//   (dense_gemv_bf16_tc_impl<NT>), so every logit is the same FP32 word.
//   The logits stay FP32 end to end (the checkpoint's
//   `moe_router_dtype: float32`): BF16 inputs, FP32 accumulation, an FP32
//   logit into the sigmoid, as the reference's GateLinear(out_dtype=float32).
//   The CTAs past them quantize the MoE input to NVFP4 with
//   quantize_bf16_to_nvfp4's per-group body: groups are independent, so the
//   same bytes.
// * glm_moe_route_select: one CTA runs the sigmoid top-k of every row (a
//   warp per row; see below), then moe_sort_by_expert_scan's and
//   moe_build_tile_worklist_scan's bodies on its result.
//
// Top-k: moe_topk_sigmoid_batched_f32 takes, eight times, the largest selection
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

// Quant CTA `qcta`: NVFP4 groups 256 * qcta .. of `A` (`[M, K]`) into
// `a_packed` (`[M, K/2]`) and `a_scale` (`[M, K/16]`) with scale2 =
// `a_scale2` (glm_moe_router_quant_impl).
__device__ __forceinline__ void glm_moe_route_quant_cta(
    const __nv_bfloat16* __restrict__ A, unsigned char* __restrict__ a_packed,
    unsigned char* __restrict__ a_scale, unsigned int M, unsigned int K, unsigned int qcta,
    float a_scale2
) {
    const unsigned int groups = K / GROUP_SIZE;
    const unsigned int q = qcta * blockDim.x + threadIdx.x;
    if (q >= M * groups) return;
    const unsigned int row = q / groups;
    nvfp4_quantize_group(A + (unsigned long long)row * K,
                         a_packed + (unsigned long long)row * (K / 2),
                         a_scale + (unsigned long long)row * groups, q % groups, a_scale2,
                         1.0f / a_scale2);
}

// Router CTAs `[0, router_ctas)`: rows of `gate_w` (`[E, K]` BF16) 16 per
// CTA into `logits` (`[M, E]` FP32). The rest: NVFP4 groups of `A`
// (`[M, K]`), 256 per CTA, into `a_packed` (`[M, K/2]`) and `a_scale`
// (`[M, K/16]`) with scale2 = `a_scale2`: 1 (gemm_fp4.rs quantize_bf16_to_nvfp4), or
// the gate/up input scale under ATLAS_GLM_FP4L_INPUT_SCALES (input_scales.rs).
template <int NT>
__device__ __forceinline__ void glm_moe_router_quant_impl(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ gate_w,
    float* __restrict__ logits, unsigned char* __restrict__ a_packed,
    unsigned char* __restrict__ a_scale, unsigned int M, unsigned int E, unsigned int K,
    unsigned int router_ctas, float a_scale2
) {
    if (blockIdx.x < router_ctas) {
        dense_gemv_bf16_tc_impl<NT>(A, reinterpret_cast<const __nv_bfloat16*>(gate_w), logits,
                                    M, E, K, E, K);
        return;
    }
    glm_moe_route_quant_cta(A, a_packed, a_scale, M, K, blockIdx.x - router_ctas, a_scale2);
}

// `touch_rows` (E, or 0 for none) router weight rows and the correction
// bias, immutable weights, are asked into L2 by the router CTAs before the
// wait. Grid (router_ctas + quant CTAs, 1, 1), block (256, 1, 1); the row
// count picks the canonical tier's activation tiles (dense_tier: NT = 1 up to
// 8 rows, 2 up to 16, 4 up to 32).
extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE) glm_moe_router_quant(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ bias, float* __restrict__ logits,
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

// Selection scores ordered as unsigned words (no NaN reaches it; -0 is +0,
// so equal scores give equal words).
__device__ __forceinline__ unsigned int glm_route_key(float v) {
    const unsigned int u = __float_as_uint(v == 0.0f ? 0.0f : v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// glm_moe_route_topk_row for E <= 32 * ROUTE_REG_SLOTS experts with each
// lane's selection scores in registers: every round a lane offers its best
// untaken score (strict `>` over its ascending experts: the lowest index on a
// tie, -1e30 and index 0 if it holds none above the sentinel, a NaN never),
// the warp takes the largest score and, among the lanes offering it, the
// lowest index (two redux.sync), and only the winner's lane rescans. The
// same winners in the same order as the shuffle reduction, and lane 0 repeats
// the weight arithmetic: the same index and weight words.
#define ROUTE_REG_SLOTS 10
__device__ __forceinline__ void glm_moe_route_topk_row_regs(
    const float* __restrict__ logits, const float* __restrict__ bias,
    unsigned int* __restrict__ ids, float* __restrict__ wts, float* s_sig,
    unsigned int row, unsigned int E, unsigned int top_k, unsigned int normalize,
    float scaling_factor
) {
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int k = top_k < ROUTE_MAX_TOP_K ? top_k : ROUTE_MAX_TOP_K;
    const float* my = logits + (unsigned long long)row * E;
    float sel[ROUTE_REG_SLOTS];
    #pragma unroll
    for (int j = 0; j < ROUTE_REG_SLOTS; j++) {
        const unsigned int i = lane + 32u * j;
        sel[j] = -1e30f;
        if (i < E) {
            const float logit = my[i];
            const float sig = 1.0f / (1.0f + __expf(-logit));
            s_sig[i] = sig;
            sel[j] = sig + bias[i];
        }
    }
    __syncwarp();
    // This lane's offer: its best score above the sentinel, lowest index.
    auto offer = [&](float& best, unsigned int& at) {
        best = -1e30f;
        at = 0u;
        #pragma unroll
        for (int j = 0; j < ROUTE_REG_SLOTS; j++) {
            if (sel[j] > best) {
                best = sel[j];
                at = lane + 32u * j;
            }
        }
    };
    float best;
    unsigned int at;
    offer(best, at);
    // Lane t keeps round t's winner.
    const unsigned int rounds = k < E ? k : E;
    unsigned int mine = 0u;
    for (unsigned int t = 0; t < rounds; t++) {
        const unsigned int key = glm_route_key(best);
        const unsigned int top_key = __reduce_max_sync(0xFFFFFFFFu, key);
        const unsigned int win = __reduce_min_sync(0xFFFFFFFFu, key == top_key ? at : 0xFFFFFFFFu);
        if (lane == t) mine = win;
        if (win % 32u == lane && key == top_key && at == win) {
            #pragma unroll
            for (int j = 0; j < ROUTE_REG_SLOTS; j++)
                if (lane + 32u * j == win) sel[j] = -1e30f;
            offer(best, at);
        }
    }
    // The weights: every lane sums the winners' sigmoids in round order (the
    // serial sum's additions), then lane t normalizes and stores round t's.
    const float val = lane < rounds ? s_sig[mine] : 0.0f;
    float topk_sum = 0.0f;
    for (unsigned int t = 0; t < rounds; t++) topk_sum += __shfl_sync(0xFFFFFFFFu, val, t);
    if (lane < rounds) {
        const float v = (normalize && topk_sum > 1e-20f) ? val / topk_sum : val;
        ids[(unsigned long long)row * top_k + lane] = mine;
        wts[(unsigned long long)row * top_k + lane] = v * scaling_factor;
    }
    __syncwarp();
}

// The sigmoid top-k of row `row` on one warp (moe_topk_sigmoid_batched_f32's
// arithmetic; see the file comment), into `ids` / `wts` at row * top_k.
__device__ __forceinline__ void glm_moe_route_topk_row(
    const float* __restrict__ logits, const float* __restrict__ bias,
    unsigned int* __restrict__ ids, float* __restrict__ wts, float* s_sig, float* s_sel,
    unsigned int row, unsigned int E, unsigned int top_k, unsigned int normalize,
    float scaling_factor
) {
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = E < ROUTE_MAX_EXPERTS ? E : ROUTE_MAX_EXPERTS;
    const unsigned int k = top_k < ROUTE_MAX_TOP_K ? top_k : ROUTE_MAX_TOP_K;
    const float* my = logits + (unsigned long long)row * E;
    for (unsigned int i = lane; i < n; i += WARP_SIZE) {
        const float logit = my[i];
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
// `worklist` / `total_tiles` / `n_tiles` / `m_tile` as
// moe_build_tile_worklist_scan.
extern "C" __global__ void __launch_bounds__(ROUTE_WARPS * WARP_SIZE) glm_moe_route_select(
    const float* __restrict__ logits, const float* __restrict__ bias,
    unsigned int* __restrict__ topk_ids, float* __restrict__ topk_weights,
    int* __restrict__ sorted_token_ids, int* __restrict__ sorted_expert_ids,
    int* __restrict__ expert_offsets, int* __restrict__ token_to_perm,
    const unsigned long long* __restrict__ B_weight_ptrs, unsigned int* __restrict__ worklist,
    int* __restrict__ total_tiles, unsigned int M, unsigned int E, unsigned int top_k,
    unsigned int normalize, float scaling_factor, unsigned int n_tiles, unsigned int m_tile,
    const unsigned int* __restrict__ row_live
) {
    atlas_pdl_enter();
    if (blockDim.x != ROUTE_WARPS * WARP_SIZE || E > ROUTE_MAX_EXPERTS) return;
    __shared__ float s_sig[ROUTE_WARPS][ROUTE_MAX_EXPERTS];
    __shared__ float s_sel[ROUTE_WARPS][ROUTE_MAX_EXPERTS];
    // The correction bias (in L2: the router touched it) and the local-expert
    // pointer table, read once into shared memory (the same words).
    __shared__ float s_bias[ROUTE_MAX_EXPERTS];
    __shared__ unsigned long long s_ptrs[ROUTE_MAX_EXPERTS];
    for (unsigned int i = threadIdx.x; i < E; i += blockDim.x) {
        s_bias[i] = bias[i];
        s_ptrs[i] = B_weight_ptrs[i];
    }
    __syncthreads();
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    for (unsigned int row = warp; row < M; row += ROUTE_WARPS) {
        if (E <= 32u * ROUTE_REG_SLOTS)
            glm_moe_route_topk_row_regs(logits, s_bias, topk_ids, topk_weights, s_sig[warp], row, E,
                                        top_k, normalize, scaling_factor);
        else
            glm_moe_route_topk_row(logits, s_bias, topk_ids, topk_weights, s_sig[warp], s_sel[warp],
                                   row, E, top_k, normalize, scaling_factor);
    }
    // The block's global writes (top-k ids, then expert offsets) are visible
    // to the whole block after each barrier.
    __syncthreads();
    // Dead verify rows (moe_route_mask_rows; `row_live` null for none).
    if (row_live != nullptr) {
        for (unsigned int i = threadIdx.x; i < M * top_k; i += blockDim.x)
            if (row_live[i / top_k] == 0u) topk_ids[i] = E;
        __syncthreads();
    }
    moe_sort_by_expert_scan_body(topk_ids, sorted_token_ids, sorted_expert_ids, expert_offsets,
                                 token_to_perm, M * top_k, E, top_k);
    __syncthreads();
    moe_build_tile_worklist_scan_body(expert_offsets, s_ptrs, worklist, total_tiles, E,
                                      n_tiles, m_tile);
}

// glm_moe_router_quant and glm_moe_route_select in ONE launch for a batch of
// at most 8 rows of at most ROUTE_ONE_EXPERTS experts. The router's 16-row
// tiles' warp shares (dense_gemv_bf16_tc_warp_acc: tile t's warp w runs
// chunks w, w + 8, ... of K) are spread ROUTE_ONE_VW to a CTA over
// `router_ctas` CTAs instead of eight to a CTA over E / 16 (a cold router
// weight streams from about three times as many SMs), each share's C
// fragments into `partials`; the CTAs past them quantize as
// glm_moe_router_quant's. The last CTA to finish (counted in `done`, which
// it returns to 0) sums each tile's eight shares in warp order, the tier's
// own reduction, into shared memory and runs glm_moe_route_select's body on
// those logits, leaving them in `logits` too as glm_moe_router_quant does.
// The same per-share chains and the same sums, so the same logit words and
// the same outputs.
// Grid (router_ctas + quant CTAs, 1, 1), router_ctas = ceil(E / 16 * 8 /
// ROUTE_ONE_VW), block (256, 1, 1); `done` a device word that is 0 between
// launches; `partials` E / 16 * 8 * 32 * 4 floats.
#define ROUTE_ONE_EXPERTS (32 * ROUTE_REG_SLOTS)
#define ROUTE_ONE_VW 3
extern "C" __global__ void __launch_bounds__(DG_TC_WARPS * WARP_SIZE) glm_moe_route_one(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ bias, float* __restrict__ logits,
    float* __restrict__ partials, unsigned char* __restrict__ a_packed,
    unsigned char* __restrict__ a_scale,
    unsigned int router_ctas, unsigned int touch_rows,
    unsigned int* __restrict__ topk_ids, float* __restrict__ topk_weights,
    int* __restrict__ sorted_token_ids, int* __restrict__ sorted_expert_ids,
    int* __restrict__ expert_offsets, int* __restrict__ token_to_perm,
    const unsigned long long* __restrict__ B_weight_ptrs, unsigned int* __restrict__ worklist,
    int* __restrict__ total_tiles, unsigned int M, unsigned int E, unsigned int K,
    unsigned int top_k, unsigned int normalize, float scaling_factor, unsigned int n_tiles,
    unsigned int m_tile, unsigned int* __restrict__ done, const unsigned int* __restrict__ row_live,
    float a_scale2
) {
    // `bias` is the f32 correction bias, passed as bytes for the touch.
    atlas_pdl_enter_touch({gate_w, K * 2u, K * 2ull}, {bias, 4u, 4ull}, touch_rows, blockIdx.x,
                          router_ctas);
    if (blockDim.x != ROUTE_WARPS * WARP_SIZE || M > 8u || E > ROUTE_ONE_EXPERTS) return;
    const unsigned int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const unsigned int shares = (E + 15u) / 16u * DG_TC_WARPS;
    if (blockIdx.x < router_ctas) {
        const unsigned int share = blockIdx.x * ROUTE_ONE_VW + warp;
        if (warp < ROUTE_ONE_VW && share < shares) {
            float acc[1][4];
            dense_gemv_bf16_tc_warp_acc<1, DG_TC_DEEP>(
                acc, A, reinterpret_cast<const __nv_bfloat16*>(gate_w), M, E, K, K,
                share / DG_TC_WARPS, share % DG_TC_WARPS);
            reinterpret_cast<float4*>(partials)[share * WARP_SIZE + lane] =
                make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);
        }
    } else {
        glm_moe_route_quant_cta(A, a_packed, a_scale, M, K, blockIdx.x - router_ctas, a_scale2);
    }
    __shared__ unsigned int s_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = atomicAdd(done, 1u) == gridDim.x - 1u;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // The correction bias and the pointer table (weights) for the last CTA.
    __shared__ float s_bias[ROUTE_ONE_EXPERTS];
    __shared__ unsigned long long s_ptrs[ROUTE_ONE_EXPERTS];
    for (unsigned int i = threadIdx.x; i < E; i += blockDim.x) {
        s_bias[i] = reinterpret_cast<const float*>(bias)[i];
        s_ptrs[i] = B_weight_ptrs[i];
    }
    // The other CTAs wrote the shares during this launch: read them through
    // L2 (never the non-coherent path). Lane `l` of tile t's fragment q is
    // weight row 16t + l / 4 (+ 8 for q >= 2), activation row 2 (l % 4) +
    // q % 2 (dense_gemv_bf16_tc_cta's C layout); summed from 0 in warp order.
    __shared__ float s_logits[8 * ROUTE_ONE_EXPERTS];
    for (unsigned int i = threadIdx.x; i < shares / DG_TC_WARPS * WARP_SIZE * 4u; i += blockDim.x) {
        const unsigned int t = i / (WARP_SIZE * 4u), l = (i / 4u) % WARP_SIZE, q = i % 4u;
        const unsigned int n = t * 16u + l / 4u + (q < 2u ? 0u : 8u), m = 2u * (l % 4u) + (q & 1u);
        float p[DG_TC_WARPS];
        #pragma unroll
        for (int w = 0; w < DG_TC_WARPS; w++)
            p[w] = __ldcg(partials + ((t * DG_TC_WARPS + w) * WARP_SIZE + l) * 4u + q);
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < DG_TC_WARPS; w++) v += p[w];
        if (n < E && m < M) s_logits[m * E + n] = logits[m * E + n] = v;
    }
    __shared__ float s_sig[ROUTE_WARPS][ROUTE_ONE_EXPERTS];
    __syncthreads();
    for (unsigned int row = warp; row < M; row += ROUTE_WARPS)
        glm_moe_route_topk_row_regs(s_logits, s_bias, topk_ids, topk_weights, s_sig[warp], row, E,
                                    top_k, normalize, scaling_factor);
    __syncthreads();
    // Dead verify rows (moe_route_mask_rows; `row_live` null for none).
    if (row_live != nullptr) {
        for (unsigned int i = threadIdx.x; i < M * top_k; i += blockDim.x)
            if (row_live[i / top_k] == 0u) topk_ids[i] = E;
        __syncthreads();
    }
    moe_sort_by_expert_scan_body(topk_ids, sorted_token_ids, sorted_expert_ids, expert_offsets,
                                 token_to_perm, M * top_k, E, top_k);
    __syncthreads();
    moe_build_tile_worklist_scan_body(expert_offsets, s_ptrs, worklist, total_tiles, E, n_tiles,
                                      m_tile);
    if (threadIdx.x == 0) *done = 0u;
}
