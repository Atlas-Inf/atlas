// SPDX-License-Identifier: AGPL-3.0-only

// The MoE output seam's unpermute as the producer of a fused chain
// reduce-scatter (`ATLAS_GLM_SP_MOE_SEAM` with `ATLAS_RDMA_RS_FUSED_SEG`,
// `layers::glm_sp_moe_seam`): the `_vec8` unpermute kernels' rows
// (glm_moe_unpermute_row.cuh), computed piece by piece in the plan's order
// and written where the op wants them -- a raw piece straight into its
// pinned staging, a kept one in place in the seam buffer -- each published
// once every block wrote its rows (rdma_rs_fused.cuh), so the network starts
// on the first finished piece. Every row is the unpermute kernel's row, so
// the reduce-scatter sums the bytes it would have summed.
// Grid: (any, 1, 1)  Block: (256, 1, 1); every block runs every piece.
// `output` is the seam buffer's first row (token 0), the rows' token
// indices count from it.

#include "../../common/atlas_pdl.cuh"
#include "../../common/rdma_rs_fused.cuh"
#include "glm_moe_unpermute_row.cuh"

template <typename Row>
__device__ __forceinline__ void rs_fused_walk(
    const RsFusedPlan& plan, const __nv_bfloat16* output, unsigned int hidden_size, Row row) {
    rs_fused_begin_dev(plan);
    const unsigned long long row_bytes = (unsigned long long)hidden_size * 2;
    const unsigned int piece_rows = (unsigned int)(plan.piece_bytes / row_bytes);
    for (unsigned int i = 0; i < plan.npieces; i++) {
        const unsigned int t0 = (unsigned int)((plan.piece[i].out - (unsigned long long)output) / row_bytes);
        unsigned char* to = rs_fused_piece_dst(plan, i);
        const bool raw = rs_fused_piece_raw(plan, i);
        for (unsigned int r = blockIdx.x; r < piece_rows; r += gridDim.x) {
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(to + r * row_bytes);
            if (raw && plan.mode == 1)
                row.template run<true>(t0 + r, dst);
            else
                row.template run<false>(t0 + r, dst);
        }
        rs_fused_piece_done(plan, i, gridDim.x);
    }
}

struct RouteArgs {
    const int* token_to_perm;
    const int* topk_ids;
    const float* topk_weights;
    unsigned int topk, local_expert_start, local_expert_end;

    __device__ __forceinline__ glm_unpermute::Routes routes(unsigned int token) const {
        return glm_unpermute::load_routes(
            token, token_to_perm, topk_ids, topk_weights, topk, local_expert_start, local_expert_end);
    }
};

struct RowBf16 {
    RouteArgs a;
    const __nv_bfloat16* expert_output;
    unsigned int hidden_size;

    template <bool WT>
    __device__ __forceinline__ void run(unsigned int token, __nv_bfloat16* dst) const {
        glm_unpermute::row_vec8<WT>(a.routes(token), expert_output, dst, hidden_size);
    }
};

struct RowE4m3 {
    RouteArgs a;
    const unsigned char* expert_output;
    const float* scales;
    unsigned int hidden_size;

    template <bool WT>
    __device__ __forceinline__ void run(unsigned int token, __nv_bfloat16* dst) const {
        glm_unpermute::row_vec8_e4m3<WT>(a.routes(token), expert_output, scales, dst, hidden_size);
    }
};

// moe_unpermute_reduce_indexed_ep_vec8's arguments (no row count: the plan's
// pieces name the rows), then the plan.
extern "C" __global__ void __launch_bounds__(256) moe_unpermute_rs_fused_vec8(
    const __nv_bfloat16* __restrict__ expert_output,
    const __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int topk,
    unsigned int local_expert_start,
    unsigned int local_expert_end,
    const RsFusedPlan plan
) {
    atlas_pdl_enter();
    const RowBf16 row{{token_to_perm, topk_ids, topk_weights, topk, local_expert_start, local_expert_end},
                      expert_output, hidden_size};
    rs_fused_walk(plan, output, hidden_size, row);
}

// The `_e4m3` twin (ATLAS_GLM_MOE_DOWN_E4M3): E4M3 expert rows and their
// FP32 scales.
extern "C" __global__ void __launch_bounds__(256) moe_unpermute_rs_fused_vec8_e4m3(
    const unsigned char* __restrict__ expert_output,
    const __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int topk,
    unsigned int local_expert_start,
    unsigned int local_expert_end,
    const float* __restrict__ scales,
    const RsFusedPlan plan
) {
    atlas_pdl_enter();
    const RowE4m3 row{{token_to_perm, topk_ids, topk_weights, topk, local_expert_start, local_expert_end},
                      expert_output, scales, hidden_size};
    rs_fused_walk(plan, output, hidden_size, row);
}
