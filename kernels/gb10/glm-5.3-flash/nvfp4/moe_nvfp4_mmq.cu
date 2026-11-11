// SPDX-License-Identifier: AGPL-3.0-only
//
// Grouped routed-expert NVFP4 W4A4 MMQ for small expert batches on GB10.
//
// This reuses Atlas's vendored Blackwell MMQ primitive rather than introducing
// another GEMM implementation.  Weights use the equal-size block_nvfp4 layout
// already proven by the dense Qwen path (36 bytes per 64 values, exactly the
// checkpoint's 32 packed bytes plus four scale bytes).  Expert routing remains
// Atlas-native: expert_offsets describe contiguous sorted rows and pointer
// tables preserve EP NULL slots.

#include <cuda_bf16.h>
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"
#include "../../qwen3.6-27b/nvfp4/q4k_vendor/quantize_impl.cuh"

// GLM's measured routed load at a 1K prompt is around 56 rows/expert/rank.
// A 64-row tile avoids the 128-row MMQ's wasted issue slots while normally
// streaming each expert weight only once.
static constexpr int ATLAS_MOE_MMQ_X = 64;

template <bool GateUp>
static __device__ __forceinline__ void atlas_moe_nvfp4_mmq64_impl(
        const unsigned long long * __restrict__ packed_ptrs_a,
        const unsigned long long * __restrict__ packed_ptrs_b,
        const int * __restrict__ y,
        __nv_bfloat16 * __restrict__ out_a,
        __nv_bfloat16 * __restrict__ out_b,
        const int * __restrict__ expert_offsets,
        int num_experts, int nrows_x, int ncols_y, int ncols_x) {
    constexpr ggml_type type = GGML_TYPE_NVFP4;
    constexpr int mmq_y = get_mmq_y_device();
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int packed_expert = (int) blockIdx.z;
    const int expert = GateUp ? packed_expert / 2 : packed_expert;
    const int projection = GateUp ? packed_expert & 1 : 0;
    if (expert >= num_experts) return;

    const unsigned long long raw = projection == 0
        ? packed_ptrs_a[expert]
        : packed_ptrs_b[expert];
    if (raw == 0) return; // EP-remote placeholder.

    const int m_start = expert_offsets[expert];
    const int m_end = expert_offsets[expert + 1];
    const int m_local = (int) blockIdx.y * ATLAS_MOE_MMQ_X;
    const int m_count = m_end - m_start;
    if (m_local >= m_count) return;

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < ATLAS_MOE_MMQ_X; j0 += nwarps * warp_size) {
        const int j = j0 + (int) threadIdx.y * warp_size + (int) threadIdx.x;
        if (j0 + nwarps * warp_size > ATLAS_MOE_MMQ_X && j >= ATLAS_MOE_MMQ_X) break;
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int it = (int) blockIdx.x;
    const int offset_x = it * mmq_y * (ncols_x / qk);
    // block_fp4_mmq is K-block-major: moving to row r advances one block
    // within every K block, while ncols_y remains the full sorted-row stride.
    const int offset_y = (m_start + m_local) * (int) (sizeof(block_fp4_mmq) / sizeof(int));
    const int offset_dst = (m_start + m_local) * nrows_x + it * mmq_y;
    const int tile_x_max_i = nrows_x - it * mmq_y - 1;
    const int tile_y_max_j = m_count - m_local - 1;
    const int kb0_stop = ncols_x / qk;

    __nv_bfloat16 * out = projection == 0 ? out_a : out_b;
    const char * x = reinterpret_cast<const char *>(raw);
    mul_mat_q_process_tile<type, ATLAS_MOE_MMQ_X, false, false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, out + offset_dst, nullptr,
        ncols_x / qk, ncols_y, nrows_x, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

extern "C" __global__ void __launch_bounds__(256, 1) atlas_moe_nvfp4_mmq64_gate_up(
        const unsigned long long * packed_gate_ptrs,
        const unsigned long long * packed_up_ptrs,
        const int * y,
        __nv_bfloat16 * gate_out,
        __nv_bfloat16 * up_out,
        const int * expert_offsets,
        int num_experts, int nrows_x, int ncols_y, int ncols_x) {
    atlas_moe_nvfp4_mmq64_impl<true>(
        packed_gate_ptrs, packed_up_ptrs, y, gate_out, up_out,
        expert_offsets, num_experts, nrows_x, ncols_y, ncols_x);
}

extern "C" __global__ void __launch_bounds__(256, 1) atlas_moe_nvfp4_mmq64_down(
        const unsigned long long * packed_down_ptrs,
        const int * y,
        __nv_bfloat16 * out,
        const int * expert_offsets,
        int num_experts, int nrows_x, int ncols_y, int ncols_x) {
    atlas_moe_nvfp4_mmq64_impl<false>(
        packed_down_ptrs, nullptr, y, out, nullptr,
        expert_offsets, num_experts, nrows_x, ncols_y, ncols_x);
}

// Same activation format as Atlas's dense NVFP4 MMQ path.
extern "C" __global__ void atlas_moe_nvfp4_quantize_bf16(
        const __nv_bfloat16 * x, const int * sorted_token_ids, void * y,
        long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_nvfp4_worker<__nv_bfloat16>(
        x, sorted_token_ids, y, ne00, s01, 0, 0, ne0, ne1, 1);
}

// Batched checkpoint-layout -> block_nvfp4 repack.  Source/destination pointer
// tables keep global expert numbering, including NULL EP-remote entries.
extern "C" __global__ void atlas_moe_nvfp4_repack_batched(
        const unsigned long long * __restrict__ packed_ptrs,
        const unsigned long long * __restrict__ scale_ptrs,
        const unsigned long long * __restrict__ out_ptrs,
        int n_rows, int k, int num_experts) {
    const int expert = (int) blockIdx.y;
    if (expert >= num_experts) return;
    const uint8_t * packed = reinterpret_cast<const uint8_t *>(packed_ptrs[expert]);
    const uint8_t * scales = reinterpret_cast<const uint8_t *>(scale_ptrs[expert]);
    uint8_t * out = reinterpret_cast<uint8_t *>(out_ptrs[expert]);
    if (packed == nullptr || scales == nullptr || out == nullptr) return;

    const int blocks_per_row = k / QK_NVFP4;
    const int64_t nblocks = (int64_t) n_rows * blocks_per_row;
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const int row = (int) (b / blocks_per_row);
    const int kb = (int) (b % blocks_per_row);
    const uint8_t * prow = packed + (int64_t) row * (k / 2);
    const uint8_t * srow = scales + (int64_t) row * (k / 16);

    // SoA within each row: all nibble blocks followed by all scale blocks.
    // This is byte-identical to the dense MMQ layout and remains 36 B/K64.
    const int qs_region = blocks_per_row * 32;
    uint8_t * row_out = out + (int64_t) row * blocks_per_row * 36;
    uint8_t * qs_out = row_out + kb * 32;
    uint8_t * d_out = row_out + qs_region + kb * 4;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int k0 = kb * QK_NVFP4 + s * 16;
        d_out[s] = srow[k0 / 16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int ka = k0 + j;
            const int kb2 = k0 + 8 + j;
            const uint8_t lo = (prow[ka >> 1] >> ((ka & 1) * 4)) & 0xF;
            const uint8_t hi = (prow[kb2 >> 1] >> ((kb2 & 1) * 4)) & 0xF;
            qs_out[s * 8 + j] = lo | (hi << 4);
        }
    }
}

// Apply projection scale2 while producing the exact GLM/DeepSeek clamped
// SwiGLU input for the down projection.  One block owns one sorted expert row.
extern "C" __global__ void atlas_moe_nvfp4_silu_scale2(
        const __nv_bfloat16 * gate, const __nv_bfloat16 * up,
        __nv_bfloat16 * out, const float * gate_scale2,
        const float * up_scale2, const int * expert_offsets,
        int width, int max_rows, int num_experts) {
    const int local_row = (int) blockIdx.x;
    const int expert = (int) blockIdx.y;
    if (expert >= num_experts || local_row >= max_rows) return;
    const int row = expert_offsets[expert] + local_row;
    if (row >= expert_offsets[expert + 1]) return;
    const float gs = gate_scale2[expert];
    const float us = up_scale2[expert];
    for (int col = (int) threadIdx.x; col < width; col += (int) blockDim.x) {
        const int64_t idx = (int64_t) row * width + col;
        float g = __bfloat162float(gate[idx]) * gs;
        float u = __bfloat162float(up[idx]) * us;
        g = fminf(fmaxf(g, -10.0f), 10.0f);
        u = fminf(fmaxf(u, -10.0f), 10.0f);
        out[idx] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
    }
}

extern "C" __global__ void atlas_moe_nvfp4_scale2_rows(
        __nv_bfloat16 * data, const float * scale2,
        const int * expert_offsets, int width, int max_rows, int num_experts) {
    const int local_row = (int) blockIdx.x;
    const int expert = (int) blockIdx.y;
    if (expert >= num_experts || local_row >= max_rows) return;
    const int row = expert_offsets[expert] + local_row;
    if (row >= expert_offsets[expert + 1]) return;
    const float s = scale2[expert];
    for (int col = (int) threadIdx.x; col < width; col += (int) blockDim.x) {
        const int64_t idx = (int64_t) row * width + col;
        data[idx] = __float2bfloat16(__bfloat162float(data[idx]) * s);
    }
}
