#include "fwd.h"
#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"

// ==================== launch_fwd ====================
template <int D, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int const* state_slot_ids,
    int state_capacity,
    cudaStream_t stream
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int CHUNK = 16;

    using K1L = K1Layouts<D, CHUNK>;
    using WS = WorkspaceSizes<CHUNK, D>;

    // TMA layouts for Kernel 1
    using TMAQKLayout = typename K1L::TMAQKLayout;
    using TMABetaSmemLayout = typename K1L::TMABetaSmemLayout;
    using TMAGTotalSmemLayout = typename K1L::TMAGTotalSmemLayout;

    // --- gmem layouts for original tensors
    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, D * H, 1));
    // 1D beta layout: [H*T] contiguous
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));

    Tensor m_q   = make_tensor(make_gmem_ptr(q_ptr), gmem_layout);
    Tensor m_k   = make_tensor(make_gmem_ptr(k_ptr), gmem_layout);
    Tensor m_v   = make_tensor(make_gmem_ptr(v_ptr), gmem_layout);
    Tensor m_out = make_tensor(make_gmem_ptr(out_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);

    // --- Workspace gmem layouts (separated arrays)
    int64_t n_ht = int64_t(H) * total_tiles;
    char* ws = reinterpret_cast<char*>(workspace_ptr);
    float* ws_inv   = reinterpret_cast<float*>(ws);
    float* ws_mqk   = reinterpret_cast<float*>(ws + n_ht * WS::kINV);
    float* ws_norms = reinterpret_cast<float*>(ws + n_ht * (WS::kINV + WS::kMqk));

    int* ws_tile_prefix = reinterpret_cast<int*>(ws + n_ht * WS::kPerTile);

    auto ws_lm_gmem_layout = make_layout(make_shape(int(n_ht), CHUNK, CHUNK), LayoutRight{});
    auto ws_norms_gmem_layout = make_layout(make_shape(int(n_ht), 2 * CHUNK), LayoutRight{});
    Tensor m_ws_inv   = make_tensor(make_gmem_ptr(ws_inv), ws_lm_gmem_layout);
    Tensor m_ws_mqk   = make_tensor(make_gmem_ptr(ws_mqk), ws_lm_gmem_layout);
    Tensor m_ws_norms = make_tensor(make_gmem_ptr(ws_norms), ws_norms_gmem_layout);

    // --- TMA descriptors for Kernel 1 (loads: q,k,beta,g,dt_bias; it stores
    // the workspace from registers); Kernel 2 loads q, k, g the same way
    auto tma_load_q    = make_tma_copy(SM90_TMA_LOAD{}, m_q, TMAQKLayout{});
    auto tma_load_k    = make_tma_copy(SM90_TMA_LOAD{}, m_k, TMAQKLayout{});
    auto tma_load_beta = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});

    Tensor m_g = make_tensor(make_gmem_ptr(g_bf16_ptr), gmem_layout);
    auto tma_load_g = make_tma_copy(SM90_TMA_LOAD{}, m_g, TMAQKLayout{});

    auto dt_bias_gmem_layout = make_layout(make_shape(H, D), LayoutRight{});
    Tensor m_dt_bias = make_tensor(make_gmem_ptr(dt_bias_ptr), dt_bias_gmem_layout);
    auto tma_load_dt_bias = make_tma_copy(SM90_TMA_LOAD{}, m_dt_bias, TMAGTotalSmemLayout{});

    // ===== Launch Kernel 1 (prepare) =====
#if BLOCK_LEVEL_K1 >= 0
    {
        constexpr int kK1Threads = 256;
        using SharedStorageK1T = SharedStorageK1<K1L>;
        int smem_size_k1 = sizeof(SharedStorageK1T);

        auto kernel1 = _flash_kda_fwd_prepare<
            decltype(tma_load_q), decltype(tma_load_k),
            decltype(tma_load_beta),
            decltype(tma_load_g), decltype(tma_load_dt_bias),
            CHUNK, D, kK1Threads, IsVarlen
        >;

        cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k1);

        if constexpr (IsVarlen) {
            _flash_kda_build_tile_prefix<<<1, 32, 0, stream>>>(
                cu_seqlens_ptr, N, CHUNK, ws_tile_prefix);
        }

        dim3 grid_k1(total_tiles, H);
        dim3 block_k1(kK1Threads);

        kernel1<<<grid_k1, block_k1, smem_size_k1, stream>>>(
            tma_load_q, tma_load_k, tma_load_beta,
            tma_load_g, tma_load_dt_bias,
            ws_inv, ws_mqk, ws_norms,
            scale, T_total, H, N, cu_seqlens_ptr, total_tiles,
            A_log_ptr, gate_scale, ws_tile_prefix
        );
    }
#endif

    // ===== Launch Kernel 2 (recurrence) =====
#if BLOCK_LEVEL_K2 >= 0
    // Atlas: VD value columns per block. Half-width blocks double the
    // recurrence's parallelism while the grid fits one wave (one block per
    // SM); wider grids keep full-width blocks with twice the math warps.
    auto launch_k2 = [&](auto vd) {
        constexpr int VD = decltype(vd)::value;
        constexpr int kInputStages = 2;
        constexpr int kOperandStages = 2;
        constexpr int kOutputStages = 2;
        using K2L = K2Layouts<D, CHUNK, VD>;

        // --- TMA descriptors for Kernel 2 (loads: v, beta, workspace; store: out)
        auto tma_load_v     = make_tma_copy(SM90_TMA_LOAD{}, m_v, typename K2L::TMAVOLayout{});
        auto tma_load_beta2 = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});
        auto tma_load_ws_inv   = make_tma_copy(SM90_TMA_LOAD{}, m_ws_inv, typename K2L::TMALMLayout{});
        auto tma_load_ws_mqk   = make_tma_copy(SM90_TMA_LOAD{}, m_ws_mqk, typename K2L::TMALMLayout{});
        auto tma_load_ws_norms = make_tma_copy(SM90_TMA_LOAD{}, m_ws_norms, typename K2L::TMANormsLayout{});
        auto tma_store_out = make_tma_copy(SM90_TMA_STORE{}, m_out, typename K2L::TMAVOLayout{});
        static_assert(std::is_same_v<typename K2L::TMAQKLayout, TMAQKLayout>);

        constexpr int kK2Threads = VD / 16 * 32 + D + 32 * 2;
        using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOperandStages, kOutputStages>;
        int smem_size_k2 = sizeof(SharedStorageK2T);

        auto kernel2 = _flash_kda_fwd_recurrence<
            decltype(tma_load_v), decltype(tma_load_beta2), decltype(tma_load_q),
            decltype(tma_load_ws_inv), decltype(tma_load_ws_mqk), decltype(tma_load_ws_norms),
            decltype(tma_store_out),
            CHUNK, D, VD, kInputStages, kOperandStages, kOutputStages, kK2Threads,
            HasStateIn, HasStateOut, StateFP32, IsVarlen
        >;

        cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k2);

        dim3 grid_k2(N * (D / VD), H);
        dim3 block_k2(kK2Threads);

        kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
            tma_load_v, tma_load_beta2,
            tma_load_q, tma_load_k, tma_load_g,
            tma_load_ws_inv, tma_load_ws_mqk, tma_load_ws_norms,
            tma_store_out,
            A_log_ptr, dt_bias_ptr, gate_scale,
            initial_state_ptr, final_state_ptr,
            out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles,
            state_slot_ids, state_capacity
        );
    };
    static int const sm_count = [] {
        int device = 0, count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
        return count;
    }();
    if (2 * N * H <= sm_count) {
        launch_k2(Int<D / 2>{});
    } else {
        launch_k2(Int<D>{});
    }
#endif
}

// Explicit instantiations
#define INSTANTIATE_LAUNCH_FWD(D, HI, HO, FP32, VL) \
    template void launch_fwd<D, HI, HO, FP32, VL>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, void const*, float, void*, \
        cutlass::bfloat16_t*, void*, int, int, int, int, \
        int64_t const*, float const*, float const*, float, \
        int const*, int, cudaStream_t);

#define INSTANTIATE_STATE_VARIANTS(VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  true,  false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  true,  true,  VL) \
    INSTANTIATE_LAUNCH_FWD(128, false, false, false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, false, true,  false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  false, false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, false, true,  true,  VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  false, true,  VL)

INSTANTIATE_STATE_VARIANTS(true)   // varlen
INSTANTIATE_STATE_VARIANTS(false)  // non-varlen
