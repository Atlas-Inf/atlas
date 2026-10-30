#pragma once

#include <type_traits>

#include "utils.cuh"

// Atlas: the per-tile [CHUNK, D] operands use 128-byte swizzled atoms instead
// of the interleaved ones, so each is one or two TMA boxes rather than dozens:
// the single load thread's TMA issue (~8.9K cycles a 16-token tile) bounded
// the recurrence. Only the shared-memory addresses move.
//
// Atlas: the recurrence matches the FP32 recurrence, not a BF16 one. The
// [K, VD] state stays in FP32 registers for the whole call (loaded from and
// stored to the FP32 state directly). Prep warps rebuild each tile's decayed
// and restored operands in FP32 from q, k and g (Kernel 1 left only INV,
// Mqk and the row normalizers) and split them into BF16 pairs; the state,
// v - k@S and U are split the same way, so every MMA runs on ~16-bit
// operands with FP32 accumulation. A CTA owns VD value columns of one head,
// one math warp per 16.
template <int D, int CHUNK = 16, int VD = D>
struct K2Layouts {
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedMMALayout = decltype(tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));
    using VOLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<VD>{}),
        LayoutLeft{}
    ));
    using QKLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<CHUNK>{}), LayoutRight{}));
    using NormsLayout = Layout<Shape<Int<2 * CHUNK>>, Stride<Int<1>>>;

    using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
    using TMAQKLayout = decltype(prepend(QKLayout{}));
    using TMAVOLayout = decltype(composition(
        VOLayout{}.layout_a(),
        VOLayout{}.offset(),
        prepend(VOLayout{}.layout_b())
    ));
    using TMALMLayout = decltype(prepend(LMLayout{}));
    using TMANormsLayout = decltype(prepend(NormsLayout{}));
};

template <class Layouts, int InputStages, int OperandStages, int OutputStages>
struct SharedStorageK2 {
    using BF16 = cutlass::bfloat16_t;
    using QKLayout = typename Layouts::QKLayout;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using NormsLayout = typename Layouts::NormsLayout;
    using MMALayout = typename Layouts::MMALayout;

    // TMA'd per tile; the prep warps read q, k, g and the normalizers, the
    // math warps the rest.
    struct InputStorage {
        alignas(1024) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> q;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> k;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> g;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<LMLayout>> Mqk;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<NormsLayout>> norms;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
    };

    // Written by the prep warps: swizzled [hi, lo] operand pairs, 1024-byte
    // (128B swizzle) aligned, and the tile's total decay.
    struct OperandStorage {
        alignas(1024) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed[2];
        alignas(1024) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed[2];
        alignas(1024) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored[2];
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
    };

    struct OutputStorage {
        alignas(1024) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> out;
    };

    InputStorage input[InputStages];
    OperandStorage operand[OperandStages];
    OutputStorage output[OutputStages];

    typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
    typename cutlass::PipelineAsync<OperandStages>::SharedStorage operand_pipeline;
    typename cutlass::PipelineAsync<OutputStages>::SharedStorage store_pipeline;
};

// ==================== Kernel 2: Recurrence ====================
template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaLoadQK,
    class TmaLoadWsINV, class TmaLoadWsMqk, class TmaLoadWsNorms,
    class TmaStoreOut,
    int CHUNK,
    int D,
    int VD,
    int InputStages,
    int OperandStages,
    int OutputStages,
    int NumThreads,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool IsVarlen = true
>
__global__ void __launch_bounds__(NumThreads) _flash_kda_fwd_recurrence(
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadQK const tma_load_q,
    CUTE_GRID_CONSTANT TmaLoadQK const tma_load_k,
    CUTE_GRID_CONSTANT TmaLoadQK const tma_load_g,
    CUTE_GRID_CONSTANT TmaLoadWsINV const tma_load_ws_inv,
    CUTE_GRID_CONSTANT TmaLoadWsMqk const tma_load_ws_mqk,
    CUTE_GRID_CONSTANT TmaLoadWsNorms const tma_load_ws_norms,
    CUTE_GRID_CONSTANT TmaStoreOut const tma_store_out,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    void const* initial_state_ptr,
    void* final_state_ptr,
    cutlass::bfloat16_t* out_raw_ptr,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles,
    int const* state_slot_ids,
    int state_capacity
) {
    using BF16 = cutlass::bfloat16_t;
    using StateT = std::conditional_t<StateFP32, float, BF16>;
    using Layouts = K2Layouts<D, CHUNK, VD>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using QKLayout = typename Layouts::QKLayout;
    using NormsLayout = typename Layouts::NormsLayout;
    using TMAQKLayout = typename Layouts::TMAQKLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMALMLayout = typename Layouts::TMALMLayout;
    using TMANormsLayout = typename Layouts::TMANormsLayout;
    constexpr int kWarpSize = 32;
    constexpr int kVSlices = D / VD;
    constexpr int kComputeThreads = VD / 16 * kWarpSize;  // one warp per 16 value columns
    constexpr int kPrepThreads = D;                       // one thread per key column
    static_assert(D % VD == 0 && VD % 16 == 0);
    static_assert(NumThreads == kComputeThreads + kPrepThreads + 2 * kWarpSize,
                  "K2 runs its math and prep warps plus one load and one store warp");

    // Transaction bytes: v + beta + q, k, g + INV + Mqk + normalizers
    constexpr uint32_t kTmaTransactionBytes =
        uint32_t(cute::cosize_v<VOLayout>) * uint32_t(sizeof(BF16)) +
        uint32_t(32) * uint32_t(sizeof(BF16)) +  // beta (bf16, sigmoid fused)
        uint32_t(cute::cosize_v<QKLayout>) * uint32_t(sizeof(BF16)) * 3 +
        uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(float)) * 2 +
        uint32_t(cute::cosize_v<NormsLayout>) * uint32_t(sizeof(float));

    // --- shared memory
    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK2<Layouts, InputStages, OperandStages, OutputStages>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    // --- warp specialization
    int warp_id = threadIdx.x / kWarpSize;
    WarpRole warp_role = WarpRole::NonParticipant;
    constexpr int kPrepWarp0 = kComputeThreads / kWarpSize;
    constexpr int kLoadWarp = kPrepWarp0 + kPrepThreads / kWarpSize;
    if (warp_id < kPrepWarp0) {
        warp_role = WarpRole::MMA;
    } else if (warp_id < kLoadWarp) {
        warp_role = WarpRole::PREP;
    } else if (warp_id < kLoadWarp + 1) {
        warp_role = WarpRole::LOAD_QKG;
    } else if (warp_id < kLoadWarp + 2) {
        warp_role = WarpRole::STORE;
    }

    using LoadPipelineState = cutlass::PipelineState<InputStages>;
    using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
    LoadPipeline load_pipeline = make_load_pipeline<InputStages>(
        shared_storage.load_pipeline,
        kTmaTransactionBytes,
        warp_role, 1, kComputeThreads + kPrepThreads
    );
    using OperandPipelineState = cutlass::PipelineState<OperandStages>;
    using OperandPipeline = cutlass::PipelineAsync<OperandStages>;
    OperandPipeline operand_pipeline = make_store_pipeline<OperandStages>(
        shared_storage.operand_pipeline,
        warp_role, kPrepThreads, kComputeThreads,
        WarpRole::PREP, WarpRole::MMA
    );
    using StorePipelineState = cutlass::PipelineState<OutputStages>;
    using StorePipeline = cutlass::PipelineAsync<OutputStages>;
    StorePipeline store_pipeline = make_store_pipeline<OutputStages>(
        shared_storage.store_pipeline,
        warp_role, kComputeThreads, 1
    );

    // --- per-block sequence info (the VD-column slices of a head are adjacent
    // blocks, so they read each tile's workspace while it is in L2)
    int seq_idx  = blockIdx.x / kVSlices;
    int v_idx    = blockIdx.x % kVSlices;
    int head_idx = blockIdx.y;
    int64_t bos, eos;
    int tile_base;

    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        // Compute tile_base via linear scan (no host-precomputed table)
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = T_total / N;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }
    int seq_len  = int(eos - bos);
    int t_tiles  = (seq_len + CHUNK - 1) / CHUNK;
    int state_seq_idx = state_slot_ids == nullptr ? seq_idx : state_slot_ids[seq_idx];
    bool lane_predicate = cute::elect_one_sync();

    // --- LOAD warp: issue TMA loads for v, beta, and workspace intermediates
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
        Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));

        Tensor g_q = tma_load_q.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_k = tma_load_k.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_g = tma_load_g.get_tma_tensor(make_shape(H, T_total, D));
        auto g_ws_inv = tma_load_ws_inv.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));
        auto g_ws_mqk = tma_load_ws_mqk.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));
        auto g_ws_norms = tma_load_ws_norms.get_tma_tensor(make_shape(H * total_tiles, 2 * CHUNK));

        LoadPipelineState load_write = cutlass::make_producer_start_state<LoadPipeline>();
        auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});
        auto cta_q = tma_load_q.get_slice(Int<0>{});
        auto cta_k = tma_load_k.get_slice(Int<0>{});
        auto cta_g = tma_load_g.get_slice(Int<0>{});
        auto cta_ws_inv = tma_load_ws_inv.get_slice(Int<0>{});
        auto cta_ws_mqk = tma_load_ws_mqk.get_slice(Int<0>{});
        auto cta_ws_norms = tma_load_ws_norms.get_slice(Int<0>{});

        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.producer_acquire(load_write);
            using LoadBarrierType = typename LoadPipeline::ProducerBarrierType;
            LoadBarrierType* tma_barrier = load_pipeline.producer_get_barrier(load_write);
            int stage = load_write.index();
            auto& input = shared_storage.input[stage];
            int ws_idx = head_idx * total_tiles + tile_base + t;

            // TMA load v (this block's value columns)
            auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, v_idx * VD);
            Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VD>{}), stride(g_v.layout())));
            Tensor s_v_tile = make_tensor(make_smem_ptr(input.v.begin()), TMAVOLayout{});
            cute::copy(tma_load_v.with(*tma_barrier),
                cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));

            // TMA load beta (1D)
            int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
            int beta_aligned = beta_linear & ~7;
            auto beta_off = g_beta.layout()(beta_aligned);
            Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
            Tensor s_beta_tile = make_tensor(make_smem_ptr(input.beta.begin()), TMABetaSmemLayout{});
            cute::copy(tma_load_beta.with(*tma_barrier),
                cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));

            // TMA load q, k, g (all key columns)
            auto load_qk = [&](auto const& tma, auto const& cta, auto const& g_x, auto& storage) {
                auto off = g_x.layout()(head_idx, int(bos) + t * CHUNK, 0);
                Tensor g_tile = make_tensor(g_x.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_x.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(storage.begin()), TMAQKLayout{});
                cute::copy(tma.with(*tma_barrier), cta.partition_S(g_tile), cta.partition_D(s_tile));
            };
            load_qk(tma_load_q, cta_q, g_q, input.q);
            load_qk(tma_load_k, cta_k, g_k, input.k);
            load_qk(tma_load_g, cta_g, g_g, input.g);
            // INV, Mqk (FP32)
            auto load_lm = [&](auto const& tma, auto const& cta, auto const& g_ws, auto& storage) {
                auto off = g_ws.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(storage.begin()), TMALMLayout{});
                cute::copy(tma.with(*tma_barrier), cta.partition_S(g_tile), cta.partition_D(s_tile));
            };
            load_lm(tma_load_ws_inv, cta_ws_inv, g_ws_inv, input.INV);
            load_lm(tma_load_ws_mqk, cta_ws_mqk, g_ws_mqk, input.Mqk);
            // q / k normalizers
            {
                auto off = g_ws_norms.layout()(ws_idx, 0);
                Tensor g_tile = make_tensor(g_ws_norms.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<2 * CHUNK>{}), stride(g_ws_norms.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(input.norms.begin()), TMANormsLayout{});
                cute::copy(tma_load_ws_norms.with(*tma_barrier), cta_ws_norms.partition_S(g_tile), cta_ws_norms.partition_D(s_tile));
            }

            ++load_write;
        }
        load_pipeline.producer_tail(load_write);
    }

    // --- PREP warps: the tile's decayed and restored operands, split
    // Thread c owns key column c: the cumulative log2 decay over the tile
    // (Kernel 1's sums), then for each row k_decayed = k_hat * 2^G,
    // q_decayed = q_hat * 2^G and k_restored = k_hat * 2^(G_total - G).
    if (warp_role == WarpRole::PREP) {
        LoadPipelineState load_read;
        OperandPipelineState operand_write = cutlass::make_producer_start_state<OperandPipeline>();
        const int col = threadIdx.x - kPrepWarp0 * kWarpSize;
        const float a = expf(A_log_ptr[head_idx]);
        const float dt = dt_bias_ptr[head_idx * D + col];
        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.consumer_wait(load_read);
            operand_pipeline.producer_acquire(operand_write);
            auto& input = shared_storage.input[load_read.index()];
            auto& operand = shared_storage.operand[operand_write.index()];
            Tensor q = make_tensor(make_smem_ptr(input.q.begin()), QKLayout{});
            Tensor k = make_tensor(make_smem_ptr(input.k.begin()), QKLayout{});
            Tensor g = make_tensor(make_smem_ptr(input.g.begin()), QKLayout{});
            Tensor norms = make_tensor(make_smem_ptr(input.norms.begin()), NormsLayout{});
            int actual_len = min(CHUNK, seq_len - t * CHUNK);

            float gamma[CHUNK];
            float sum = 0.0f;
            CUTE_UNROLL
            for (int row = 0; row < CHUNK; ++row) {
                if (row < actual_len) {
                    sum += kda_log2_decay(bf16_to_f32(g(row, col)), dt, a, gate_scale);
                }
                gamma[row] = sum;
            }
            operand.g_total.begin()[col] = ex2_approx_ftz_f32(sum);
            auto plane = [&](auto& storage, int part) {
                return make_tensor(make_smem_ptr(storage[part].begin()), MMALayout{});
            };
            CUTE_UNROLL
            for (int row = 0; row < CHUNK; ++row) {
                float k_hat = bf16_to_f32(k(row, col)) * norms(CHUNK + row);
                float q_hat = bf16_to_f32(q(row, col)) * norms(row);
                float decay = ex2_approx_ftz_f32(gamma[row]);
                BF16 hi, lo;
                split_bf16(k_hat * decay, hi, lo);
                plane(operand.k_decayed, 0)(row, col) = hi;
                plane(operand.k_decayed, 1)(row, col) = lo;
                split_bf16(q_hat * decay, hi, lo);
                plane(operand.q_decayed, 0)(row, col) = hi;
                plane(operand.q_decayed, 1)(row, col) = lo;
                split_bf16(k_hat * ex2_approx_ftz_f32(sum - gamma[row]), hi, lo);
                plane(operand.k_restored, 0)(row, col) = hi;
                plane(operand.k_restored, 1)(row, col) = lo;
            }
            operand_pipeline.producer_commit(operand_write);
            load_pipeline.consumer_release(load_read);
            ++load_read;
            ++operand_write;
        }
    }

    // --- MMA warps
    if (warp_role == WarpRole::MMA) {
        LoadPipelineState load_read;
        OperandPipelineState operand_read;
        StorePipelineState out_write = cutlass::make_producer_start_state<StorePipeline>();
        int compute_tid = threadIdx.x;
        const int mma_warp = compute_tid / 32;
        const int lane_id = compute_tid % 32;
        const int group_id = (lane_id / 4) % 8;

        auto mma = make_tiled_mma(
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
            Layout<Shape<_1,_1>>{},
            Tile<_16,_16,_16>{}
        );
        ThrMMA thr_mma = mma.get_slice(lane_id);

        // A copy: K-major tiles → LDSM_N (k_decayed, q_decayed)
        auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
        auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);
        // A copy: MN-major → LDSM_T (k_restored^T in the state update)
        auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
        auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);
        // C load/store (v in, out)
        auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
        auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
        auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
        auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);

        auto block16 = make_shape(Int<16>{}, Int<16>{});
        Tensor A_ref = make_tensor(make_smem_ptr(shared_storage.operand[0].k_decayed[0].begin()), MMALayout{});
        Tensor A_blk = local_tile(A_ref, block16, make_coord(0, 0));
        Tensor C_blk = local_tile(make_tensor(make_smem_ptr(shared_storage.input[0].v.begin()), VOLayout{}),
                                  block16, make_coord(0, 0));
        auto tCrC_ref = thr_mma.partition_C(C_blk);

        using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
        using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
        using AFragT = decltype(thr_mma.partition_fragment_A(A_blk));
        using BFragT = decltype(thr_mma.partition_fragment_B(A_blk));

        // [hi, lo] A fragments of a 16x16 block of split planes.
        auto load_a = [&](auto const& copy_atom, auto const& thr_copy, auto const& planes_layout,
                          auto& planes, int mb, int kb, AFragT (&frag)[2]) {
            CUTE_UNROLL
            for (int part = 0; part < 2; ++part) {
                Tensor plane = make_tensor(make_smem_ptr(planes[part].begin()), planes_layout);
                copy(copy_atom, thr_copy.partition_S(local_tile(plane, block16, make_coord(mb, kb))),
                     thr_copy.retile_D(frag[part]));
            }
        };
        // FP32 C fragment → [hi, lo] B fragments.
        auto split_to_b = [&](AccFragT const& c, BFragT (&b)[2]) {
            SFragT hi, lo;
            split_fragment(c, hi, lo);
            movm_c_to_b(hi, b[0]);
            movm_c_to_b(lo, b[1]);
        };

        // This warp's 16 value columns of the FP32 [K, V] state, as eight
        // 16x16 C fragments over K; global state is [V, K] per head.
        constexpr int kStateBlocks = D / 16;
        AccFragT state[kStateBlocks];
        Tensor coord = thr_mma.partition_C(make_identity_tensor(block16));
        int64_t state_base = (int64_t(state_seq_idx) * H + head_idx) * D * D
                           + int64_t(v_idx * VD + mma_warp * 16) * D;
        CUTE_UNROLL
        for (int m = 0; m < kStateBlocks; ++m) {
            CUTE_UNROLL
            for (int i = 0; i < int(size(state[m])); ++i) {
                if constexpr (HasStateIn) {
                    auto const* s = static_cast<StateT const*>(initial_state_ptr);
                    state[m](i) = float(s[state_base + get<1>(coord(i)) * D + m * 16 + get<0>(coord(i))]);
                } else {
                    state[m](i) = 0.0f;
                }
            }
        }

        for (int t = 0; t < t_tiles; ++t) {
            store_pipeline.producer_acquire(out_write);
            load_pipeline.consumer_wait(load_read);
            operand_pipeline.consumer_wait(operand_read);
            int out_stage = out_write.index();
            auto& input = shared_storage.input[load_read.index()];
            auto& operand = shared_storage.operand[operand_read.index()];
            Tensor v_tile = make_tensor(make_smem_ptr(input.v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(input.beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head_idx * T_total + int(bos) + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()), VOLayout{});
            Tensor g_total = make_tensor(make_smem_ptr(operand.g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(input.INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(input.Mqk.begin()), LMLayout{});

            // ======== Phase 1: u = k_decayed @ S, out = q_decayed @ S ========
            AccFragT u_acc, out_acc;
            clear(u_acc);
            clear(out_acc);
            CUTE_UNROLL
            for (int k = 0; k < kStateBlocks; ++k) {
                AFragT kd[2], qd[2];
                load_a(smem_tiled_copy_A, smem_thr_copy_A, MMALayout{}, operand.k_decayed, 0, k, kd);
                load_a(smem_tiled_copy_A, smem_thr_copy_A, MMALayout{}, operand.q_decayed, 0, k, qd);
                BFragT s[2];
                split_to_b(state[k], s);
                gemm_split(thr_mma, kd[0], kd[1], s[0], s[1], u_acc);
                gemm_split(thr_mma, qd[0], qd[1], s[0], s[1], out_acc);
            }

            // ======== Phase 2: w = (v - u) * beta, FP32 ========
            float beta0 = sigmoid_f32(bf16_to_f32(beta_tile(beta_smem_offset + group_id)));
            float beta1 = sigmoid_f32(bf16_to_f32(beta_tile(beta_smem_offset + group_id + 8)));
            {
                SFragT v_bf16;
                copy(smem_tiled_load_C,
                     smem_thr_load_C.partition_S(local_tile(v_tile, block16, make_coord(0, mma_warp))),
                     smem_thr_load_C.retile_D(v_bf16));
                CUTE_UNROLL
                for (int a = 0; a < 2; ++a) {
                    CUTE_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_acc(c0) = (bf16_to_f32(v_bf16(c0)) - u_acc(c0)) * beta0;
                        u_acc(c1) = (bf16_to_f32(v_bf16(c1)) - u_acc(c1)) * beta1;
                    }
                }
            }

            // ======== Phase 3: U = INV @ w ========
            auto lm_a = [&](auto const& lm, AFragT (&frag)[2]) {
                split_fragment(thr_mma.partition_A(lm), frag[0], frag[1]);
            };
            BFragT w[2], U[2];
            split_to_b(u_acc, w);
            {
                AFragT inv[2];
                lm_a(INV, inv);
                clear(u_acc);
                gemm_split(thr_mma, inv[0], inv[1], w[0], w[1], u_acc);
            }
            split_to_b(u_acc, U);

            // ======== Phase 4: out += Mqk @ U; store ========
            {
                AFragT mqk[2];
                lm_a(Mqk, mqk);
                gemm_split(thr_mma, mqk[0], mqk[1], U[0], U[1], out_acc);
                SFragT out_bf16;
                cute::transform(out_acc, out_bf16, [] __device__ (float x) { return BF16(x); });
                Tensor out_block = local_tile(out_tile, block16, make_coord(0, mma_warp));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16), smem_thr_store_C.partition_D(out_block));
            }

            // ======== Phase 5: S = S * g_total + k_restored^T @ U ========
            CUTE_UNROLL
            for (int m = 0; m < kStateBlocks; ++m) {
                AFragT kr[2];
                load_a(smem_tiled_copy_A_T, smem_thr_copy_A_T, TransposedMMALayout{}, operand.k_restored, m, 0, kr);
                float g0 = g_total(m * 16 + group_id);
                float g1 = g_total(m * 16 + group_id + 8);
                CUTE_UNROLL
                for (int a = 0; a < 2; ++a) {
                    CUTE_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        state[m](make_coord(make_coord(a, 0), 0, d)) *= g0;
                        state[m](make_coord(make_coord(a, 1), 0, d)) *= g1;
                    }
                }
                gemm_split(thr_mma, kr[0], kr[1], U[0], U[1], state[m]);
            }

            cutlass::arch::fence_view_async_shared();
            store_pipeline.producer_commit(out_write);
            operand_pipeline.consumer_release(operand_read);
            load_pipeline.consumer_release(load_read);
            ++load_read;
            ++operand_read;
            ++out_write;
        }

        if constexpr (HasStateOut) {
            auto* s = static_cast<StateT*>(final_state_ptr);
            CUTE_UNROLL
            for (int m = 0; m < kStateBlocks; ++m) {
                CUTE_UNROLL
                for (int i = 0; i < int(size(state[m])); ++i) {
                    s[state_base + get<1>(coord(i)) * D + m * 16 + get<0>(coord(i))] = StateT(state[m](i));
                }
            }
        }
    }

    if (warp_role == WarpRole::STORE && lane_predicate) {
        Tensor g_out = tma_store_out.get_tma_tensor(make_shape(H, T_total, D));
        auto cta_tma_store = tma_store_out.get_slice(Int<0>{});
        StorePipelineState out_read;
        for (int t = 0; t < t_tiles; ++t) {
            store_pipeline.consumer_wait(out_read);
            int stage = out_read.index();
            int actual_len = min(CHUNK, seq_len - t * CHUNK);

            BF16* out_stage_ptr = shared_storage.output[stage].out.begin();

            if (actual_len < CHUNK) {
                // Manual store for tail tile to avoid overwriting next sequence
                // Only one thread (lane_predicate) runs here, so loop over all VD
                Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), VOLayout{});
                for (int row = 0; row < actual_len; ++row) {
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D + v_idx * VD;
                    for (int col = 0; col < VD; ++col) {
                        out_raw_ptr[global_base + col] = s_out(row, col);
                    }
                }
            } else {
                // TMA store for full tiles
                auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, v_idx * VD);
                Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VD>{}), stride(g_out.layout())));
                Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAVOLayout{});
                cute::copy(
                    tma_store_out,
                    cta_tma_store.partition_S(s_out_tile),
                    cta_tma_store.partition_D(g_out_tile)
                );
                tma_store_arrive();
            }

            tma_store_wait<0>();
            store_pipeline.consumer_release(out_read);
            ++out_read;
        }
    }

    __syncthreads();
}
