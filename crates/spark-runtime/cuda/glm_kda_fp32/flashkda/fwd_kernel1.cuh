#pragma once

#include "utils.cuh"

// Atlas: Kernel 1 computes in FP32 what the FP32 recurrence computes:
// normalized q/k, the exact logistic gate and beta, and the decayed operands
// of L and Mqk as split BF16 pairs (see split_bf16) for split-operand MMAs.
// INV = (I + L)^-1 is an FP32 forward substitution. It writes INV, Mqk and
// the row normalizers from registers; Kernel 2 rebuilds the decayed
// operands from q, k and g, which costs fewer bytes than storing them.
template <int D, int CHUNK = 16>
struct K1Layouts {
    using QKLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
    using GLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<CHUNK>{}), LayoutRight{}));

    using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
    using TMAQKLayout = decltype(prepend(QKLayout{}));
    using TMAGTotalSmemLayout = decltype(prepend(GTotalLayout{}));
};

template <class Layouts>
struct SharedStorageK1 {
    using BF16 = cutlass::bfloat16_t;
    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    // Phase A: q, k, g alive
    // Phase B: the split k_decayed, q_decayed and k_inv operands of L and Mqk
    union {
        struct {
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> q;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> k;
            alignas(128) cute::ArrayEngine<float, cute::cosize_v<GLayout>> g;
        };
        struct {
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed[2];
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed[2];
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_inv[2];
        };
    };
    alignas(128) cute::ArrayEngine<float, cute::cosize_v<LMLayout>> L;
    alignas(128) cute::ArrayEngine<float, cute::cosize_v<LMLayout>> Mqk;

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> g_bf16;      // TMA load target
    alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> dt_bias;  // TMA load target
    alignas(16) cutlass::arch::ClusterTransactionBarrier tma_load_barrier;
};

// Atlas: one warp's [CHUNK, CHUNK] = A[CHUNK, D] @ B[CHUNK, D]^T from split
// [hi, lo] operands in shared memory (MMALayout), into an FP32 matrix.
template <class MMALayout, class TensorC>
CUTLASS_DEVICE void mma_split_16x16_1warp(
    cutlass::bfloat16_t* const (&a)[2],
    cutlass::bfloat16_t* const (&b)[2],
    TensorC& c,
    int lane
) {
    using BF16 = cutlass::bfloat16_t;
    auto mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );
    ThrMMA thr_mma = mma.get_slice(lane);
    auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_copy_a = copy_a.get_thread_slice(lane);
    auto thr_copy_b = copy_b.get_thread_slice(lane);

    auto block = [](BF16* plane, int k) {
        return local_tile(make_tensor(make_smem_ptr(plane), MMALayout{}),
                          make_shape(Int<16>{}, Int<16>{}), make_coord(0, k));
    };
    auto acc = thr_mma.partition_fragment_C(c);
    clear(acc);
    constexpr int kBlocks = decltype(cute::size<1>(MMALayout{}))::value / 16;
    CUTE_UNROLL
    for (int k = 0; k < kBlocks; ++k) {
        auto a_hi = thr_mma.partition_fragment_A(block(a[0], k));
        auto a_lo = thr_mma.partition_fragment_A(block(a[1], k));
        auto b_hi = thr_mma.partition_fragment_B(block(b[0], k));
        auto b_lo = thr_mma.partition_fragment_B(block(b[1], k));
        copy(copy_a, thr_copy_a.partition_S(block(a[0], k)), thr_copy_a.retile_D(a_hi));
        copy(copy_a, thr_copy_a.partition_S(block(a[1], k)), thr_copy_a.retile_D(a_lo));
        copy(copy_b, thr_copy_b.partition_S(block(b[0], k)), thr_copy_b.retile_D(b_hi));
        copy(copy_b, thr_copy_b.partition_S(block(b[1], k)), thr_copy_b.retile_D(b_lo));
        gemm_split(thr_mma, a_hi, a_lo, b_hi, b_lo, acc);
    }
    auto c_out = thr_mma.partition_C(c);
    CUTE_UNROLL
    for (int i = 0; i < int(size(acc)); ++i) {
        c_out(i) = acc(i);
    }
}

// Build prefix-sum of per-sequence tile counts, so Kernel 1 can map
// global_tile_idx -> seq_idx with an O(log N) binary search instead of an
// O(N) linear scan per CTA in varlen mode.
__global__ void _flash_kda_build_tile_prefix(
    int64_t const* __restrict__ cu_seqlens,
    int N,
    int chunk,
    int* __restrict__ tile_prefix
) {
    if (threadIdx.x == 0) {
        int acc = 0;
        tile_prefix[0] = 0;
        for (int i = 0; i < N; ++i) {
            int slen = int(cu_seqlens[i + 1] - cu_seqlens[i]);
            acc += (slen + chunk - 1) / chunk;
            tile_prefix[i + 1] = acc;
        }
    }
}

// ==================== Kernel 1: Prepare ====================
template <
    class TmaLoadQ,
    class TmaLoadK,
    class TmaLoadBeta,
    class TmaLoadG,
    class TmaLoadDtBias,
    int CHUNK,
    int D,
    int NumThreads,
    bool IsVarlen = true
>
__global__ void __launch_bounds__(NumThreads, 3) _flash_kda_fwd_prepare(
    CUTE_GRID_CONSTANT TmaLoadQ const tma_load_q,
    CUTE_GRID_CONSTANT TmaLoadK const tma_load_k,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadG const tma_load_g,
    CUTE_GRID_CONSTANT TmaLoadDtBias const tma_load_dt_bias,
    float* ws_inv,
    float* ws_mqk,
    float* ws_norms,
    float scale,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles,
    float const* A_log_ptr,
    float gate_scale,
    int const* tile_prefix
) {
    // --- constants
    using BF16 = cutlass::bfloat16_t;
    using Layouts = K1Layouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using TMAQKLayout = typename Layouts::TMAQKLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMAGTotalSmemLayout = typename Layouts::TMAGTotalSmemLayout;
    constexpr uint32_t kTmaTransactionBytes =
        uint32_t(cute::cosize_v<QKLayout>) * uint32_t(3 * sizeof(BF16)) +  // q + k + g_bf16
        uint32_t(32) * uint32_t(sizeof(BF16)) +  // beta (bf16, sigmoid fused)
        uint32_t(D) * uint32_t(sizeof(float));  // dt_bias

    // --- shared memory
    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK1<Layouts>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    // --- per-CTA tile info
    int global_tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx, tiles_before, local_t;
    int64_t bos, eos;
    int seq_len, t_tiles_this_seq;

    if constexpr (IsVarlen) {
        int lo = 0, hi = N;
        while (lo + 1 < hi) {
            int mid = (lo + hi) >> 1;
            if (tile_prefix[mid] <= global_tile_idx) lo = mid;
            else hi = mid;
        }
        seq_idx = lo;
        tiles_before = tile_prefix[lo];
        local_t = global_tile_idx - tiles_before;
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
    } else {
        int T_seq = T_total / N;
        int tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
        seq_idx = global_tile_idx / tiles_per_seq;
        tiles_before = seq_idx * tiles_per_seq;
        local_t = global_tile_idx - tiles_before;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
    }
    seq_len = int(eos - bos);
    t_tiles_this_seq = (seq_len + CHUNK - 1) / CHUNK;
    // Early exit for excess CTAs (total_tiles is an upper bound)
    if (local_t >= t_tiles_this_seq) return;
    // --- TMA load inputs (single-shot, no pipeline)
    // Only thread 0 issues TMA loads (not elect_one_sync which is per-warp)
    if (threadIdx.x == 0) {
        using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
        shared_storage.tma_load_barrier.init(1);
        cutlass::arch::fence_barrier_init();  // generic init -> visible to async proxy (TMA complete-tx)
        shared_storage.tma_load_barrier.arrive_and_expect_tx(kTmaTransactionBytes);

        Tensor g_q = tma_load_q.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_k = tma_load_k.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));

        auto cta_tma_load_q = tma_load_q.get_slice(Int<0>{});
        auto cta_tma_load_k = tma_load_k.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});

        auto qk_off = g_q.layout()(head_idx, int(bos) + local_t * CHUNK, 0);
        auto tile_shape_3d = make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{});
        auto tile_stride_3d = stride(g_q.layout());
        Tensor g_q_tile = make_tensor(g_q.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));
        Tensor g_k_tile = make_tensor(g_k.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));

        int beta_linear = head_idx * T_total + (int(bos) + local_t * CHUNK);
        int beta_aligned = beta_linear & ~7;
        auto beta_off = g_beta.layout()(beta_aligned);
        Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});

        Tensor s_q_tile = make_tensor(make_smem_ptr(shared_storage.q.begin()), TMAQKLayout{});
        Tensor s_k_tile = make_tensor(make_smem_ptr(shared_storage.k.begin()), TMAQKLayout{});
        Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.beta.begin()), TMABetaSmemLayout{});

        cute::copy(tma_load_q.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_q.partition_S(g_q_tile), cta_tma_load_q.partition_D(s_q_tile));
        cute::copy(tma_load_k.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_k.partition_S(g_k_tile), cta_tma_load_k.partition_D(s_k_tile));
        cute::copy(tma_load_beta.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));

        // TMA load g_bf16 (same gmem layout as q/k)
        Tensor g_g = tma_load_g.get_tma_tensor(make_shape(H, T_total, D));
        auto cta_tma_load_g = tma_load_g.get_slice(Int<0>{});
        Tensor g_g_tile = make_tensor(g_g.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));
        Tensor s_g_bf16_tile = make_tensor(make_smem_ptr(shared_storage.g_bf16.begin()), TMAQKLayout{});
        cute::copy(tma_load_g.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_g.partition_S(g_g_tile), cta_tma_load_g.partition_D(s_g_bf16_tile));

        // TMA load dt_bias [H, D] → [D] slice for current head
        Tensor g_dt = tma_load_dt_bias.get_tma_tensor(make_shape(H, D));
        auto cta_tma_load_dt = tma_load_dt_bias.get_slice(Int<0>{});
        auto dt_off = g_dt.layout()(head_idx, 0);
        Tensor g_dt_tile = make_tensor(g_dt.data() + dt_off,
            make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_dt.layout())));
        Tensor s_dt_tile = make_tensor(make_smem_ptr(shared_storage.dt_bias.begin()), TMAGTotalSmemLayout{});
        cute::copy(tma_load_dt_bias.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_dt.partition_S(g_dt_tile), cta_tma_load_dt.partition_D(s_dt_tile));
    }

    // --- Compute a_log_exp (overlaps with TMA)
    float a_log_exp = expf(A_log_ptr[head_idx]);
    // --- Wait for TMA (q, k, beta, g_bf16, dt_bias)
    __syncthreads();
    shared_storage.tma_load_barrier.wait(0);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    // --- QK L2 Normalization (FP32, kept in registers) ---
    // Atlas: each thread owns 8 consecutive columns of one row, the layout it
    // later writes the workspace in.
    int compute_tid = threadIdx.x;
    constexpr int ELEMS_PER_THREAD = 8;
    constexpr int THREADS_PER_ROW = D / ELEMS_PER_THREAD;  // 16
    static_assert(CHUNK * THREADS_PER_ROW == NumThreads);
    int my_row = threadIdx.x / THREADS_PER_ROW;
    int my_col = (threadIdx.x % THREADS_PER_ROW) * ELEMS_PER_THREAD;
    int actual_len = min(CHUNK, seq_len - local_t * CHUNK);
    float q_vals[ELEMS_PER_THREAD], k_vals[ELEMS_PER_THREAD];
    {
        BF16* q_smem = shared_storage.q.begin();
        BF16* k_smem = shared_storage.k.begin();
        float q_sq = 0.0f, k_sq = 0.0f;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            float qv = bf16_to_f32(q_smem[my_row * D + my_col + i]);
            float kv = bf16_to_f32(k_smem[my_row * D + my_col + i]);
            q_vals[i] = qv;
            k_vals[i] = kv;
            q_sq += qv * qv;
            k_sq += kv * kv;
        }

        #pragma unroll
        for (int delta = 8; delta >= 1; delta >>= 1) {
            q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, delta);
            k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, delta);
        }

        // Tail rows (past the sequence) contribute zero keys and queries.
        bool live = my_row < actual_len;
        float q_inv = live ? rsqrtf(q_sq + 1e-6f) * scale : 0.0f;
        float k_inv = live ? rsqrtf(k_sq + 1e-6f) : 0.0f;
        if (threadIdx.x % THREADS_PER_ROW == 0) {
            float* norms = ws_norms + int64_t(head_idx * total_tiles + global_tile_idx) * 2 * CHUNK;
            norms[my_row] = q_inv;
            norms[CHUNK + my_row] = k_inv;
        }
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            q_vals[i] *= q_inv;
            k_vals[i] *= k_inv;
        }
    }

    // --- Gate activation + cumsum (log2 decay) ---
    int ws_idx = head_idx * total_tiles + global_tile_idx;
    if (compute_tid < D) {
        // Kernel 2 accumulates the same sums (kda_log2_decay).
        int col = compute_tid;
        BF16 const* g_bf16_smem = shared_storage.g_bf16.begin();
        float dt = shared_storage.dt_bias.begin()[col];
        float* g_smem = shared_storage.g.begin();
        float sum = 0.0f;
        #pragma unroll
        for (int row = 0; row < CHUNK; ++row) {
            if (row < actual_len) {
                sum += kda_log2_decay(bf16_to_f32(g_bf16_smem[row * D + col]), dt, a_log_exp, gate_scale);
            }
            g_smem[row * D + col] = sum;
        }
    }
    __syncthreads();

    // --- Decayed operands of L and Mqk, split ---
    // Row i of k_decayed / q_decayed carries the decay since the tile start
    // through token i, k_inv its inverse.
    struct alignas(16) Pack { BF16 v[ELEMS_PER_THREAD]; CUTLASS_DEVICE BF16& operator[](int i) { return v[i]; } };
    Pack kd[2], qd[2], ki[2];
    {
        float const* g_row = shared_storage.g.begin() + my_row * D + my_col;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            float g = g_row[i];
            float decay = ex2_approx_ftz_f32(g);
            split_bf16(k_vals[i] * decay, kd[0][i], kd[1][i]);
            split_bf16(q_vals[i] * decay, qd[0][i], qd[1][i]);
            split_bf16(k_vals[i] * ex2_approx_ftz_f32(-g), ki[0][i], ki[1][i]);
        }
    }
    __syncthreads();  // q/k/g -> split planes (union)

    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        auto plane = [&](auto& storage) {
            return make_tensor(make_smem_ptr(storage[part].begin()), MMALayout{});
        };
        auto put = [&](auto& storage, Pack const& x) {
            *reinterpret_cast<Pack*>(&plane(storage)(my_row, my_col)) = x;
        };
        put(shared_storage.k_decayed, kd[part]);
        put(shared_storage.q_decayed, qd[part]);
        put(shared_storage.k_inv, ki[part]);
    }
    __syncthreads();

    // --- L = k_decayed @ k_inv^T, Mqk = q_decayed @ k_inv^T (split MMAs) ---
    Tensor L = make_tensor(make_smem_ptr(shared_storage.L.begin()), LMLayout{});
    Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), LMLayout{});
    {
        BF16* const kd_planes[2] = {shared_storage.k_decayed[0].begin(), shared_storage.k_decayed[1].begin()};
        BF16* const qd_planes[2] = {shared_storage.q_decayed[0].begin(), shared_storage.q_decayed[1].begin()};
        BF16* const ki_planes[2] = {shared_storage.k_inv[0].begin(), shared_storage.k_inv[1].begin()};
        if (compute_tid < 32) {
            mma_split_16x16_1warp<MMALayout>(kd_planes, ki_planes, L, compute_tid);
        } else if (compute_tid < 64) {
            mma_split_16x16_1warp<MMALayout>(qd_planes, ki_planes, Mqk, compute_tid - 32);
        }
    }
    __syncthreads();

    Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.beta.begin()), BetaSmemLayout{});
    int beta_smem_offset = (head_idx * T_total + int(bos) + local_t * CHUNK) & 7;
    if (compute_tid < CHUNK) {
        // INV = (I + diag(beta) tril(L, -1))^-1 by forward substitution;
        // lane j owns column j.
        int j = compute_tid;
        float x[CHUNK];
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            float acc = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < i; ++kk) {
                acc = fmaf(L(i, kk), x[kk], acc);
            }
            float b = sigmoid_f32(bf16_to_f32(beta_tile(beta_smem_offset + i)));
            x[i] = i == j ? 1.0f : -b * acc;
        }
        float* inv = ws_inv + int64_t(ws_idx) * CHUNK * CHUNK;
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            inv[i * CHUNK + j] = x[i];
        }
    } else if (compute_tid >= 32 && compute_tid < 64) {
        // Mqk keeps the diagonal: token i reads its own update.
        float* mqk = ws_mqk + int64_t(ws_idx) * CHUNK * CHUNK;
        for (int e = compute_tid - 32; e < CHUNK * CHUNK; e += 32) {
            int i = e / CHUNK, jj = e % CHUNK;
            mqk[e] = jj <= i ? Mqk(i, jj) : 0.0f;
        }
    }
}
