#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/stride.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/tfloat32.h>

#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor_impl.hpp"

#ifndef BLOCK_LEVEL_K1
#define BLOCK_LEVEL_K1 1
#endif

#ifndef BLOCK_LEVEL_K2
#define BLOCK_LEVEL_K2 1
#endif

__device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {
    float result;
    asm("cvt.f32.bf16 %0, %1;\n" : "=f"(result) : "h"(x.storage));
    return result;
}

// Atlas: the gate and beta logistic to FP32 accuracy, as the FP32 recurrence
// computes it (tanh.approx is good to ~5e-4).
__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Atlas: x as hi + lo, two BF16 terms (16 significant bits). Products of
// split operands, hi*hi + hi*lo + lo*hi on the BF16 tensor cores with FP32
// accumulation, carry FP32 operands to ~1e-5 relative.
__device__ __forceinline__ void split_bf16(float x, cutlass::bfloat16_t& hi, cutlass::bfloat16_t& lo) {
    hi = cutlass::bfloat16_t(x);
    lo = cutlass::bfloat16_t(x - bf16_to_f32(hi));
}

// Atlas: log2 of KDA's per-token decay exp(lower * sigmoid(a * (g + dt_bias)))
// (gate_scale = lower * log2(e)), as both kernels accumulate it over a tile.
__device__ __forceinline__ float kda_log2_decay(float g, float dt_bias, float a, float gate_scale) {
    return gate_scale * sigmoid_f32(a * (g + dt_bias));
}

using namespace cute;

// Atlas: FP32 values (registers or shared memory) as split BF16 fragments.
template <class Src, class Hi, class Lo>
CUTLASS_DEVICE void split_fragment(Src const& x, Hi& hi, Lo& lo) {
    CUTE_UNROLL
    for (int i = 0; i < int(size(x)); ++i) {
        split_bf16(float(x(i)), hi(i), lo(i));
    }
}

// Atlas: c += a * b from split operands; the lo * lo term (~2^-18 relative)
// is dropped.
template <class Mma, class A, class B, class C>
CUTLASS_DEVICE void gemm_split(Mma const& mma, A const& a_hi, A const& a_lo,
                               B const& b_hi, B const& b_lo, C& c) {
    gemm(mma, a_lo, b_hi, c);
    gemm(mma, a_hi, b_lo, c);
    gemm(mma, a_hi, b_hi, c);
}

// A 16x16 BF16 C fragment (rows = MMA K) as the B operand of the next MMA,
// without a shared-memory round trip.
template <class CFragment, class BFragment>
CUTLASS_DEVICE void movm_c_to_b(CFragment const& c, BFragment& b) {
    static_assert(sizeof(CFragment) == 4 * sizeof(uint32_t));
    static_assert(sizeof(BFragment) == 4 * sizeof(uint32_t));
    uint32_t const* src = reinterpret_cast<uint32_t const*>(&c(0));
    uint32_t* dst = reinterpret_cast<uint32_t*>(&b(0));
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        SM75_U32x1_MOVM_T::copy(src[i], dst[i]);
    }
}

// Workspace per-tile byte sizes (all naturally 128-byte aligned)
// Atlas: Kernel 1 leaves only what needs a whole tile: the FP32 INV and Mqk
// and the per-row q/k normalizers; Kernel 2 derives the decayed operands
// from q, k and g itself.
template <int CHUNK, int D>
struct WorkspaceSizes {
    static_assert(CHUNK * CHUNK * 4 % 128 == 0);
    static_assert(2 * CHUNK * 4 % 128 == 0);

    static constexpr int kINV       = CHUNK * CHUNK * 4;     // 1024
    static constexpr int kMqk       = CHUNK * CHUNK * 4;     // 1024
    static constexpr int kNorms     = 2 * CHUNK * 4;         // 128: q (with scale), k
    static constexpr int64_t kPerTile = kINV + kMqk + kNorms;
};

enum class WarpRole {
    MMA,
    PREP,
    LOAD_QKG,
    STORE,
    NonParticipant,
};

template <int Stages>
CUTLASS_DEVICE
cutlass::PipelineTmaAsync<Stages> make_load_pipeline(
    typename cutlass::PipelineTmaAsync<Stages>::SharedStorage& storage,
    uint32_t transaction_bytes,
    WarpRole warp_role,
    uint32_t num_producers,
    uint32_t num_consumers
) {
    using Pipeline = cutlass::PipelineTmaAsync<Stages>;
    typename Pipeline::Params params;

    auto role = Pipeline::ThreadCategory::NonParticipant;
    bool is_leader = false;
    if (warp_role == WarpRole::LOAD_QKG) {
        role = Pipeline::ThreadCategory::Producer;
        is_leader = cute::elect_one_sync();
    } else if (warp_role == WarpRole::MMA || warp_role == WarpRole::PREP) {
        role = Pipeline::ThreadCategory::Consumer;
    }

    params.transaction_bytes = transaction_bytes;
    params.role = role;
    params.is_leader = is_leader;
    params.num_consumers = num_consumers;
    params.num_producers = num_producers;

    Pipeline pipeline(storage, params, Shape<_1,_1>{});
    cutlass::pipeline_init_wait(1);
    return pipeline;
}

template <int Stages>
CUTLASS_DEVICE
cutlass::PipelineAsync<Stages> make_store_pipeline(
    typename cutlass::PipelineAsync<Stages>::SharedStorage& storage,
    WarpRole warp_role,
    uint32_t num_producers,
    uint32_t num_consumers,
    WarpRole producer = WarpRole::MMA,
    WarpRole consumer = WarpRole::STORE
) {
    using Pipeline = cutlass::PipelineAsync<Stages>;
    typename Pipeline::Params params;

    auto role = Pipeline::ThreadCategory::NonParticipant;
    if (warp_role == producer) {
        role = Pipeline::ThreadCategory::Producer;
    } else if (warp_role == consumer) {
        role = Pipeline::ThreadCategory::Consumer;
    }

    params.role = role;
    params.producer_arv_count = num_producers;
    params.consumer_arv_count = num_consumers;

    Pipeline pipeline(storage, params);
    cutlass::pipeline_init_wait(1);
    return pipeline;
}
