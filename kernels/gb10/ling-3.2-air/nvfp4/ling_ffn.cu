// SPDX-License-Identifier: AGPL-3.0-only

// Original Ling 3.2 expert epilogue.  The checkpoint's routed and shared
// matrices are row-major E4M3 with 128-wide dynamic activation groups and
// [N/128,K/128] BF16/F32 weight_scale_inv tensors.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

#define FP8_GROUP_K 128
#define FP8_E4M3_MAX 448.0f

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ unsigned char ling_encode_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1u;
    int exponent = (int)((bits >> 23) & 0xFF) - 127;
    unsigned int mantissa = bits & 0x7FFFFFu;
    int encoded_exp = exponent + 7;
    unsigned int encoded_mantissa;
    if (encoded_exp < 1) {
        encoded_exp = 0;
        encoded_mantissa = 0;
        if (exponent >= -10) {
            float abs_v = v < 0.0f ? -v : v;
            encoded_mantissa = (unsigned int)(abs_v / 0.001953125f + 0.5f);
            if (encoded_mantissa > 7u) encoded_mantissa = 7u;
        }
    } else if (encoded_exp > 15) {
        encoded_exp = 15;
        encoded_mantissa = 6;
    } else {
        encoded_mantissa = (mantissa + (1u << 19)) >> 20;
        if (encoded_mantissa > 7u) {
            encoded_mantissa = 0;
            ++encoded_exp;
            if (encoded_exp > 15) {
                encoded_exp = 15;
                encoded_mantissa = 6;
            }
        }
    }
    return (unsigned char)((sign << 7) | ((unsigned)encoded_exp << 3) | encoded_mantissa);
}
#endif

__device__ __forceinline__ unsigned char ling_encode_e4m3(float v) {
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    return ling_encode_fp8(v);
#else
    __nv_fp8_storage_t encoded = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return (unsigned char)encoded;
#endif
}

__device__ __forceinline__ float ling_decode_e4m3(unsigned char bits) {
    const unsigned int exponent = (bits >> 3) & 0x0Fu;
    const unsigned int mantissa = bits & 0x07u;
    float value;
    if (exponent == 0) {
        value = (float)mantissa * 0.001953125f;
    } else if (exponent == 15 && mantissa == 7) {
        value = 0.0f; // E4M3 NaN; checkpoint weights never intentionally use it.
    } else {
        value = (1.0f + (float)mantissa * 0.125f) * exp2f((float)exponent - 7.0f);
    }
    return (bits & 0x80u) ? -value : value;
}

// Each lane owns four contiguous FP8 coefficients. Fetch a single word
// and convert two pairs with Blackwell's native E4M3 instructions.
__device__ __forceinline__ void ling_load_four(const unsigned char* src, float (&out)[4]) {
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = ling_decode_e4m3(src[i]);
#else
    // Safetensors can place an E4M3 matrix at any byte offset. GB10 reads
    // those expert pages directly from the original file mapping.
    unsigned int word;
    if ((reinterpret_cast<size_t>(src) & 3u) == 0) {
        word = *reinterpret_cast<const unsigned int*>(src);
    } else {
        word = static_cast<unsigned int>(src[0])
             | (static_cast<unsigned int>(src[1]) << 8)
             | (static_cast<unsigned int>(src[2]) << 16)
             | (static_cast<unsigned int>(src[3]) << 24);
    }
    const __half2 low = __nv_cvt_fp8x2_to_halfraw2(static_cast<unsigned short>(word), __NV_E4M3);
    const __half2 high = __nv_cvt_fp8x2_to_halfraw2(static_cast<unsigned short>(word >> 16), __NV_E4M3);
    const float2 a = __half22float2(low);
    const float2 b = __half22float2(high);
    out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
#endif
}

// W8A8 gate/up decode for one token.  Each warp owns two output rows and
// folds the dynamic activation scale with the original 128x128 weight scale
// after the FP8 dot product, matching fp8_gemm_t_blockscaled numerics.
extern "C" __global__ void ling_fp8_gate_up_shared(
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ a_scale,
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ shared_gate_weight,
    const float* __restrict__ shared_gate_scale,
    __nv_bfloat16* __restrict__ shared_gate_out,
    const unsigned char* __restrict__ shared_up_weight,
    const float* __restrict__ shared_up_scale,
    __nv_bfloat16* __restrict__ shared_up_out,
    unsigned int N,
    unsigned int K,
    unsigned int top_k) {
    const unsigned int slot = blockIdx.y;
    const bool is_shared = slot == top_k;
    const unsigned int projection = blockIdx.z;
    const unsigned char* weight;
    const float* block_scale;
    __nv_bfloat16* output;
    if (is_shared) {
        if (projection == 0) {
            weight = shared_gate_weight;
            block_scale = shared_gate_scale;
            output = shared_gate_out;
        } else {
            weight = shared_up_weight;
            block_scale = shared_up_scale;
            output = shared_up_out;
        }
    } else {
        const unsigned int expert = expert_indices[slot];
        if (projection == 0) {
            weight = (const unsigned char*)gate_weight_ptrs[expert];
            block_scale = (const float*)gate_scale_ptrs[expert];
            output = gate_out + (unsigned long long)slot * N;
        } else {
            weight = (const unsigned char*)up_weight_ptrs[expert];
            block_scale = (const float*)up_scale_ptrs[expert];
            output = up_out + (unsigned long long)slot * N;
        }
        if (weight == 0) {
            const unsigned int base = blockIdx.x * 8;
            for (unsigned int i = threadIdx.x; i < 8 && base + i < N; i += 128)
                output[base + i] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int local_output = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int n1 = blockIdx.x * 8 + local_output * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = n2 < N;
    const unsigned int groups = K / FP8_GROUP_K;
    const unsigned int n1_block = n1 / FP8_GROUP_K;
    const unsigned int n2_block = n2 / FP8_GROUP_K;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (unsigned int group = 0; group < groups; ++group) {
        const unsigned int base = group * FP8_GROUP_K + lane * 4;
        const float as = a_scale[group];
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float activation[4], w1[4], w2[4];
        ling_load_four(A_fp8 + base, activation);
        ling_load_four(weight + (unsigned long long)n1 * K + base, w1);
        if (have_n2) ling_load_four(weight + (unsigned long long)n2 * K + base, w2);
        #pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            dot1 += activation[i] * w1[i];
            if (have_n2) dot2 += activation[i] * w2[i];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot1 += __shfl_down_sync(0xffffffff, dot1, offset);
            dot2 += __shfl_down_sync(0xffffffff, dot2, offset);
        }
        if (lane == 0) {
            acc1 += dot1 * as * block_scale[n1_block * groups + group];
            if (have_n2)
                acc2 += dot2 * as * block_scale[n2_block * groups + group];
        }
    }
    if (lane == 0) {
        output[n1] = __float2bfloat16(acc1);
        if (have_n2) output[n2] = __float2bfloat16(acc2);
    }
}

// Clamp follows the reference implementation: SiLU(gate) is computed first,
// then bounded above, while up is bounded on both sides.  The product is
// rounded through BF16 before the group maximum and FP8 encoding, matching the
// original BF16 activation path.
extern "C" __global__ void ling_silu_mul_quant_fp8(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ output_fp8,
    float* __restrict__ output_scale,
    __nv_bfloat16* __restrict__ output_bf16,
    unsigned int rows,
    unsigned int cols,
    float limit) {
    const unsigned int row = blockIdx.x;
    const unsigned int group = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || group >= cols / FP8_GROUP_K || lane >= FP8_GROUP_K) return;

    const unsigned int index = row * cols + group * FP8_GROUP_K + lane;
    float g = __bfloat162float(gate[index]);
    float u = __bfloat162float(up[index]);
    const float sigmoid = 1.0f / (1.0f + __expf(-g));
    // HF computes SiLU first, then applies the asymmetric clamp.  Clamping
    // the raw gate before SiLU is observably wrong at the layer limits.
    float activated = __bfloat162float(__float2bfloat16(g * sigmoid));
    if (limit > 0.0f) {
        activated = fminf(activated, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    const float product = __bfloat162float(__float2bfloat16(activated * u));
    if (output_bf16 != nullptr) output_bf16[index] = __float2bfloat16(product);

    float max_value = fabsf(product);
    for (int offset = 16; offset > 0; offset >>= 1)
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
    __shared__ float warp_max[4];
    const unsigned int warp = lane >> 5;
    const unsigned int warp_lane = lane & 31;
    if (warp_lane == 0) warp_max[warp] = max_value;
    __syncthreads();
    __shared__ float scale;
    if (lane == 0) {
        float group_max = fmaxf(fmaxf(warp_max[0], warp_max[1]),
                                fmaxf(warp_max[2], warp_max[3]));
        scale = fmaxf(group_max / FP8_E4M3_MAX, 1.0e-12f);
        output_scale[row * (cols / FP8_GROUP_K) + group] = scale;
    }
    __syncthreads();
    float normalized = fmaxf(fminf(product / scale, FP8_E4M3_MAX), -FP8_E4M3_MAX);
    output_fp8[index] = ling_encode_e4m3(normalized);
}

// W4A16 keeps the gated activation in BF16.  Keep the same asymmetric clamp
// as the original Ling implementation: gate is bounded above and up is
// bounded on both sides.  The intermediate is rounded through BF16 so this
// path agrees with the reference before the NVFP4 GEMV consumes it.
extern "C" __global__ void ling_silu_mul_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    unsigned int cols,
    float limit) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = rows * cols;
    if (index >= total) return;
    float g = __bfloat162float(gate[index]);
    float u = __bfloat162float(up[index]);
    const float sigmoid = 1.0f / (1.0f + __expf(-g));
    float activated = __bfloat162float(__float2bfloat16(g * sigmoid));
    if (limit > 0.0f) {
        activated = fminf(activated, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    output[index] = __float2bfloat16(activated * u);
}

// Reduce the eight original expert outputs using the router's already
// normalized probabilities.  The output is intentionally a fresh row because
// the shared expert row is kept in the attention arena until this launch.
extern "C" __global__ void ling_weighted_sum(
    const __nv_bfloat16* __restrict__ expert_output,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden,
    unsigned int top_k) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden) return;
    float sum = 0.0f;
    for (unsigned int expert = 0; expert < top_k; ++expert)
        sum += weights[expert] * __bfloat162float(expert_output[expert * hidden + index]);
    output[index] = __float2bfloat16(sum);
}

// Original-FP8 down projection with dynamic group-128 activation quantization
// and the Ling clamp order.  The activation is quantized inside the same
// warp that consumes it, so the fast path never materializes eight BF16
// activation rows or stages a selected weight through host memory.
extern "C" __global__ void ling_silu_down_shared_fp8(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ output,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ shared_gate_in,
    const __nv_bfloat16* __restrict__ shared_up_in,
    const unsigned char* __restrict__ shared_down_weight,
    const float* __restrict__ shared_down_scale,
    __nv_bfloat16* __restrict__ shared_down_out,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    float routed_limit,
    float shared_limit) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = expert_slot == top_k;
    const unsigned char* weight;
    const float* block_scale;
    const __nv_bfloat16* gate;
    const __nv_bfloat16* up;
    if (is_shared) {
        weight = shared_down_weight;
        block_scale = shared_down_scale;
        gate = shared_gate_in;
        up = shared_up_in;
    } else {
        const unsigned int expert = expert_indices[expert_slot];
        weight = (const unsigned char*)weight_ptrs[expert];
        block_scale = (const float*)block_scale_ptrs[expert];
        gate = gate_out + (unsigned long long)expert_slot * K;
        up = up_out + (unsigned long long)expert_slot * K;
        if (weight == 0) {
            const unsigned int base = blockIdx.x * 8;
            for (unsigned int i = threadIdx.x; i < 8 && base + i < N; i += 128)
                output[(unsigned long long)expert_slot * N + base + i] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int threads_per_output = 32;
    const unsigned int local_output = threadIdx.x / threads_per_output;
    const unsigned int lane = threadIdx.x % threads_per_output;
    const unsigned int n1 = blockIdx.x * 8 + local_output * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = n2 < N;
    const unsigned int groups = K / FP8_GROUP_K;
    const unsigned int n1_block = n1 / 128;
    const unsigned int n2_block = n2 / 128;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    const float limit = is_shared ? shared_limit : routed_limit;
    for (unsigned int group = 0; group < groups; ++group) {
        const unsigned int base = group * FP8_GROUP_K + lane * 4;
        float values[4];
        float local_max = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            const float g = __bfloat162float(gate[base + i]);
            float u = __bfloat162float(up[base + i]);
            float activated = __bfloat162float(__float2bfloat16(g / (1.0f + __expf(-g))));
            if (limit > 0.0f) {
                activated = fminf(activated, limit);
                u = fminf(fmaxf(u, -limit), limit);
            }
            values[i] = __bfloat162float(__float2bfloat16(activated * u));
            local_max = fmaxf(local_max, fabsf(values[i]));
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        float activation_scale = fmaxf(local_max / FP8_E4M3_MAX, 1.0e-12f);
        activation_scale = __shfl_sync(0xffffffff, activation_scale, 0);
        float w1[4], w2[4];
        ling_load_four(weight + (unsigned long long)n1 * K + base, w1);
        if (have_n2) ling_load_four(weight + (unsigned long long)n2 * K + base, w2);
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            const float normalized = fmaxf(fminf(values[i] / activation_scale,
                                                  FP8_E4M3_MAX),
                                           -FP8_E4M3_MAX);
            const float activation = ling_decode_e4m3(ling_encode_e4m3(normalized));
            dot1 += activation * w1[i];
            if (have_n2)
                dot2 += activation * w2[i];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot1 += __shfl_down_sync(0xffffffff, dot1, offset);
            dot2 += __shfl_down_sync(0xffffffff, dot2, offset);
        }
        if (lane == 0) {
            acc1 += dot1 * activation_scale * block_scale[n1_block * groups + group];
            if (have_n2)
                acc2 += dot2 * activation_scale * block_scale[n2_block * groups + group];
        }
    }
    __nv_bfloat16* out = is_shared ? shared_down_out : output + (unsigned long long)expert_slot * N;
    if (lane == 0) out[n1] = __float2bfloat16(acc1);
    if (have_n2) {
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}
