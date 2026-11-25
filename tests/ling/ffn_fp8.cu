// SPDX-License-Identifier: AGPL-3.0-only
// Direct parity probe for the native Ling strict W8A8 expert kernels.
//
// Build from the Atlas root on a GB10:
//   nvcc -std=c++17 -O2 -arch=sm_121 tests/ling/ffn_fp8.cu -o /tmp/ling-ffn-fp8
//   /tmp/ling-ffn-fp8
//
// The probe uses one selected routed expert and the shared expert. It checks
// both gate/up projection rows and the down projection after deliberately
// placing values above the routed (4) and shared (5) post-SiLU clamp limits.

#include "../../kernels/gb10/common/per_token_group_quant_fp8.cu"
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_ffn.cu"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kExperts = 512;
constexpr int kTopK = 1;
constexpr int kHidden = 2560;
constexpr int kIntermediate = 768;
constexpr int kGroup = 128;
constexpr int kSelectedExpert = 7;
constexpr float kFp8Max = 448.0f;

void check(cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", expression, cudaGetErrorString(error));
        std::exit(2);
    }
}

#define CHECK(call) check((call), #call)

template <typename T>
T* managed(std::size_t count) {
    T* pointer = nullptr;
    CHECK(cudaMallocManaged(&pointer, count * sizeof(T)));
    return pointer;
}

template <typename T>
void release(T* pointer) {
    CHECK(cudaFree(pointer));
}

float as_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__nv_bfloat16 as_bf16(float value) {
    return __float2bfloat16(value);
}

float as_fp8(float value) {
    return float(__nv_fp8_e4m3(value));
}

float decode_fp8(unsigned char bits) {
    __nv_fp8_e4m3 encoded;
    std::memcpy(&encoded, &bits, sizeof(bits));
    return float(encoded);
}

void reference_linear(const unsigned char* activations, const float* activation_scales,
                      const __nv_fp8_e4m3* weights, const float* weight_scales,
                      int n, int k, std::vector<__nv_bfloat16>& result) {
    const int groups = k / kGroup;
    result.resize(n);
    for (int row = 0; row < n; ++row) {
        float accumulator = 0.0f;
        for (int group = 0; group < groups; ++group) {
            float dot = 0.0f;
            for (int offset = 0; offset < kGroup; ++offset) {
                const int index = group * kGroup + offset;
                dot += decode_fp8(activations[index]) * float(weights[row * k + index]);
            }
            accumulator += dot * activation_scales[group] *
                           weight_scales[(row / kGroup) * groups + group];
        }
        result[row] = as_bf16(accumulator);
    }
}

void reference_down(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                    const __nv_fp8_e4m3* weights, const float* weight_scales,
                    int n, int k, float limit, std::vector<__nv_bfloat16>& result) {
    const int groups = k / kGroup;
    result.resize(n);
    std::vector<float> values(kGroup);
    for (int row = 0; row < n; ++row) {
        float accumulator = 0.0f;
        for (int group = 0; group < groups; ++group) {
            float maximum = 0.0f;
            for (int offset = 0; offset < kGroup; ++offset) {
                const int index = group * kGroup + offset;
                const float g = as_float(gate[index]);
                float u = as_float(up[index]);
                float activated = as_float(as_bf16(g / (1.0f + std::exp(-g))));
                if (limit > 0.0f) {
                    activated = std::min(activated, limit);
                    u = std::max(-limit, std::min(u, limit));
                }
                values[offset] = as_float(as_bf16(activated * u));
                maximum = std::max(maximum, std::fabs(values[offset]));
            }
            const float activation_scale = std::max(maximum / kFp8Max, 1.0e-12f);
            float dot = 0.0f;
            for (int offset = 0; offset < kGroup; ++offset) {
                const float normalized = std::max(-kFp8Max,
                    std::min(kFp8Max, values[offset] / activation_scale));
                const float activation = as_fp8(normalized);
                const int index = group * kGroup + offset;
                dot += activation * float(weights[row * k + index]);
            }
            accumulator += dot * activation_scale *
                           weight_scales[(row / kGroup) * groups + group];
        }
        result[row] = as_bf16(accumulator);
    }
}

void compare(const char* label, const __nv_bfloat16* actual,
             const std::vector<__nv_bfloat16>& expected) {
    float maximum_error = 0.0f;
    int worst = -1;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float a = as_float(actual[i]);
        const float e = as_float(expected[i]);
        const float error = std::fabs(a - e);
        const float allowed = 0.05f + 0.01f * std::fabs(e);
        if (error > maximum_error) {
            maximum_error = error;
            worst = static_cast<int>(i);
        }
        if (!std::isfinite(error) || error > allowed) {
            std::fprintf(stderr, "%s mismatch at %d: actual=%g expected=%g error=%g allowed=%g\n",
                         label, static_cast<int>(i), a, e, error, allowed);
            std::exit(1);
        }
    }
    std::printf("%s matched %zu BF16 outputs (max_abs=%g at %d)\n",
                label, expected.size(), maximum_error, worst);
}

void fill_weights(__nv_fp8_e4m3* weights, float* scales, int n, int k, int salt) {
    const int groups = k / kGroup;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            const int pattern = (row * 17 + col * 13 + salt) % 29 - 14;
            weights[row * k + col] = __nv_fp8_e4m3(float(pattern) * 0.18f);
        }
    }
    for (int row = 0; row < (n + kGroup - 1) / kGroup; ++row)
        for (int group = 0; group < groups; ++group)
            scales[row * groups + group] = 0.0025f * float(1 + (row + group + salt) % 11);
}

} // namespace

int main() {
    const int input_groups = kHidden / kGroup;
    const int gate_groups = kHidden / kGroup;
    const int down_groups = kIntermediate / kGroup;
    const int gate_scale_count = (kIntermediate / kGroup) * gate_groups;
    const int down_scale_count = (kHidden / kGroup) * down_groups;

    auto input = managed<__nv_bfloat16>(kHidden);
    auto activations = managed<unsigned char>(kHidden);
    auto activation_scales = managed<float>(input_groups);
    auto gate_weight_base = managed<__nv_fp8_e4m3>(kIntermediate * kHidden + 1);
    auto gate_weight = gate_weight_base + 1;
    auto up_weight_base = managed<__nv_fp8_e4m3>(kIntermediate * kHidden + 2);
    auto up_weight = up_weight_base + 2;
    auto down_weight_base = managed<__nv_fp8_e4m3>(kHidden * kIntermediate + 3);
    auto down_weight = down_weight_base + 3;
    auto gate_scale = managed<float>(gate_scale_count);
    auto up_scale = managed<float>(gate_scale_count);
    auto down_scale = managed<float>(down_scale_count);
    auto gate_ptrs = managed<unsigned long long>(kExperts);
    auto up_ptrs = managed<unsigned long long>(kExperts);
    auto down_ptrs = managed<unsigned long long>(kExperts);
    auto gate_scale_ptrs = managed<unsigned long long>(kExperts);
    auto up_scale_ptrs = managed<unsigned long long>(kExperts);
    auto down_scale_ptrs = managed<unsigned long long>(kExperts);
    auto expert_indices = managed<unsigned int>(kTopK);
    auto routed_gate = managed<__nv_bfloat16>(kIntermediate * kTopK);
    auto routed_up = managed<__nv_bfloat16>(kIntermediate * kTopK);
    auto routed_down = managed<__nv_bfloat16>(kHidden * kTopK);
    auto shared_gate = managed<__nv_bfloat16>(kIntermediate);
    auto shared_up = managed<__nv_bfloat16>(kIntermediate);
    auto shared_down = managed<__nv_bfloat16>(kHidden);
    auto shared_gate_weight = managed<__nv_fp8_e4m3>(kIntermediate * kHidden);
    auto shared_up_weight = managed<__nv_fp8_e4m3>(kIntermediate * kHidden);
    auto shared_down_weight = managed<__nv_fp8_e4m3>(kHidden * kIntermediate);
    auto shared_gate_scale = managed<float>(gate_scale_count);
    auto shared_up_scale = managed<float>(gate_scale_count);
    auto shared_down_scale = managed<float>(down_scale_count);

    for (int i = 0; i < kHidden; ++i)
        input[i] = as_bf16(std::sin(float(i) * 0.037f) * 3.0f + ((i % 31) - 15) * 0.01f);
    input[0] = as_bf16(17.0f);
    input[128] = as_bf16(-13.0f);
    fill_weights(gate_weight, gate_scale, kIntermediate, kHidden, 3);
    fill_weights(up_weight, up_scale, kIntermediate, kHidden, 9);
    fill_weights(down_weight, down_scale, kHidden, kIntermediate, 15);
    fill_weights(shared_gate_weight, shared_gate_scale, kIntermediate, kHidden, 21);
    fill_weights(shared_up_weight, shared_up_scale, kIntermediate, kHidden, 27);
    fill_weights(shared_down_weight, shared_down_scale, kHidden, kIntermediate, 33);

    for (int i = 0; i < kExperts; ++i) {
        gate_ptrs[i] = up_ptrs[i] = down_ptrs[i] = 0;
        gate_scale_ptrs[i] = up_scale_ptrs[i] = down_scale_ptrs[i] = 0;
    }
    expert_indices[0] = kSelectedExpert;
    gate_ptrs[kSelectedExpert] = reinterpret_cast<unsigned long long>(gate_weight);
    up_ptrs[kSelectedExpert] = reinterpret_cast<unsigned long long>(up_weight);
    down_ptrs[kSelectedExpert] = reinterpret_cast<unsigned long long>(down_weight);
    gate_scale_ptrs[kSelectedExpert] = reinterpret_cast<unsigned long long>(gate_scale);
    up_scale_ptrs[kSelectedExpert] = reinterpret_cast<unsigned long long>(up_scale);
    down_scale_ptrs[kSelectedExpert] = reinterpret_cast<unsigned long long>(down_scale);

    per_token_group_quant_fp8<<<dim3(1, input_groups), kGroup>>>(
        input, activations, activation_scales, 1, kHidden);
    CHECK(cudaGetLastError());
    ling_fp8_gate_up_shared<<<dim3((kIntermediate + 7) / 8, kTopK + 1, 2), 128>>>(
        activations, activation_scales, gate_ptrs, gate_scale_ptrs, routed_gate,
        up_ptrs, up_scale_ptrs, routed_up, expert_indices,
        reinterpret_cast<unsigned char*>(shared_gate_weight), shared_gate_scale, shared_gate,
        reinterpret_cast<unsigned char*>(shared_up_weight), shared_up_scale, shared_up,
        kIntermediate, kHidden, kTopK);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> expected;
    reference_linear(activations, activation_scales, gate_weight, gate_scale,
                     kIntermediate, kHidden, expected);
    compare("routed gate", routed_gate, expected);
    reference_linear(activations, activation_scales, up_weight, up_scale,
                     kIntermediate, kHidden, expected);
    compare("routed up", routed_up, expected);
    reference_linear(activations, activation_scales, shared_gate_weight, shared_gate_scale,
                     kIntermediate, kHidden, expected);
    compare("shared gate", shared_gate, expected);
    reference_linear(activations, activation_scales, shared_up_weight, shared_up_scale,
                     kIntermediate, kHidden, expected);
    compare("shared up", shared_up, expected);

    // Make clamp order observable. The down kernel must compute SiLU first,
    // then clamp the activated gate and the raw up value, before quantizing.
    for (int i = 0; i < kIntermediate; ++i) {
        const float sign = (i & 1) ? -1.0f : 1.0f;
        routed_gate[i] = as_bf16(sign * (5.0f + (i % 7) * 0.17f));
        routed_up[i] = as_bf16(sign * (6.0f + (i % 5) * 0.21f));
        shared_gate[i] = as_bf16(sign * (6.0f + (i % 7) * 0.13f));
        shared_up[i] = as_bf16(sign * (7.0f + (i % 5) * 0.19f));
    }
    ling_silu_down_shared_fp8<<<dim3((kHidden + 7) / 8, kTopK + 1), 128>>>(
        routed_gate, routed_up, down_ptrs, down_scale_ptrs, routed_down, expert_indices,
        shared_gate, shared_up, reinterpret_cast<unsigned char*>(shared_down_weight),
        shared_down_scale, shared_down, kHidden, kIntermediate, kTopK, 4.0f, 5.0f);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    reference_down(routed_gate, routed_up, down_weight, down_scale,
                   kHidden, kIntermediate, 4.0f, expected);
    compare("routed down (limit 4 after SiLU)", routed_down, expected);
    reference_down(shared_gate, shared_up, shared_down_weight, shared_down_scale,
                   kHidden, kIntermediate, 5.0f, expected);
    compare("shared down (limit 5 after SiLU)", shared_down, expected);

    release(shared_down_scale); release(shared_up_scale); release(shared_gate_scale);
    release(shared_down_weight); release(shared_up_weight); release(shared_gate_weight);
    release(shared_down); release(shared_up); release(shared_gate);
    release(routed_down); release(routed_up); release(routed_gate);
    release(expert_indices);
    release(down_scale_ptrs); release(up_scale_ptrs); release(gate_scale_ptrs);
    release(down_ptrs); release(up_ptrs); release(gate_ptrs);
    release(down_scale); release(up_scale); release(gate_scale);
    release(down_weight_base); release(up_weight_base); release(gate_weight_base);
    release(activation_scales); release(activations); release(input);
    return 0;
}
