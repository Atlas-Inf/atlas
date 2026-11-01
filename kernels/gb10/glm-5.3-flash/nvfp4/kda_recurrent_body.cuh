// SPDX-License-Identifier: AGPL-3.0-only

// Private shared scalar/indexed arithmetic. Preserve operation order and runtime
// dimension arithmetic; see scripts/dev/glm_kda_legacy_reference.cuh for the
// independent pre-extraction numerical oracle.
#pragma once
#include <cuda_bf16.h>

static __device__ __forceinline__ void atlas_kda_recurrent_body(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim,
    float lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int vrow = threadIdx.x;
    if (head >= heads || dim > 128 || blockDim.x < dim) return;

    __shared__ float kda_shared_qv[128];
    __shared__ float kda_shared_kv[128];
    __shared__ float kda_shared_gate_exp[128];
    __shared__ float kda_shared_red_q[128];
    __shared__ float kda_shared_red_k[128];
    __shared__ float kda_shared_inv_q;
    __shared__ float kda_shared_inv_k;
    __shared__ float kda_shared_beta;

    // H is [key_dim, value_dim], with value_dim contiguous. Each thread owns
    // one value column, so every warp reads and writes adjacent FP32 elements.
    // The original [value_dim, key_dim] traversal issued one 32-byte memory
    // transaction per lane for every state access on decode.
    float* H = state + (unsigned long long)head * dim * dim;
    const float a = expf(a_log[head]);
    const float scale = rsqrtf((float)dim);

    for (unsigned int t = 0; t < tokens; ++t) {
        const unsigned long long qbase = (unsigned long long)t * 3 * heads * dim;
        if (vrow < dim) {
            float q = (float)qkv[qbase + (unsigned long long)head * dim + vrow];
            float k = (float)qkv[qbase + (unsigned long long)heads * dim
                              + (unsigned long long)head * dim + vrow];
            kda_shared_qv[vrow] = q;
            kda_shared_kv[vrow] = k;
            kda_shared_red_q[vrow] = q * q;
            kda_shared_red_k[vrow] = k * k;
            float g = (float)raw_gate[((unsigned long long)t * heads + head) * dim + vrow];
            float log_decay = lower_bound /
                (1.0f + expf(-a * (g + dt_bias[(unsigned long long)head * dim + vrow])));
            kda_shared_gate_exp[vrow] = expf(log_decay);
        }
        __syncthreads();

        for (unsigned int stride = 64; stride > 0; stride >>= 1) {
            if (vrow < stride) {
                kda_shared_red_q[vrow] += kda_shared_red_q[vrow + stride];
                kda_shared_red_k[vrow] += kda_shared_red_k[vrow + stride];
            }
            __syncthreads();
        }
        if (vrow == 0) {
            kda_shared_inv_q = rsqrtf(kda_shared_red_q[0] + 1.0e-6f) * scale;
            kda_shared_inv_k = rsqrtf(kda_shared_red_k[0] + 1.0e-6f);
            float b = (float)raw_beta[(unsigned long long)t * heads + head];
            kda_shared_beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();

        if (vrow < dim) {
            float dot_k = 0.0f;
            #pragma unroll 4
            for (unsigned int k = 0; k < dim; k += 4) {
                float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * kda_shared_gate_exp[k + 0];
                float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * kda_shared_gate_exp[k + 1];
                float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * kda_shared_gate_exp[k + 2];
                float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * kda_shared_gate_exp[k + 3];
                dot_k += h0 * (kda_shared_kv[k + 0] * kda_shared_inv_k) + h1 * (kda_shared_kv[k + 1] * kda_shared_inv_k)
                       + h2 * (kda_shared_kv[k + 2] * kda_shared_inv_k) + h3 * (kda_shared_kv[k + 3] * kda_shared_inv_k);
            }
            const unsigned long long vbase = qbase + (unsigned long long)2 * heads * dim;
            float delta = ((float)qkv[vbase + (unsigned long long)head * dim + vrow]
                           - dot_k) * kda_shared_beta;
            float out = 0.0f;
            #pragma unroll 4
            for (unsigned int k = 0; k < dim; k += 4) {
                float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * kda_shared_gate_exp[k + 0]
                         + delta * (kda_shared_kv[k + 0] * kda_shared_inv_k);
                float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * kda_shared_gate_exp[k + 1]
                         + delta * (kda_shared_kv[k + 1] * kda_shared_inv_k);
                float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * kda_shared_gate_exp[k + 2]
                         + delta * (kda_shared_kv[k + 2] * kda_shared_inv_k);
                float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * kda_shared_gate_exp[k + 3]
                         + delta * (kda_shared_kv[k + 3] * kda_shared_inv_k);
                H[(unsigned long long)(k + 0) * dim + vrow] = h0;
                H[(unsigned long long)(k + 1) * dim + vrow] = h1;
                H[(unsigned long long)(k + 2) * dim + vrow] = h2;
                H[(unsigned long long)(k + 3) * dim + vrow] = h3;
                out += h0 * (kda_shared_qv[k + 0] * kda_shared_inv_q) + h1 * (kda_shared_qv[k + 1] * kda_shared_inv_q)
                     + h2 * (kda_shared_qv[k + 2] * kda_shared_inv_q) + h3 * (kda_shared_qv[k + 3] * kda_shared_inv_q);
            }
            output[((unsigned long long)t * heads + head) * dim + vrow] =
                __float2bfloat16(out);
        }
        __syncthreads();
    }
}

