// SPDX-License-Identifier: AGPL-3.0-only
// Keep the logical attention width while reading the shared RoPE tail once per token.
#include "ling_cache_layout.cuh"
#define HDIM 192
#define ATLAS_PREFILL_GROUP_QUERY_TILES
#define inferspark_prefill_paged ling_mla_prefill
#include "../../common/inferspark_prefill_paged.cu"

// The learned NoPE channels differ per head. All heads share one RoPE key,
// and padding in the mathematical V head contributes zero.
extern "C" __global__ void ling_cache_write(const __nv_bfloat16* key, const __nv_bfloat16* value,
        __nv_bfloat16* k_pool, __nv_bfloat16* v_pool, const long long* slots) {
    const unsigned row = blockIdx.x;
    if (slots[row] < 0) return;
    constexpr unsigned heads = 32, n = heads*128, stride = n+64;
    const unsigned long long dst = static_cast<unsigned long long>(slots[row]) * stride;
    for (unsigned i=threadIdx.x; i<stride; i+=blockDim.x) {
        const unsigned source = i<n ? (i/128)*192+i%128 : 128+i-n;
        k_pool[dst+i] = key[row*heads*192+source];
        v_pool[dst+i] = i<n ? value[row*n+i] : __float2bfloat16(0.f);
    }
}
