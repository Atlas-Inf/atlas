// SPDX-License-Identifier: AGPL-3.0-only

// Test-only kernels for the one-shot NIC stress (loopback_tests.rs,
// nic_loopback_sm_stage_stress): fill a rank's payload for op k on the GPU and
// check the landed all-reduce against the same values recomputed, so millions
// of ops run without a host round trip each. Every element changes from op to
// op, so a stale staged byte read by the NIC shows up as a wrong value.
//
//   nvcc -cubin -arch=sm_121a -O3 oneshot_stress.cu -o oneshot_stress.cubin

#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ unsigned short stress_val(uint64_t k, uint32_t rank, uint32_t i) {
    uint64_t x = (k << 33) ^ ((uint64_t)rank << 32) ^ i;
    x = (x ^ (x >> 31)) * 0x7fb5d329728ea185ull;
    x = (x ^ (x >> 27)) * 0x81dadef4bc2dd44dull;
    return (unsigned short)(x ^ (x >> 33));
}

extern "C" __global__ void stress_fill(unsigned short* p, uint32_t n, uint32_t rank, uint64_t k) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] = stress_val(k, rank, i);
    }
}

// bad[0]: wrong elements; bad[1]: blocks that saw one; bad[2]: lowest bad op
// (initialise to ~0); bad[3]: ops checked (block 0).
extern "C" __global__ void stress_check(
    const unsigned short* dst, uint32_t n, uint32_t rank, uint64_t k, uint32_t add,
    unsigned long long* bad) {
    __shared__ unsigned int wrong;
    if (threadIdx.x == 0) {
        wrong = 0;
    }
    __syncthreads();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned short peer = stress_val(k, 1 - rank, i);
        unsigned short want = peer;
        if (add) {
            // The landing's operand order: own + peer.
            const __nv_bfloat16 s = __hadd(__ushort_as_bfloat16(stress_val(k, rank, i)),
                                           __ushort_as_bfloat16(peer));
            want = __bfloat16_as_ushort(s);
        }
        if (dst[i] != want) {
            atomicAdd(&wrong, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (wrong != 0) {
            atomicAdd(bad, (unsigned long long)wrong);
            atomicAdd(bad + 1, 1ull);
            atomicMin(bad + 2, (unsigned long long)k);
        }
        if (blockIdx.x == 0) {
            atomicAdd(bad + 3, 1ull);
        }
    }
}
