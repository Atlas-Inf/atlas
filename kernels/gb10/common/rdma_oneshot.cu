// SPDX-License-Identifier: AGPL-3.0-only

// Device half of the graph-capturable two-rank one-shot RDMA exchange
// (crates/spark-comm/src/nccl_backend/rdma_pair/oneshot.rs). Before this
// kernel the stream has copied this rank's payload into the staging buffer
// (copy engine). Block 0 publishes its size in the host `stage` word (unless
// the stream already did, `stage` == null), which tells the RDMA proxy to
// send it to the peer followed by one flag per rail. Then: wait until every
// rail's flag from the peer reaches this op, and land the peer payload in
// `dst` -- an in-place add bit-identical to bf16_add_inplace
// (dst = __hadd(dst, peer)) or a copy.
//
// The op sequence number lives in device memory (`state[0]`), so eager
// launches and CUDA-graph replays share one sequence: every block reads
// seq = state[0] + 1 and the last block to finish stores it back. Receive
// slots alternate by seq parity. A flag word is (seq << 24 | bytes); a flag
// at this seq carrying another size means the ranks issued different ops.
// Only block 0 polls the flags in host memory (many SMs polling one host line
// cost ~4 us at 16 blocks) and republishes seq in device memory (`state[2]`)
// for the other blocks, which still check the host flags every 256 polls so
// no block depends on another's residency. A peer that never arrives trips a
// %globaltimer limit, and a flag more than one op ahead means the sequences
// diverged (the peer stages seq + 2 only after its kernel seq + 1, which
// waited for our flag seq + 1). Each fault writes the poison word
// (reason | seq) for the host health check, then traps. The host writes the
// same word when it can no longer send for this rank (its RDMA proxy failed,
// or a launch failed after a fenced stage): block 0 traps on it at launch and
// while it waits, rather than complete ops the peer never sees.
//
// Not on the PDL list: it must start after its stream predecessors, and its
// successors must not read dst before it completes.
//
// Prior art (ideas only, no code; docs/glm-prior-art.md): the protocol --
// pinned parity receive slots, a host proxy that WRITEs each stripe then a
// seq flag on the same QP, poison on timeout -- follows b12x RoCEnante
// (github.com/local-inference-lab/b12x, b12x/comm/roce/ and docs/rocenante.md;
// Jason Cook, local-inference-lab/b12x#295; Apache-2.0), with the graph-replay
// wedge hazard of local-inference-lab/b12x#313. The device-resident sequence,
// block-0 flag polling with a device go word, and last-block-out seq store
// follow mmastrac's arx one-shot all-reduce (github.com/mmastrac/
// glm-5.3-flash-4x-gx10, experimental/arx/arx_vllm.cu, PR #4; that directory
// carries no license, so nothing of it is reproduced here).

#include <cuda_bf16.h>
#include <stdint.h>

#define ONESHOT_POISON_DESYNC (1ull << 61)
#define ONESHOT_POISON_TIMEOUT (1ull << 62)
#define ONESHOT_POISON_MISMATCH (1ull << 63)
#define ONESHOT_FLAG_STRIDE 8 // u64 words: one 64-byte line per rail
#define ONESHOT_UNROLL 4       // receive loads in flight per thread

__device__ __forceinline__ uint64_t oneshot_ld_acquire(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t oneshot_ld_acquire_gpu(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// Receive-slot loads: system-scope (never the non-coherent path), and without
// a memory clobber so a thread keeps several in flight.
__device__ __forceinline__ uint4 oneshot_ld_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned short oneshot_ld_u16(const unsigned short* p) {
    unsigned short v;
    asm volatile("ld.relaxed.sys.global.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint64_t oneshot_now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void oneshot_poison(uint64_t* poison, uint64_t word) {
    *(volatile uint64_t*)poison = word;
    __threadfence_system();
    __trap();
}

// Whether every rail's flag has reached `seq` (poisons on a size mismatch or
// a flag from beyond the next op).
__device__ __forceinline__ bool oneshot_arrived(
    const uint64_t* flags, uint32_t rails, uint64_t seq, uint32_t bytes, uint64_t* poison) {
    for (uint32_t r = 0; r < rails; ++r) {
        const uint64_t f = oneshot_ld_acquire(flags + r * ONESHOT_FLAG_STRIDE);
        if ((f >> 24) < seq) {
            return false;
        }
        if ((f >> 24) == seq && (f & 0xffffff) != bytes) {
            oneshot_poison(poison, ONESHOT_POISON_MISMATCH | seq);
        }
        if ((f >> 24) > seq + 1) {
            oneshot_poison(poison, ONESHOT_POISON_DESYNC | seq);
        }
    }
    return true;
}

extern "C" __global__ void rdma_oneshot_bf16(
    __nv_bfloat16* __restrict__ dst,
    const unsigned char* recv,  // receive slot 0; slot 1 is slot_stride bytes on
    uint64_t slot_stride,
    uint32_t bytes,
    uint32_t add,
    const uint64_t* flags,      // one per rail, ONESHOT_FLAG_STRIDE words apart
    uint32_t rails,
    uint64_t* state,            // device: [0] seq, [1] finished blocks (u32), [2] go
    uint32_t* stage,            // host `stage` word to publish, or null
    uint64_t* poison,
    uint64_t timeout_ns          // 0 = wait forever
) {
    uint64_t* go = state + 2;
    const uint64_t seq = *(volatile uint64_t*)state + 1;
    if (threadIdx.x == 0) {
        if (stage != nullptr && blockIdx.x == 0) {
            // The staging copy completed before this kernel started (stream
            // order), so the proxy may read it once it sees the size.
            asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(stage), "r"(bytes) : "memory");
        }
        // After the publish, which starts our send.
        if (blockIdx.x == 0 && oneshot_ld_acquire(poison) != 0) {
            __trap();
        }
        const uint64_t t0 = oneshot_now();
        for (uint32_t spin = 0;; ++spin) {
            const bool own = blockIdx.x == 0 || spin % 256 == 255;
            if (own ? oneshot_arrived(flags, rails, seq, bytes, poison)
                    : oneshot_ld_acquire_gpu(go) >= seq) {
                break;
            }
            if (spin % 64 == 63) {
                if (blockIdx.x == 0 && oneshot_ld_acquire(poison) != 0) {
                    __trap();
                }
                if (timeout_ns != 0 && oneshot_now() - t0 > timeout_ns) {
                    oneshot_poison(poison, ONESHOT_POISON_TIMEOUT | seq);
                }
            }
        }
        if (blockIdx.x == 0 && gridDim.x > 1) {
            asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(go), "l"(seq) : "memory");
        }
    }
    __syncthreads();

    const unsigned char* src = recv + (seq & 1) * slot_stride;
    const uint32_t n = bytes / 2;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    uint32_t head = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        const uint32_t n8 = n / 8;
        head = n8 * 8;
        for (uint32_t base = tid; base < n8; base += ONESHOT_UNROLL * stride) {
            uint4 p[ONESHOT_UNROLL];
#pragma unroll
            for (int u = 0; u < ONESHOT_UNROLL; ++u) {
                if (base + u * stride < n8) {
                    p[u] = oneshot_ld_v4(s4 + base + u * stride);
                }
            }
#pragma unroll
            for (int u = 0; u < ONESHOT_UNROLL; ++u) {
                const uint32_t i = base + u * stride;
                if (i < n8) {
                    if (add) {
                        const uint4 q = d4[i];
                        __nv_bfloat16* pv = reinterpret_cast<__nv_bfloat16*>(&p[u]);
                        const __nv_bfloat16* qv = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            pv[k] = __hadd(qv[k], pv[k]);
                        }
                    }
                    d4[i] = p[u];
                }
            }
        }
    }
    const unsigned short* s1 = reinterpret_cast<const unsigned short*>(src);
    for (uint32_t i = head + tid; i < n; i += stride) {
        const __nv_bfloat16 p = __ushort_as_bfloat16(oneshot_ld_u16(s1 + i));
        dst[i] = add ? __hadd(dst[i], p) : p;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t* done = reinterpret_cast<uint32_t*>(state + 1);
        __threadfence();
        if (atomicAdd(done, 1u) == gridDim.x - 1) {
            *done = 0;
            *(volatile uint64_t*)state = seq;
        }
    }
}
