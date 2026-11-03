// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// The chain one-shot channel's device helpers (`rdma_oneshot.cu`), and the
// producer side of the fused reduce-scatter (`oneshot::RsFused`): a kernel
// that computes the reduce-scatter's input rows (the MoE seam's unpermute)
// writes each piece straight into the op's buffers and publishes it, so the
// network starts on finished rows and the chain kernel
// (`rdma_chain_bf16`, root == ranks + 4) sums them as they land.
//
// Producer contract, every block:
//   rs_fused_begin_dev(plan);                   // once, first
//   for i in 0..plan.npieces, in order:
//     write the BF16 rows of piece i (plan.piece_bytes from the row at
//       plan.piece[i].out, the seam buffer) to rs_fused_piece_dst(plan, i)
//       (rs_fused_piece_raw: pinned staging, store with st.wt);
//     rs_fused_piece_done(plan, i, gridDim.x);
// The producer must be stream-ordered after every earlier op of the channel
// (it reads the channel's device sequence). Device-only, no host symbols.

#include <stdint.h>

#define ONESHOT_POISON_DESYNC (1ull << 61)
#define ONESHOT_POISON_TIMEOUT (1ull << 62)
#define ONESHOT_POISON_MISMATCH (1ull << 63)
#define ONESHOT_FLAG_STRIDE 8 // u64 words: one 64-byte line per rail
#define ONESHOT_MAX_RAILS 8    // a pair's rails (oneshot.rs MAX_RAILS)
#define ONESHOT_ORIGIN_LINES 32 // flag lines per origin: a chain op's pieces (MAX_PIECES)
#define ONESHOT_MAX_RANKS 4    // oneshot.rs MAX_RANKS
#define ONESHOT_UNROLL 4       // receive loads in flight per thread
#define ONESHOT_READY 14464    // bytes from `stage`: oneshot.rs READY (one u64 per piece)
// A `stage` word with this bit: the proxy sends piece p once ready[p] == seq
// (a pipelined all-gather stages piece by piece), not the whole op at once.
#define ONESHOT_PIPE_BIT (1u << 31)

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

__device__ __forceinline__ uint32_t oneshot_ld_acquire_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Staging stores into pinned host memory read by the NIC: write-through
// (mode 1) or plain (mode 2).
__device__ __forceinline__ void oneshot_st_v4(uint4* p, uint4 v, uint32_t mode) {
    if (mode == 1) {
        asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
                     ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    } else {
        *p = v;
    }
}

__device__ __forceinline__ void oneshot_st_u16(unsigned short* p, unsigned short v, uint32_t mode) {
    if (mode == 1) {
        asm volatile("st.global.wt.u16 [%0], %1;" ::"l"(p), "h"(v) : "memory");
    } else {
        *p = v;
    }
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

// The pipelined ops' start (all-gather, reduce-scatter): block 0 waits until
// the proxy drained the previous op's sends (`stage` == 0), publishes `code`
// (with the PIPE bit) and then `go` = seq in device memory; every other block
// waits for `go`, never for `stage`, which already holds this op's code by
// the time a late block would look (it would wait forever).
__device__ __forceinline__ void oneshot_pipe_gate(
    uint32_t* stage, uint64_t* state, uint64_t seq, uint32_t code, uint64_t* poison,
    uint64_t timeout_ns) {
    uint64_t* go = state + 2;
    if (threadIdx.x == 0) {
        const uint64_t t0 = oneshot_now();
        for (uint32_t spin = 1;; ++spin) {
            const bool ready = blockIdx.x == 0 ? oneshot_ld_acquire_u32(stage) == 0
                                               : oneshot_ld_acquire_gpu(go) >= seq;
            if (ready) {
                break;
            }
            if (spin % 64 == 0) {
                if (oneshot_ld_acquire(poison) != 0) {
                    __trap();
                }
                if (timeout_ns != 0 && oneshot_now() - t0 > timeout_ns) {
                    oneshot_poison(poison, ONESHOT_POISON_TIMEOUT | seq);
                }
            }
        }
        if (blockIdx.x == 0) {
            asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(stage),
                         "r"(code | ONESHOT_PIPE_BIT) : "memory");
            asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(go), "l"(seq) : "memory");
        }
    }
    __syncthreads();
}

// ---- The fused reduce-scatter's producer side ----
#define ONESHOT_RS_MAX_PIECES 32
// Byte offsets in the channel's device state (oneshot.rs STATE_*).
#define ONESHOT_STATE_PRODUCED 256 // u32 per piece: producer blocks done
#define ONESHOT_STATE_OWN 512      // u64 per piece: = seq once the producer wrote it

struct RsFusedPiece {
    uint64_t out;   // the piece's first row in the seam buffer (device)
    uint64_t s;     // pinned staging address of a raw piece, 0 for a kept one
    uint32_t p;     // the op's piece index
    uint32_t pad;
};

// `oneshot::RsFusedPlan`, by value: 64 + 32 * 24 bytes.
struct RsFusedPlan {
    uint64_t ready;      // u64 per piece, pinned (the proxy sends piece p at ready[p] == seq)
    uint64_t state;      // the channel's device state
    uint64_t stage;      // u32, pinned
    uint64_t poison;     // u64, pinned
    uint64_t timeout_ns;
    uint32_t code;       // the op's size code (oneshot::FUSED_BIT)
    uint32_t npieces;
    uint32_t piece_bytes;
    uint32_t mode;       // 1: staging stores write through (st.wt)
    uint64_t pad;
    RsFusedPiece piece[ONESHOT_RS_MAX_PIECES];
};

__device__ __forceinline__ uint64_t rs_fused_seq(const RsFusedPlan& plan) {
    return *(volatile uint64_t*)plan.state + 1;
}

__device__ __forceinline__ void rs_fused_begin_dev(const RsFusedPlan& plan) {
    oneshot_pipe_gate(reinterpret_cast<uint32_t*>(plan.stage),
                      reinterpret_cast<uint64_t*>(plan.state), rs_fused_seq(plan), plan.code,
                      reinterpret_cast<uint64_t*>(plan.poison), plan.timeout_ns);
}

__device__ __forceinline__ bool rs_fused_piece_raw(const RsFusedPlan& plan, uint32_t i) {
    return plan.piece[i].s != 0;
}

__device__ __forceinline__ unsigned char* rs_fused_piece_dst(const RsFusedPlan& plan, uint32_t i) {
    return reinterpret_cast<unsigned char*>(plan.piece[i].s != 0 ? plan.piece[i].s
                                                                  : plan.piece[i].out);
}

// Every block, once it wrote its rows of piece i: the last of `writers`
// publishes the piece (a raw one to the proxy, a kept one to the chain
// kernel).
__device__ __forceinline__ void rs_fused_piece_done(
    const RsFusedPlan& plan, uint32_t i, uint32_t writers) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t p = plan.piece[i].p;
        unsigned char* st = reinterpret_cast<unsigned char*>(plan.state);
        uint32_t* produced = reinterpret_cast<uint32_t*>(st + ONESHOT_STATE_PRODUCED) + p;
        uint32_t prior;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(prior) : "l"(produced) : "memory");
        if (prior + 1 == writers) {
            *produced = 0;
            const uint64_t seq = rs_fused_seq(plan);
            if (plan.piece[i].s != 0) {
                asm volatile("st.release.sys.global.u64 [%0], %1;"
                             ::"l"(reinterpret_cast<uint64_t*>(plan.ready) + p), "l"(seq)
                             : "memory");
            } else {
                asm volatile("st.release.gpu.global.u64 [%0], %1;"
                             ::"l"(reinterpret_cast<uint64_t*>(st + ONESHOT_STATE_OWN) + p),
                             "l"(seq) : "memory");
            }
        }
    }
    __syncthreads();
}
