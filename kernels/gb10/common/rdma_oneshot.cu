// SPDX-License-Identifier: AGPL-3.0-only

// Device half of the graph-capturable one-shot RDMA exchange
// (crates/spark-comm/src/nccl_backend/rdma_pair/oneshot.rs). Before this
// kernel the stream has copied this rank's payload into the staging buffer
// (copy engine). Block 0 publishes its size in the host `stage` word (unless
// the stream already did, `stage` == null), which tells the RDMA proxy to
// send it to the peer followed by one flag per rail. Then: wait until every
// rail's flag from the peer reaches this op, and land the peer payload in
// `dst` -- an in-place add bit-identical to bf16_add_inplace
// (dst = __hadd(dst, peer)) or a copy.
//
// rdma_chain_bf16 is the same kernel for a chain of three or four ranks:
// every rank's payload reaches every other rank (the proxies forward what
// arrives from one neighbour to the other), into a receive slot pair per
// origin rank. It waits for every origin's flags (one per piece, `rails`)
// and lands their sum in the same pairwise tree on every rank:
// (x0 + x1) + (x2 + x3) for four, each add a __hadd as above. Every rank
// computes the same expression of the same inputs, so the result is
// bit-identical on all of them. `add` is ignored: a chain sums, or with
// `root` < `ranks` copies rank `root`'s payload (a broadcast; the root keeps
// its own), or with `root` == `ranks` + 1 copies each other rank's payload to
// `dst` + origin * `gather_stride` bytes (an all-gather; each rank's own part
// is its staged payload, already in place).
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
// Programmatic dependent launch: every pair block starts with atlas_pdl_enter(),
// so a PDL-launched successor (which waits in its own entry) is scheduled
// while this kernel waits for the peer, and never reads dst before it
// completes. Chain blocks (three or more ranks) start with
// atlas_pdl_enter_serial() instead: a seam issues 8-10 chain ops back to
// back, and with the trigger first all of them were resident at once
// (~the whole register file), slowing a concurrent stream's GEMM 5x
// (pipe2-off.sqlite: the shared expert's 708 us GEMM took 3580 us). The runtime's generic PDL list leaves this kernel out: spark-comm
// launches it itself, with the PDL attribute only when the kernel stages the
// payload (sm_stage != 0); after a copy-engine stage the predecessor is a copy
// and the attribute would buy nothing.
//
// SM staging (sm_stage != 0, ATLAS_RDMA_ONESHOT_SM_STAGE): the kernel copies
// src into the staging buffer itself instead of the copy engine, so the
// producer -> all-reduce -> consumer chain is kernels only. Each block waits
// until the proxy has drained the previous send (`stage` == 0), stores its
// part of the payload (each thread exactly the elements it later lands, so an
// in-place src == dst is read before it is overwritten), fences at system
// scope and counts itself in state[3]; block 0 publishes `stage` once every
// block has. sm_stage 1 stores with st.global.wt (write through the L2 to
// system memory: what the NIC reads), 2 with plain stores (a stress-test
// control for the stale reads seen with SM staging before).
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

#include "atlas_pdl.cuh"
#include "rdma_rs_fused.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Whether every origin's every rail flag has reached `seq` (poisons on a size
// mismatch or a flag from beyond the next op). Origin slot i's flags start
// ONESHOT_ORIGIN_LINES lines after slot i - 1's; with two ranks, slot 0 is the
// peer.
template <int R>
__device__ __forceinline__ bool oneshot_arrived(
    const uint64_t* flags, uint32_t rails, uint64_t seq, uint32_t bytes, uint64_t* poison) {
#pragma unroll
    for (uint32_t i = 0; i + 1 < R; ++i) {
        // Multi-rank routes can change their incoming QP between ops.
        // Match the payload parity so seq+1 cannot satisfy seq's wait or
        // be overwritten by a late seq flag. Preserve the pair's ABI.
        const uint64_t* of = flags + i * ONESHOT_ORIGIN_LINES * ONESHOT_FLAG_STRIDE
                             + (R > 2 ? (seq & 1) : 0);
        for (uint32_t r = 0; r < rails; ++r) {
            const uint64_t f = oneshot_ld_acquire(of + r * ONESHOT_FLAG_STRIDE);
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
    }
    return true;
}

// Land one origin's payload (`peer`, this op's receive slot) in `dst`: added
// in place (dst = __hadd(dst, peer)) when `add`, else copied. With `vec`
// (`dst` 16-byte aligned) the first `n8 * 8` elements go by uint4, the rest
// one by one.
__device__ __forceinline__ void oneshot_land_one(
    __nv_bfloat16* dst, const unsigned char* peer, uint32_t add, uint32_t n, uint32_t n8,
    bool vec, uint32_t tid, uint32_t stride) {
    const uint32_t head = n8 * 8;
    if (vec) {
        const uint4* s4 = reinterpret_cast<const uint4*>(peer);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
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
    const unsigned short* s1 = reinterpret_cast<const unsigned short*>(peer);
    for (uint32_t i = head + tid; i < n; i += stride) {
        const __nv_bfloat16 p = __ushort_as_bfloat16(oneshot_ld_u16(s1 + i));
        dst[i] = add ? __hadd(dst[i], p) : p;
    }
}

// One add of the chain's tree: __hadd per BF16 lane, as in the landing above.
__device__ __forceinline__ uint4 oneshot_add(uint4 a, uint4 b) {
    __nv_bfloat16* av = reinterpret_cast<__nv_bfloat16*>(&a);
    const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        av[k] = __hadd(av[k], bv[k]);
    }
    return a;
}

__device__ __forceinline__ __nv_bfloat16 oneshot_add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

// The pairwise tree over every rank's value, in rank order: ranks 0+1, 2+3,
// ... then their sums likewise. `own` is this rank's; `p[i]` origin slot i's.
template <int R, typename T>
__device__ __forceinline__ T oneshot_tree(T own, const T* p, uint32_t rank) {
    T v[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        v[r] = (uint32_t)r < rank ? p[r] : (uint32_t)r == rank ? own : p[r - 1];
    }
#pragma unroll
    for (int w = 1; w < R; w *= 2) {
#pragma unroll
        for (int r = 0; r + w < R; r += 2 * w) {
            v[r] = oneshot_add(v[r], v[r + w]);
        }
    }
    return v[0];
}

// Land the sum over a chain of `R` ranks in `dst`, element for element as the
// two-rank landing below (`peer` is origin slot 0's receive slot of this op;
// slot i's is i * 2 * slot_stride on).
template <int R>
__device__ __forceinline__ void oneshot_land_sum(
    __nv_bfloat16* dst, const unsigned char* peer, uint64_t slot_stride, uint32_t rank,
    uint32_t n, uint32_t n8, uint32_t tid, uint32_t stride) {
    const uint64_t origin = 2 * slot_stride;
    if (n8 != 0) {
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        for (uint32_t base = tid; base < n8; base += ONESHOT_UNROLL * stride) {
            uint4 p[ONESHOT_UNROLL][R - 1];
#pragma unroll
            for (int u = 0; u < ONESHOT_UNROLL; ++u) {
                if (base + u * stride < n8) {
#pragma unroll
                    for (int j = 0; j < R - 1; ++j) {
                        p[u][j] = oneshot_ld_v4(
                            reinterpret_cast<const uint4*>(peer + j * origin) + base + u * stride);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < ONESHOT_UNROLL; ++u) {
                const uint32_t i = base + u * stride;
                if (i < n8) {
                    d4[i] = oneshot_tree<R>(d4[i], p[u], rank);
                }
            }
        }
    }
    for (uint32_t i = n8 * 8 + tid; i < n; i += stride) {
        __nv_bfloat16 p[R - 1];
#pragma unroll
        for (int j = 0; j < R - 1; ++j) {
            p[j] = __ushort_as_bfloat16(oneshot_ld_u16(
                reinterpret_cast<const unsigned short*>(peer + j * origin) + i));
        }
        dst[i] = oneshot_tree<R>(dst[i], p, rank);
    }
}

// Copy `len` bytes (a multiple of 2) `from` -> `to` on the block's threads,
// by uint4 where both are 16-byte aligned; `sys` stores write through to
// system memory (staging), loads read system scope (receive slots).
__device__ __forceinline__ void oneshot_block_copy(
    unsigned char* to, const unsigned char* from, uint32_t len, uint32_t sm_stage, bool stage) {
    const uint32_t tid = threadIdx.x, nt = blockDim.x;
    const bool vec = ((reinterpret_cast<uintptr_t>(to) | reinterpret_cast<uintptr_t>(from)) & 15) == 0;
    const uint32_t n16 = vec ? len / 16 : 0;
    for (uint32_t base = tid; base < n16; base += ONESHOT_UNROLL * nt) {
        uint4 v[ONESHOT_UNROLL];
#pragma unroll
        for (int u = 0; u < ONESHOT_UNROLL; ++u) {
            if (base + u * nt < n16) {
                const uint4* q = reinterpret_cast<const uint4*>(from) + base + u * nt;
                v[u] = stage ? *q : oneshot_ld_v4(q);
            }
        }
#pragma unroll
        for (int u = 0; u < ONESHOT_UNROLL; ++u) {
            if (base + u * nt < n16) {
                uint4* d = reinterpret_cast<uint4*>(to) + base + u * nt;
                if (stage) {
                    oneshot_st_v4(d, v[u], sm_stage);
                } else {
                    *d = v[u];
                }
            }
        }
    }
    for (uint32_t i = n16 * 8 + tid; i < len / 2; i += nt) {
        const unsigned short* f = reinterpret_cast<const unsigned short*>(from) + i;
        unsigned short* d = reinterpret_cast<unsigned short*>(to) + i;
        if (stage) {
            oneshot_st_u16(d, *f, sm_stage);
        } else {
            *d = oneshot_ld_u16(f);
        }
    }
}

// Wait (thread 0) until flag line `line` reaches `seq` with size `bytes`.
__device__ __forceinline__ void oneshot_wait_line(
    const uint64_t* line, uint64_t seq, uint32_t bytes, uint64_t* poison, uint64_t timeout_ns) {
    // This helper serves the multi-rank gather/all-to-all only.
    line += seq & 1;
    const uint64_t t0 = oneshot_now();
    for (uint32_t spin = 0;; ++spin) {
        const uint64_t f = oneshot_ld_acquire(line);
        if ((f >> 24) > seq + 1) {
            oneshot_poison(poison, ONESHOT_POISON_DESYNC | seq);
        }
        if ((f >> 24) >= seq) {
            if ((f >> 24) == seq && (f & 0xffffff) != bytes) {
                oneshot_poison(poison, ONESHOT_POISON_MISMATCH | seq);
            }
            return;
        }
        if (spin % 64 == 63) {
            if (oneshot_ld_acquire(poison) != 0) {
                __trap();
            }
            if (timeout_ns != 0 && oneshot_now() - t0 > timeout_ns) {
                oneshot_poison(poison, ONESHOT_POISON_TIMEOUT | seq);
            }
        }
    }
}

// The chain's all-gather, pipelined (root == ranks + 1). Under SM staging the
// payload goes to S piece by piece (`rails` pieces, the proxy's `pieces`
// split): every block copies its share of piece p, fences at system scope and
// counts itself in its u32 at state + 32 bytes + 4 p; the last publishes ready[p] =
// seq, and the proxy, told by the PIPE bit in `stage` (published first),
// sends each piece as it is published. Each other origin's piece is landed as
// soon as its own flag arrives, a block per (piece, origin) item, so landing
// overlaps the transfer instead of following it. This rank's part is its
// staged payload, already in place.
template <int R>
__device__ __noinline__ void oneshot_gather(
    unsigned char* dst, const unsigned char* recv, uint64_t slot_stride, uint32_t bytes,
    const uint64_t* flags, uint32_t pieces, uint64_t* state, uint32_t* stage, uint64_t* poison,
    uint64_t timeout_ns, const unsigned char* src, unsigned char* sbuf, uint32_t sm_stage,
    uint32_t rank, int64_t gather_stride) {
    const uint64_t seq = *(volatile uint64_t*)state + 1;
    const uint32_t part = (((bytes + pieces - 1) / pieces) + 63u) & ~63u;
    auto len_of = [&](uint32_t p) { return min(part, bytes - p * part); };
    const uint32_t tid = threadIdx.x;
    if (sm_stage != 0) {
        oneshot_pipe_gate(stage, state, seq, bytes, poison, timeout_ns);
        uint64_t* ready =
            reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(stage) + ONESHOT_READY);
        uint32_t* pending = reinterpret_cast<uint32_t*>(state + 4);
        for (uint32_t p = 0; p < pieces; ++p) {
            // The block's contiguous share of piece p, 16-byte aligned.
            const uint32_t len = len_of(p);
            const uint32_t share = ((len + gridDim.x - 1) / gridDim.x + 15u) & ~15u;
            const uint32_t lo = min(len, blockIdx.x * share), hi = min(len, lo + share);
            oneshot_block_copy(sbuf + p * part + lo, src + p * part + lo, hi - lo, sm_stage, true);
            __threadfence_system();
            __syncthreads();
            if (tid == 0) {
                uint32_t prior;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                             : "=r"(prior) : "l"(pending + p) : "memory");
                if (prior + 1 == gridDim.x) {
                    pending[p] = 0;
                    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ready + p), "l"(seq)
                                 : "memory");
                }
            }
        }
    } else if (stage != nullptr && blockIdx.x == 0 && tid == 0) {
        // Staged by the stream before this kernel: the whole op at once.
        asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(stage), "r"(bytes) : "memory");
    }
    if (blockIdx.x == 0 && tid == 0 && oneshot_ld_acquire(poison) != 0) {
        __trap();
    }
    // Land: items piece-major, nearest origin first within a piece (the
    // order they arrive in).
    const unsigned char* slots = recv + (seq & 1) * slot_stride;
    const uint32_t items = (R - 1) * pieces;
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
        const uint32_t p = item / (R - 1), k = item % (R - 1);
        // k-th nearest origin: alternate below / above, skipping missing ones.
        uint32_t origin = 0, found = 0;
        for (uint32_t d = 1; d < R && found <= k; ++d) {
            if (rank >= d && found++ == k) { origin = rank - d; break; }
            if (rank + d < R && found++ == k) { origin = rank + d; break; }
        }
        const uint32_t slot = origin - (origin > rank);
        if (tid == 0) {
            oneshot_wait_line(flags + (slot * ONESHOT_ORIGIN_LINES + p) * ONESHOT_FLAG_STRIDE, seq,
                              bytes, poison, timeout_ns);
        }
        __syncthreads();
        oneshot_block_copy(dst + (int64_t)origin * gather_stride + p * part,
                           slots + slot * 2 * slot_stride + p * part, len_of(p), 0, false);
        __syncthreads();
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t* done = reinterpret_cast<uint32_t*>(state + 1);
        __threadfence();
        if (atomicAdd(done, 1u) == gridDim.x - 1) {
            *done = 0;
            *(volatile uint64_t*)state = seq;
        }
    }
}

template <int R>
__device__ __forceinline__ void oneshot_run(
    __nv_bfloat16* dst, const unsigned char* recv, uint64_t slot_stride, uint32_t bytes,
    uint32_t add, const uint64_t* flags, uint32_t rails, uint64_t* state, uint32_t* stage,
    uint64_t* poison, uint64_t timeout_ns, const unsigned char* src, unsigned char* sbuf,
    uint32_t sm_stage, uint32_t rank, uint32_t root, int64_t gather_stride) {
    if constexpr (R > 2) {
        atlas_pdl_enter_serial();
        if (root == R + 1) {
            oneshot_gather<R>(reinterpret_cast<unsigned char*>(dst), recv, slot_stride, bytes,
                              flags, rails, state, stage, poison, timeout_ns, src, sbuf, sm_stage,
                              rank, gather_stride);
            return;
        }
    } else {
        atlas_pdl_enter();
    }
    uint64_t* go = state + 2;
    const uint64_t seq = *(volatile uint64_t*)state + 1;
    const uint32_t n = bytes / 2;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    // The 16-byte path lands n8 * 8 elements by uint4, the rest one by one.
    const bool vec = (reinterpret_cast<uintptr_t>(dst) & 15) == 0;
    const uint32_t n8 = vec ? n / 8 : 0;
    if (sm_stage != 0) {
        if (threadIdx.x == 0) {
            const uint64_t t0 = oneshot_now();
            for (uint32_t spin = 1; oneshot_ld_acquire_u32(stage) != 0; ++spin) {
                if (spin % 64 == 0) {
                    if (oneshot_ld_acquire(poison) != 0) {
                        __trap();
                    }
                    if (timeout_ns != 0 && oneshot_now() - t0 > timeout_ns) {
                        oneshot_poison(poison, ONESHOT_POISON_TIMEOUT | seq);
                    }
                }
            }
        }
        __syncthreads();
        // The same element -> thread mapping as the landing below, so an
        // in-place src == dst (same alignment) is read before it is
        // overwritten. A src misaligned under an aligned dst is another
        // buffer (peer exchange), copied element by element.
        const bool vsrc = vec && (reinterpret_cast<uintptr_t>(src) & 15) == 0;
        const uint32_t first = vsrc ? n8 * 8 : 0;
        if (vsrc) {
            const uint4* s4 = reinterpret_cast<const uint4*>(src);
            uint4* o4 = reinterpret_cast<uint4*>(sbuf);
            for (uint32_t i = tid; i < n8; i += stride) {
                oneshot_st_v4(o4 + i, s4[i], sm_stage);
            }
        }
        const unsigned short* s1 = reinterpret_cast<const unsigned short*>(src);
        unsigned short* o1 = reinterpret_cast<unsigned short*>(sbuf);
        for (uint32_t i = first + tid; i < n; i += stride) {
            oneshot_st_u16(o1 + i, s1[i], sm_stage);
        }
        // Every thread's stores are performed at system scope before its
        // block counts itself; block 0 publishes once all blocks have.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t* staged = reinterpret_cast<uint32_t*>(state + 3);
            asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(staged) : "memory");
            if (blockIdx.x == 0) {
                for (;;) {
                    uint32_t c;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(c) : "l"(staged) : "memory");
                    if (c == gridDim.x) {
                        break;
                    }
                }
                // No block counts the next op before this kernel completes.
                *(volatile uint32_t*)staged = 0;
                __threadfence_system();
            }
        }
    }
    if (threadIdx.x == 0) {
        if (stage != nullptr && blockIdx.x == 0) {
            // The payload is in S: staged by the stream before this kernel
            // started, or by every block above. The proxy may read it once
            // it sees the size.
            asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(stage), "r"(bytes) : "memory");
        }
        // After the publish, which starts our send.
        if (blockIdx.x == 0 && oneshot_ld_acquire(poison) != 0) {
            __trap();
        }
        const uint64_t t0 = oneshot_now();
        for (uint32_t spin = 0;; ++spin) {
            const bool own = blockIdx.x == 0 || spin % 256 == 255;
            if (own ? oneshot_arrived<R>(flags, rails, seq, bytes, poison)
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

    const unsigned char* peer = recv + (seq & 1) * slot_stride;
    if constexpr (R == 2) {
        oneshot_land_one(dst, peer, add, n, n8, vec, tid, stride);
    } else if (root == R) {
        oneshot_land_sum<R>(dst, peer, slot_stride, rank, n, n8, tid, stride);
    } else if (root != rank) {
        oneshot_land_one(dst, peer + (root - (root > rank)) * 2 * slot_stride, 0, n, n8, vec,
                         tid, stride);
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

// ---- The four-rank chain reduce-scatter (root == ranks + 2) ----
//
// Rank d of the chain 0-1-2-3 ends with segment d of the pairwise sum
// (x0 + x1) + (x2 + x3), the same words as the all-reduce, with each link
// carrying three quarters of the payload instead of three payloads. The op's
// staged payload is the four segments [seg0 | seg1 | seg2 | seg3] (segment j
// of this rank's partial at src + j * seg_stride), in eight pieces, two a
// segment. The pieces' routes (the proxy's `relay::rs_role` is the same
// table):
//   rank 0 sends its raw seg1..3 up; rank 3 its raw seg0..2 down;
//   rank 1 sends raw seg0 down, rank 2 raw seg3 up;
//   rank 1 computes a = x0 + x1 of seg2..3 into S and sends it up, piece by
//   piece as x0 lands; rank 2 likewise b = x2 + x3 of seg0..1, down;
//   rank 1's proxy forwards b of seg0 down, rank 2's a of seg3 up.
// Then rank 0 lands (x0 + x1) + b, rank 1 (x0 + x1) + b, rank 2 a + (x2 + x3)
// and rank 3 a + (x2 + x3), each in its own segment of dst = src. Every
// receive slot keeps its origin's index on the way; a flag carries
// bytes | ONESHOT_RS_BIT, so neither a proxy nor a kernel takes a
// reduce-scatter piece for an all-reduce's. A computed piece is published to
// the proxy in the control page's ready words (`ready[p] = seq`) once every
// block that wrote it has fenced at system scope.
#define ONESHOT_RS_BIT (1u << 23)
// The op's pieces are `k` a segment (`oneshot::rs_per_segment`: 2, 4 or 8),
// piece p of segment p / k. Rank r's tasks, computes first, at segment level
// as `relay::rs_role` routes them: rank 1 computes segments 2 and 3 and lands
// 1, rank 2 computes 0 and 1 and lands 2, the ends land their own.
struct OneshotRsTask {
    uint32_t piece;
    bool compute;
};

__device__ __forceinline__ uint32_t oneshot_rs_ntasks(uint32_t r, uint32_t k) {
    return (r == 1 || r == 2) ? 3 * k : k;
}

__device__ __forceinline__ OneshotRsTask oneshot_rs_task(uint32_t r, uint32_t k, uint32_t i) {
    switch (r) {
        case 1: return i < 2 * k ? OneshotRsTask{2 * k + i, true} : OneshotRsTask{i - k, false};
        case 2: return i < 2 * k ? OneshotRsTask{i, true} : OneshotRsTask{i, false};
        case 0: return {i, false};
        default: return {3 * k + i, false};
    }
}

// Rank r's raw pieces (sent as staged) in the proxy's send order: rank 0's
// segments 2, 3 then 1 (the far side needs them first), rank 1's 0, rank
// 2's 3, rank 3's 0, 1 and 2.
__device__ __forceinline__ uint32_t oneshot_rs_nraw(uint32_t r, uint32_t k) {
    return (r == 0 || r == 3) ? 3 * k : k;
}

__device__ __forceinline__ uint32_t oneshot_rs_raw_piece(uint32_t r, uint32_t k, uint32_t i) {
    switch (r) {
        case 0: return i < 2 * k ? 2 * k + i : i - k;
        case 2: return 3 * k + i;
        default: return i;
    }
}

// Wait (thread 0) until origin slot `slot`'s flag of piece `p` reaches `seq`
// with the op's `code`.
__device__ __forceinline__ void oneshot_rs_wait(
    const uint64_t* flags, uint32_t slot, uint32_t p, uint64_t seq, uint32_t code,
    uint64_t* poison, uint64_t timeout_ns) {
    const uint64_t* line =
        flags + (slot * ONESHOT_ORIGIN_LINES + p) * ONESHOT_FLAG_STRIDE + (seq & 1);
    const uint64_t t0 = oneshot_now();
    for (uint32_t spin = 0;; ++spin) {
        const uint64_t f = oneshot_ld_acquire(line);
        if ((f >> 24) > seq + 1) {
            oneshot_poison(poison, ONESHOT_POISON_DESYNC | seq);
        }
        if ((f >> 24) >= seq) {
            if ((f >> 24) == seq && (f & 0xffffff) != code) {
                oneshot_poison(poison, ONESHOT_POISON_MISMATCH | seq);
            }
            return;
        }
        if (spin % 64 == 63) {
            if (oneshot_ld_acquire(poison) != 0) {
                __trap();
            }
            if (timeout_ns != 0 && oneshot_now() - t0 > timeout_ns) {
                oneshot_poison(poison, ONESHOT_POISON_TIMEOUT | seq);
            }
        }
    }
}

__device__ __forceinline__ uint4 oneshot_rs_ld(const unsigned char* p, bool sys) {
    return sys ? oneshot_ld_v4(reinterpret_cast<const uint4*>(p))
               : *reinterpret_cast<const uint4*>(p);
}

__device__ __noinline__ void oneshot_rs4(
    unsigned char* src, int64_t seg_stride, const unsigned char* recv, uint64_t slot_stride,
    uint32_t bytes, const uint64_t* flags, uint64_t* state, uint32_t* stage, uint64_t* poison,
    uint64_t timeout_ns, unsigned char* sbuf, uint32_t sm_stage, uint32_t rank, uint32_t pieces) {
    const uint64_t seq = *(volatile uint64_t*)state + 1;
    const uint32_t code = bytes | ONESHOT_RS_BIT;
    const uint32_t k = pieces / 4;                    // pieces a segment
    const uint32_t piece = bytes / pieces;            // a multiple of 64 bytes
    const uint32_t vecs = piece / 16;
    // This rank's piece p of segment p / k in device memory.
    auto own = [&](uint32_t p) {
        return src + (int64_t)(p / k) * seg_stride + (p % k) * piece;
    };
    const unsigned char* slots = recv + (seq & 1) * slot_stride;
    auto slot_of = [&](uint32_t origin) {
        return slots + (origin - (origin > rank)) * 2 * slot_stride;
    };
    const uint32_t tid = threadIdx.x;

    // Stage the raw pieces, after the proxy drained the previous op's sends
    // (which also frees S for the computed pieces). `stage` goes first, with
    // the PIPE bit: the proxy sends each raw piece as its ready word shows it
    // staged, as it sends the computed ones.
    uint64_t* ready = reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(stage) + ONESHOT_READY);
    uint32_t* pending = reinterpret_cast<uint32_t*>(state + 4);
    oneshot_pipe_gate(stage, state, seq, code, poison, timeout_ns);
    for (uint32_t i = 0; i < oneshot_rs_nraw(rank, k); ++i) {
        const uint32_t p = oneshot_rs_raw_piece(rank, k, i);
        const uint4* from = reinterpret_cast<const uint4*>(own(p));
        uint4* to = reinterpret_cast<uint4*>(sbuf + p * piece);
        for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vecs; i += gridDim.x * blockDim.x) {
            oneshot_st_v4(to + i, from[i], sm_stage);
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            uint32_t prior;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                         : "=r"(prior) : "l"(pending + p) : "memory");
            if (prior + 1 == gridDim.x) {
                pending[p] = 0;
                asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ready + p), "l"(seq)
                             : "memory");
            }
        }
    }

    // The tasks, each cut into `split` element ranges, one per item.
    const uint32_t ntasks = oneshot_rs_ntasks(rank, k);
    const uint32_t split = gridDim.x / ntasks > 0 ? gridDim.x / ntasks : 1;
    const uint32_t per = (vecs + split - 1) / split;
    for (uint32_t item = blockIdx.x; item < ntasks * split; item += gridDim.x) {
        const OneshotRsTask task = oneshot_rs_task(rank, k, item / split);
        const uint32_t p = task.piece;
        const uint32_t lo = (item % split) * per;
        const uint32_t hi = min(vecs, lo + per);
        // Inputs: the pieces this task combines with this rank's own.
        //   rank 0: x1 (origin 1), b (origin 2)   (x0 + x1) + b
        //   rank 1: x0 (origin 0); final also b (origin 2)
        //   rank 2: x3 (origin 3); final also a (origin 1)
        //   rank 3: x2 (origin 2), a (origin 1)   a + (x2 + x3)
        const uint32_t near = rank == 0 ? 1 : rank == 1 ? 0 : rank == 2 ? 3 : 2;
        const uint32_t far = rank < 2 ? 2 : 1;
        if (tid == 0) {
            oneshot_rs_wait(flags, near - (near > rank), p, seq, code, poison, timeout_ns);
            if (!task.compute) {
                oneshot_rs_wait(flags, far - (far > rank), p, seq, code, poison, timeout_ns);
            }
        }
        __syncthreads();
        const unsigned char* x = own(p);
        const unsigned char* n = slot_of(near) + p * piece;
        const unsigned char* f = slot_of(far) + p * piece;
        for (uint32_t i = lo + tid; i < hi; i += blockDim.x) {
            const uint4 mine = oneshot_rs_ld(x + 16 * i, false);
            const uint4 other = oneshot_rs_ld(n + 16 * i, true);
            // The lower rank's value goes first, as in the tree.
            const uint4 pair = rank < 2 ? oneshot_add(rank == 0 ? mine : other, rank == 0 ? other : mine)
                                        : oneshot_add(rank == 2 ? mine : other, rank == 2 ? other : mine);
            if (task.compute) {
                oneshot_st_v4(reinterpret_cast<uint4*>(sbuf + p * piece) + i, pair, sm_stage);
            } else {
                const uint4 rest = oneshot_rs_ld(f + 16 * i, true);
                // (x0 + x1) + (x2 + x3): the low pair first.
                const uint4 sum = rank < 2 ? oneshot_add(pair, rest) : oneshot_add(rest, pair);
                *reinterpret_cast<uint4*>(own(p) +
                                          16 * i) = sum;
            }
        }
        if (task.compute) {
            __threadfence_system();
            __syncthreads();
            if (tid == 0) {
                uint32_t prior;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                             : "=r"(prior) : "l"(pending + p) : "memory");
                if (prior + 1 == split) {
                    pending[p] = 0;
                    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ready + p), "l"(seq)
                                 : "memory");
                }
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {
        uint32_t* done = reinterpret_cast<uint32_t*>(state + 1);
        __threadfence();
        if (atomicAdd(done, 1u) == gridDim.x - 1) {
            *done = 0;
            *(volatile uint64_t*)state = seq;
        }
    }
}


// ---- The four-rank reduce-scatter in FP8 (root == ranks + 2, `add` set) ----
//
// oneshot_rs4's routes, pieces and pairing, with every piece on the wire as
// E4M3 values and one f32 scale (amax / 448) per 128 values: a BF16 piece of
// P bytes (P % 1024 == 0) travels as its P / 2 E4M3 bytes, then its P / 256
// scales, `oneshot_fp8_wire(P)` = 33 P / 64 bytes at 33 / 64 of its BF16
// offset (`oneshot::rs_wire_pieces`). A hop dequantizes what lands, adds its
// own partial in f32 and quantizes only what it sends on; the owner rounds the
// f32 sum (x0 + x1) + (x2 + x3) to BF16 once. Lossy, so opt-in
// (`CommBackend::reduce_scatter_fp8_async`); the op's size code carries
// ONESHOT_RS_FP8_BIT beside ONESHOT_RS_BIT. A half-warp handles a 128-value
// group, 8 values a lane.
#define ONESHOT_RS_FP8_BIT 2u

__device__ __forceinline__ uint32_t oneshot_fp8_wire(uint32_t p) { return p / 2 + p / 64; }

struct OneshotF8 {
    float v[8];
};

__device__ __forceinline__ OneshotF8 oneshot_f8_bf16(uint4 u) {
    OneshotF8 r;
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        r.v[2 * j] = f.x;
        r.v[2 * j + 1] = f.y;
    }
    return r;
}

__device__ __forceinline__ uint4 oneshot_f8_to_bf16(const OneshotF8& x) {
    uint4 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        h[j] = __floats2bfloat162_rn(x.v[2 * j], x.v[2 * j + 1]);
    }
    return u;
}

__device__ __forceinline__ OneshotF8 oneshot_f8_add(const OneshotF8& a, const OneshotF8& b) {
    OneshotF8 r;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        r.v[j] = a.v[j] + b.v[j];
    }
    return r;
}

// Group `g` of a landed wire piece at `w` (`nvals` value bytes), this lane's
// 8 values, dequantized.
__device__ __forceinline__ OneshotF8 oneshot_f8_load(
    const unsigned char* w, uint32_t nvals, uint32_t g, uint32_t lane) {
    uint32_t q0, q1;
    float s;
    asm volatile("ld.relaxed.sys.global.v2.u32 {%0, %1}, [%2];"
                 : "=r"(q0), "=r"(q1) : "l"(w + g * 128 + lane * 8));
    asm volatile("ld.relaxed.sys.global.f32 %0, [%1];" : "=f"(s) : "l"(w + nvals + g * 4));
    OneshotF8 r;
    const uint32_t q[2] = {q0, q1};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned short pair = (unsigned short)(q[j / 2] >> (16 * (j % 2)));
        uint32_t h2;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
        r.v[2 * j] = f.x * s;
        r.v[2 * j + 1] = f.y * s;
    }
    return r;
}

// Quantize this lane's 8 values of group `g` (the half-warp's 128) into the
// wire piece at `w`. Every lane of the warp calls it (the amax is a
// half-warp shuffle); only lanes with `in` store.
__device__ __forceinline__ void oneshot_f8_store(
    unsigned char* w, uint32_t nvals, uint32_t g, uint32_t lane, const OneshotF8& x, bool in,
    uint32_t mode) {
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        amax = fmaxf(amax, fabsf(x.v[j]));
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float scale = amax > 0.f ? amax / 448.f : 1.f;
    const float inv = amax > 0.f ? 448.f / amax : 1.f;
    uint32_t q[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        unsigned short lo, hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(lo) : "f"(x.v[4 * j + 1] * inv), "f"(x.v[4 * j] * inv));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(hi) : "f"(x.v[4 * j + 3] * inv), "f"(x.v[4 * j + 2] * inv));
        q[j] = (uint32_t)lo | (uint32_t)hi << 16;
    }
    if (!in) {
        return;
    }
    unsigned char* vp = w + g * 128 + lane * 8;
    unsigned char* sp = w + nvals + g * 4;
    if (mode == 1) {
        asm volatile("st.global.wt.v2.u32 [%0], {%1, %2};" ::"l"(vp), "r"(q[0]), "r"(q[1])
                     : "memory");
        if (lane == 0) {
            asm volatile("st.global.wt.f32 [%0], %1;" ::"l"(sp), "f"(scale) : "memory");
        }
    } else {
        *reinterpret_cast<uint2*>(vp) = make_uint2(q[0], q[1]);
        if (lane == 0) {
            *reinterpret_cast<float*>(sp) = scale;
        }
    }
}

__device__ __noinline__ void oneshot_rs4_fp8(
    unsigned char* src, int64_t seg_stride, const unsigned char* recv, uint64_t slot_stride,
    uint32_t bytes, const uint64_t* flags, uint64_t* state, uint32_t* stage, uint64_t* poison,
    uint64_t timeout_ns, unsigned char* sbuf, uint32_t sm_stage, uint32_t rank, uint32_t pieces) {
    const uint64_t seq = *(volatile uint64_t*)state + 1;
    const uint32_t code = bytes | ONESHOT_RS_BIT | ONESHOT_RS_FP8_BIT;
    const uint32_t k = pieces / 4;            // pieces a segment
    const uint32_t piece = bytes / pieces;    // BF16 bytes, a multiple of 1024
    const uint32_t wpiece = oneshot_fp8_wire(piece);
    const uint32_t nvals = piece / 2;         // E4M3 bytes of a wire piece
    const uint32_t groups = piece / 256;      // 128-value groups a piece
    auto own = [&](uint32_t p) {
        return src + (int64_t)(p / k) * seg_stride + (p % k) * piece;
    };
    const unsigned char* slots = recv + (seq & 1) * slot_stride;
    auto slot_of = [&](uint32_t origin) {
        return slots + (origin - (origin > rank)) * 2 * slot_stride;
    };
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % 16;
    const uint32_t half = tid / 16;
    const uint32_t halves = blockDim.x / 16;

    uint64_t* ready = reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(stage) + ONESHOT_READY);
    uint32_t* pending = reinterpret_cast<uint32_t*>(state + 4);
    auto publish = [&](uint32_t p, uint32_t writers) {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            uint32_t prior;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                         : "=r"(prior) : "l"(pending + p) : "memory");
            if (prior + 1 == writers) {
                pending[p] = 0;
                asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ready + p), "l"(seq)
                             : "memory");
            }
        }
    };
    oneshot_pipe_gate(stage, state, seq, code, poison, timeout_ns);
    // The raw pieces, quantized into S as oneshot_rs4 stages them.
    for (uint32_t i = 0; i < oneshot_rs_nraw(rank, k); ++i) {
        const uint32_t p = oneshot_rs_raw_piece(rank, k, i);
        const unsigned char* from = own(p);
        unsigned char* to = sbuf + p * wpiece;
        for (uint32_t g0 = blockIdx.x * halves; g0 < groups; g0 += gridDim.x * halves) {
            const uint32_t g = g0 + half;
            const bool in = g < groups;
            OneshotF8 x = {};
            if (in) {
                x = oneshot_f8_bf16(*reinterpret_cast<const uint4*>(from + g * 256 + lane * 16));
            }
            oneshot_f8_store(to, nvals, g, lane, x, in, sm_stage);
        }
        publish(p, gridDim.x);
    }

    // The tasks, each cut into `split` group ranges, one per item.
    const uint32_t ntasks = oneshot_rs_ntasks(rank, k);
    const uint32_t split = gridDim.x / ntasks > 0 ? gridDim.x / ntasks : 1;
    const uint32_t per = (groups + split - 1) / split;
    for (uint32_t item = blockIdx.x; item < ntasks * split; item += gridDim.x) {
        const OneshotRsTask task = oneshot_rs_task(rank, k, item / split);
        const uint32_t p = task.piece;
        const uint32_t lo = (item % split) * per;
        const uint32_t hi = min(groups, lo + per);
        const uint32_t near = rank == 0 ? 1 : rank == 1 ? 0 : rank == 2 ? 3 : 2;
        const uint32_t far = rank < 2 ? 2 : 1;
        if (tid == 0) {
            oneshot_rs_wait(flags, near - (near > rank), p, seq, code, poison, timeout_ns);
            if (!task.compute) {
                oneshot_rs_wait(flags, far - (far > rank), p, seq, code, poison, timeout_ns);
            }
        }
        __syncthreads();
        unsigned char* x = own(p);
        const unsigned char* n = slot_of(near) + p * wpiece;
        const unsigned char* f = slot_of(far) + p * wpiece;
        for (uint32_t g0 = lo; g0 < hi; g0 += halves) {
            const uint32_t g = g0 + half;
            const bool in = g < hi;
            OneshotF8 pair = {};
            OneshotF8 rest = {};
            if (in) {
                // f32 adds of two values are the same either way round.
                pair = oneshot_f8_add(oneshot_f8_bf16(*reinterpret_cast<const uint4*>(x + g * 256 + lane * 16)),
                                      oneshot_f8_load(n, nvals, g, lane));
                if (!task.compute) {
                    rest = oneshot_f8_load(f, nvals, g, lane);
                }
            }
            if (task.compute) {
                oneshot_f8_store(sbuf + p * wpiece, nvals, g, lane, pair, in, sm_stage);
            } else if (in) {
                // (x0 + x1) + (x2 + x3), rounded to BF16 once.
                *reinterpret_cast<uint4*>(x + g * 256 + lane * 16) =
                    oneshot_f8_to_bf16(oneshot_f8_add(pair, rest));
            }
        }
        if (task.compute) {
            publish(p, split);
        }
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {
        uint32_t* done = reinterpret_cast<uint32_t*>(state + 1);
        __threadfence();
        if (atomicAdd(done, 1u) == gridDim.x - 1) {
            *done = 0;
            *(volatile uint64_t*)state = seq;
        }
    }
}

// ---- The fused four-rank reduce-scatter (root == ranks + 4) ----
//
// oneshot_rs4's routes and sums over the chain's fused region
// (`oneshot::fused_*`), whose producer kernel (`rdma_rs_fused.cuh`) already
// gated the op, staged this rank's raw pieces and published each one: here
// only the computed and final tasks run, each as soon as the producer
// published the piece (`ONESHOT_STATE_OWN`) and its inputs landed. A
// rank's region holds its raw segments' staging, then one segment per
// (origin, segment) it receives (`oneshot_fused_recv`); a middle rank's
// computed pair lands in place over the near payload it sums, and goes on
// from there. Every final piece j of this rank's segment, once written,
// sets done[j] = epoch for the consumer's stream waits. One op of the whole
// payload: the size code counts 512-byte units (ONESHOT_RS_FUSED_BIT).
#define ONESHOT_RS_FUSED_BIT 4u

struct OneshotRsFused {
    uint32_t* done;  // u32 per piece of this rank's segment
    uint32_t epoch;
    uint32_t pad;
};

__device__ __forceinline__ uint32_t oneshot_fused_code(uint32_t bytes) {
    return ONESHOT_RS_BIT | ONESHOT_RS_FUSED_BIT | (bytes / 512) << 3;
}

// Raw segments staged at rank r (oneshot::fused_raw).
__device__ __forceinline__ uint32_t oneshot_fused_nraw(uint32_t r) {
    return (r == 0 || r == 3) ? 3 : 1;
}

// Entry of origin `o`'s segment `g` among rank r's received ones, after its
// raw staging (oneshot::fused_recv_index).
__device__ __forceinline__ uint32_t oneshot_fused_recv(uint32_t r, uint32_t o, uint32_t g) {
    switch (r) {
        case 0: return o == 1 ? 0 : 1;                 // x1 seg 0, b seg 0
        case 1: return o == 0 ? g - 1 : 3 + g;         // x0 segs 1..3, b segs 0..1
        case 2: return o == 3 ? g : 3 + (g - 2);       // x3 segs 0..2, a segs 2..3
        default: return o == 2 ? 0 : 1;                // x2 seg 3, a seg 3
    }
}

__device__ __noinline__ void oneshot_rs4_fused(
    unsigned char* src, int64_t seg_stride, unsigned char* region, uint64_t seg_cap,
    uint32_t bytes, const uint64_t* flags, uint64_t* state, uint32_t* stage, uint64_t* poison,
    uint64_t timeout_ns, uint32_t sm_stage, uint32_t rank, uint32_t pieces,
    const OneshotRsFused fused) {
    const uint64_t seq = *(volatile uint64_t*)state + 1;
    const uint32_t code = oneshot_fused_code(bytes);
    const uint32_t k = pieces / 4;
    const uint32_t piece = bytes / pieces;
    const uint32_t vecs = piece / 16;
    auto own = [&](uint32_t p) {
        return src + (int64_t)(p / k) * seg_stride + (p % k) * piece;
    };
    // Origin `o`'s landed piece p (segment p / k) in this rank's region.
    auto landed = [&](uint32_t o, uint32_t p) {
        const uint32_t e = oneshot_fused_nraw(rank) + oneshot_fused_recv(rank, o, p / k);
        return region + e * seg_cap + (p % k) * piece;
    };
    const uint32_t tid = threadIdx.x;
    uint64_t* ready = reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(stage) + ONESHOT_READY);
    uint32_t* pending = reinterpret_cast<uint32_t*>(state + 4);
    const uint64_t* produced =
        reinterpret_cast<const uint64_t*>(reinterpret_cast<unsigned char*>(state) + ONESHOT_STATE_OWN);

    const uint32_t ntasks = oneshot_rs_ntasks(rank, k);
    const uint32_t split = gridDim.x / ntasks > 0 ? gridDim.x / ntasks : 1;
    const uint32_t per = (vecs + split - 1) / split;
    for (uint32_t item = blockIdx.x; item < ntasks * split; item += gridDim.x) {
        const OneshotRsTask task = oneshot_rs_task(rank, k, item / split);
        const uint32_t p = task.piece;
        const uint32_t lo = (item % split) * per;
        const uint32_t hi = min(vecs, lo + per);
        const uint32_t near = rank == 0 ? 1 : rank == 1 ? 0 : rank == 2 ? 3 : 2;
        const uint32_t far = rank < 2 ? 2 : 1;
        if (tid == 0) {
            const uint64_t t0 = oneshot_now();
            for (uint32_t spin = 1; oneshot_ld_acquire_gpu(produced + p) < seq; ++spin) {
                if (spin % 64 == 0) {
                    if (oneshot_ld_acquire(poison) != 0) {
                        __trap();
                    }
                    if (timeout_ns != 0 && oneshot_now() - t0 > timeout_ns) {
                        oneshot_poison(poison, ONESHOT_POISON_TIMEOUT | seq);
                    }
                }
            }
            oneshot_rs_wait(flags, near - (near > rank), p, seq, code, poison, timeout_ns);
            if (!task.compute) {
                oneshot_rs_wait(flags, far - (far > rank), p, seq, code, poison, timeout_ns);
            }
        }
        __syncthreads();
        const unsigned char* x = own(p);
        unsigned char* n = landed(near, p);
        const unsigned char* f = task.compute ? nullptr : landed(far, p);
        for (uint32_t i = lo + tid; i < hi; i += blockDim.x) {
            const uint4 mine = oneshot_rs_ld(x + 16 * i, false);
            const uint4 other = oneshot_rs_ld(n + 16 * i, true);
            // The lower rank's value goes first, as in the tree.
            const uint4 pair = rank < 2 ? oneshot_add(rank == 0 ? mine : other, rank == 0 ? other : mine)
                                        : oneshot_add(rank == 2 ? mine : other, rank == 2 ? other : mine);
            if (task.compute) {
                // In place over the near payload, which goes on from there.
                oneshot_st_v4(reinterpret_cast<uint4*>(n) + i, pair, sm_stage);
            } else {
                const uint4 rest = oneshot_rs_ld(f + 16 * i, true);
                const uint4 sum = rank < 2 ? oneshot_add(pair, rest) : oneshot_add(rest, pair);
                *reinterpret_cast<uint4*>(own(p) + 16 * i) = sum;
            }
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            uint32_t prior;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                         : "=r"(prior) : "l"(pending + p) : "memory");
            if (prior + 1 == split) {
                pending[p] = 0;
                if (task.compute) {
                    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ready + p), "l"(seq)
                                 : "memory");
                } else {
                    asm volatile("st.release.sys.global.u32 [%0], %1;"
                                 ::"l"(fused.done + p % k), "r"(fused.epoch) : "memory");
                }
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {
        uint32_t* done = reinterpret_cast<uint32_t*>(state + 1);
        __threadfence();
        if (atomicAdd(done, 1u) == gridDim.x - 1) {
            *done = 0;
            *(volatile uint64_t*)state = seq;
        }
    }
}

// ---- The four-rank chain all-to-all (root == ranks + 3) ----
//
// Each rank sends a distinct segment to every other rank: segment d of the
// op (the rank's part for rank d, both planes, `bytes / 4`) travels toward
// rank d, the middle ranks forwarding the segments for the ranks beyond them
// (`relay` OpKind::AllToAll). Pieces are the reduce-scatter's (`k` a
// segment), staged and published one by one like its raw pieces; rank d
// lands segment d of every origin's slot as each piece's flag arrives, and
// copies its own part from its source. A segment is plane 0's part, then
// plane 1's; a part is whole rows, which land `pitch` apart at the plane's
// `dst + origin * dst_stride`.
struct OneshotA2aPlane {
    const unsigned char* src;
    int64_t src_stride;
    unsigned char* dst;
    int64_t dst_stride;
    uint32_t row_bytes;
    uint32_t pitch;
    uint32_t bytes;
    uint32_t pad;
};
struct OneshotA2a {
    OneshotA2aPlane plane[2];
};

// Copy segment bytes [x0, x0 + len) (16-byte units) between a contiguous
// buffer `flat` (segment offset 0 at flat) and the planes' rows of rank
// `who` (its source part for rank `who`, or its landing for origin `who`).
// `to_rows`: flat -> rows (landing), else rows -> flat (staging).
__device__ __forceinline__ void oneshot_a2a_copy(
    const OneshotA2a& a, uint32_t who, unsigned char* flat, const unsigned char* flat_src,
    uint32_t x0, uint32_t len, bool to_rows, uint32_t sm_stage, bool sys_src) {
    const uint32_t a0 = a.plane[0].bytes;
    for (uint32_t x = x0 + 16 * threadIdx.x; x < x0 + len; x += 16 * blockDim.x) {
        const bool second = x >= a0;
        const OneshotA2aPlane& p = a.plane[second ? 1 : 0];
        const uint32_t y = second ? x - a0 : x;
        if (to_rows) {
            const uint4 v = sys_src ? oneshot_ld_v4(reinterpret_cast<const uint4*>(flat_src + x))
                                    : *reinterpret_cast<const uint4*>(flat_src + x);
            unsigned char* d = p.dst + (int64_t)who * p.dst_stride +
                               (uint64_t)(y / p.row_bytes) * p.pitch + y % p.row_bytes;
            *reinterpret_cast<uint4*>(d) = v;
        } else {
            const uint4 v = *reinterpret_cast<const uint4*>(p.src + (int64_t)who * p.src_stride + y);
            oneshot_st_v4(reinterpret_cast<uint4*>(flat + x), v, sm_stage);
        }
    }
}

__device__ __noinline__ void oneshot_a2a4(
    const OneshotA2a a, const unsigned char* recv, uint64_t slot_stride, uint32_t bytes,
    const uint64_t* flags, uint64_t* state, uint32_t* stage, uint64_t* poison,
    uint64_t timeout_ns, unsigned char* sbuf, uint32_t sm_stage, uint32_t rank, uint32_t pieces) {
    const uint64_t seq = *(volatile uint64_t*)state + 1;
    const uint32_t code = bytes | ONESHOT_RS_BIT | 1u;
    const uint32_t k = pieces / 4;
    const uint32_t seg = bytes / 4, piece = seg / k;  // whole 64-byte units
    const uint32_t tid = threadIdx.x;
    uint64_t* ready = reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(stage) + ONESHOT_READY);
    uint32_t* pending = reinterpret_cast<uint32_t*>(state + 4);
    oneshot_pipe_gate(stage, state, seq, code, poison, timeout_ns);
    // Stage the other ranks' segments, the farthest first (they have the
    // most hops ahead), each piece published once every block copied its
    // share.
    for (uint32_t dist = 3; dist >= 1; --dist) {
        for (uint32_t side = 0; side < 2; ++side) {
            if (side == 0 ? rank < dist : rank + dist >= 4) continue;
            const uint32_t d = side == 0 ? rank - dist : rank + dist;
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t p = d * k + j;
                const uint32_t share = ((piece + gridDim.x - 1) / gridDim.x + 15u) & ~15u;
                const uint32_t lo = min(piece, blockIdx.x * share), hi = min(piece, lo + share);
                oneshot_a2a_copy(a, d, sbuf + d * seg, nullptr, j * piece + lo, hi - lo, false,
                                 sm_stage, false);
                __threadfence_system();
                __syncthreads();
                if (tid == 0) {
                    uint32_t prior;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                                 : "=r"(prior) : "l"(pending + p) : "memory");
                    if (prior + 1 == gridDim.x) {
                        pending[p] = 0;
                        asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ready + p),
                                     "l"(seq) : "memory");
                    }
                }
            }
        }
    }
    // Land: this rank's own part (from its source), then every origin's
    // segment `rank`, nearest origin first within a piece.
    const unsigned char* slots = recv + (seq & 1) * slot_stride;
    const uint32_t items = 4 * k;
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
        const uint32_t j = item / 4, n = item % 4;
        const uint32_t x0 = j * piece;
        if (n == 0) {
            // Own part: source rows for rank `rank` straight to its landing.
            const OneshotA2aPlane* pl = a.plane;
            const uint32_t a0 = pl[0].bytes;
            for (uint32_t x = x0 + 16 * tid; x < x0 + piece; x += 16 * blockDim.x) {
                const bool second = x >= a0;
                const OneshotA2aPlane& p = pl[second ? 1 : 0];
                const uint32_t y = second ? x - a0 : x;
                const uint4 v = *reinterpret_cast<const uint4*>(p.src + (int64_t)rank * p.src_stride + y);
                *reinterpret_cast<uint4*>(p.dst + (int64_t)rank * p.dst_stride +
                                          (uint64_t)(y / p.row_bytes) * p.pitch +
                                          y % p.row_bytes) = v;
            }
            continue;
        }
        // The n-th nearest origin.
        uint32_t origin = 0, found = 0;
        for (uint32_t dd = 1; dd < 4 && found < n; ++dd) {
            if (rank >= dd && ++found == n) { origin = rank - dd; break; }
            if (rank + dd < 4 && ++found == n) { origin = rank + dd; break; }
        }
        const uint32_t slot = origin - (origin > rank);
        const uint32_t p = rank * k + j;
        if (tid == 0) {
            oneshot_wait_line(flags + (slot * ONESHOT_ORIGIN_LINES + p) * ONESHOT_FLAG_STRIDE, seq,
                              code, poison, timeout_ns);
        }
        __syncthreads();
        oneshot_a2a_copy(a, origin, nullptr, slots + slot * 2 * slot_stride + rank * seg, x0, piece,
                         true, 0, true);
        __syncthreads();
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t* done = reinterpret_cast<uint32_t*>(state + 1);
        __threadfence();
        if (atomicAdd(done, 1u) == gridDim.x - 1) {
            *done = 0;
            *(volatile uint64_t*)state = seq;
        }
    }
}

extern "C" __global__ void rdma_oneshot_bf16(
    __nv_bfloat16* dst,         // may alias src (in place)
    const unsigned char* recv,  // receive slot 0; slot 1 is slot_stride bytes on
    uint64_t slot_stride,
    uint32_t bytes,
    uint32_t add,
    const uint64_t* flags,      // one per rail, ONESHOT_FLAG_STRIDE words apart
    uint32_t rails,
    uint64_t* state,            // device: [0] seq, [1] finished blocks (u32), [2] go,
                                // [3] staged blocks (u32)
    uint32_t* stage,            // host `stage` word to publish, or null
    uint64_t* poison,
    uint64_t timeout_ns,         // 0 = wait forever
    const unsigned char* src,   // sm_stage: this rank's payload (device)
    unsigned char* sbuf,        // sm_stage: the staging buffer S (pinned)
    uint32_t sm_stage           // 0: S already staged; 1: stage, write-through; 2: plain
) {
    oneshot_run<2>(dst, recv, slot_stride, bytes, add, flags, rails, state, stage, poison,
                   timeout_ns, src, sbuf, sm_stage, 0, 2, 0);
}

// rdma_oneshot_bf16's arguments, then the chain's shape. Receive slots and
// flags are per origin slot: slot i's pair starts 2 * i * slot_stride after
// `recv`, its flags ONESHOT_ORIGIN_LINES lines after slot i - 1's; `rails` is
// the pieces per origin (a reduce-scatter's: 4 x its pieces a segment).
// Three 256-thread blocks an SM (at most 85 registers): two channels of four
// ranks sharing one GPU (the NIC loopback test) need 8 x 16 blocks resident
// at once, and every block of an op waits on blocks of the others (so that
// test runs the reduce-scatter's grid at ATLAS_RDMA_RS_BLOCKS=16, not 48).
extern "C" __global__ void __launch_bounds__(256, 3) rdma_chain_bf16(
    __nv_bfloat16* dst, const unsigned char* recv, uint64_t slot_stride, uint32_t bytes,
    uint32_t add, const uint64_t* flags, uint32_t rails, uint64_t* state, uint32_t* stage,
    uint64_t* poison, uint64_t timeout_ns, const unsigned char* src, unsigned char* sbuf,
    uint32_t sm_stage,
    uint32_t ranks,             // 3..=ONESHOT_MAX_RANKS
    uint32_t rank,
    uint32_t root,              // < ranks: land only rank `root`'s payload (a broadcast);
                                // ranks: the sum; ranks + 1: an all-gather; ranks + 2
                                // (four ranks): the reduce-scatter (oneshot_rs4, or
                                // with `add` set oneshot_rs4_fp8); ranks + 4 the
                                // fused one (oneshot_rs4_fused)
    int64_t gather_stride,      // all-gather: bytes from one rank's part to the next's
    const OneshotA2a a2a,       // ranks + 3 (four ranks): the all-to-all (oneshot_a2a4)
    const OneshotRsFused fused  // ranks + 4 (four ranks): the fused reduce-scatter
) {
    if (ranks == ONESHOT_MAX_RANKS && root == ranks + 4) {
        atlas_pdl_enter_serial();
        // `recv` is the fused region, `slot_stride` its segment capacity.
        oneshot_rs4_fused(reinterpret_cast<unsigned char*>(dst), gather_stride,
                          const_cast<unsigned char*>(recv), slot_stride, bytes, flags, state, stage,
                          poison, timeout_ns, sm_stage != 0 ? sm_stage : 1, rank, rails, fused);
        return;
    }
    if (ranks == ONESHOT_MAX_RANKS && root == ranks + 3) {
        atlas_pdl_enter_serial();
        oneshot_a2a4(a2a, recv, slot_stride, bytes, flags, state, stage, poison, timeout_ns, sbuf,
                     sm_stage != 0 ? sm_stage : 1, rank, rails);
        return;
    }
    if (ranks == ONESHOT_MAX_RANKS && root == ranks + 2) {
        atlas_pdl_enter_serial();
        if (add) {
            oneshot_rs4_fp8(reinterpret_cast<unsigned char*>(dst), gather_stride, recv,
                            slot_stride, bytes, flags, state, stage, poison, timeout_ns, sbuf,
                            sm_stage != 0 ? sm_stage : 1, rank, rails);
        } else {
            oneshot_rs4(reinterpret_cast<unsigned char*>(dst), gather_stride, recv, slot_stride,
                        bytes, flags, state, stage, poison, timeout_ns, sbuf,
                        sm_stage != 0 ? sm_stage : 1, rank, rails);
        }
        return;
    }
    if (ranks == 3) {
        oneshot_run<3>(dst, recv, slot_stride, bytes, add, flags, rails, state, stage, poison,
                       timeout_ns, src, sbuf, sm_stage, rank, root, gather_stride);
    } else {
        oneshot_run<ONESHOT_MAX_RANKS>(dst, recv, slot_stride, bytes, add, flags, rails, state,
                                       stage, poison, timeout_ns, src, sbuf, sm_stage, rank, root,
                                       gather_stride);
    }
}

// A fused reduce-scatter's producer for a payload already in place
// (`rdma_rs_fused.cuh`): each raw piece copied from its rows into its
// staging, each kept one published as it is. The NIC loopback test and the
// bench drive the fused op with it; a model's producer computes the rows
// instead.
extern "C" __global__ void __launch_bounds__(256) rdma_rs_fused_copy(const RsFusedPlan plan) {
    rs_fused_begin_dev(plan);
    const uint32_t vecs = plan.piece_bytes / 16;
    for (uint32_t i = 0; i < plan.npieces; ++i) {
        if (rs_fused_piece_raw(plan, i)) {
            const uint4* from = reinterpret_cast<const uint4*>(plan.piece[i].out);
            uint4* to = reinterpret_cast<uint4*>(rs_fused_piece_dst(plan, i));
            for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < vecs;
                 v += gridDim.x * blockDim.x) {
                oneshot_st_v4(to + v, from[v], plan.mode);
            }
        }
        rs_fused_piece_done(plan, i, gridDim.x);
    }
}
