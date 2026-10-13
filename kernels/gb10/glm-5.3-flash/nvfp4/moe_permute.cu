// SPDX-License-Identifier: AGPL-3.0-only

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include <assert.h>   // device-side assert() for the work-list packing guard

// Atlas MoE token permutation kernels.
//
// Reorders tokens by expert assignment for batched GEMM,
// and scatters results back with weighted accumulation.
//
// Memory layout:
//   hidden_states: [num_tokens, hidden_size] BF16
//   topk_ids:      [num_tokens, topk] int32
//   topk_weights:  [num_tokens, topk] float32
//   permuted_out:  [num_tokens * topk, hidden_size] BF16
//   expert_offsets: [num_experts + 1] int32 (prefix sum of tokens per expert)

// Permute tokens: gather from hidden_states into expert-sorted order.
// Each thread handles one element of one token.
extern "C" __global__ void moe_permute_tokens(
    const __nv_bfloat16* __restrict__ hidden_states,  // [num_tokens, hidden_size]
    __nv_bfloat16* __restrict__ permuted,              // [total_expanded, hidden_size]
    const int* __restrict__ sorted_token_ids,          // [total_expanded] maps permuted row -> original token
    unsigned int hidden_size,
    unsigned int total_expanded
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (row >= total_expanded) return;

    int src_token = sorted_token_ids[row];
    // Copy hidden_size elements — each thread handles one element,
    // loop if hidden_size > blockDim.x
    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        permuted[row * hidden_size + c] = hidden_states[src_token * hidden_size + c];
    }
}

// Unpermute and weighted reduce: scatter expert outputs back to original token order
// and accumulate with topk weights.
//
// For each original token, sums over its topk expert outputs:
//   output[token, :] = sum_k( topk_weight[token, k] * expert_output[permuted_idx[token,k], :] )
extern "C" __global__ void moe_unpermute_reduce(
    const __nv_bfloat16* __restrict__ expert_output,   // [total_expanded, hidden_size]
    __nv_bfloat16* __restrict__ output,                 // [num_tokens, hidden_size]
    const int* __restrict__ sorted_token_ids,           // [total_expanded]
    const float* __restrict__ topk_weights,             // [num_tokens, topk]
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int total_expanded
) {
    // Each block handles one original token
    unsigned int token = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (token >= num_tokens) return;

    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            unsigned int perm_row = token * topk + k;
            if (perm_row < total_expanded) {
                float w = topk_weights[token * topk + k];
                float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
                acc += w * val;
            }
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// Count tokens per expert (for building sorted indices).
// Uses atomicAdd to count how many tokens are assigned to each expert.
extern "C" __global__ void moe_count_experts(
    const int* __restrict__ topk_ids,   // [num_tokens, topk]
    int* __restrict__ expert_counts,     // [num_experts] — zeroed before launch
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_tokens * topk;
    if (idx < total) {
        int expert_id = topk_ids[idx];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// Unpermute with pre-built reverse map: token_to_perm[token, k] = row in expert_output.
// Each block handles one original token, accumulates topk expert outputs with weights.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_unpermute_reduce_indexed(
    const __nv_bfloat16* __restrict__ expert_output,  // [total_expanded, hidden_size]
    __nv_bfloat16* __restrict__ output,                // [num_tokens, hidden_size]
    const int* __restrict__ token_to_perm,             // [num_tokens, topk] reverse map
    const float* __restrict__ topk_weights,            // [num_tokens, topk]
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    for (unsigned int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            int perm_row = token_to_perm[token * topk + k];
            float w = topk_weights[token * topk + k];
            float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
            acc += w * val;
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// EP-aware indexed reduce. Each rank accumulates only the expert IDs it owns;
// the following cross-rank all-reduce reconstructs the complete routed output.
// Skipping remote IDs makes initialization of their unwritten rows unnecessary.
extern "C" __global__ void moe_unpermute_reduce_indexed_ep(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int local_expert_start,
    unsigned int local_expert_end
) {
    atlas_pdl_enter();
    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    for (unsigned int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            unsigned int slot = token * topk + k;
            int expert = topk_ids[slot];
            if (expert >= (int)local_expert_start && expert < (int)local_expert_end) {
                int perm_row = token_to_perm[slot];
                float value = __bfloat162float(expert_output[perm_row * hidden_size + c]);
                acc += topk_weights[slot] * value;
            }
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// Batched blend: for each token, compute sigmoid(dot(normed, gate_weight)) and
// blend shared expert output into routed output.
//
// output[token] += sigmoid(gate_scalar) * shared_out[token]
// where gate_scalar = dot(normed[token], gate_weight)
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_batched_blend(
    __nv_bfloat16* __restrict__ output,          // [num_tokens, hidden_size] (in/out)
    const __nv_bfloat16* __restrict__ shared_out, // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ normed,     // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate_weight, // [hidden_size] shared expert gate
    unsigned int hidden_size,
    unsigned int num_tokens
) {
    atlas_pdl_enter();
    __shared__ float s_dot_partial[8]; // one per warp (256/32=8)

    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;

    const __nv_bfloat16* my_normed = normed + token * hidden_size;
    const __nv_bfloat16* my_shared = shared_out + token * hidden_size;
    __nv_bfloat16* my_output = output + token * hidden_size;

    // Phase 1: dot product normed[token] . gate_weight
    // NULL gate_weight = no gate modulation → sigmoid=1.0 (always include shared expert)
    float local_dot = 0.0f;
    if (gate_weight != 0) {
        for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
            float n = __bfloat162float(my_normed[i]);
            float g = __bfloat162float(gate_weight[i]);
            local_dot += n * g;
        }
    }

    // Warp-level sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    }
    if (lane == 0) s_dot_partial[warp_id] = local_dot;
    __syncthreads();

    // Cross-warp sum (thread 0)
    float gate_scalar;
    if (tid == 0) {
        if (gate_weight == 0) {
            // No gate: always include shared expert at full weight
            gate_scalar = 1.0f;
        } else {
            float total = 0.0f;
            for (unsigned int w = 0; w < blockDim.x / 32; w++) {
                total += s_dot_partial[w];
            }
            gate_scalar = 1.0f / (1.0f + __expf(-total));
        }
        s_dot_partial[0] = gate_scalar;
    }
    __syncthreads();
    gate_scalar = s_dot_partial[0];

    // Phase 2: output[i] += gate_scalar * shared_out[i]
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float o = __bfloat162float(my_output[i]);
        float s = __bfloat162float(my_shared[i]);
        my_output[i] = __float2bfloat16(o + gate_scalar * s);
    }
}

// Sort tokens by expert assignment: counting sort producing sorted indices,
// expert prefix sum offsets, and a reverse map for unpermute.
//
// Grid: (1, 1, 1)  Block: (256, 1, 1)
//
// Output layout: sorted_token_ids groups all slots for the same expert
// contiguously. expert_offsets[e] is the first sorted position for expert e.
// token_to_perm[slot] = sorted position of original slot.
//
// CONTRACT: row order inside an expert group is unspecified. A slot's position
// is offsets[e] plus what its atomicAdd returned, and nothing orders the warps
// in the scatter, so two launches (or two TP ranks) may order one expert's
// rows differently. expert_offsets is fixed, and sorted_token_ids and
// token_to_perm always describe the same permutation. Consumers must address
// rows only through sorted_token_ids / token_to_perm and treat each row on its
// own; a bytewise comparison or hash of a route-major buffer is not meaningful.
extern "C" __global__ void moe_sort_by_expert(
    const unsigned int* __restrict__ topk_ids,      // [total_expanded] expert ids
    int* __restrict__ sorted_token_ids,              // [total_expanded] → original token index
    int* __restrict__ sorted_expert_ids,             // [total_expanded] → expert id per sorted pos
    int* __restrict__ expert_offsets,                 // [num_experts + 1] prefix sum
    int* __restrict__ token_to_perm,                  // [total_expanded] → sorted position
    unsigned int total_expanded,
    unsigned int num_experts,
    unsigned int topk
) {
    atlas_pdl_enter();
    // Supports up to 512 experts (counts[512] + offsets[513] = 4100 bytes shared mem).
    __shared__ unsigned int counts[1024];
    __shared__ unsigned int offsets[1025];

    // Clear counts
    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();

    // Phase 1: histogram
    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x)
        atomicAdd(&counts[topk_ids[i]], 1);
    __syncthreads();

    // Phase 2: prefix sum (single thread)
    if (threadIdx.x == 0) {
        offsets[0] = 0;
        for (unsigned int e = 0; e < num_experts; e++)
            offsets[e + 1] = offsets[e] + counts[e];
        for (unsigned int e = 0; e <= num_experts; e++)
            expert_offsets[e] = (int)offsets[e];
    }
    __syncthreads();

    // Phase 3: reset counts for scatter
    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();

    // Phase 4: scatter to sorted positions
    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x) {
        unsigned int expert_id = topk_ids[i];
        unsigned int pos = offsets[expert_id] + atomicAdd(&counts[expert_id], 1);
        sorted_token_ids[pos] = (int)(i / topk);
        sorted_expert_ids[pos] = (int)expert_id;
        token_to_perm[i] = (int)pos;
    }
}

// Build the COMPACTED (expert, m_tile, n_tile) work-list for the persistent
// grouped-GEMM grid (moe_fp8_grouped_gemm). Replaces a dense 3D launch
// `[ceil(N/64), max_m_tiles, num_experts]` (≈16M CTAs/layer, 99.5% early-exit)
// with exactly one work-item per (expert, m_tile, n_tile) that actually has
// tokens, so the persistent 96-CTA grid never spawns a tile that would
// early-exit.
//
// Output packing (matches the grouped-GEMM kernel decode):
//   worklist[w*2 + 0] = expert_id
//   worklist[w*2 + 1] = (m_tile << 6) | n_tile      (n_tile < 64, 6 bits)
//   total_tiles[0]    = number of emitted work-items
//
// One block: each thread counts one expert's items, a block scan (in chunks
// of blockDim.x experts, expert order) places them, and the thread writes its
// expert's items — the same list, in the same expert/m/n order, as a serial
// walk (which cost ~45 us of dependent loads at 288 experts). Experts with no
// tokens (M_e <= 0) OR a NULL weight pointer (remote expert under EP) are
// skipped, exactly like the grouped-GEMM per-tile `if (M_expert <= 0)` /
// `if (B_exp == 0) continue;` guards — so the emitted work-list is the set of
// tiles the dense grid would NOT have early-exited on.
//
// SAME-STREAM INVARIANT (R3): the launcher MUST enqueue moe_fp8_grouped_gemm on
// the SAME stream as this builder so the read of total_tiles/worklist
// happens-after this write. No cross-stream event is inserted.
//
// Grid: (1, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_build_tile_worklist(
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const unsigned long long* __restrict__ B_weight_ptrs,    // [num_experts] → [N, K] FP8 (0 = remote)
    unsigned int* __restrict__ worklist,                     // [worst_case_tiles * 2]
    int* __restrict__ total_tiles,                           // [1]
    unsigned int num_experts,
    unsigned int n_tiles,                                    // ceil(N / PM4_N_TILE)
    unsigned int m_tile                                      // PM4_M_TILE (=128)
) {
    atlas_pdl_enter();
    __shared__ unsigned int s_scan[1024];
    __shared__ unsigned int s_base;
    const unsigned int tid = threadIdx.x;
    if (tid == 0) s_base = 0;
    for (unsigned int e0 = 0; e0 < num_experts; e0 += blockDim.x) {
        const unsigned int e = e0 + tid;
        unsigned int mt_e = 0;
        if (e < num_experts) {
            const int M_e = expert_offsets[e + 1] - expert_offsets[e];
            if (M_e > 0 && B_weight_ptrs[e] != 0)   // mirror grouped-GEMM early-exit guards
                mt_e = ((unsigned int)M_e + m_tile - 1) / m_tile;
        }
        const unsigned int items = mt_e * n_tiles;
        s_scan[tid] = items;
        __syncthreads();
        for (unsigned int d = 1; d < blockDim.x; d <<= 1) {
            const unsigned int add = tid >= d ? s_scan[tid - d] : 0;
            __syncthreads();
            s_scan[tid] += add;
            __syncthreads();
        }
        unsigned int w = s_base + s_scan[tid] - items;
        for (unsigned int mt = 0; mt < mt_e; mt++) {
            for (unsigned int nt = 0; nt < n_tiles; nt++) {
                // R2: packing overflow guard. n_tile must fit in 6 bits and
                // m_tile in the remaining 26 bits of a 32-bit word.
                assert(mt < (1u << 26) && nt < 64u);
                worklist[w * 2 + 0] = e;
                worklist[w * 2 + 1] = (mt << 6) | nt;
                w++;
            }
        }
        __syncthreads();
        if (tid == blockDim.x - 1) s_base += s_scan[tid];
        __syncthreads();
    }
    if (tid == 0) total_tiles[0] = (int)s_base;
}

// moe_unpermute_reduce_indexed_ep with 16-byte accesses
// (ATLAS_GLM_MOE_UNPERMUTE_VEC): each thread owns 8 consecutive columns and
// issues one 16-byte load per local route before accumulating, where the
// scalar kernel walks its 16 columns one 2-byte load chain at a time (latency
// bound at decode row counts). Every element sums the same products in the
// same slot order, so the output bytes are identical.
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1); hidden_size % 8 == 0,
// topk <= 8.
extern "C" __global__ void moe_unpermute_reduce_indexed_ep_vec8(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int local_expert_start,
    unsigned int local_expert_end
) {
    atlas_pdl_enter();
    constexpr unsigned int MAX_TOPK = 8;
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    // The token's routes in slot order, each live if local. Every index is
    // a compile-time slot, so the arrays stay in registers.
    unsigned int rows[MAX_TOPK];
    float weights[MAX_TOPK];
    bool live[MAX_TOPK];
    #pragma unroll
    for (unsigned int k = 0; k < MAX_TOPK; k++) {
        const unsigned int slot = token * topk + k;
        const int expert = k < topk ? topk_ids[slot] : -1;
        live[k] = k < topk && expert >= (int)local_expert_start && expert < (int)local_expert_end;
        rows[k] = live[k] ? (unsigned int)token_to_perm[slot] : 0u;
        weights[k] = live[k] ? topk_weights[slot] : 0.0f;
    }
    for (unsigned int c = threadIdx.x * 8; c < hidden_size; c += blockDim.x * 8) {
        uint4 raw[MAX_TOPK];
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++)
            if (live[i]) raw[i] = *(const uint4*)&expert_output[rows[i] * hidden_size + c];
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++) {
            if (!live[i]) continue;
            const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw[i]);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float value = __bfloat162float(v[j]);
                acc[j] += weights[i] * value;
            }
        }
        uint4 out;
        __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
        #pragma unroll
        for (int j = 0; j < 8; j++) o[j] = __float2bfloat16(acc[j]);
        *(uint4*)&output[token * hidden_size + c] = out;
    }
}

// moe_unpermute_reduce_indexed_ep_vec8 followed by moe_batched_blend, one
// launch (ATLAS_GLM_DECODE_FUSE). The gate is the blend's dot: the row and the
// gate weight are staged with 16-byte loads and summed per thread, per warp
// and across warps in the blend's order. Each 8-column group then reduces its
// local routes as the vec8 kernel does, rounds to BF16 (the value the blend
// read back) and adds gate * shared. Same bytes as the pair.
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1); hidden 4096, topk <= 8;
// every row pointer and gate_weight 16-byte aligned.
extern "C" __global__ void __launch_bounds__(256) moe_unpermute_blend_ep_vec8(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ normed,
    const __nv_bfloat16* __restrict__ gate_weight,   // nullable: gate 1.0
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int local_expert_start,
    unsigned int local_expert_end
) {
    atlas_pdl_enter();
    constexpr unsigned int MAX_TOPK = 8, H = 4096, BLOCK = 256;
    const unsigned int token = blockIdx.x, tid = threadIdx.x;
    if (token >= num_tokens || blockDim.x != BLOCK) return;
    __shared__ __align__(16) __nv_bfloat16 s_normed[H];
    __shared__ __align__(16) __nv_bfloat16 s_gate[H];
    __shared__ float s_dot_partial[BLOCK / 32];

    float local_dot = 0.0f;
    if (gate_weight != 0) {
        #pragma unroll
        for (unsigned int i = tid * 8; i < H; i += BLOCK * 8) {
            *(uint4*)&s_normed[i] = *(const uint4*)&normed[token * H + i];
            *(uint4*)&s_gate[i] = *(const uint4*)&gate_weight[i];
        }
        __syncthreads();
        for (unsigned int i = tid; i < H; i += BLOCK) {
            float n = __bfloat162float(s_normed[i]);
            float g = __bfloat162float(s_gate[i]);
            local_dot += n * g;
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    if (tid % 32 == 0) s_dot_partial[tid / 32] = local_dot;
    __syncthreads();
    if (tid == 0) {
        float gate = 1.0f;
        if (gate_weight != 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < BLOCK / 32; w++) total += s_dot_partial[w];
            gate = 1.0f / (1.0f + __expf(-total));
        }
        s_dot_partial[0] = gate;
    }
    __syncthreads();
    const float gate_scalar = s_dot_partial[0];

    unsigned int rows[MAX_TOPK];
    float weights[MAX_TOPK];
    bool live[MAX_TOPK];
    #pragma unroll
    for (unsigned int k = 0; k < MAX_TOPK; k++) {
        const unsigned int slot = token * topk + k;
        const int expert = k < topk ? topk_ids[slot] : -1;
        live[k] = k < topk && expert >= (int)local_expert_start && expert < (int)local_expert_end;
        rows[k] = live[k] ? (unsigned int)token_to_perm[slot] : 0u;
        weights[k] = live[k] ? topk_weights[slot] : 0.0f;
    }
    for (unsigned int c = tid * 8; c < H; c += BLOCK * 8) {
        uint4 raw[MAX_TOPK];
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++)
            if (live[i]) raw[i] = *(const uint4*)&expert_output[rows[i] * H + c];
        const uint4 sraw = *(const uint4*)&shared_out[token * H + c];
        const __nv_bfloat16* sv = reinterpret_cast<const __nv_bfloat16*>(&sraw);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (unsigned int i = 0; i < MAX_TOPK; i++) {
            if (!live[i]) continue;
            const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw[i]);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float value = __bfloat162float(v[j]);
                acc[j] += weights[i] * value;
            }
        }
        uint4 out;
        __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float routed = __bfloat162float(__float2bfloat16(acc[j]));
            const float s = __bfloat162float(sv[j]);
            o[j] = __float2bfloat16(routed + gate_scalar * s);
        }
        *(uint4*)&output[token * H + c] = out;
    }
}

// moe_sort_by_expert with its prefix sum spread over the block
// (ATLAS_GLM_DECODE_FUSE): one thread walked all expert counts and then stored
// every offset. Here each thread sums up to four consecutive experts, a warp
// then a block shuffle scan places those sums, and each thread writes its own
// offsets. Integer sums, so expert_offsets are the same words; the histogram
// and the scatter are the original's, so rows keep its contract (any order
// within an expert group). Grid: (1, 1, 1)  Block: (256, 1, 1); at most 1024
// experts.
extern "C" __global__ void moe_sort_by_expert_scan(
    const unsigned int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ sorted_expert_ids,
    int* __restrict__ expert_offsets,
    int* __restrict__ token_to_perm,
    unsigned int total_expanded,
    unsigned int num_experts,
    unsigned int topk
) {
    atlas_pdl_enter();
    __shared__ unsigned int counts[1024];
    __shared__ unsigned int offsets[1025];
    __shared__ unsigned int warp_sums[8];
    const unsigned int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if (blockDim.x != 256 || num_experts > 1024) return;

    for (unsigned int i = tid; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();
    for (unsigned int i = tid; i < total_expanded; i += blockDim.x)
        atomicAdd(&counts[topk_ids[i]], 1);
    __syncthreads();

    const unsigned int per = (num_experts + 255) / 256;
    const unsigned int e0 = min(tid * per, num_experts), e1 = min(e0 + per, num_experts);
    unsigned int local = 0;
    for (unsigned int e = e0; e < e1; e++) local += counts[e];
    unsigned int incl = local;
    #pragma unroll
    for (unsigned int o = 1; o < 32; o <<= 1) {
        const unsigned int up = __shfl_up_sync(0xFFFFFFFF, incl, o);
        if (lane >= o) incl += up;
    }
    if (lane == 31) warp_sums[warp] = incl;
    __syncthreads();
    unsigned int base = 0;
    for (unsigned int w = 0; w < warp; w++) base += warp_sums[w];
    unsigned int running = base + incl - local;
    for (unsigned int e = e0; e < e1; e++) {
        offsets[e] = running;
        expert_offsets[e] = (int)running;
        running += counts[e];
    }
    if (e1 == num_experts && e0 < e1) {
        offsets[num_experts] = running;
        expert_offsets[num_experts] = (int)running;
    }
    if (num_experts == 0 && tid == 0) {
        offsets[0] = 0;
        expert_offsets[0] = 0;
    }
    __syncthreads();

    for (unsigned int i = tid; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();
    for (unsigned int i = tid; i < total_expanded; i += blockDim.x) {
        unsigned int expert_id = topk_ids[i];
        unsigned int pos = offsets[expert_id] + atomicAdd(&counts[expert_id], 1);
        sorted_token_ids[pos] = (int)(i / topk);
        sorted_expert_ids[pos] = (int)expert_id;
        token_to_perm[i] = (int)pos;
    }
}

// moe_build_tile_worklist with the chunked Hillis-Steele scan replaced by the
// shuffle scan of moe_sort_by_expert_scan (ATLAS_GLM_DECODE_FUSE): each thread
// owns up to four consecutive experts, so one scan places every expert where
// the original walked 256-expert chunks of eight barrier pairs each. Integer
// sums and the same expert/m/n emission order, so the work-list and
// total_tiles are the same words. Grid: (1, 1, 1)  Block: (256, 1, 1); at most
// 1024 experts.
extern "C" __global__ void moe_build_tile_worklist_scan(
    const int* __restrict__ expert_offsets,
    const unsigned long long* __restrict__ B_weight_ptrs,
    unsigned int* __restrict__ worklist,
    int* __restrict__ total_tiles,
    unsigned int num_experts,
    unsigned int n_tiles,
    unsigned int m_tile
) {
    atlas_pdl_enter();
    __shared__ unsigned int warp_sums[8];
    const unsigned int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if (blockDim.x != 256 || num_experts > 1024) return;
    const unsigned int per = (num_experts + 255) / 256;
    const unsigned int e0 = min(tid * per, num_experts), e1 = min(e0 + per, num_experts);
    unsigned int mt[4] = {0u, 0u, 0u, 0u};
    unsigned int local = 0;
    #pragma unroll
    for (unsigned int k = 0; k < 4; k++) {
        const unsigned int e = e0 + k;
        if (e < e1) {
            const int M_e = expert_offsets[e + 1] - expert_offsets[e];
            if (M_e > 0 && B_weight_ptrs[e] != 0)   // mirror grouped-GEMM early-exit guards
                mt[k] = ((unsigned int)M_e + m_tile - 1) / m_tile;
            local += mt[k] * n_tiles;
        }
    }
    unsigned int incl = local;
    #pragma unroll
    for (unsigned int o = 1; o < 32; o <<= 1) {
        const unsigned int up = __shfl_up_sync(0xFFFFFFFF, incl, o);
        if (lane >= o) incl += up;
    }
    if (lane == 31) warp_sums[warp] = incl;
    __syncthreads();
    unsigned int w = 0;
    for (unsigned int v = 0; v < warp; v++) w += warp_sums[v];
    w += incl - local;
    #pragma unroll
    for (unsigned int k = 0; k < 4; k++) {
        const unsigned int e = e0 + k;
        for (unsigned int m = 0; m < mt[k]; m++) {
            for (unsigned int nt = 0; nt < n_tiles; nt++) {
                assert(m < (1u << 26) && nt < 64u);
                worklist[w * 2 + 0] = e;
                worklist[w * 2 + 1] = (m << 6) | nt;
                w++;
            }
        }
    }
    if (tid == 255) {
        unsigned int total = 0;
        for (unsigned int v = 0; v < 8; v++) total += warp_sums[v];
        total_tiles[0] = (int)total;
    }
}
