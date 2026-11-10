// SPDX-License-Identifier: AGPL-3.0-only
//
// Exact DFlash tree verify (ATLAS_DFLASH_TREE), qwen3.6-27b tree (so the
// qwen3.6-27b and qwen3.8-27b targets): the KV side of a tree.
//
// A tree verify gives every row its own block table: the committed prefix's
// blocks, then private blocks for the positions from the block holding the
// first verify position on. Each row writes its own K/V at its slot there
// (position = prefix + depth), and `kv_slot_copy_bf16` then copies into the
// row's private blocks the entries its attention must see besides its own:
// the committed prefix's entries in the first private block and every
// ancestor's K/V at the ancestor's position. A row's paged attention then
// reads, slot by slot, the keys and values serial decode reads at that
// position (same positions, same count, so the same split bounds): the
// attention arithmetic is untouched, and the copies move bytes only. After
// the verdict the same kernel copies the accepted path's entries from the
// deepest accepted row's private blocks into the sequence's own slots.
//
// `entries` holds (src_slot, dst_slot) int64 pairs into one layer's BF16
// K and V pools ([slot, n_elems]); `count` (device) says how many of the
// `gridDim.x` blocks have an entry, so a captured graph replays any tree of
// the same width. Entries must not depend on each other (no entry's dst is
// another's src).
//
// Grid: (max entries, 1, 1)  Block: (128, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void kv_slot_copy_bf16(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ entries,  // [2 * n]: src slot, dst slot
    const unsigned int* __restrict__ count,
    const unsigned int n_elems              // num_kv_heads * head_dim
) {
    const unsigned int e = blockIdx.x;
    if (e >= *count) return;
    const long long src = entries[2 * e];
    const long long dst = entries[2 * e + 1];
    if (src < 0 || dst < 0) return;
    const unsigned long long s = (unsigned long long)src * n_elems;
    const unsigned long long d = (unsigned long long)dst * n_elems;
    if (n_elems % 8 == 0) {
        const uint4* ks = reinterpret_cast<const uint4*>(k_cache + s);
        const uint4* vs = reinterpret_cast<const uint4*>(v_cache + s);
        uint4* kd = reinterpret_cast<uint4*>(k_cache + d);
        uint4* vd = reinterpret_cast<uint4*>(v_cache + d);
        for (unsigned int i = threadIdx.x; i < n_elems / 8; i += blockDim.x) {
            kd[i] = ks[i];
            vd[i] = vs[i];
        }
    } else {
        for (unsigned int i = threadIdx.x; i < n_elems; i += blockDim.x) {
            k_cache[d + i] = k_cache[s + i];
            v_cache[d + i] = v_cache[s + i];
        }
    }
}
