// SPDX-License-Identifier: AGPL-3.0-only
//
// Append a single token's K and V projections into a contiguous KV
// cache at slot `cache_pos`. Mirrors the CUDA `kv_cache_append`
// kernel's role on the decode hot path; the paged variant follows in
// a separate kernel.
//
// Layout:
//   new_k   : bfloat [num_kv_heads, head_dim]
//   new_v   : bfloat [num_kv_heads, head_dim]
//   k_cache : bfloat [max_seq, num_kv_heads, head_dim]
//   v_cache : bfloat [max_seq, num_kv_heads, head_dim]
//
// Grid: (head_dim, num_kv_heads, 1) — one thread per cache element.

#include <metal_stdlib>
using namespace metal;

kernel void kv_cache_append(
    constant uint &num_kv_heads [[buffer(0)]],
    constant uint &head_dim     [[buffer(1)]],
    constant uint &cache_pos    [[buffer(2)]],
    device const bfloat *new_k  [[buffer(3)]],
    device const bfloat *new_v  [[buffer(4)]],
    device bfloat       *k_cache [[buffer(5)]],
    device bfloat       *v_cache [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint h = gid.y;
    if (h >= num_kv_heads || d >= head_dim) {
        return;
    }
    uint cache_off = cache_pos * num_kv_heads * head_dim + h * head_dim + d;
    uint new_off   = h * head_dim + d;
    k_cache[cache_off] = new_k[new_off];
    v_cache[cache_off] = new_v[new_off];
}
