// SPDX-License-Identifier: AGPL-3.0-only
// Shared host/device byte mapping; no layout activation by this header.
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace glm_native_btile {
constexpr unsigned n = 2048, k = 4096, group = 16;
constexpr size_t packed_bytes = size_t(n) * k / 2;
constexpr size_t scale_bytes = size_t(n) * k / group;
#ifdef __CUDACC__
#define GLM_NATIVE_HD __host__ __device__
#else
#define GLM_NATIVE_HD
#endif
GLM_NATIVE_HD constexpr size_t native_source_for_tile(size_t destination) {
    const size_t tile = destination / 4096;
    const size_t row = (tile / (k / 64)) * 128 + (destination % 4096) / 32;
    const size_t packed_col = (tile % (k / 64)) * 32 + destination % 32;
    return row * (k / 2) + packed_col;
}
GLM_NATIVE_HD constexpr size_t tile_for_native(size_t source) {
    const size_t row = source / (k / 2), packed_col = source % (k / 2);
    return (((row / 128) * (k / 64) + packed_col / 32) * 128 + row % 128) * 32 + packed_col % 32;
}
#undef GLM_NATIVE_HD

struct Span { uint64_t address; size_t bytes; };
struct Projection { Span packed, scales; uint32_t scalar_bits; };
inline bool valid_span(Span span, size_t bytes) {
    return span.address && span.address % 16 == 0 && span.bytes == bytes
        && span.address <= std::numeric_limits<uint64_t>::max() - span.bytes;
}
inline bool disjoint(Span a, Span b) {
    return a.address + a.bytes <= b.address || b.address + b.bytes <= a.address;
}
inline bool valid_repack(unsigned rows, unsigned cols, unsigned gs, Projection p, Span scratch) {
    return rows == n && cols == k && gs == group
        && (p.scalar_bits & 0x7f800000u) != 0x7f800000u
        && valid_span(p.packed, packed_bytes) && valid_span(p.scales, scale_bytes)
        && valid_span(scratch, packed_bytes)
        && disjoint(p.packed, p.scales) && disjoint(p.packed, scratch)
        && disjoint(p.scales, scratch);
}
} // namespace glm_native_btile
