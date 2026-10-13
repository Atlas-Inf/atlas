// SPDX-License-Identifier: AGPL-3.0-only
// Native packed bytes -> B-tile SSOT. Source MUST be separate scratch.
#pragma once
#include "glm_moe_btile_native_layout.h"
extern "C" __global__ void glm_native_to_btile_u8(
    const unsigned char* __restrict__ source,
    unsigned char* __restrict__ destination,
    unsigned int rows, unsigned int cols
) {
    if (rows != glm_native_btile::n || cols != glm_native_btile::k
        || !source || !destination || source == destination
        || blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) return;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < glm_native_btile::packed_bytes)
        destination[index] = source[glm_native_btile::native_source_for_tile(index)];
}
