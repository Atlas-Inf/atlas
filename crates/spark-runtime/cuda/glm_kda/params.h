// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cstddef>
#include <cstdint>

// Internal Atlas CUDA operator contract, mirrored by src/glm_kda.rs.
// All offsets are bytes into caller-owned workspace; no allocation is hidden.
struct AtlasKdaScratch {
  std::uint64_t kd, qd, kr, gt, inv, mqk, beta, bytes;
};
struct AtlasKdaParams {
  std::uint64_t qkv, gate, beta, a_log, dt_bias, state, output, workspace;
  AtlasKdaScratch layout;
  std::uint32_t tokens, reserved;
  std::uint64_t stream;
};
static_assert(sizeof(AtlasKdaScratch) == 64);
static_assert(sizeof(AtlasKdaParams) == 144);
static_assert(alignof(AtlasKdaParams) == 8);
static_assert(offsetof(AtlasKdaParams, layout) == 64);
static_assert(offsetof(AtlasKdaParams, tokens) == 128);
static_assert(offsetof(AtlasKdaParams, stream) == 136);
