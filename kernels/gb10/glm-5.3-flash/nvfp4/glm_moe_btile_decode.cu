// SPDX-License-Identifier: AGPL-3.0-only
// Reuse the existing scalar LUT/decoder without editing the common family.
// Legacy exports from this include are contained in this new module; their
// established module lookup and every serving caller remain unchanged.
#include "../../common/moe_shared_expert_fused_t.cu"
#include "glm_moe_btile_decode_register.cuh"
