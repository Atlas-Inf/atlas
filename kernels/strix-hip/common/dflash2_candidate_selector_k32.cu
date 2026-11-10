// SPDX-License-Identifier: AGPL-3.0-only
// Top-k=32 twin of the DFlash2 candidate selector for strix-hip (Ripple-65B-
// DFlash2 ships `selector_top_k: 32`), mirroring
// kernels/gb10/common/dflash2_candidate_selector_k32.cu. The production K=16
// instantiation is untouched. Every extern "C" entry point is renamed onto
// the _k32 namespace; the walk scores candidate c on warp c, so 32 warps.
#define DF2_SEL_MAX_TOP_K 32
#define DF2_SEL_THREADS 1024
// K=32 per-thread register lists spill on gfx1151; take the round form.
#define DF2_SEL_ROUND_TOPK 1
#define dflash2_candidate_selector dflash2_candidate_selector_k32
#define dflash2_candidate_selector_batched dflash2_candidate_selector_batched_k32
#include "dflash2_candidate_selector.cu"
