// SPDX-License-Identifier: AGPL-3.0-only
// Top-k=32 twin of the DFlash2 candidate selector (Ripple-65B-DFlash2 ships
// `selector_top_k: 32`; the production K=16 instantiation is untouched and
// byte-identical). Every extern "C" entry point is renamed onto the _k32
// namespace; `dflash2_candidate_selector_batched` is not needed — the batch
// path keeps the K=16 cap for now and mixed drafters use the serial path.
#define DF2_SEL_MAX_TOP_K 32
#define DF2_SEL_THREADS 1024
#define DF2_SEL_LAUNCH_BOUNDS __launch_bounds__(DF2_SEL_THREADS)
#define dflash2_candidate_selector dflash2_candidate_selector_k32
#define dflash2_selector_topk dflash2_selector_topk_k32
#define dflash2_selector_walk dflash2_selector_walk_k32
#define dflash2_selector_walk_ranks dflash2_selector_walk_ranks_k32
#define dflash2_candidate_selector_batched dflash2_candidate_selector_batched_k32
#include "dflash2_candidate_selector.cu"
