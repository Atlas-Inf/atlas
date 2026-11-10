// SPDX-License-Identifier: AGPL-3.0-only
// The DFlash2 candidate selector with a 32-candidate cap: the common kernel's
// body under other entry names, so the 16-candidate module every existing
// DFlash2 drafter launches is untouched. Ricursive's Ripple-65B DFlash2
// drafter ships `selector_top_k: 32`. The walk scores one candidate per warp,
// so the block grows to 32 warps (and the candidate pool to their 32 x 32
// lists); it stages 8 rows at a time (not 16) to fit the 48 KB static
// shared-memory limit. Picks follow the same total order and
// greedy chain as the 16-candidate kernel over the wider top-k prefix. The
// one-CTA-per-sequence batched entry is not built at this cap (its per-thread
// K-sized register lists would spill at 1024 threads); the per-sequence
// fused launch serves batches instead.
#define DF2_SEL_MAX_TOP_K 32
#define DF2_SEL_THREADS 1024
#define DF2_SEL_WALK_ROWS 8
// The overflow path stages 32 warps x 32 candidates in the pool.
#define DF2_SEL_POOL 1024
#define DF2_SEL_BOUNDS __launch_bounds__(DF2_SEL_THREADS)
#define DF2_SEL_NO_BATCHED 1
#define dflash2_candidate_selector dflash2_candidate_selector_k32
#define dflash2_selector_topk dflash2_selector_topk_k32
#define dflash2_selector_walk dflash2_selector_walk_k32
#define dflash2_selector_walk_ranks dflash2_selector_walk_ranks_k32
#include "dflash2_candidate_selector.cu"
