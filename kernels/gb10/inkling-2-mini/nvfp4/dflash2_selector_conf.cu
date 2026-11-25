// SPDX-License-Identifier: AGPL-3.0-only
// The DFlash2 candidate selector with per-draft confidence for the Inkling 2
// target (a DFlash2 drafter such as incoai/Inkling-2-Mini-DFlash2 under
// ATLAS_DFLASH_CONF_WIDTH / ATLAS_DFLASH_CONF_LOG): GLM's twin
// (`glm_dflash2_selector_conf.cu`) under a target-neutral module name. Picks
// are identical to the common kernel's; out_tokens grows to
// [gamma tokens | gamma f32 confidences].
#define DF2_SEL_CONF 1
#define dflash2_candidate_selector dflash2_candidate_selector_conf
#define dflash2_selector_walk dflash2_selector_walk_conf
#define dflash2_selector_walk_ranks dflash2_selector_walk_ranks_conf
#include "../../common/dflash2_candidate_selector.cu"
