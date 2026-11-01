// SPDX-License-Identifier: AGPL-3.0-only
// The DFlash2 candidate selector with per-draft confidence
// (ATLAS_DFLASH_CONF_WIDTH / ATLAS_DFLASH_CONF_LOG): the common kernel's
// body under another entry name, so the production entry point and its
// module are untouched. Picks are identical; out_tokens grows to
// [gamma tokens | gamma f32 confidences].
#define DF2_SEL_CONF 1
#define dflash2_candidate_selector dflash2_candidate_selector_conf
#include "../../common/dflash2_candidate_selector.cu"
