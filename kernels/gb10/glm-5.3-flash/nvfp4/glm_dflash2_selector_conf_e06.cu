// SPDX-License-Identifier: AGPL-3.0-only
// Edge-weight twin of the confidence selector (ATLAS_DFLASH2_EDGE=0.6):
// Score(c) = Unary(c) + 0.6 * <Pred[prev] ⊙ H_proj(h_t), Succ[c]>. Changes
// drafts only; verification is unchanged and exact. Same entry names and
// arity as glm_dflash2_selector_conf. Prior art: MiaAI-Lab's TensorFold
// recipe scores DFlash2 candidates with an edge weight of 0.6
// (dflash2.py, EDGE; idea only, see docs/glm-prior-art.md).
#define DF2_SEL_CONF 1
#define DF2_SEL_EDGE 0.6f
#define dflash2_candidate_selector dflash2_candidate_selector_conf
#define dflash2_selector_walk dflash2_selector_walk_conf
#include "../../common/dflash2_candidate_selector.cu"
