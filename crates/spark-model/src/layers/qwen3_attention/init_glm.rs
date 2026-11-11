// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 kernel selection for `Qwen3AttentionLayer::new_with_gating`: the
//! env-selected semantic-index / sparse-attention variants and the exact GLM
//! MLA paged-decode kernel.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::init_arch_gates::{ArchProbes, gated as gate};
use crate::layers::ops::{GLM_INDEX_LOGITS_V2_POOLS, GLM_INDEX_LOGITS_V2_ROWS};

/// Env-selected GLM semantic-index / sparse-attention kernel variants.
pub(super) struct GlmIndexSelection {
    pub(super) glm_sparse_attn_heads_per_cta: u32,
    pub(super) glm_sparse_attn_fn: &'static str,
    pub(super) glm_index_logits_rows_per_cta: u32,
    pub(super) glm_index_logits_pools_per_cta: u32,
    pub(super) glm_index_wmma: bool,
    pub(super) glm_sparse_graphs: bool,
    pub(super) glm_index_logits_fn: &'static str,
}

impl GlmIndexSelection {
    pub(super) fn from_config(config: &ModelConfig, probes: &ArchProbes) -> Result<Self> {
        // Multi-head sparse MLA reuses GLM's shared compressed K/V rows. Keep
        // the original one-head kernel available as an operational fallback.
        let glm_sparse_attn_heads_per_cta =
            if std::env::var("ATLAS_GLM_SPARSE_HEAD_GROUP").ok().as_deref() == Some("1") {
                1
            } else {
                8
            };
        let glm_sparse_attn_fn = if glm_sparse_attn_heads_per_cta == 1 {
            "glm_sparse_mla_prefill_bf16"
        } else {
            "glm_sparse_mla_prefill_bf16_head8"
        };
        let kpool_shape = probes.glm_kpool_indexer
            && config.index_n_heads == 32
            && config.index_head_dim == 128
            && config.index_kpool == 4;
        let (
            glm_index_logits_fn,
            glm_index_logits_rows_per_cta,
            glm_index_logits_pools_per_cta,
            glm_index_wmma,
        ) = index_logits_choice(kpool_shape, |name| {
            std::env::var(name).ok().as_deref() == Some("1")
        });
        let glm_sparse_graphs = probes.glm_kpool_indexer
            && super::glm_multi_seq_sparse_graphs_enabled(&config.model_type)?;
        Ok(Self {
            glm_sparse_attn_heads_per_cta,
            glm_sparse_attn_fn,
            glm_index_logits_rows_per_cta,
            glm_index_logits_pools_per_cta,
            glm_index_wmma,
            glm_sparse_graphs,
            glm_index_logits_fn,
        })
    }
}

/// Env-free core of the prefill semantic-scorer choice: `(kernel, rows per
/// CTA, pools per CTA, glm_indexer_wmma module)`. `env_on(flag)`: flag == "1".
fn index_logits_choice(
    kpool_shape: bool,
    env_on: impl Fn(&str) -> bool,
) -> (&'static str, u32, u32, bool) {
    // Tile semantic-index scoring over eight query rows during prefill so
    // both queries and pooled keys are reused. Decode retains the original
    // eight-pool kernel because it has only one live row.
    let row_group = !env_on("ATLAS_GLM_INDEX_ROW_GROUP");
    let wmma = kpool_shape && row_group && env_on("ATLAS_GLM_INDEX_WMMA");
    // v2 reproduces the WMMA scorer's bits, so it may only stand in for WMMA;
    // without it, v2 would change the scalar scorers' reduction order.
    if wmma && env_on("ATLAS_GLM_INDEX_LOGITS_V2") {
        let (rows, pools) = (GLM_INDEX_LOGITS_V2_ROWS, GLM_INDEX_LOGITS_V2_POOLS);
        ("glm_index_logits_bf16_mma_v2", rows, pools, true)
    } else if wmma {
        ("glm_index_logits_bf16_wmma_row8_pool32", 8, 32, true)
    } else if row_group {
        ("glm_index_logits_bf16_row8", 8, 8, false)
    } else {
        ("glm_index_logits_bf16", 1, 8, false)
    }
}

/// `paged_decode_mla_k`. `#[track_caller]` so the boot audit still names the
/// constructor's field, as with `gated`.
#[track_caller]
pub(super) fn paged_decode_mla(
    probes: &ArchProbes,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    mla_decode_mod: &str,
) -> Result<KernelHandle> {
    // GLM's latent width is 512, not the inherited DeepSeek 576.
    // Require its exact kernel; a missing module must not fall back.
    Ok(if config.model_type == "glm5_next" {
        gpu.kernel(mla_decode_mod, "paged_decode_attn")?
    } else {
        gate(probes.mla, gpu, mla_decode_mod, "paged_decode_attn")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW_GROUP: &str = "ATLAS_GLM_INDEX_ROW_GROUP";
    const WMMA: &str = "ATLAS_GLM_INDEX_WMMA";
    const V2: &str = "ATLAS_GLM_INDEX_LOGITS_V2";

    fn choice(kpool_shape: bool, on: &[&str]) -> (&'static str, u32, u32, bool) {
        index_logits_choice(kpool_shape, |name| on.contains(&name))
    }

    #[test]
    fn logits_v2_only_replaces_the_wmma_scorer_it_reproduces() {
        let (rows, pools) = (GLM_INDEX_LOGITS_V2_ROWS, GLM_INDEX_LOGITS_V2_POOLS);
        let v2 = ("glm_index_logits_bf16_mma_v2", rows, pools, true);
        let wmma = ("glm_index_logits_bf16_wmma_row8_pool32", 8, 32, true);
        let row8 = ("glm_index_logits_bf16_row8", 8, 8, false);
        let scalar = ("glm_index_logits_bf16", 1, 8, false);
        assert_eq!(choice(true, &[WMMA]), wmma);
        assert_eq!(choice(true, &[]), row8);
        assert_eq!(choice(true, &[ROW_GROUP, WMMA]), scalar);
        // Anywhere but in place of WMMA, V2 must be inert: it would change the
        // scalar scorers' reduction order (both production profiles set WMMA,
        // but a profile or rank without it must not silently turn lossy).
        for kpool_shape in [false, true] {
            for on in [&[][..], &[ROW_GROUP], &[WMMA], &[ROW_GROUP, WMMA]] {
                let off = choice(kpool_shape, on);
                let with_v2 = choice(kpool_shape, &[on, &[V2]].concat());
                let want = if off == wmma { v2 } else { off };
                assert_eq!(with_v2, want, "kpool shape {kpool_shape}, flags {on:?}");
            }
        }
    }
}
