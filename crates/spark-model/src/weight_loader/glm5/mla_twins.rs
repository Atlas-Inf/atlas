// SPDX-License-Identifier: AGPL-3.0-only

//! Lossy, opt-in MXFP8 twins of GLM MLA weights beyond the four projections
//! of `ATLAS_GLM_MLA_MXFP8`, for target layers only. They serve every paged
//! GLM pass short enough for their tier (index wq_b: <= 32 rows, W_uk / W_uv:
//! <= 16): verify blocks, and paged prefill chunks that short too (a prefix
//! hit's short suffix, a prompt's last chunk). Longer chunks, serial decode
//! and the multi-seq batched decode keep BF16.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;

/// `(bf16 weight, n, k)` twins for `Qwen3AttentionLayer::install_mla_mxfp8`
/// from `[index wq_b, W_uk, W_uv]` and the `[index, kvb]` opt-ins; none for
/// the replicated MTP body (`mtp_body`), which keeps its own BF16 paths:
/// - `ATLAS_GLM_INDEX_MXFP8=1`: the indexer query projection. weights_proj
///   stays BF16; its per-head weights decide near-tie top-k ranks.
/// - `ATLAS_GLM_MLA_KVB_MXFP8=1`: the absorbed per-head W_uk
///   `[heads, kv_lora, nope]` and W_uv `[heads, v_dim, kv_lora]`, whose
///   32-value blocks never straddle a head, for the head-grouped GEMV.
pub(super) fn opt_in_mxfp8_twins(
    config: &ModelConfig,
    [index_wq_b, w_uk_t, w_uv]: [DevicePtr; 3],
    [index, kvb]: [bool; 2],
    mtp_body: bool,
) -> Vec<(DevicePtr, usize, usize)> {
    let heads = config.num_key_value_heads;
    let mut twins = Vec::new();
    if mtp_body {
        return twins;
    }
    if index {
        let n = config.index_n_heads * config.index_head_dim;
        twins.push((index_wq_b, n, config.q_lora_rank));
    }
    if kvb {
        twins.push((w_uk_t, heads * config.kv_lora_rank, config.qk_nope_head_dim));
        twins.push((w_uv, heads * config.v_head_dim, config.kv_lora_rank));
    }
    twins
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig {
            model_type: "glm5_next".into(),
            num_key_value_heads: 32,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 256,
            v_head_dim: 256,
            index_n_heads: 32,
            index_head_dim: 128,
            ..ModelConfig::qwen3_next_80b_nvfp4()
        }
    }

    const WEIGHTS: [DevicePtr; 3] = [DevicePtr(0x100), DevicePtr(0x200), DevicePtr(0x300)];

    #[test]
    fn no_twins_unless_opted_in() {
        assert!(opt_in_mxfp8_twins(&config(), WEIGHTS, [false, false], false).is_empty());
    }

    #[test]
    fn mtp_body_keeps_bf16_even_when_opted_in() {
        assert!(opt_in_mxfp8_twins(&config(), WEIGHTS, [true, true], true).is_empty());
    }

    #[test]
    fn index_opt_in_twins_only_the_query_projection() {
        assert_eq!(
            opt_in_mxfp8_twins(&config(), WEIGHTS, [true, false], false),
            [(DevicePtr(0x100), 32 * 128, 1536)]
        );
    }

    #[test]
    fn kvb_opt_in_twins_the_per_head_absorb_weights() {
        assert_eq!(
            opt_in_mxfp8_twins(&config(), WEIGHTS, [false, true], false),
            [
                (DevicePtr(0x200), 32 * 512, 256),
                (DevicePtr(0x300), 32 * 256, 512),
            ]
        );
    }
}
