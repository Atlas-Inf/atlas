// SPDX-License-Identifier: AGPL-3.0-only

//! The loader's small private helpers.
//!
//! Split out of `qwen4_exp.rs` for the file-size cap. The seam is a real one:
//! none of these touches the loader's state or its trait impl — they are the
//! free functions the impl and its submodules call.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use crate::weight_map::DenseWeight;

/// A ones-filled `[n]` BF16 norm scale.
///
/// BF16 1.0 is `0x3F80`, so the buffer cannot be produced with `memset`.
pub(super) fn ones_norm(n: usize, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let host: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3Fu8], n).flatten().collect();
    let ptr = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// `model.language_model` for the multimodal layout, `model` otherwise.
pub(super) fn embed_prefix(config: &ModelConfig) -> String {
    if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    }
}

/// The model-level hyper-connection mixer that collapses the residual streams.
pub(super) fn mixer_prefix(config: &ModelConfig) -> String {
    format!("{}.hyper_connection_mixer", embed_prefix(config))
}

/// The MTP drafter's own precision (`mtp.rs`), each switch on only at `1`.
/// Drafts only: the target verifies every token and keeps its own layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MtpPrecision {
    /// `ATLAS_QWEN4EXP_MTP_BF16_ATTN=1`: the drafter's Q/K/V/O stay at the
    /// checkpoint's BF16 (the `--attn-proj-dtype bf16` arms).
    pub(super) bf16_attn: bool,
    /// `ATLAS_QWEN4EXP_MTP_BF16_ROUTER=1`, or a serving mode (Enntity's only
    /// drafter MoE): the router stays the checkpoint's BF16 and the shared
    /// expert is an NVFP4 copy fused into the expert kernels. Off: an NVFP4
    /// router copy and the BF16-native shared expert on its own chain.
    pub(super) bf16_router: bool,
}

impl MtpPrecision {
    pub(super) fn from_env() -> Result<Self> {
        let mode = super::mode::selected()?.is_some();
        Ok(Self::from_lookup(|key| std::env::var(key).ok(), mode))
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>, mode: bool) -> Self {
        let on = |key: &str| lookup(key).as_deref() == Some("1");
        Self {
            bf16_attn: on("ATLAS_QWEN4EXP_MTP_BF16_ATTN"),
            bf16_router: mode || on("ATLAS_QWEN4EXP_MTP_BF16_ROUTER"),
        }
    }

    /// The config the drafter is built with: the TP=1 view (`mtp.rs`), with
    /// [`Self::bf16_attn`] on its own copy only.
    pub(super) fn drafter_config(self, config: &ModelConfig) -> ModelConfig {
        let mut config = config.tp1_view();
        config.attn_proj_bf16 |= self.bf16_attn;
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| {
            set.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    }

    const ATTN: &str = "ATLAS_QWEN4EXP_MTP_BF16_ATTN";
    const ROUTER: &str = "ATLAS_QWEN4EXP_MTP_BF16_ROUTER";

    #[test]
    fn unset_switches_keep_this_stacks_drafter() {
        let off = MtpPrecision {
            bf16_attn: false,
            bf16_router: false,
        };
        assert_eq!(MtpPrecision::from_lookup(lookup(&[]), false), off);
        for value in ["0", "", "true", "on", " 1", "1 "] {
            let set = [(ATTN, value), (ROUTER, value)];
            assert_eq!(MtpPrecision::from_lookup(lookup(&set), false), off);
        }
    }

    #[test]
    fn each_switch_moves_only_its_own_precision() {
        let attn = MtpPrecision::from_lookup(lookup(&[(ATTN, "1")]), false);
        assert!(attn.bf16_attn && !attn.bf16_router);
        let router = MtpPrecision::from_lookup(lookup(&[(ROUTER, "1")]), false);
        assert!(!router.bf16_attn && router.bf16_router);
    }

    /// Enntity's modes run the BF16-router drafter MoE (their only one); the
    /// attention switch stays the mode contract's to set (`mode.rs`).
    #[test]
    fn a_serving_mode_takes_the_bf16_router_drafter_moe() {
        for set in [&[][..], &[(ROUTER, "0")][..]] {
            let p = MtpPrecision::from_lookup(lookup(set), true);
            assert!(p.bf16_router && !p.bf16_attn);
        }
    }

    #[test]
    fn drafter_config_is_the_tp1_view_with_only_its_attention_widened() {
        let full = ModelConfig::qwen3_next_80b_nvfp4();
        let mut rank1 = full.clone();
        rank1.num_attention_heads /= 2;
        rank1.num_key_value_heads /= 2;
        rank1.linear_num_key_heads /= 2;
        rank1.linear_num_value_heads /= 2;
        (rank1.tp_rank, rank1.tp_world_size) = (1, 2);
        (rank1.ep_rank, rank1.ep_world_size) = (1, 2);
        for target_bf16 in [false, true] {
            rank1.attn_proj_bf16 = target_bf16;
            for bf16_attn in [false, true] {
                let p = MtpPrecision {
                    bf16_attn,
                    bf16_router: false,
                };
                let drafter = p.drafter_config(&rank1);
                let mut want = rank1.tp1_view();
                want.attn_proj_bf16 = target_bf16 || bf16_attn;
                assert_eq!(format!("{drafter:?}"), format!("{want:?}"));
                assert_eq!(rank1.attn_proj_bf16, target_bf16, "target untouched");
            }
        }
    }
}
