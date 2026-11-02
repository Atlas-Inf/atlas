// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit bounded nonspeculative independent rows, never temporal K5.
use crate::layer::ForwardContext;
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferSizes;

pub fn enabled(model_type: &str) -> Result<bool> {
    let value = match std::env::var("ATLAS_GLM_INDEPENDENT_DECODE") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    parse(model_type, value.as_deref())
}
fn parse(model_type: &str, value: Option<&str>) -> Result<bool> {
    ensure!(
        matches!(value, None | Some("0" | "1")),
        "ATLAS_GLM_INDEPENDENT_DECODE must be 0 or 1"
    );
    ensure!(
        value != Some("1") || model_type == "glm5_next",
        "independent decode requires glm5_next"
    );
    Ok(value == Some("1"))
}
pub struct Launch<'a> {
    pub model_type: &'a str,
    pub world: usize,
    pub tp: usize,
    pub ep: usize,
    pub ep_v2: bool,
    pub active: usize,
    pub admitted: usize,
    pub context: usize,
    pub bf16: bool,
    pub independent: bool,
    pub lora: bool,
    pub hss: bool,
    pub swap: bool,
}
impl Launch<'_> {
    pub fn validate(&self) -> Result<()> {
        validate_topology(
            self.model_type,
            self.world,
            self.tp,
            self.ep,
            self.ep_v2,
            self.independent,
        )?;
        ensure!(
            (2..=8).contains(&self.active)
                && self.active == self.admitted
                && (1..=2048).contains(&self.context)
                && self.bf16
                && self.independent
                && !self.lora
                && !self.hss
                && !self.swap,
            "independent decode requires equal active/admitted2..8, BF16 context1..2048, nonspeculative resident base model"
        );
        Ok(())
    }
}
fn validate_environment() -> Result<()> {
    ensure!(
        !crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
        "independent decode requires FP32 KDA state"
    );
    ensure!(
        !matches!(
            std::env::var("ATLAS_GLM_K5_HC_CUBLAS").as_deref(),
            Ok("1" | "true" | "yes")
        ),
        "independent decode excludes temporal K5 mHC selection"
    );
    for name in [
        "ATLAS_GLM_MULTI_SEQ_SPARSE",
        "ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS",
        "ATLAS_GLM_C4_SPARSE",
        "ATLAS_GLM_MOE_GATE_UP_M16",
        "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY",
    ] {
        ensure!(
            std::env::var(name).as_deref() != Ok("1"),
            "independent decode excludes {name}"
        );
    }
    crate::layers::moe::validate_independent_environment()
}
pub fn validate_runtime(
    config: &ModelConfig,
    world: usize,
    ep_v2: bool,
    independent: bool,
) -> Result<()> {
    validate_topology(
        &config.model_type,
        world,
        config.tp_world_size,
        config.ep_world_size,
        ep_v2,
        independent,
    )?;
    ensure!(
        config.linear_num_key_heads == 32
            && config.linear_num_value_heads == 32
            && config.linear_key_head_dim == 128
            && config.linear_value_head_dim == 128
            && config.linear_conv_kernel_dim == 4,
        "independent decode requires actual local KDA32/d128/conv4"
    );
    super::glm_c4::validate_geometry(config)
}
fn validate_topology(
    model_type: &str,
    world: usize,
    tp: usize,
    ep: usize,
    ep_v2: bool,
    independent: bool,
) -> Result<()> {
    ensure!(
        enabled(model_type)?,
        "independent decode requires explicit opt-in"
    );
    ensure!(
        world == 2 && tp == 2 && ep == 2 && ep_v2 && independent,
        "independent decode requires nonspeculative world2/TP2/EP2 and EP-v2"
    );
    validate_environment()
}
pub fn validate_positions(
    positions: impl IntoIterator<Item = usize>,
    rows: usize,
    capacity: usize,
) -> Result<()> {
    ensure!(
        (2..=8).contains(&capacity) && (1..=capacity).contains(&rows),
        "independent width exceeds live capacity2..8"
    );
    let mut count = 0;
    for p in positions {
        ensure!(p < 2048, "independent position must be below2048");
        count += 1;
    }
    ensure!(count == rows, "independent token/state row count mismatch");
    Ok(())
}
pub fn validate_scratch(sizes: &BufferSizes, hc_mult: usize, rows: usize) -> Result<()> {
    ensure!(
        (1..=8).contains(&rows),
        "independent scratch width outside1..8"
    );
    super::glm_c4::validate_scratch_rows(sizes, hc_mult, rows)
}
pub fn validate_projection_handles(nvfp4: [u64; 7], bf16: u64) -> Result<()> {
    ensure!(
        !nvfp4.contains(&0) && bf16 != 0,
        "independent decode requires every drain-width2..8 projection handle"
    );
    Ok(())
}
/// Rows whose FFN may take the exact independent 2..8-row MoE: the actual
/// independent decode lane, or one GLM DFlash verify block.
pub fn ffn_rows_selected(ctx: &ForwardContext, rows: usize) -> Result<bool> {
    if selected(ctx, rows)? {
        return Ok(true);
    }
    Ok(crate::speculative::glm_repair_policy::dflash_enabled()
        && ctx.config.model_type == "glm5_next"
        && ctx.config.tp_world_size == 2
        && ctx.config.ep_world_size == 2
        && (2..=crate::speculative::glm_repair_policy::MAX_DFLASH_VERIFY_ROWS).contains(&rows))
}

/// The indexed view is produced only by the actual independent model path.
/// Temporal verifier contexts have no such view, even when their width is5.
pub fn selected(ctx: &ForwardContext, rows: usize) -> Result<bool> {
    if !enabled(&ctx.config.model_type)? {
        return Ok(false);
    }
    let Some(view) = ctx.ssm_batch else {
        return Ok(false);
    };
    ensure!(
        (2..=8).contains(&rows)
            && rows <= ctx.levers.max_decode_seqs as usize
            && view.layer(0)?.rows() as usize == rows
            && ctx
                .attn_metadata
                .is_some_and(|m| m.num_seqs as usize == rows),
        "independent layer row metadata disagrees with live indexed view"
    );
    Ok(true)
}

impl super::TransformerModel {
    pub(super) fn validate_independent_decode(
        &self,
        tokens: &[u32],
        seqs: &[&mut crate::traits::SequenceState],
    ) -> Result<()> {
        if !enabled(&self.config.model_type)? {
            return Ok(());
        }
        validate_runtime(
            &self.config,
            self.comm.as_ref().map_or(0, |c| c.world_size()),
            self.ep_protocol_v2,
            self.proposer.is_none() && !self.self_speculative,
        )?;
        validate_positions(
            seqs.iter().map(|s| s.seq_len),
            tokens.len(),
            self.levers.max_decode_seqs as usize,
        )?;
        ensure!(
            tokens
                .iter()
                .all(|t| (*t as usize) < self.config.vocab_size)
                && self.lora.is_none()
                && self.overlays.is_none()
                && self
                    .comm
                    .as_ref()
                    .is_some_and(|c| c.rank() == self.config.ep_rank)
                && self.config.tp_rank == self.config.ep_rank,
            "independent decode requires valid base-model tokens and actual local rank"
        );
        for (i, seq) in seqs.iter().enumerate() {
            ensure!(
                seq.slot_idx < self.levers.max_decode_seqs as usize
                    && !seqs[..i].iter().any(|s| s.slot_idx == seq.slot_idx)
                    && seq.adapter_id == 0
                    && seq.adapter_slot < 0
                    && seq.hss_window_start() == 0
                    && seq
                        .ssm_slot
                        .as_ref()
                        .is_some_and(|g| g.belongs_to(&self.ssm_pool)),
                "independent decode requires unique resident base-model slots"
            );
        }
        validate_scratch(self.buffers.sizes(), self.config.hc_mult, tokens.len())?;
        {
            use spark_runtime::kv_cache::{KvCacheDtype, SparseIndexCacheDtype};
            let cache = self.kv_cache.lock();
            ensure!(
                cache.dtype() == KvCacheDtype::Bf16
                    && (0..cache.num_layers())
                        .all(|i| cache.dtype_for_layer(i) == KvCacheDtype::Bf16)
                    && cache.config().cache_blocks_per_seq.is_none()
                    && cache
                        .sparse_index_config()
                        .is_some_and(|s| s.dtype == SparseIndexCacheDtype::Bf16),
                "independent decode requires actual resident BF16 pools and semantic index"
            );
            let capacity = (self.max_blocks_per_seq as usize)
                .checked_mul(cache.block_size())
                .ok_or_else(|| anyhow::anyhow!("independent KV extent overflow"))?;
            ensure!(
                cache.block_size() > 0 && seqs.iter().all(|s| s.seq_len < capacity),
                "independent position exceeds actual block table"
            );
        }
        let refs: Vec<_> = seqs.iter().map(|s| &**s).collect();
        let kinds: Vec<_> = (0..self.layers.len())
            .map(|i| self.config.layer_type(i))
            .collect();
        let bytes = self
            .buffers
            .scratch_bytes()
            .checked_sub(32768)
            .ok_or_else(|| anyhow::anyhow!("independent metadata arena truncated"))?;
        ensure!(
            super::ssm_indexed_decode::prepare_indexed_decode(
                Some(&self.ssm_pool),
                &refs,
                &kinds,
                self.buffers.scratch().offset(32768),
                bytes,
                self.buffers.decode_meta()
            )?
            .is_some(),
            "independent SSM pool absent"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_flag_and_all_drain_positions() {
        assert!(!parse("other", None).unwrap());
        assert!(parse("glm5_next", Some("1")).unwrap());
        assert!(parse("other", Some("1")).is_err());
        assert!(parse("glm5_next", Some("yes")).is_err());
        for cap in 2..=8 {
            for rows in 1..=cap {
                assert!(validate_positions(vec![2047; rows], rows, cap).is_ok());
                assert!(validate_positions(vec![2048; rows], rows, cap).is_err());
            }
        }
        for rows in [0, 9] {
            assert!(validate_positions(vec![0; rows], rows, 8).is_err());
        }
        assert!(validate_projection_handles([1; 7], 1).is_ok());
        for i in 0..7 {
            let mut h = [1; 7];
            h[i] = 0;
            assert!(validate_projection_handles(h, 1).is_err());
        }
    }
}
