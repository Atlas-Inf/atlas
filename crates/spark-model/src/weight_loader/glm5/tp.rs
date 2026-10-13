// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-parallel dimension plans for GLM-5 MLA projections.

use atlas_core::config::ModelConfig;

use crate::tp_shard::TpShardKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MlaTpPlan {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub hidden: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub local_q_heads: usize,
    pub local_kv_heads: usize,
    pub q_head_dim: usize,
    pub kv_head_dim: usize,
    pub v_head_dim: usize,
}

impl MlaTpPlan {
    pub(super) fn from_config(config: &ModelConfig) -> Self {
        Self {
            tp_rank: config.tp_rank,
            tp_size: config.tp_world_size.max(1),
            hidden: config.hidden_size,
            q_lora: config.q_lora_rank,
            kv_lora: config.kv_lora_rank,
            local_q_heads: config.num_attention_heads,
            local_kv_heads: config.num_key_value_heads,
            q_head_dim: config.qk_nope_head_dim + config.qk_rope_head_dim,
            kv_head_dim: config.qk_nope_head_dim + config.v_head_dim,
            v_head_dim: config.v_head_dim,
        }
    }

    pub(super) fn q_b(self) -> (usize, usize, TpShardKind) {
        (
            self.local_q_heads * self.tp_size * self.q_head_dim,
            self.q_lora,
            TpShardKind::ColumnParallel,
        )
    }

    pub(super) fn kv_b(self) -> (usize, usize, TpShardKind) {
        (
            self.local_kv_heads * self.tp_size * self.kv_head_dim,
            self.kv_lora,
            TpShardKind::ColumnParallel,
        )
    }

    pub(super) fn o(self) -> (usize, usize, TpShardKind) {
        (
            self.hidden,
            self.local_q_heads * self.tp_size * self.v_head_dim,
            TpShardKind::RowParallel,
        )
    }

    pub(super) fn local_q_b_shape(self) -> [usize; 2] {
        [self.local_q_heads * self.q_head_dim, self.q_lora]
    }

    pub(super) fn local_kv_b_shape(self) -> [usize; 2] {
        [self.local_kv_heads * self.kv_head_dim, self.kv_lora]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glm53_tp2() -> MlaTpPlan {
        MlaTpPlan {
            tp_rank: 1,
            tp_size: 2,
            hidden: 4096,
            q_lora: 1536,
            kv_lora: 512,
            local_q_heads: 32,
            local_kv_heads: 32,
            q_head_dim: 256,
            kv_head_dim: 512,
            v_head_dim: 256,
        }
    }

    #[test]
    fn glm53_mla_tp2_matches_checkpoint_and_local_shapes() {
        let plan = glm53_tp2();
        assert_eq!(plan.q_b(), (16384, 1536, TpShardKind::ColumnParallel));
        assert_eq!(plan.kv_b(), (32768, 512, TpShardKind::ColumnParallel));
        assert_eq!(plan.o(), (4096, 16384, TpShardKind::RowParallel));
        assert_eq!(plan.local_q_b_shape(), [8192, 1536]);
        assert_eq!(plan.local_kv_b_shape(), [16384, 512]);
    }
}
