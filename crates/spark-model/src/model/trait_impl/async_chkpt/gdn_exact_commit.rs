// SPDX-License-Identifier: AGPL-3.0-only

//! The commit of `ATLAS_QWEN4EXP_EXACT_DEFER` (`layers/ops/qwen4exp_gdn_defer.rs`):
//! the accepted prefix of one sequence replayed from H0 into every GDN layer
//! its verify deferred, one launch.

use super::*;
use crate::layers::ops::qwen4exp_gdn_defer::{GdnCommitLayer, gdn_commit_layers};

impl TransformerModel {
    /// Whether a pending GDN layer is the exact deferral's (else the wyN
    /// deferral's): the exact verify bypasses the wyN arm, so the two never
    /// mark the same step.
    pub(in crate::model::trait_impl) fn gdn_pending_is_exact(&self) -> bool {
        crate::layers::qwen3_ssm::verify_exact_for(&self.levers)
    }

    /// The commit entry of GDN layer `layer_idx` for `ssm`.
    pub(in crate::model::trait_impl) fn gdn_exact_commit_entry(
        &self,
        layer_idx: usize,
        ssm: &crate::layer::SsmLayerState,
    ) -> GdnCommitLayer {
        GdnCommitLayer {
            h: ssm.h_state,
            conv: ssm.conv_state,
            stage_qkv: ssm.gdn_commit_qkv,
            stage_gb: ssm.gdn_commit_gb,
            conv_w: self.layers[layer_idx].gdn_conv_weight(),
        }
    }

    /// Replay `num_accepted` staged tokens into `layers` on `stream`.
    pub(in crate::model::trait_impl) fn commit_gdn_exact(
        &self,
        layers: &[GdnCommitLayer],
        num_accepted: usize,
        stream: u64,
    ) -> Result<()> {
        let rows: Vec<usize> = (0..num_accepted).collect();
        self.commit_gdn_exact_rows(layers, &rows, stream)
    }

    /// Replay the staged verify rows `rows` (a prefix, or on qwen3_5 also a
    /// tree's root path) into `layers` on `stream`: qwen3_5's deferred fused
    /// rows (`ATLAS_QWEN38_VERIFY_DEFER`, `qwen35_gdn_commit_rows`) where its
    /// kernels are in the set, else qwen4_exp's prefix commit.
    pub(in crate::model::trait_impl) fn commit_gdn_exact_rows(
        &self,
        layers: &[GdnCommitLayer],
        rows: &[usize],
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            layers
                .iter()
                .all(|l| !l.stage_qkv.is_null() && !l.stage_gb.is_null() && !l.conv_w.is_null()),
            "deferred GDN commit: a pending layer without staging or conv weight"
        );
        let c = &self.config;
        let q38 = crate::layers::ops::qwen38_decode_fuse::defer_on()
            && crate::layers::ops::qwen38_decode_fuse::defer_kernels(self.gpu.as_ref())
                .1
                .0
                != 0;
        if q38 {
            return crate::layers::ops::qwen38_decode_fuse::gdn_commit_rows(
                self.gpu.as_ref(),
                layers,
                rows,
                c.linear_num_key_heads as u32,
                c.linear_num_value_heads as u32,
                c.linear_key_head_dim as u32,
                c.linear_value_head_dim as u32,
                c.linear_conv_kernel_dim as u32,
                1e-6,
                stream,
            );
        }
        anyhow::ensure!(
            rows.iter().enumerate().all(|(i, &r)| r == i),
            "deferred GDN commit: rows {rows:?} are not a prefix"
        );
        let num_accepted = rows.len();
        gdn_commit_layers(
            self.gpu.as_ref(),
            layers,
            num_accepted as u32,
            c.linear_num_key_heads as u32,
            c.linear_num_value_heads as u32,
            c.linear_key_head_dim as u32,
            c.linear_value_head_dim as u32,
            c.linear_conv_kernel_dim as u32,
            1e-6,
            stream,
        )
    }
}
