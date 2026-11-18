// SPDX-License-Identifier: AGPL-3.0-only

use super::weights::{descriptors, mlp, tensor};
use crate::weight_map::ExpertWeight;
use anyhow::{Context, Result, ensure};
use atlas_core::{
    config::{Hy4Mlp, ModelConfig},
    hy4_reference::topology::EpPlan,
};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    weights::{WeightDtype, WeightStore},
};

/// No buffers borrowed from the generic MoE: it assumes routed width == H.
/// Checkpoint allocations stay owned by WeightStore. Only table is owned here.
pub struct Hy4Ffn<'a> {
    _store: &'a WeightStore,
    pub(super) gpu: &'a dyn GpuBackend,
    pub(super) table: DevicePtr,
    pub(super) h: usize,
    pub(super) latent: usize,
    pub(super) inter: usize,
    pub(super) shared_inter: usize,
    pub(super) ep: EpPlan,
    pub(super) dense: bool,
    pub(super) routing: atlas_core::hy4_reference::moe::Routing,
    pub(super) limit: f32,
    pub(super) eps: f32,
    pub(super) router: DevicePtr,
    pub(super) bias: DevicePtr,
    pub(super) down: DevicePtr,
    pub(super) up: DevicePtr,
    pub(super) norm: DevicePtr,
    pub(super) kernels: [KernelHandle; 5],
}

impl<'a> Hy4Ffn<'a> {
    pub fn load(
        store: &'a WeightStore,
        layer: usize,
        c: &ModelConfig,
        gpu: &'a dyn GpuBackend,
    ) -> Result<Self> {
        let hy = c.hy4.as_ref().context("Hy4 metadata required")?;
        let kind = hy
            .mlp_layer_types
            .get(layer)
            .context("Hy4 layer out of range")?;
        ensure!(
            c.tp_rank == c.ep_rank && !c.expert_tp,
            "Hy4 requires coincident ranks and whole experts"
        );
        let ep = EpPlan::new(c.ep_rank, c.tp_world_size, c.ep_world_size, c.num_experts)?;
        let dense = *kind == Hy4Mlp::Dense;
        let (h, latent) = (c.hidden_size, hy.routed_expert_hidden_size);
        let p = format!("{}.mlp", c.layer_prefix(layer));
        let inter = if dense {
            c.intermediate_size
        } else {
            c.moe_intermediate_size
        };
        ensure!(
            [h, latent, inter, c.shared_expert_intermediate_size]
                .iter()
                .all(|n| *n > 0 && n.is_multiple_of(16) && u32::try_from(*n).is_ok()),
            "Hy4 NVFP4 dimensions must be positive multiples of 16"
        );
        ensure!(
            c.num_experts <= 512
                && c.num_experts_per_tok > 0
                && c.num_experts_per_tok <= c.num_experts,
            "Hy4 unsupported top-k/expert count"
        );
        ensure!(
            [
                hy.swiglu_limit as f32,
                c.routed_scaling_factor as f32,
                c.rms_norm_eps as f32
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0),
            "Hy4 invalid FFN scale/epsilon/clamp"
        );
        let mut experts = Vec::new();
        let (mut router, mut bias, mut down, mut up) = (
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
        );
        if dense {
            experts.push(mlp(store, &p, h, inter, gpu)?);
        } else {
            ensure!(
                c.scoring_func == "sigmoid" && c.use_routing_bias && c.router_logits_f32,
                "Hy4 requires FP32 sigmoid routing with correction bias"
            );
            router = tensor(
                store,
                &format!("{p}.gate.weight"),
                &[c.num_experts, h],
                WeightDtype::BF16,
            )?;
            bias = tensor(
                store,
                &format!("{p}.gate.e_score_correction_bias"),
                &[c.num_experts],
                WeightDtype::FP32,
            )?;
            down = tensor(
                store,
                &format!("{p}.routed_expert_down_proj.weight"),
                &[latent, h],
                WeightDtype::BF16,
            )?;
            up = tensor(
                store,
                &format!("{p}.routed_expert_up_proj.weight"),
                &[h, latent],
                WeightDtype::BF16,
            )?;
            for e in 0..c.num_experts {
                experts.push(if ep.owns(e) {
                    mlp(store, &format!("{p}.experts.{e}"), latent, inter, gpu)?
                } else {
                    ExpertWeight::null()
                });
            }
            experts.push(mlp(
                store,
                &format!("{p}.shared_experts"),
                h,
                c.shared_expert_intermediate_size,
                gpu,
            )?);
        }
        let norm = if hy.sandwich_norm {
            tensor(
                store,
                &format!(
                    "{p}.{}.weight",
                    if dense {
                        "post_fc2_norm"
                    } else {
                        "combine_output_norm"
                    }
                ),
                &[h],
                WeightDtype::BF16,
            )?
        } else {
            DevicePtr::NULL
        };
        let kernels = [
            gpu.kernel("hy4_ffn", "hy4_linear")?,
            gpu.kernel("hy4_ffn", "hy4_topk")?,
            gpu.kernel("hy4_ffn", "hy4_qgate_up")?,
            gpu.kernel("hy4_ffn", "hy4_qdown")?,
            gpu.kernel("hy4_ffn", "hy4_norm_add")?,
        ];
        let bytes = descriptors(&experts);
        let table = gpu.alloc(bytes.len())?;
        if let Err(e) = gpu.copy_h2d(&bytes, table) {
            let _ = gpu.free(table);
            return Err(e);
        }
        Ok(Self {
            _store: store,
            gpu,
            table,
            h,
            latent,
            inter,
            shared_inter: c.shared_expert_intermediate_size,
            ep,
            dense,
            routing: atlas_core::hy4_reference::moe::Routing {
                top_k: c.num_experts_per_tok,
                renormalize: c.norm_topk_prob,
                scaling: c.routed_scaling_factor as f32,
            },
            limit: hy.swiglu_limit as f32,
            eps: c.rms_norm_eps as f32,
            router,
            bias,
            down,
            up,
            norm,
            kernels,
        })
    }

    /// Explicit release after the compute stream is drained; retryable on failure.
    pub fn release(&mut self) -> Result<()> {
        if !self.table.is_null() {
            self.gpu.free(self.table)?;
            self.table = DevicePtr::NULL;
        }
        Ok(())
    }
}

impl Drop for Hy4Ffn<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            tracing::error!("Hy4 FFN descriptor cleanup: {e:#}");
        }
    }
}
