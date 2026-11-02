// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sparse_ep_reduce_enabled(ep_world_size: usize, has_comm: bool, has_kernel: bool) -> bool {
    ep_world_size > 1 && has_comm && has_kernel
}

/// Whether `ATLAS_GLM_MOE_UNPERMUTE_VEC=1` may take the 16-byte-access EP
/// reduce: its threads own 8 columns and it holds at most 8 routes a token.
fn unpermute_vec_eligible(flag: bool, hidden_size: usize, top_k: usize) -> bool {
    flag && hidden_size.is_multiple_of(8) && top_k <= 8
}

/// The EP unpermute-reduce kernel: `moe_unpermute_reduce_indexed_ep`, or its
/// `_vec8` twin (same launch, same bytes) where eligible and shipped.
/// GLM requires the scalar kernel; other model families use their dense reduce.
pub(super) fn unpermute_ep_kernel(
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<KernelHandle> {
    // Only the GLM target ships this sparse EP reduction. Other MoE families
    // retain the dense reduction and its zero-fill for remote expert rows.
    if config.model_type != "glm5_next" {
        return Ok(KernelHandle(0));
    }
    let max_top_k = config
        .num_experts_per_toks
        .iter()
        .fold(config.num_experts_per_tok, |a, &b| a.max(b));
    if config.model_type == "glm5_next"
        && unpermute_vec_eligible(
            std::env::var("ATLAS_GLM_MOE_UNPERMUTE_VEC").as_deref() == Ok("1"),
            config.hidden_size,
            max_top_k,
        )
    {
        let vec = super::super::try_kernel(gpu, "moe", "moe_unpermute_reduce_indexed_ep_vec8");
        if gpu.op_cache().once("moe:unpermute_vec8") {
            if vec.0 != 0 {
                tracing::info!("ATLAS_GLM_MOE_UNPERMUTE_VEC: 16-byte EP unpermute-reduce");
            } else {
                tracing::warn!(
                    "ATLAS_GLM_MOE_UNPERMUTE_VEC=1 ignored: target lacks the vec8 reduce"
                );
            }
        }
        if vec.0 != 0 {
            return Ok(vec);
        }
    }
    gpu.kernel("moe", "moe_unpermute_reduce_indexed_ep")
}

impl MoeLayer {
    fn use_sparse_ep_reduce(&self, ctx: &ForwardContext) -> bool {
        sparse_ep_reduce_enabled(
            ctx.config.ep_world_size,
            ctx.comm.is_some(),
            self.moe_unpermute_reduce_ep.0 != 0,
        )
    }

    /// Whether [`Self::prepare_ep_prefill_outputs`] zeroes the expert
    /// buffers (remote rows read by a dense reduce, or forced).
    pub(super) fn ep_prefill_zeroes(&self, ctx: &ForwardContext) -> bool {
        let force_zero = std::env::var("ATLAS_MOE_PREFILL_ZERO").ok().as_deref() == Some("1");
        force_zero || (ctx.comm.is_some() && !self.use_sparse_ep_reduce(ctx))
    }

    pub(super) fn prepare_ep_prefill_outputs(
        &self,
        total_expanded: u32,
        inter: u32,
        hidden: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let force_zero = std::env::var("ATLAS_MOE_PREFILL_ZERO").as_deref() == Ok("1");
        if !self.ep_prefill_zeroes(ctx)
            || (self.q38_routed_serves(hidden, inter, ctx) && !force_zero)
        {
            return Ok(());
        }
        let rows = total_expanded as usize;
        let gate_bytes = rows * inter as usize * 2;
        ctx.gpu
            .memset_async(ctx.buffers.expert_gate_out(), 0, gate_bytes, stream)?;
        ctx.gpu
            .memset_async(ctx.buffers.expert_up_out(), 0, gate_bytes, stream)?;
        ctx.gpu.memset_async(
            ctx.buffers.expert_down_out(),
            0,
            rows * hidden as usize * 2,
            stream,
        )
    }

    /// The unpermute-reduce into `output`; with `blend_input` (the normed MoE
    /// input) it may also apply the shared-expert blend of `attn_output` in
    /// the same launch (`ATLAS_GLM_DECODE_FUSE`), and returns whether it did.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn unpermute_ep_prefill(
        &self,
        expert_output: DevicePtr,
        output: DevicePtr,
        token_to_perm: DevicePtr,
        topk_ids: DevicePtr,
        topk_weights: DevicePtr,
        hidden: u32,
        tokens: u32,
        topk: u32,
        blend_input: Option<DevicePtr>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if self.try_q38_unpermute(
            expert_output,
            output,
            token_to_perm,
            topk_ids,
            topk_weights,
            [hidden, tokens, topk],
            ctx,
            stream,
        )? {
            return Ok(false);
        }
        if self.use_sparse_ep_reduce(ctx) {
            let (local_start, local_end) = ctx.config.local_expert_range();
            let (local_start, local_end) = (local_start as u32, local_end as u32);
            if let Some(normed) = blend_input {
                let ptrs = [
                    expert_output,
                    output,
                    token_to_perm,
                    topk_ids,
                    topk_weights,
                    ctx.buffers.attn_output(),
                    normed,
                    self.weights.shared_expert_gate.weight,
                ];
                let dims = [hidden, tokens, topk, local_start, local_end];
                if ops::glm_decode_fuse::moe_unpermute_blend(ctx.gpu, ptrs, dims, stream)? {
                    return Ok(true);
                }
            }
            ops::moe_unpermute_reduce_indexed_ep(
                ctx.gpu,
                self.moe_unpermute_reduce_ep,
                expert_output,
                output,
                token_to_perm,
                topk_ids,
                topk_weights,
                hidden,
                tokens,
                topk,
                local_start,
                local_end,
                stream,
            )?;
            return Ok(false);
        }
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce,
            expert_output,
            output,
            token_to_perm,
            topk_weights,
            hidden,
            tokens,
            topk,
            stream,
        )?;
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::{sparse_ep_reduce_enabled, unpermute_ep_kernel, unpermute_vec_eligible};
    use crate::layers::moe::MoeLayer;
    use crate::layers::moe::gate_up_repack::recording::{Event, Gpu};
    use crate::weight_map::MoeWeights;
    use atlas_core::config::ModelConfig;

    #[test]
    fn qwen_moe_constructor_does_not_require_glm_ep_kernels() {
        for ep_world_size in [1, 2] {
            let gpu = Gpu::new();
            *gpu.missing.lock().unwrap() = vec![
                "moe_unpermute_reduce_indexed_ep".into(),
                "moe_unpermute_reduce_indexed_ep_vec8".into(),
            ];
            let mut config = ModelConfig::qwen3_next_80b_nvfp4();
            config.model_type = "qwen3_5_moe".into();
            config.ep_world_size = ep_world_size;
            let layer = MoeLayer::new(
                MoeWeights::empty(config.num_experts),
                config.num_experts,
                None,
                &gpu,
                &config,
            )
            .expect("generic MoE uses its existing dense unpermute kernel");
            assert_eq!(layer.moe_unpermute_reduce_ep.0, 0);
            assert!(!gpu.trace().iter().any(|event| matches!(
                event,
                Event::Lookup(_, name) if name.starts_with("moe_unpermute_reduce_indexed_ep")
            )));
        }
    }

    #[test]
    fn glm_still_requires_its_ep_kernel() {
        for ep_world_size in [1, 2] {
            let mut config = ModelConfig::qwen3_next_80b_nvfp4();
            config.model_type = "glm5_next".into();
            config.ep_world_size = ep_world_size;
            let gpu = Gpu::new();
            assert_ne!(unpermute_ep_kernel(&gpu, &config).unwrap().0, 0);
            *gpu.missing.lock().unwrap() = vec![
                "moe_unpermute_reduce_indexed_ep".into(),
                "moe_unpermute_reduce_indexed_ep_vec8".into(),
            ];
            assert!(unpermute_ep_kernel(&gpu, &config).is_err());
        }
    }

    #[test]
    fn sparse_reduce_requires_real_ep_communication_and_kernel() {
        assert!(sparse_ep_reduce_enabled(2, true, true));
        assert!(!sparse_ep_reduce_enabled(1, true, true));
        assert!(!sparse_ep_reduce_enabled(2, false, true));
        assert!(!sparse_ep_reduce_enabled(2, true, false));
    }

    #[test]
    fn vec_reduce_needs_flag_8_column_rows_and_at_most_8_routes() {
        assert!(unpermute_vec_eligible(true, 4096, 8));
        assert!(!unpermute_vec_eligible(false, 4096, 8));
        assert!(!unpermute_vec_eligible(true, 4100, 8));
        assert!(!unpermute_vec_eligible(true, 4096, 9));
    }
}
