// SPDX-License-Identifier: AGPL-3.0-only

//! Guarded independent C4 MoE with a scalar EP control and grouped opt-in.

use super::*;
use spark_runtime::buffers::BufferSizes;

fn c4_scalar_rows() -> impl Iterator<Item = usize> {
    (0..4).rev()
}

#[cfg(test)]
fn c4_moe_arenas(
    config: &atlas_core::config::ModelConfig,
    sizes: &BufferSizes,
) -> Result<[(&'static str, usize, usize); 12]> {
    independent_moe_arenas(config, sizes, 4)
}

fn independent_moe_arenas(
    config: &atlas_core::config::ModelConfig,
    sizes: &BufferSizes,
    rows: usize,
) -> Result<[(&'static str, usize, usize); 12]> {
    anyhow::ensure!((2..=8).contains(&rows), "independent MoE rows must be 2..8");
    let bytes = |factors: &[usize]| -> Result<usize> {
        factors.iter().try_fold(1usize, |n, &factor| {
            n.checked_mul(factor)
                .ok_or_else(|| anyhow::anyhow!("C4 MoE byte count overflow"))
        })
    };
    let add = |a: usize, b: usize| {
        a.checked_add(b)
            .ok_or_else(|| anyhow::anyhow!("C4 MoE byte count overflow"))
    };
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let shared = config.shared_expert_intermediate_size;
    let routes = bytes(&[rows, config.num_experts_per_tok])?;
    let output = bytes(&[rows, h, 2])?;
    let gate_up = bytes(&[routes, inter, 2])?;
    let routed_down = bytes(&[routes, h, 2])?;
    let input_elements = bytes(&[rows, h])?;
    let down_elements = bytes(&[routes, inter])?;
    let input_pack = add(input_elements / 2, input_elements / 16)?;
    let down_pack = add(down_elements / 2, down_elements / 16)?;
    let sort = add(
        bytes(&[3, routes, 4])?,
        bytes(&[add(config.num_experts, 1)?, 4])?,
    )?;
    let worklist = add(16, bytes(&[routes, inter.div_ceil(128), 8])?)?;
    Ok([
        ("normed input", sizes.norm_output, output),
        ("output", sizes.moe_output, output),
        (
            "router / sort metadata",
            sizes.gate_logits,
            sort.max(bytes(&[rows, config.num_experts, 2])?),
        ),
        ("compact worklist", sizes.moe_router_in_f32, worklist),
        ("routed gate", sizes.expert_gate_out, gate_up),
        ("routed up", sizes.expert_up_out, gate_up.max(down_pack)),
        (
            "routed down / FP4 staging",
            sizes.expert_down_out,
            routed_down.max(input_pack).max(down_pack),
        ),
        (
            "shared gate",
            sizes.ssm_deinterleaved,
            bytes(&[rows, shared, 2])?,
        ),
        ("shared up", sizes.ssm_qkvz, bytes(&[rows, shared, 2])?),
        ("shared down", sizes.attn_output, output),
        ("scalar shared gate", sizes.logits, bytes(&[shared, 2])?),
        ("routing scratch", sizes.scratch, bytes(&[routes, 8])?),
    ])
}

fn validate_c4_moe_arenas(
    config: &atlas_core::config::ModelConfig,
    sizes: &BufferSizes,
) -> Result<()> {
    validate_independent_moe_arenas(config, sizes, 4)
}

pub(super) fn validate_independent_moe_arenas(
    config: &atlas_core::config::ModelConfig,
    sizes: &BufferSizes,
    rows: usize,
) -> Result<()> {
    for (name, available, required) in independent_moe_arenas(config, sizes, rows)? {
        anyhow::ensure!(
            available >= required,
            "C4 MoE {name} requires {required} bytes, arena has {available}"
        );
    }
    Ok(())
}

/// `ATLAS_GLM_C4_GROUPED_MOE=1`: the grouped arm of [`MoeLayer::forward_c4`].
/// Both ranks must run the same value (`model::startup_parity`): it issues
/// one four-row EP reduce where the scalar control issues four.
pub(crate) fn c4_grouped_requested() -> bool {
    std::env::var("ATLAS_GLM_C4_GROUPED_MOE").as_deref() == Ok("1")
}

impl MoeLayer {
    /// Four independent rows, never sequential-token verification. The caller
    /// validates short-context/non-speculative C4 policy before layer dispatch.
    /// Scalar control issues four per-row EP reductions; grouped issues one
    /// four-row reduction. Both add the replicated shared expert only once.
    pub fn forward_c4(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.btile_input_guard(input, 4, ctx, stream)?;
        anyhow::ensure!(
            crate::model::glm_c4::enabled(&ctx.config.model_type)
                && super::prequant_fp4::c4_grouped_shape(ctx.config, 4, ctx.levers.max_decode_seqs)
                && ctx.attn_metadata.is_some_and(|m| m.num_seqs == 4)
                && self.glm_native_moe_resources(ctx)
                && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
                && !self.is_dflash_capture_layer,
            "C4 MoE requires opted-in independent TP2/EP2 GLM native NVFP4 rows"
        );
        // Both current KDA and MLA callers put all four live inputs here.
        // Neither scalar forward nor grouped forward writes this arena under
        // the no-LoRA/no-pre-expert-norm eligibility above.
        anyhow::ensure!(
            input == ctx.buffers.norm_output(),
            "C4 MoE input must use the normed row arena"
        );
        validate_c4_moe_arenas(ctx.config, ctx.buffers.sizes())?;
        let output = ctx.buffers.moe_output();
        if c4_grouped_requested() {
            anyhow::ensure!(
                self.glm_c4_grouped(ctx, 4),
                "C4 grouped MoE requires prequant kernels, sparse EP reduction and exact-M4 shared GEMV"
            );
            self.forward_prefill(input, 4, ctx, stream)?;
        } else {
            let row_bytes = ctx.config.hidden_size * 2;
            // Scalar forward writes only output row0. Preserve each completed
            // row at its final offset; processing row0 last avoids clobbering it.
            for row in c4_scalar_rows() {
                let result = self.forward(input.offset(row * row_bytes), ctx, stream)?;
                anyhow::ensure!(result == output, "scalar C4 control changed output arena");
                if row > 0 {
                    ctx.gpu.copy_d2d_async(
                        result,
                        output.offset(row * row_bytes),
                        row_bytes,
                        stream,
                    )?;
                }
            }
        }
        Ok(output)
    }

    /// Match the scalar control's GEMV arithmetic and BF16 routing logits for
    /// every independent C4 row. C3 retains its distinct dense-GEMM control.
    pub(super) fn c4_router_logits(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.independent_router_logits(input, output, 4, ctx, stream)
    }

    pub(super) fn independent_router_logits(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let experts = ctx.config.num_experts;
        // One weight pass for every row; per-row arithmetic equals dense_gemv.
        if self.dense_gemv_batchm.0 != 0
            && (1..=ops::DENSE_GEMV_BATCHM_MAX_M as usize).contains(&rows)
        {
            return ops::dense_gemv_batchm(
                ctx.gpu,
                self.dense_gemv_batchm,
                input,
                &self.weights.gate,
                output,
                rows as u32,
                experts as u32,
                h as u32,
                experts as u32,
                stream,
            );
        }
        for row in 0..rows {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv,
                input.offset(row * h * 2),
                &self.weights.gate,
                output.offset(row * experts * 2),
                experts as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn c4_shared_expert(
        &self,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.independent_shared_batchm(input, gate_out, up_out, down_out, 4, h, inter, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn independent_shared_batchm(
        &self,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        rows: u32,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let kernel = self.w4a16_batchm.kernel(rows);
        anyhow::ensure!(
            kernel.0 != 0,
            "C4 shared expert exact-M4 GEMV is unavailable"
        );
        for (weight, output) in [
            (&self.weights.shared_expert.gate_proj, gate_out),
            (&self.weights.shared_expert.up_proj, up_out),
        ] {
            ops::w4a16_gemv_batchm(
                ctx.gpu, kernel, input, weight, output, rows, inter, h, stream,
            )?;
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            gate_out,
            up_out,
            gate_out,
            rows * inter,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kernel,
            gate_out,
            &self.weights.shared_expert.down_proj,
            down_out,
            rows,
            h,
            inter,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "forward_c4_tests.rs"]
mod tests;
