// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_SP_MLA_OPROJ_AHEAD=1`: a four-rank sequence-parallel GLM MLA
//! layer's row-parallel o_proj runs a seam block at a time, each block's
//! reduce-scatter right behind it (`glm_sp_overlap::reduce_ahead`).
//!
//! The o_proj made every row's BF16 partial in one GEMM, and the FFN seam's
//! reduce-scatter started only after it: its first block's link time sat
//! exposed before the seam's first post. Here W_uv's rows are cast into a
//! `[total, heads * v]` FP8 image (`glm_owner_cast`: the per-token cast the
//! W8A8 o_proj runs anyway, per owner as W_uv runs), and the o_proj GEMM
//! runs per seam block of every rank's part on its cast rows
//! (`ops::Fp8bRows::Cast`), the block's reduce-scatter issued as soon as it
//! is written. The FFN seam over the same blocks then only waits for each.
//!
//! Exact: the W8A8 block FP8 GEMM is row-invariant (a row's words depend on
//! that row alone, at any height), and W_uv and the cast run as before.

use anyhow::{Context, Result};
use spark_runtime::gpu::DevicePtr;

use super::super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::glm_sp::{self, SpRows};
use crate::layers::{glm_sp_overlap, ops};

fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_SP_MLA_OPROJ_AHEAD").as_deref() == Ok("1"))
}

impl Qwen3AttentionLayer {
    /// This chunk's SP rows and the FFN seam's blocks when its o_proj runs
    /// ahead a block at a time (never with the all-to-all o_proj).
    pub(super) fn glm_oproj_ahead(
        &self,
        num_tokens: usize,
        ctx: &ForwardContext,
    ) -> Option<(SpRows, Vec<(usize, usize)>)> {
        let mla = self.mla.as_ref()?;
        let sp = glm_sp::current().filter(|sp| {
            requested()
                && sp.ranks > 2
                && num_tokens == sp.total()
                && self.post_attn_out_norm.is_none()
                && !ctx.graph_capture
                && !crate::det_trace::on()
        })?;
        let blocks = glm_sp_overlap::blocks(sp, ctx)?;
        blocks
            .iter()
            .all(|&(_, n)| self.mla_native_cast_serves(&mla.wo, n))
            .then_some((sp, blocks))
    }

    /// The o_proj of the cast image into `o_out` (`[sp.total(), h]`
    /// partials), a block of every part at a time, each block reduced
    /// ahead for the FFN seam over `blocks`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_oproj_reduce_ahead(
        &self,
        mla: &MlaWeights,
        o_out: DevicePtr,
        (sp, blocks): (SpRows, &[(usize, usize)]),
        nq: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (h, kl) = (h as usize, nq as usize * mla.v_dim);
        let (q, sf) = super::a2a::fp8_image(ctx, super::a2a::SRC_IMAGE, sp.total(), kl)?;
        if ctx.gpu.op_cache().once("glm:sp_mla_oproj_ahead") {
            tracing::info!(
                "ATLAS_GLM_SP_MLA_OPROJ_AHEAD: MLA o_proj a seam block at a time ({} blocks), each reduced behind it",
                blocks.len()
            );
        }
        let kernels = ops::native_kernels(ctx.gpu);
        let scratch = ops::Fp8ActScratch::of(ctx.buffers);
        glm_sp_overlap::reduce_ahead(sp, blocks, o_out, h, ctx, stream, |(r0, n)| {
            for part in 0..sp.ranks {
                let row = part * sp.rows + r0;
                let ([tn, tk], wo) = self
                    .mla_native_o_twin(mla)
                    .context("MLA o_proj ahead without the o twin")?;
                anyhow::ensure!(
                    [tn, tk] == [h, kl] && self.mla_native_cast_serves(&mla.wo, n),
                    "MLA o twin [{tn}, {tk}] for {n} cast rows of [{h}, {kl}]"
                );
                let cast = ops::Fp8bRows::Cast {
                    act: q.offset(row * kl),
                    scale: sf.offset(row * kl / 128 * 4),
                };
                ops::fp8b_w8a8_gemm(
                    ctx.gpu,
                    &kernels,
                    scratch,
                    wo,
                    cast,
                    o_out.offset(row * h * 2),
                    [n, h, kl],
                    stream,
                )?;
            }
            Ok(())
        })
    }
}
