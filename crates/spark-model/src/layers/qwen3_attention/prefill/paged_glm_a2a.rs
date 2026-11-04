// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_SP_MLA_OPROJ_A2A=1`: a four-rank sequence-parallel GLM MLA
//! layer runs its o_proj on this rank's rows over every head.
//!
//! The row-parallel o_proj made BF16 partials of every row from this
//! rank's 16 heads (16 x 256 = 4096 values a row) and met them in the
//! seam's reduce-scatter. Here each owner's W_uv rows are cast the way the
//! FP8 block o_proj casts them anyway (`per_token_group_quant_fp8`, 1 x 128
//! groups along K: a 4096-wide K slice casts as in the whole row) into one
//! `[total, 4096]` image, and the chain all-to-all
//! (`CommBackend::all_to_all_async`) lands every rank's slice of this
//! rank's rows in the `[rows, 4 x 4096]` operand: E4M3 at its K offset, the
//! f32 group scales at their columns. The whole FP8 block o_proj
//! (`MlaNative::o_full`) then makes this rank's rows in one CUTLASS
//! blockwise GEMM, and the seam reduces nothing (`glm_sp::mark_attn_whole`).
//! The sum over heads runs in one FP32 accumulation instead of four BF16
//! partials and their adds: not the reduce-scatter's bytes.
//!
//! When the exchange declines, the rank's own twin multiplies the image
//! (every row, its heads) into the usual partials and the seam reduces them.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::glm_sp::{self, SpRows};
use crate::layers::ops;

/// The op-cache scratch of the chunk's cast W_uv rows (`glm_owner_cast`).
pub(super) const SRC_IMAGE: &str = "glm_mla_a2a_src";

/// The `[rows, k]` E4M3 image and its `[rows, k / 128]` f32 scales in one
/// op-cache scratch named `tag`.
pub(super) fn fp8_image(
    ctx: &ForwardContext,
    tag: &'static str,
    rows: usize,
    k: usize,
) -> Result<(DevicePtr, DevicePtr)> {
    let q_bytes = (rows * k).next_multiple_of(256);
    let at = ctx
        .gpu
        .op_cache()
        .scratch(ctx.gpu, tag, q_bytes + rows * k / 128 * 4)?;
    Ok((at, at.offset(q_bytes)))
}

impl Qwen3AttentionLayer {
    /// This chunk's SP rows when its MLA o_proj takes the all-to-all.
    pub(super) fn glm_oproj_a2a(&self, num_tokens: usize, ctx: &ForwardContext) -> Option<SpRows> {
        glm_sp::current().filter(|sp| {
            self.mla_native.o_full.is_some()
                && num_tokens == sp.total()
                && sp.ranks == 4
                && sp.rows * (sp.ranks - 1) == sp.peer_rows
                && sp.rows.is_multiple_of(64)
                && self.post_attn_out_norm.is_none()
                && !ctx.graph_capture
                && !crate::det_trace::on()
                && ctx.comm.is_some()
        })
    }

    /// W_uv of one owner's `rows` attention rows (`latent`) cast into rows
    /// `[row0, row0 + rows)` of the chunk's FP8 o_proj image.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_owner_cast(
        &self,
        mla: &MlaWeights,
        latent: DevicePtr,
        (row0, rows): (usize, usize),
        total: usize,
        nq: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (kv_lora, v_dim) = (mla.kv_lora_rank as u32, mla.v_dim as u32);
        let kl = (nq * v_dim) as usize;
        let v = ctx.buffers.qkv_output();
        self.glm_head_gemm(
            latent,
            mla.w_uv.weight,
            v,
            [rows as u32, nq, kv_lora, v_dim, nq * kv_lora, nq * v_dim],
            ctx,
            stream,
        )?;
        let (q, sf) = fp8_image(ctx, SRC_IMAGE, total, kl)?;
        ops::per_token_group_quant_fp8_warp(
            ctx.gpu,
            ops::native_kernels(ctx.gpu).act_quant,
            v,
            q.offset(row0 * kl),
            sf.offset(row0 * kl / 128 * 4),
            rows as u32,
            kl as u32,
            stream,
        )
    }

    /// The o_proj of the cast image into `o_out` (`[sp.total(), h]`): this
    /// rank's rows whole after the all-to-all (marked for the seam), or,
    /// when the exchange declines, every row's partial on this rank's twin.
    pub(super) fn glm_oproj_exchange(
        &self,
        mla: &MlaWeights,
        o_out: DevicePtr,
        sp: SpRows,
        nq: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let full = self
            .mla_native
            .o_full
            .as_ref()
            .context("MLA all-to-all o_proj without the whole weight")?;
        let kl = nq as usize * mla.v_dim;
        let (rows, k) = (sp.rows, sp.ranks * kl);
        let (q, sf) = fp8_image(ctx, SRC_IMAGE, sp.total(), kl)?;
        let (a, a_sf) = fp8_image(ctx, "glm_mla_a2a_dst", rows, k)?;
        let (q_part, sf_part) = (rows * kl, rows * kl / 128 * 4);
        let sf_row = kl / 128 * 4;
        // Part d (rank 3 - d's rows) sits at d * rows of the image.
        let planes = [
            spark_comm::AllToAllPlane {
                src: q.offset(3 * q_part).0,
                src_stride: -(q_part as i64),
                dst: a.0,
                dst_stride: kl as i64,
                rows,
                row_bytes: kl,
                pitch: k,
            },
            spark_comm::AllToAllPlane {
                src: sf.offset(3 * sf_part).0,
                src_stride: -(sf_part as i64),
                dst: a_sf.0,
                dst_stride: sf_row as i64,
                rows,
                row_bytes: sf_row,
                pitch: sp.ranks * sf_row,
            },
        ];
        let comm = ctx
            .comm
            .context("MLA all-to-all o_proj without a communicator")?;
        if comm.all_to_all_async(&planes, stream)? {
            if ctx.gpu.op_cache().once("glm:sp_mla_oproj_a2a") {
                tracing::info!(
                    "ATLAS_GLM_SP_MLA_OPROJ_A2A: MLA o_proj over every head on this rank's {rows} rows"
                );
            }
            spark_runtime::cutlass::fp8_blockwise_gemm_act_weight_t(
                a.0,
                a_sf.0,
                full.data.0,
                full.scales.0,
                sp.local(o_out, h as usize).0,
                [rows as u32, h, k as u32],
                stream,
            )?;
            glm_sp::mark_attn_whole();
            return Ok(());
        }
        let ([tn, tk], own) = self
            .mla_native_o_twin(mla)
            .context("MLA all-to-all o_proj fallback without the o twin")?;
        ensure!(
            [tn, tk] == [h as usize, kl],
            "MLA o twin [{tn}, {tk}] for [{h}, {kl}]"
        );
        spark_runtime::cutlass::fp8_blockwise_gemm_act_weight_t(
            q.0,
            sf.0,
            own.data.0,
            own.scales.0,
            o_out.0,
            [sp.total() as u32, h, kl as u32],
            stream,
        )
    }
}
