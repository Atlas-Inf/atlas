// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_SP_KDA_QUANT_GATHER=1`: a four-rank sequence-parallel KDA layer
//! gathers its attention input already cast and side-projected.
//!
//! The attention seam all-gathers the BF16 normed rows (8 KiB a row at
//! hidden 4096). On the W8A8 prefill path every reader of a peer row is
//! row-local: the MXFP8 cast Q, K and V share (`mxfp8_quant_act_bf16`, on
//! the 128-row scale tile grid) and the b / f_a / g_a side projections (the
//! pipelined triple, or its bytes from the packed CUTLASS GEMM). So each
//! rank casts and side-projects its own rows, and the gather moves, part by
//! part, the E4M3 rows, their tiled scales, every head's beta (b_proj is
//! TP-sharded by head, so a peer's own beta is the wrong heads: the whole
//! b_proj makes all of them, on the packed GEMM's CUTLASS tiles) and the
//! replicated f_a / g_a planes: 4864 bytes a row at GLM's shape. Q, K and V
//! then project the gathered image, this rank's heads of beta go into the
//! beta plane, and the rest of the layer reads the planes the BF16 gather
//! led to.

use super::*;
use crate::layers::glm_sp::SpRows;

fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_SP_KDA_QUANT_GATHER").as_deref() == Ok("1"))
}

/// One gathered block: `(rows' first byte, bytes a row)` of each piece, in
/// staging order.
type Pieces = [(DevicePtr, usize); 5];

impl Glm5KdaLayer {
    /// Whether this prefill over `sp` gathers its attention input cast and
    /// side-projected: the seam then gathers no BF16 rows.
    pub(super) fn sp_kda_quant_gather(&self, sp: SpRows, ctx: &ForwardContext) -> bool {
        requested()
            && sp.ranks > 2
            && sp.rows * (sp.ranks - 1) == sp.peer_rows
            && sp.rows.is_multiple_of(ops::MX_LT_TILE_ROWS)
            && !ctx.graph_capture
            && !crate::det_trace::on()
            && self.qkv_w8a8()
            && self.weights.b_proj_full.is_some()
            && ctx.config.tp_rank < sp.ranks
            && self.side_proj_rows_serve(sp.total() as u32, false)
            && self.side_proj_rows_serve(sp.rows as u32, false)
    }

    /// Where rows `row0..` of part `part`'s pieces sit: the E4M3 rows and
    /// tiled scales of the activation scratch, every head's beta
    /// (`beta_full`, `[total, ranks * heads]`), then the f_a / g_a planes of
    /// `projected` (after its three `plane_bytes` Q/K/V planes and the local
    /// beta plane). beta is per head, the rank's own heads; f_a / g_a are
    /// not. `row0` is on the 128-row scale tile grid.
    fn kda_gather_pieces(
        &self,
        sp: SpRows,
        projected: DevicePtr,
        plane_bytes: usize,
        (part, row0): (usize, usize),
        ctx: &ForwardContext,
    ) -> Result<Pieces> {
        let h = self.hidden_size;
        let row = part * sp.rows + row0;
        let act = ops::mx_act_scratch(ctx.buffers, sp.total(), h)?.rows_from(row, h)?;
        let beta = projected.offset(3 * plane_bytes);
        let fa = beta.offset(sp.total() * self.heads * 2);
        let ga = fa.offset(sp.total() * self.dim * 2);
        let all_heads = sp.ranks * self.heads * 2;
        let beta_full = beta_full_scratch(ctx, sp, all_heads)?;
        let d = self.dim * 2;
        Ok([
            (act.q, h),
            (act.sf, ops::mx_lt_inner_pad(h)),
            (beta_full.offset(row * all_heads), all_heads),
            (fa.offset(row * d), d),
            (ga.offset(row * d), d),
        ])
    }

    /// Bytes a row of the gathered pieces, and the staging of every part
    /// (`[total, that]`): each block of a part holds its rows' pieces back to
    /// back, so a block of every part is one gather of `[total, width]` rows.
    fn kda_gather_staging(&self, sp: SpRows, ctx: &ForwardContext) -> Result<(usize, DevicePtr)> {
        let h = self.hidden_size;
        let row = h + ops::mx_lt_inner_pad(h) + sp.ranks * self.heads * 2 + 4 * self.dim;
        ensure!(
            row.is_multiple_of(2),
            "GLM KDA quantized gather: a {row}-byte row is not whole BF16 elements"
        );
        let staged =
            ctx.gpu
                .op_cache()
                .scratch(ctx.gpu, "glm_sp_kda_quant_gather", sp.total() * row)?;
        Ok((row, staged))
    }

    /// The per-block form of [`Self::sp_kda_gather_quantized`] for an
    /// overlapped attention seam over `blocks`: the staging to gather (its
    /// `[total, width]` BF16-width view), when every block is on the scale
    /// tile grid. Each block's rows are then cast, side-projected and staged
    /// by [`Self::sp_kda_stage_rows`] as the seam norms them, and the seam
    /// gathers them; [`Self::sp_kda_unstage`] then places every row.
    pub(super) fn sp_kda_block_staging(
        &self,
        sp: SpRows,
        blocks: &[(usize, usize)],
        ctx: &ForwardContext,
    ) -> Result<Option<(DevicePtr, usize)>> {
        let on_grid = blocks.iter().all(|&(r0, n)| {
            r0.is_multiple_of(ops::MX_LT_TILE_ROWS)
                && n.is_multiple_of(ops::MX_LT_TILE_ROWS)
                && self.side_proj_rows_serve(n as u32, false)
        });
        if !on_grid {
            return Ok(None);
        }
        if ctx.gpu.op_cache().once("glm:sp_kda_block_gather") {
            tracing::info!(
                "ATLAS_GLM_SP_KDA_QUANT_GATHER: KDA attention input cast and gathered per seam block ({} blocks)",
                blocks.len()
            );
        }
        let (row, staged) = self.kda_gather_staging(sp, ctx)?;
        Ok(Some((staged, row / 2)))
    }

    /// Cast and side-project this rank's local rows `[r0, r0 + n)` of
    /// `normed` (`[sp.total(), h]`) and stage them for the gather.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sp_kda_stage_rows(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        plane_bytes: usize,
        sp: SpRows,
        (r0, n): (usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size;
        let mine = sp.row0 / sp.rows;
        let ours = self.kda_gather_pieces(sp, projected, plane_bytes, (mine, r0), ctx)?;
        let act = ops::mx_act_scratch(ctx.buffers, sp.total(), h)?;
        let first = sp.row0 + r0;
        ops::mx_quant_rows(
            ctx.gpu,
            self.mx_quant_k,
            normed,
            act,
            first..first + n,
            h,
            stream,
        )?;
        let input = normed.offset(first * h * 2);
        let beta_own = projected.offset(3 * plane_bytes + first * self.heads * 2);
        self.side_proj_rows(
            input,
            [beta_own, ours[3].0, ours[4].0],
            n as u32,
            false,
            ctx,
            stream,
        )?;
        // Every head's beta of these rows: the packed side GEMM's bytes
        // (`bf16_gemm_pipelined_twin`, the pipelined kernel's at any N).
        let b_full = self
            .weights
            .b_proj_full
            .as_ref()
            .expect("checked by sp_kda_quant_gather");
        let all_heads = (sp.ranks * self.heads) as u32;
        if !ops::bf16_gemm_pipelined_twin(
            input,
            b_full.weight.0,
            ours[2].0,
            [n as u32, all_heads, h as u32],
            stream,
        )? {
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_pipelined_k,
                input,
                b_full,
                ours[2].0,
                n as u32,
                all_heads,
                h as u32,
                stream,
            )?;
        }
        let (row, staged) = self.kda_gather_staging(sp, ctx)?;
        let mut at = staged.offset(first * row);
        for &(src, bytes) in &ours {
            ctx.gpu.copy_d2d_async(src, at, n * bytes, stream)?;
            at = at.offset(n * bytes);
        }
        Ok(())
    }

    /// Place every part's gathered rows of `blocks` where a BF16 gather and
    /// the whole-block cast and side projections would have left them, and
    /// this rank's heads of every row's beta into the beta plane.
    pub(super) fn sp_kda_unstage(
        &self,
        projected: DevicePtr,
        plane_bytes: usize,
        sp: SpRows,
        blocks: &[(usize, usize)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (row, staged) = self.kda_gather_staging(sp, ctx)?;
        for &(r0, n) in blocks {
            let first = self.kda_gather_pieces(sp, projected, plane_bytes, (0, r0), ctx)?;
            let mut at = staged.offset(r0 * row);
            for &(dst, bytes) in &first {
                ctx.gpu.copy_d2d_2d_async(
                    at,
                    sp.rows * row,
                    dst,
                    sp.rows * bytes,
                    n * bytes,
                    sp.ranks,
                    stream,
                )?;
                at = at.offset(n * bytes);
            }
        }
        let row = self.heads * 2;
        let beta_full = beta_full_scratch(ctx, sp, sp.ranks * row)?;
        ctx.gpu.copy_d2d_2d_async(
            beta_full.offset(ctx.config.tp_rank * row),
            sp.ranks * row,
            projected.offset(3 * plane_bytes),
            row,
            row,
            sp.total(),
            stream,
        )?;
        Ok(())
    }

    /// Cast and side-project this rank's rows of `normed` (`[sp.total(), h]`,
    /// local rows written), then gather every rank's: the activation scratch
    /// and the side planes of `projected` hold all rows, as after a BF16
    /// gather and the whole-block cast and side projections. Callers check
    /// [`Self::sp_kda_quant_gather`]. The whole seam's form of the per-block
    /// gather ([`Self::sp_kda_block_staging`]): one block of every row.
    pub(super) fn sp_kda_gather_quantized(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        plane_bytes: usize,
        sp: SpRows,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if ctx.gpu.op_cache().once("glm:sp_kda_quant_gather") {
            tracing::info!(
                "ATLAS_GLM_SP_KDA_QUANT_GATHER: KDA attention input gathered as MXFP8 rows + side planes ({} rows a part)",
                sp.rows
            );
        }
        let whole = [(0, sp.rows)];
        self.sp_kda_stage_rows(normed, projected, plane_bytes, sp, whole[0], ctx, stream)?;
        let (row, staged) = self.kda_gather_staging(sp, ctx)?;
        sp.all_gather(staged, row / 2, ctx, stream)?;
        self.sp_kda_unstage(projected, plane_bytes, sp, &whole, ctx, stream)
    }
}

/// The `[total, all_heads]` (bytes a row) every-head beta scratch.
fn beta_full_scratch(ctx: &ForwardContext, sp: SpRows, all_heads: usize) -> Result<DevicePtr> {
    ctx.gpu
        .op_cache()
        .scratch(ctx.gpu, "glm_sp_kda_beta_full", sp.total() * all_heads)
}
#[cfg(test)]
#[path = "sp_quant_gather_test_gpu.rs"]
mod gpu_tests;
