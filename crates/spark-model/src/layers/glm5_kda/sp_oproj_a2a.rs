// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_SP_KDA_OPROJ_A2A=1`: a four-rank sequence-parallel KDA layer
//! runs its o_proj on this rank's rows over every head.
//!
//! The TP o_proj is row-parallel: each rank multiplies all N rows of its 16
//! heads by its K slice of the weight, and the BF16 partials meet in a
//! reduce-scatter (8 KiB a row at hidden 4096). Here each rank casts its
//! heads of every row to MXFP8 (the cast the W8A8 o_proj makes anyway, a
//! 32-wide block along K, so a K slice of 2048 casts as it would in the
//! whole row), and one all-to-all hands every rank its rows of every
//! rank's heads: E4M3 bytes and their tiled scales, ~2.1 KiB a row from each
//! origin, landing as the `[rows, 4 * 2048]` W8A8 operand. The whole o_proj
//! (`Glm5KdaWeights::o_proj_full`) then makes this rank's output rows in
//! one GEMM. The sum over heads runs in one FP32 accumulation instead of
//! four BF16 partials and their adds: not the reduce-scatter's bytes.

use super::*;
use crate::layers::glm_sp::SpRows;
use anyhow::Context;

impl Glm5KdaLayer {
    /// Whether this prefill over `sp` takes the all-to-all o_proj: the seam
    /// then reduces nothing (this rank's rows are whole).
    pub(super) fn sp_oproj_a2a(&self, sp: SpRows, ctx: &ForwardContext) -> bool {
        self.weights.o_proj_full.is_some()
            && self.mx_quant_k.0 != 0
            && sp.ranks == 4
            && sp.rows * (sp.ranks - 1) == sp.peer_rows
            && sp.rows.is_multiple_of(ops::MX_LT_TILE_ROWS)
            && (self.heads * self.dim).is_multiple_of(128)
            && !ctx.graph_capture
            && !crate::det_trace::on()
            && ctx.comm.is_some()
    }

    /// This rank's o_proj rows of `gated` (`[sp.total(), heads * dim]`, this
    /// rank's heads) into `out` (`[sp.rows, hidden]`, this rank's rows).
    /// `false`: the exchange declined the shape and nothing was launched
    /// past the cast; the caller runs the row-parallel o_proj and its
    /// reduce-scatter.
    pub(super) fn sp_oproj_all_to_all(
        &self,
        gated: DevicePtr,
        out: DevicePtr,
        sp: SpRows,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let full = self
            .weights
            .o_proj_full
            .as_ref()
            .context("GLM KDA all-to-all o_proj without the whole weight")?;
        let (rows, kl) = (sp.rows, self.heads * self.dim);
        let k = sp.ranks * kl;
        ensure!(
            full.k == k && full.n == self.hidden_size,
            "GLM KDA all-to-all o_proj: a [{}, {}] weight for [{}, {k}]",
            full.n,
            full.k,
            self.hidden_size
        );
        if ctx.gpu.op_cache().once("glm:sp_kda_oproj_a2a") {
            tracing::info!(
                "ATLAS_GLM_SP_KDA_OPROJ_A2A: KDA o_proj over every head on this rank's {rows} rows"
            );
        }
        // The cast of this rank's heads, every row: row part d (rank 3 - d's
        // rows) contiguous at d * rows of both images.
        let src = ops::mx_act_scratch(ctx.buffers, sp.total(), kl)?;
        ops::mx_quant_rows(
            ctx.gpu,
            self.mx_quant_k,
            gated,
            src,
            0..sp.total(),
            kl,
            stream,
        )?;
        // The operand: this rank's rows of every rank's heads.
        let (q_bytes, sf_bytes) = (rows * k, ops::mx_lt_scale_bytes(rows, k));
        let sf_at = q_bytes.next_multiple_of(256);
        let a = ctx
            .gpu
            .op_cache()
            .scratch(ctx.gpu, "glm_sp_kda_oproj_a2a", sf_at + sf_bytes)?;
        let operand = ops::MxAct {
            q: a,
            sf: a.offset(sf_at),
        };
        let part = (rows * kl, rows * kl / 32);
        // Row tiles of one origin's scales: 16 tiles of 512 B a 2048-wide
        // slice (`mx_lt_sf_offset`), at the origin's K tiles of each row tile.
        let sf_row = ops::mx_lt_inner_pad(kl) / 4 * 512;
        let planes = [
            spark_comm::AllToAllPlane {
                src: src.q.offset(3 * part.0).0,
                src_stride: -(part.0 as i64),
                dst: operand.q.0,
                dst_stride: kl as i64,
                rows,
                row_bytes: kl,
                pitch: k,
            },
            spark_comm::AllToAllPlane {
                src: src.sf.offset(3 * part.1).0,
                src_stride: -(part.1 as i64),
                dst: operand.sf.0,
                dst_stride: sf_row as i64,
                rows: rows / ops::MX_LT_TILE_ROWS,
                row_bytes: sf_row,
                pitch: sp.ranks * sf_row,
            },
        ];
        let comm = ctx.comm.expect("checked by sp_oproj_a2a");
        if !comm.all_to_all_async(&planes, stream)? {
            return Ok(false);
        }
        full.gemm(operand, out, rows, stream)?;
        Ok(true)
    }
}
