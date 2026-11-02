// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_SP_MLA_QUANT_GATHER=1`: a four-rank sequence-parallel GLM MLA
//! layer gathers its attention input already cast for the W8A8 q_a / kv_a.
//!
//! The attention seam all-gathers the BF16 normed rows (8 KiB a row at
//! hidden 4096). Their readers are q_a and kv_a, whose native FP8 block
//! prefill (`mla_native`, W8A8) casts every row with
//! `per_token_group_quant_fp8` (1 x 128 groups, row-local) before its GEMM,
//! and the semantic indexer's BF16 key, gate and head-weight projections.
//! Here each rank casts its own rows as the seam norms each block, and the
//! seam gathers, block by block, the E4M3 rows and their f32 group scales
//! (4224 bytes a row): q_a and kv_a then run their GEMMs' passes on those
//! bytes, the cast's own, instead of casting twice.
//!
//! The indexer's projections run on this rank's BF16 rows: every owner's
//! GEMM over its whole `[row0, row0 + rows)` as before, of which only this
//! rank's rows are read (a GEMM row depends on its own input row alone, and
//! the call keeps the shape and so the kernel), and the projected keys,
//! gates and weights of this rank's rows (576 bytes a row) are gathered on
//! the side stream while q_a and kv_a run.

use std::cell::Cell;

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use super::GlmChunkOwner;
use crate::layer::ForwardContext;
use crate::layers::glm_sp::SpRows;
use crate::layers::{glm_sp_overlap, ops};
use crate::weight_map::FP8_BLOCK;

fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_SP_MLA_QUANT_GATHER").as_deref() == Ok("1"))
}

/// The gathered cast rows of a chunk: E4M3 `[total, h]` and f32
/// `[total, h / 128]` group scales (`ops::Fp8bRows::Cast`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::layers::qwen3_attention) struct MlaCast {
    pub act: DevicePtr,
    pub scale: DevicePtr,
    sp: SpRows,
}

thread_local! {
    static CAST: Cell<Option<MlaCast>> = const { Cell::new(None) };
}

/// Clears the chunk attention's cast input when the attention ends (or
/// unwinds).
pub(in crate::layers::qwen3_attention) struct CastScope(());

impl Drop for CastScope {
    fn drop(&mut self) {
        CAST.with(|c| c.set(None));
    }
}

impl MlaCast {
    /// Let the chunk attention until the scope drops read its input as these
    /// cast rows.
    pub(in crate::layers::qwen3_attention) fn enter(self) -> CastScope {
        CAST.with(|c| c.set(Some(self)));
        CastScope(())
    }

    /// The bytes a row of the E4M3 rows and of the scales, as
    /// `[total, width]` BF16-width tensors for the gathers.
    pub(in crate::layers::qwen3_attention) fn widths(h: usize) -> [usize; 2] {
        [h / 2, h / FP8_BLOCK * 2]
    }

    /// Both tensors and their widths.
    pub(in crate::layers::qwen3_attention) fn tensors(&self, h: usize) -> [(DevicePtr, usize); 2] {
        let [a, s] = Self::widths(h);
        [(self.act, a), (self.scale, s)]
    }

    pub(super) fn rows(&self) -> ops::Fp8bRows {
        ops::Fp8bRows::Cast {
            act: self.act,
            scale: self.scale,
        }
    }
}

/// The cast input of the chunk attention running on this thread, if any.
pub(super) fn cast_input() -> Option<MlaCast> {
    CAST.with(Cell::get)
}

/// An `[n, width]` BF16 scratch of every row of the chunk for the gathered
/// index projections.
fn index_scratch(
    ctx: &ForwardContext,
    tag: &'static str,
    sp: SpRows,
    width: usize,
) -> Result<DevicePtr> {
    ctx.gpu
        .op_cache()
        .scratch(ctx.gpu, tag, sp.total() * width * 2)
}

/// Every row's semantic-index keys (projected and normed), pool gates and
/// head weights, gathered: an owner's rows start at `row0` of each.
#[derive(Clone, Copy)]
pub(super) struct GatheredIndex {
    keys: DevicePtr,
    gates: DevicePtr,
    weights: DevicePtr,
    key_row: usize,
    weight_row: usize,
}

impl GatheredIndex {
    /// The keys and gates of the owner at `row0`.
    pub(super) fn keys(&self, row0: usize) -> (DevicePtr, DevicePtr) {
        let at = row0 * self.key_row;
        (self.keys.offset(at), self.gates.offset(at))
    }

    /// The head weights of the owner at `row0`.
    pub(super) fn weights(&self, row0: usize) -> DevicePtr {
        self.weights.offset(row0 * self.weight_row)
    }
}

impl Qwen3AttentionLayer {
    /// The cast rows this prefill chunk of `num_tokens` rows over `sp`
    /// gathers its attention input as, when it does: the seam then gathers
    /// no BF16 rows. Every input is mirrored on every rank.
    pub(in crate::layers::qwen3_attention) fn sp_mla_cast(
        &self,
        sp: SpRows,
        num_tokens: usize,
        diag: bool,
        ctx: &ForwardContext,
    ) -> Result<Option<MlaCast>> {
        let Some(mla) = self.mla.as_ref() else {
            return Ok(None);
        };
        let h = ctx.config.hidden_size;
        let total = sp.total();
        let even = sp.rows * (sp.ranks - 1) == sp.peer_rows;
        let row_bytes = MlaCast::widths(h).map(|w| w * 2);
        let index_rows = [ctx.config.index_head_dim * 2, ctx.config.index_n_heads * 2];
        let gathers = ctx.comm.is_some_and(|c| {
            row_bytes
                .iter()
                .chain(&index_rows)
                .all(|b| c.supports_gather_async(sp.rows * b))
                && glm_sp_overlap::blocks(sp, ctx).is_none_or(|blocks| {
                    blocks
                        .iter()
                        .all(|&(_, n)| row_bytes.iter().all(|b| c.supports_gather_async(n * b)))
                })
        });
        let on = requested()
            && ctx.config.model_type == "glm5_next"
            && sp.ranks > 2
            && even
            && num_tokens == total
            && !diag
            && !ctx.graph_capture
            && !crate::det_trace::on()
            && !crate::layers::glm_packed_scope::active()
            && self.qsa.is_none()
            && self
                .hc
                .as_ref()
                .is_some_and(|hc| ops::HcVariant::of(hc).applies_block_input_norm())
            && mla.glm_indexer.is_some()
            && mla.rope == 0
            && self.glm_paged_prefill(ctx)
            && self.mla_native_cast_serves(&mla.wq_a, total)
            && self.mla_native_cast_serves(&mla.wkv_a, total)
            && h.is_multiple_of(FP8_BLOCK)
            && gathers;
        if !on {
            return Ok(None);
        }
        let (act, scale) = super::a2a::fp8_image(ctx, "glm_sp_mla_cast", total, h)?;
        if ctx.gpu.op_cache().once("glm:sp_mla_quant_gather") {
            tracing::info!(
                "ATLAS_GLM_SP_MLA_QUANT_GATHER: MLA attention input gathered as FP8 block rows + scales ({} rows a part{})",
                sp.rows,
                match glm_sp_overlap::blocks(sp, ctx) {
                    Some(b) => format!(", {} seam blocks", b.len()),
                    None => String::new(),
                }
            );
        }
        Ok(Some(MlaCast { act, scale, sp }))
    }

    /// Cast this rank's local rows `[r0, r0 + n)` of `normed`
    /// (`[total, h]`) into the same rows of `cast`.
    pub(in crate::layers::qwen3_attention) fn sp_mla_cast_rows(
        &self,
        normed: DevicePtr,
        cast: MlaCast,
        (r0, n): (usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let first = cast.sp.row0 + r0;
        ops::per_token_group_quant_fp8_warp(
            ctx.gpu,
            ops::native_kernels(ctx.gpu).act_quant,
            normed.offset(first * h * 2),
            cast.act.offset(first * h),
            cast.scale.offset(first * (h / FP8_BLOCK) * 4),
            n as u32,
            h as u32,
            stream,
        )
    }

    /// The whole seam's form of the per-block cast and gather: cast every
    /// local row of `normed`, then gather every rank's.
    pub(in crate::layers::qwen3_attention) fn sp_mla_cast_gather_whole(
        &self,
        normed: DevicePtr,
        cast: MlaCast,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.sp_mla_cast_rows(normed, cast, (0, cast.sp.rows), ctx, stream)?;
        for (ptr, width) in cast.tensors(ctx.config.hidden_size) {
            cast.sp.all_gather(ptr, width, ctx, stream)?;
        }
        Ok(())
    }

    /// The semantic-index projections of every owner from this rank's
    /// BF16 rows of `normed` (the rest unset), each owner's at its whole
    /// height, then gathered beside the compute stream until
    /// `glm_sp_overlap::join_beside`. Head weights only for owners that
    /// select, as the selection projects them.
    pub(super) fn sp_mla_index_gather(
        &self,
        owners: &[GlmChunkOwner],
        normed: DevicePtr,
        cast: MlaCast,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<GatheredIndex> {
        let indexer = self
            .mla
            .as_ref()
            .and_then(|m| m.glm_indexer.as_ref())
            .expect("checked by sp_mla_cast");
        let (c, sp) = (ctx.config, cast.sp);
        let (dim, heads, h) = (c.index_head_dim, c.index_n_heads, c.hidden_size);
        let gathered = GatheredIndex {
            keys: index_scratch(ctx, "glm_sp_mla_index_keys", sp, dim)?,
            gates: index_scratch(ctx, "glm_sp_mla_index_gates", sp, dim)?,
            weights: index_scratch(ctx, "glm_sp_mla_index_weights", sp, heads)?,
            key_row: dim * 2,
            weight_row: heads * 2,
        };
        let mut selects = false;
        for o in owners {
            let input = normed.offset(o.row0 * h * 2);
            let (keys, gates) = gathered.keys(o.row0);
            let on = o.rows as u32;
            self.glm_index_project_keys(input, on, keys, gates, ctx, stream)?;
            if !o.dense_is_exact(c.index_topk) {
                selects = true;
                self.mla_prefill_dense(
                    input,
                    &indexer.weights_proj,
                    gathered.weights(o.row0),
                    on,
                    heads as u32,
                    h as u32,
                    ctx,
                    stream,
                )?;
            }
        }
        let mut tensors = vec![(gathered.keys, dim), (gathered.gates, dim)];
        if selects {
            tensors.push((gathered.weights, heads));
        }
        glm_sp_overlap::gather_beside(sp, &tensors, ctx, stream)?;
        Ok(gathered)
    }
}

/// `ensure!` that `cast` covers `num_tokens` rows.
pub(super) fn check_rows(cast: MlaCast, num_tokens: usize) -> Result<()> {
    ensure!(
        cast.sp.total() == num_tokens,
        "GLM MLA cast input of {} rows for a {num_tokens}-row attention",
        cast.sp.total()
    );
    Ok(())
}

#[cfg(test)]
#[path = "paged_glm_sp_gather_test_gpu.rs"]
mod gpu_tests;
