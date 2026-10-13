// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched long-context verify: 1..=4 owners x `rows` temporal rows in
//! one target traversal instead of one traversal per owner. `rows` is chosen
//! per call and uniform across that call's owners: 3 on the repaired MTP K3
//! lane, 2..=8 (a DFlash block) on the GLM DFlash lane.
//!
//! Rows are flat and owner-major: owner `o` holds rows `[rows*o, rows*o +
//! rows)`. Every per-owner state (KDA recurrence and its rollback snapshots,
//! MLA cache writes and causal semantic index) advances only against that
//! owner's own rows, in the same order as the single-owner verifier.
//! Stateless work (projections, mHC, norms, the FFN and the LM head) runs once
//! over all rows, so each weight is read once per step rather than once per
//! owner.

use super::{AttnMetadataDev, LayerState};
use anyhow::{Result, ensure};
use spark_runtime::buffers::BufferSizes;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Verify rows per owner on the served `--num-drafts=2` repaired MTP lane.
pub const K3_ROWS: usize = 3;
/// Widest per-owner verify block: one GLM DFlash block (bonus + drafts).
pub const MAX_OWNER_ROWS: usize = crate::speculative::glm_repair_policy::MAX_DFLASH_VERIFY_ROWS;
/// The long-context lane batches at most eight owners...
pub const MAX_OWNERS: usize = 8;
/// ...within one budget of verify rows across all owners (8x4, 6x5, 4x8).
/// Stage, metadata, EOS-ban and split-head buffers are sized to it.
pub const MAX_ROWS: usize = 32;

/// Whether one call may verify `owners` owners of `rows` rows each.
pub fn width_supported(owners: usize, rows: usize) -> bool {
    (1..=MAX_OWNERS).contains(&owners)
        && (2..=MAX_OWNER_ROWS).contains(&rows)
        && owners * rows <= MAX_ROWS
}

/// Widest per-owner block `owners` owners can verify together (0 if none).
pub fn max_rows_per_owner(owners: usize) -> usize {
    if (1..=MAX_OWNERS).contains(&owners) {
        (MAX_ROWS / owners).min(MAX_OWNER_ROWS)
    } else {
        0
    }
}

/// The uniform per-owner row count of one call, validated.
pub fn owner_rows(owners: &[GlmLongOwner<'_>]) -> Result<usize> {
    let rows = owners.first().map_or(0, |o| o.positions.len());
    ensure!(
        width_supported(owners.len(), rows) && owners.iter().all(|o| o.positions.len() == rows),
        "GLM long owner batch needs 1..={MAX_OWNERS} owners of one 2..={MAX_OWNER_ROWS} row width, at most {MAX_ROWS} rows"
    );
    Ok(rows)
}

/// `ATLAS_GLM_LONG_BATCH_VERIFY=1` opts in; anything but 0/1 is refused.
pub fn enabled() -> Result<bool> {
    let raw = std::env::var("ATLAS_GLM_LONG_BATCH_VERIFY").ok();
    ensure!(
        matches!(raw.as_deref(), None | Some("0") | Some("1")),
        "ATLAS_GLM_LONG_BATCH_VERIFY must be 0 or 1"
    );
    Ok(raw.as_deref() == Some("1"))
}

/// One owner's inputs for one layer.
pub struct GlmLongOwner<'a> {
    pub state: &'a mut (dyn LayerState + 'static),
    /// Absolute positions of this owner's rows (consecutive); its length is
    /// the call's uniform per-owner row count.
    pub positions: Vec<usize>,
    /// This owner's own attention metadata for exactly those rows.
    pub meta: AttnMetadataDev,
}

/// Row geometry shared by the stage copies.
#[derive(Clone, Copy, Debug)]
pub struct RowBytes {
    pub hidden: usize,
    pub highway: usize,
    pub post: usize,
    pub comb: usize,
    pub logits: usize,
}

impl RowBytes {
    pub fn new(hidden_size: usize, hc_mult: usize, vocab: usize) -> Self {
        Self {
            hidden: hidden_size * 2,
            highway: hc_mult * hidden_size * crate::layers::ops::hc_elem_bytes("glm5_next"),
            post: hc_mult * 4,
            comb: hc_mult * hc_mult * 4,
            logits: vocab * 2,
        }
    }
    fn total(&self) -> usize {
        MAX_ROWS * (3 * self.hidden + self.highway + self.post + self.comb + self.logits)
    }

    /// Whether every arena span the stage copies touch holds `rows` rows.
    pub fn arena_fits(&self, sizes: &BufferSizes, rows: usize) -> bool {
        [
            (sizes.hidden_states, self.hidden),
            (sizes.norm_output, self.hidden),
            (sizes.moe_output, self.hidden),
            (sizes.hc_streams, self.highway),
            (sizes.hc_post, self.post),
            (sizes.hc_comb, self.comb),
            (sizes.logits, self.logits),
        ]
        .into_iter()
        .all(|(capacity, row)| rows.checked_mul(row).is_some_and(|b| b <= capacity))
    }
}

/// Dedicated device staging, disjoint from the arena, for up to `MAX_ROWS`.
///
/// During the traversal MLA layers verify each owner in place at arena rows
/// `[0, rows)`; `highway`/`post`/`comb`/`norm` hold every owner's rows between
/// that per-owner attention and the joint FFN. After the traversal the same
/// storage (plus `hidden`/`logits`) holds each owner's final rows, which
/// `restore` copies back to rows `[0, rows)` so the unchanged single-owner
/// verdict, repair and propose tail sees exactly a single-owner verify.
#[derive(Clone, Copy, Debug)]
pub struct GlmLongStage {
    pub rows: RowBytes,
    pub hidden: DevicePtr,
    pub norm: DevicePtr,
    pub highway: DevicePtr,
    pub post: DevicePtr,
    pub comb: DevicePtr,
    pub logits: DevicePtr,
    /// Joint FFN output, one owner's rows at a time.
    pub ffn: DevicePtr,
}

impl GlmLongStage {
    pub fn alloc(gpu: &dyn GpuBackend, rows: RowBytes) -> Result<Self> {
        let base = gpu.alloc(rows.total())?;
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let ptr = base.offset(at);
            at += MAX_ROWS * bytes;
            ptr
        };
        Ok(Self {
            rows,
            hidden: take(rows.hidden),
            norm: take(rows.hidden),
            highway: take(rows.highway),
            post: take(rows.post),
            comb: take(rows.comb),
            logits: take(rows.logits),
            ffn: take(rows.hidden),
        })
    }

    /// Copy `count` rows of each `(arena, stage, row_bytes)` span between
    /// arena row `arena_row` and stage row `stage_row`.
    pub fn copy(
        &self,
        gpu: &dyn GpuBackend,
        spans: &[(DevicePtr, DevicePtr, usize)],
        arena_row: usize,
        stage_row: usize,
        count: usize,
        to_stage: bool,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            stage_row + count <= MAX_ROWS && arena_row + count <= MAX_ROWS,
            "GLM long owner stage rows out of range"
        );
        for &(arena, stage, bytes) in spans {
            let (a, s) = (
                arena.offset(arena_row * bytes),
                stage.offset(stage_row * bytes),
            );
            let (src, dst) = if to_stage { (a, s) } else { (s, a) };
            gpu.copy_d2d_async(src, dst, count * bytes, stream)?;
        }
        Ok(())
    }
}

/// The FFN of every owner's `rows` rows; returns the `[owners * rows, H]`
/// output. Numerics per width:
///
/// - `rows == 3` (repaired MTP K3): exactly the single-owner K=3 verifier's
///   arithmetic. Routed MoE runs once over all rows through the served C3
///   grouped path; other FFNs run per owner from arena rows `[0, 3)`.
/// - other widths, at most `MAX_DFLASH_VERIFY_ROWS` rows in total: the exact
///   independent grouped FFN (`FfnComponent::forward_independent`) once over
///   every row. Its arithmetic is row-independent, so each row matches the
///   single-owner DFlash block verify of the MLA layers.
/// - other widths, more rows in total: the plain grouped `forward_prefill`
///   over every row. Row-local but NOT the single-owner verifier's arithmetic
///   (like the `prefill` diagnostic); `owner` keeps it per owner.
pub fn ffn_per_owner(
    ffn: &crate::layers::FfnComponent,
    owners: usize,
    rows: usize,
    stage: &GlmLongStage,
    ctx: &super::ForwardContext,
    stream: u64,
) -> Result<DevicePtr> {
    let b = ctx.buffers;
    let total = owners * rows;
    match (ffn_mode(), ffn) {
        (FfnMode::Grouped, crate::layers::FfnComponent::Moe(_)) if rows == K3_ROWS => {
            // The served K3 MoE is the C3 grouped path; run it once over every
            // owner's rows with the same row-for-row arithmetic.
            crate::layers::moe::with_owner_rows(total as u32, || {
                ffn.forward_prefill(b.norm_output(), total, ctx, stream)
            })?;
            return Ok(b.moe_output());
        }
        (FfnMode::Grouped, _) if rows != K3_ROWS => return rows_ffn(ffn, total, ctx, stream),
        (FfnMode::Prefill, _) => {
            ffn.forward_prefill(b.norm_output(), total, ctx, stream)?;
            return Ok(b.moe_output());
        }
        _ => {}
    }
    let bytes = stage.rows.hidden;
    let input = [(b.norm_output(), stage.norm, bytes)];
    stage.copy(ctx.gpu, &input, 0, 0, total, true, stream)?;
    for owner in 0..owners {
        stage.copy(ctx.gpu, &input, 0, owner * rows, rows, false, stream)?;
        let out = if rows == K3_ROWS {
            ffn.forward_k3(b.norm_output(), ctx, stream)?;
            b.moe_output()
        } else {
            rows_ffn(ffn, rows, ctx, stream)?
        };
        let output = [(out, stage.ffn, bytes)];
        stage.copy(ctx.gpu, &output, 0, owner * rows, rows, true, stream)?;
    }
    Ok(stage.ffn)
}

/// FFN of `rows` rows at `norm_output` for a non-K3 width (see
/// [`ffn_per_owner`]): exact independent grouped when selected, else prefill.
fn rows_ffn(
    ffn: &crate::layers::FfnComponent,
    rows: usize,
    ctx: &super::ForwardContext,
    stream: u64,
) -> Result<DevicePtr> {
    let b = ctx.buffers;
    // Selected only for 2..=MAX_DFLASH_VERIFY_ROWS rows on the DFlash lane.
    if crate::model::glm_independent::ffn_rows_selected(ctx, rows)? {
        // `forward_independent` requires its input at `norm_output`.
        return ffn.forward_independent(b.norm_output(), rows, ctx, stream);
    }
    ffn.forward_prefill(b.norm_output(), rows, ctx, stream)?;
    Ok(b.moe_output())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FfnMode {
    /// Default: at K3 routed MoE jointly through the exact C3 grouped
    /// arithmetic and dense FFN layers per owner; other widths jointly (see
    /// [`ffn_per_owner`]).
    Grouped,
    /// Every FFN per owner through the single-owner FFN of its width
    /// (`forward_k3` at K3) (diagnostic).
    Owner,
    /// Generic prefill FFN over all rows. Measurably NOT the verifier's
    /// arithmetic (diagnostic only).
    Prefill,
}

/// `ATLAS_GLM_LONG_BATCH_FFN=grouped|owner|prefill`. Both ranks must run the
/// same mode (`model::startup_parity`): `owner` issues one EP reduce an owner
/// where the others issue one for every row.
pub(crate) fn ffn_mode() -> FfnMode {
    static MODE: std::sync::OnceLock<FfnMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("ATLAS_GLM_LONG_BATCH_FFN").as_deref() {
            Ok("owner") => FfnMode::Owner,
            Ok("prefill") => FfnMode::Prefill,
            _ => FfnMode::Grouped,
        },
    )
}

#[cfg(test)]
#[path = "glm_long_owner_tests.rs"]
mod tests;
