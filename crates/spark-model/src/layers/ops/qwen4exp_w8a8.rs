// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_QWEN4EXP_PREFILL_W8A8=1`: FP4-Local's MXFP8 prefill projections as
//! MXFP8 W8A8 GEMMs ([`super::MxW8a8`]) instead of BF16 GEMMs over their
//! exact BF16 expansion (`mxfp8_proj`'s GEMM tier), SparkQwen's prefill. The
//! W8A8 weight IS the checkpoint's E4M3 bytes; only its scales are relaid,
//! once at load. Not under `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ`, which exists
//! to prefill at decode precision. Unset keeps the BF16 prefill. Inert on the
//! BF16/NVFP4 releases (nothing installs it), so their prefill is unchanged.
//!
//! Which projections: the GDN qkvz (`in_proj_qkv` + `in_proj_z`) and
//! `out_proj`, the QSA q/k/v/o, each layer's set an [`MxPrefill`]. Not `in_proj_a`/`in_proj_b` (the fused BA +
//! gates kernel reads BF16; 0.2% of the GDN FLOPs), not the indexer's
//! `index_qk_proj` (its kernels read BF16; N = 320 a rank). The shared
//! expert's W8A8 operands are its own (`layers/moe/shared_mxfp8.rs`), built
//! with its twins under the same switch.
//!
//! Speed: `spark_runtime::cublaslt::fp8_gemm_act_weight_t_mx` (its doc
//! holds the measured numbers and their source).
//!
//! The activations come from the mHC collapse where it is the fused prefill
//! arm (`hc_up_mix_bf16_nt_mx`, `ATLAS_QWEN4EXP_PREFILL_HC` at >= 1024 rows:
//! its epilogue writes `y` as E4M3 + scales beside the BF16 `y`), else from
//! one quant pass ([`activation`]). Both give the same bytes, so where the
//! rows come from never shows in the output. The plumbing is two
//! thread-local slots, like the pass state of `qwen4exp_rowinv`:
//!
//! * a layer arms a [`Producer`] around its attention-site collapse; the
//!   collapse (`hc_pre_gemm`) takes the target (`take_target`) and, when
//!   every slab ran the fused kernel, reports it (`mark_produced`);
//! * [`Producer::finish`] leaves the produced rows PENDING for the first
//!   [`activation`] call, which keeps them only if it quantizes exactly the
//!   buffer, height and width the producer named, and quantizes the rest.
//!
//! Under `ATLAS_QWEN4EXP_PREFILL_SP` the collapse runs this rank's rows only
//! (from row 0 or from the slab-aligned split, both on the 128-row scale
//! tile grid) and the BF16 rows are gathered; the peer's rows are then
//! quantized here after the gather, rather than gathering E4M3 and scales
//! too (half again the wire bytes for a ~0.3 ms local pass).

use std::cell::Cell;

use anyhow::{Result, ensure};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{
    MX_LT_TILE_ROWS, MX_QUANT_KERNEL, MXFP8_ACT_MODULE, MxAct, MxW8a8, mx_act_scratch,
    mx_lt_inner_pad, mx_quant_rows, undo_on_err,
};
use crate::weight_map::Mxfp8Proj;

/// A layer's W8A8 prefill operands, one per MXFP8 projection, and the
/// activation quant they share.
#[derive(Debug, Clone, Copy)]
pub struct MxPrefill<const N: usize> {
    pub ops: [MxW8a8; N],
    pub quant: KernelHandle,
}

impl<const N: usize> MxPrefill<N> {
    /// The operands of `projs`, their row buckets tuned up to `m_max` rows.
    /// Load time; synchronizes `stream`. Frees what it built on an error.
    pub fn build(
        gpu: &dyn GpuBackend,
        projs: [&Mxfp8Proj; N],
        m_max: usize,
        stream: u64,
    ) -> Result<Self> {
        let quant = gpu.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL)?;
        let mut built: Vec<MxW8a8> = Vec::with_capacity(N);
        for p in projs {
            let dims = [p.n as usize, p.k as usize];
            let op = undo_on_err(MxW8a8::from_twin(gpu, &p.w, dims, m_max, stream), || {
                built.iter().try_for_each(|b| b.free(gpu))
            })?;
            built.push(op);
        }
        let ops = built
            .try_into()
            .map_err(|_| anyhow::anyhow!("MXFP8 W8A8: operand count"))?;
        Ok(Self { ops, quant })
    }

    /// The operand over `proj`'s weight bytes at its shape, if this set has
    /// one.
    pub fn find(&self, proj: &Mxfp8Proj) -> Option<&MxW8a8> {
        self.ops
            .iter()
            .find(|o| o.data == proj.w.data && (o.n, o.k) == (proj.n as usize, proj.k as usize))
    }

    /// `out[m, n] = x[m, k] @ W^T` on `proj`'s W8A8 operand: the activation's
    /// MXFP8 image ([`activation`]), then the GEMM. `Ok(false)`, nothing
    /// launched, when `proj` is not in this set.
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &self,
        gpu: &dyn GpuBackend,
        buffers: &BufferArena,
        proj: &Mxfp8Proj,
        x: DevicePtr,
        out: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(w) = self.find(proj) else {
            return Ok(false);
        };
        if m > 0 {
            let act = activation(gpu, self.quant, buffers, x, m, w.k, stream)?;
            w.gemm(act, out, m, stream)?;
        }
        Ok(true)
    }
}

/// `ATLAS_QWEN4EXP_PREFILL_W8A8` (read once): `1` asks for it.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse(std::env::var("ATLAS_QWEN4EXP_PREFILL_W8A8").ok().as_deref()))
}

fn parse(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Whether the loader installs W8A8 on a layer that has MXFP8 twins:
/// requested and not under `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ`.
pub fn enabled() -> bool {
    requested() && !super::qwen4exp_rowinv::bf16_proj()
}

/// Where a collapse writes its rows' MXFP8 image: `act` at the collapse's
/// row 0, `rows` x `hidden`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Target {
    pub act: MxAct,
    pub rows: u32,
    pub hidden: u32,
}

impl Target {
    /// The slab starting `t0` rows into the collapse (`t0` on the tile grid:
    /// slabs are 2048 rows).
    pub fn slab(self, t0: usize) -> Result<MxAct> {
        self.act.rows_from(t0, self.hidden as usize)
    }

    /// `round_up(hidden / 32, 4)`, the kernel's `sf_inner_pad`.
    pub fn inner_pad(self) -> u32 {
        mx_lt_inner_pad(self.hidden as usize) as u32
    }
}

/// Rows `[row0, row0 + rows)` of the `[total, hidden]` BF16 buffer `x`
/// already have their MXFP8 image in the activation scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pending {
    x: DevicePtr,
    total: usize,
    hidden: usize,
    row0: usize,
    rows: usize,
}

thread_local! {
    static TARGET: Cell<Option<Target>> = const { Cell::new(None) };
    static PRODUCED: Cell<bool> = const { Cell::new(false) };
    static PENDING: Cell<Option<Pending>> = const { Cell::new(None) };
}

/// A collapse armed to emit its rows' MXFP8 image; see the module docs.
/// Dropping it unfinished disarms it.
pub struct Producer {
    pending: Pending,
}

impl Producer {
    /// Arm the next collapse of `rows` rows x `hidden`, which writes rows
    /// `[row0, row0 + rows)` of `x` (`[total, hidden]` BF16, the input the
    /// W8A8 projections then read). `None` (and nothing armed) when `row0`
    /// is off the scale tile grid or the scratch cannot hold `total` rows;
    /// an error when the rows overrun `total` (the collapse would write past
    /// the activation image the scratch was checked for).
    pub fn arm(
        buffers: &BufferArena,
        x: DevicePtr,
        total: usize,
        row0: usize,
        rows: usize,
        hidden: usize,
    ) -> Result<Option<Self>> {
        PENDING.with(|p| p.set(None));
        PRODUCED.with(|p| p.set(false));
        ensure!(
            row0.checked_add(rows).is_some_and(|end| end <= total),
            "W8A8 producer: rows [{row0}, {row0} + {rows}) of a {total}-row pass"
        );
        let Ok(act) = mx_act_scratch(buffers, total, hidden) else {
            return Ok(None);
        };
        let Ok(act) = act.rows_from(row0, hidden) else {
            return Ok(None);
        };
        TARGET.with(|t| {
            t.set(Some(Target {
                act,
                rows: rows as u32,
                hidden: hidden as u32,
            }))
        });
        Ok(Some(Self {
            pending: Pending {
                x,
                total,
                hidden,
                row0,
                rows,
            },
        }))
    }

    /// The collapse ran: leave its rows pending if it produced them.
    pub fn finish(self) {
        if PRODUCED.with(|p| p.replace(false)) {
            PENDING.with(|p| p.set(Some(self.pending)));
        }
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        TARGET.with(|t| t.set(None));
        PRODUCED.with(|p| p.set(false));
    }
}

/// The armed target, for a collapse of exactly `rows` x `hidden` (taken:
/// one collapse writes it).
pub(crate) fn take_target(rows: u32, hidden: u32) -> Option<Target> {
    TARGET
        .with(|t| t.take())
        .filter(|t| t.rows == rows && t.hidden == hidden)
}

/// The collapse wrote every row of the target it took.
pub(crate) fn mark_produced() {
    PRODUCED.with(|p| p.set(true));
}

/// The MXFP8 image of `x` (`[m, k]` BF16, contiguous) in the activation
/// scratch: rows a collapse produced for exactly this `x`, `m` and `k` are
/// kept, every other row is quantized now. `kernel` is
/// [`super::MX_QUANT_KERNEL`].
pub fn activation(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    buffers: &BufferArena,
    x: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<MxAct> {
    let act = mx_act_scratch(buffers, m, k)?;
    let pending = PENDING.with(|p| p.take());
    for rows in quant_ranges(pending, x, m, k) {
        mx_quant_rows(gpu, kernel, x, act, rows, k, stream)?;
    }
    Ok(act)
}

/// The rows [`activation`] quantizes (empty ranges skipped): all of
/// `[0, m)`, or the complement of a matching pending range when both of its
/// pieces start on the tile grid.
fn quant_ranges(
    pending: Option<Pending>,
    x: DevicePtr,
    m: usize,
    k: usize,
) -> [std::ops::Range<usize>; 2] {
    let kept = pending.filter(|p| {
        let end = p.row0 + p.rows;
        p.x == x
            && p.total == m
            && p.hidden == k
            && end <= m
            && p.row0.is_multiple_of(MX_LT_TILE_ROWS)
            && (end == m || end.is_multiple_of(MX_LT_TILE_ROWS))
    });
    match kept {
        Some(p) => [0..p.row0, p.row0 + p.rows..m],
        None => [0..m, m..m],
    }
}

#[cfg(test)]
#[path = "qwen4exp_w8a8_tests.rs"]
mod tests;
