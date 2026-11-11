// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MXFP8_SHARED=1`: FP4-Local's MXFP8 shared expert as
//! SparkQwen serves it, beside `SeparateSharedExpert::Mxfp8`.
//!
//! * Decode, verify and batch rows ([`Mxfp8SharedExpert::decode`]): `[gate;
//!   up]` on the `mxfp8_gemv_tc{8,16,32}_swiglu` tiers (BF16 activations,
//!   SiLU * up in the epilogue, rounded as vLLM's `silu_and_mul` rounds it),
//!   down on the plain tiers. Both are row-invariant, so a row computes the
//!   same bytes alone, in a verify or in a batch (spec == serial).
//! * Prefill ([`Mxfp8SharedExpert::prefill`]): MXFP8 W8A8 GEMMs over the same
//!   bytes under `ATLAS_QWEN4EXP_PREFILL_W8A8=1` (`ops::MxW8a8`), with the
//!   SiLU * up quantized for the down GEMM in one pass
//!   (`mxfp8_swiglu_quant_act`); otherwise the decode tiers.
//!
//! Under EP the intermediate is split over the ranks, as the reference's TP2
//! layout splits it (vLLM `Qwen2MoeMLP`: a column-parallel gate/up, a
//! row-parallel down without its reduce, then the gated partial,
//! `sigmoid(shared_expert_gate(x)) * partial`, added to the routed partial
//! before the one all-reduce): rank
//! `r` of `n` holds gate/up rows `[r * I/n, (r + 1) * I/n)` and the matching
//! down columns, and its partial joins the routed partial in the blend before
//! the EP all-reduce (or reduce-scatter); nothing is added after it
//! ([`MoeLayer::shared_joins_reduce`]). Each rank reads `1/n` of the weights.
//! The gate scales the partials, which by linearity is the gate times their
//! sum.
//!
//! There (TP=EP=2 decode, verify and batch), it runs on a side stream
//! ([`SideLane`]) forked at the MoE's entry and joined before the blend: it
//! reads only the MoE input, so it overlaps the router and the routed
//! experts. The same kernels on the same operands: the bytes do not change.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;
use crate::layers::ops::{MX_QUANT_KERNEL, MXFP8_ACT_MODULE, MxAct, MxW8a8};

/// `ATLAS_QWEN4EXP_MXFP8_SHARED` (read once): `1` loads FP4-Local's shared
/// expert as [`Mxfp8SharedExpert`].
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MXFP8_SHARED").as_deref() == Ok("1"))
}

/// The kernel module of the decode tiers and the prefill SiLU quant.
const MODULE: &str = "mxfp8_gemv";

#[derive(Debug, Clone, Copy)]
struct Kernels {
    /// `mxfp8_gemv_tc{8,16,32}`: down.
    rows: [KernelHandle; 3],
    /// `mxfp8_gemv_tc{8,16,32}_swiglu`: gate/up + SiLU * up.
    swiglu: [KernelHandle; 3],
    /// Prefill: the input's MXFP8 quant, and the SiLU's.
    quant: KernelHandle,
    swiglu_quant: KernelHandle,
}

thread_local! {
    /// The forked step's output rows on the side lane (`[base, base +
    /// bytes)`), until [`MoeLayer::end_mxfp8_shared_fork`].
    static FORKED: std::cell::Cell<Option<(DevicePtr, usize)>> =
        const { std::cell::Cell::new(None) };
}

/// Rows the side lane's own buffers hold: any decode, verify or batch step
/// (a wider one runs in line).
const SIDE_ROWS: usize = 256;

/// The EP decode's side stream for the shared expert, its fork / join
/// events and its own SiLU * up and output rows (no buffer the routed path
/// writes). One per process: the layers fork and join it in turn.
#[derive(Debug, Clone, Copy)]
struct SideLane {
    stream: u64,
    fork: u64,
    join: u64,
    act: DevicePtr,
    out: DevicePtr,
}

impl SideLane {
    fn get(gpu: &dyn GpuBackend, [hidden, inter]: [usize; 2]) -> Result<Self> {
        static LANE: std::sync::Mutex<Option<SideLane>> = std::sync::Mutex::new(None);
        let mut lane = LANE.lock().map_err(|_| anyhow::anyhow!("side lane lock"))?;
        if let Some(l) = *lane {
            return Ok(l);
        }
        let l = Self {
            stream: gpu.create_stream()?,
            fork: gpu.create_event()?,
            join: gpu.create_event()?,
            act: gpu.alloc(SIDE_ROWS * inter * 2)?,
            out: gpu.alloc(SIDE_ROWS * hidden * 2)?,
        };
        *lane = Some(l);
        Ok(l)
    }
}

/// One layer's MXFP8 shared expert; see the module docs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mxfp8SharedExpert {
    /// `[gate; up]`, `[2 * inter, hidden]`.
    gate_up: Mxfp8Proj,
    /// `[hidden, inter]`.
    down: Mxfp8Proj,
    /// The W8A8 prefill operands over the same bytes (gate/up, down).
    w8a8: Option<(MxW8a8, MxW8a8)>,
    /// Under EP (see the module docs).
    side: Option<SideLane>,
    /// This rank holds a slice: its output is a partial, blended before the
    /// EP reduce.
    split: bool,
    k: Kernels,
    hidden: usize,
    inter: usize,
}

impl Mxfp8SharedExpert {
    /// This rank's `[gate; up]` (`[2 * inter, hidden]`) and down (`[hidden,
    /// inter]`) twins; `split` when they are a `1/n` slice under EP. With
    /// `w8a8` the prefill GEMMs are tuned up to that many rows. Load time;
    /// synchronizes the default stream.
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        gate_up: Mxfp8Proj,
        down: Mxfp8Proj,
        split: bool,
        w8a8: Option<usize>,
    ) -> Result<Self> {
        let (hidden, inter) = (down.n as usize, down.k as usize);
        ensure!(
            gate_up.n as usize == 2 * inter
                && gate_up.k as usize == hidden
                && inter.is_multiple_of(ops::MXFP8_BLOCK),
            "MXFP8 shared expert: gate/up [{}, {}] and down [{hidden}, {inter}] do not pair \
             (inter a whole number of 32-value blocks)",
            gate_up.n,
            gate_up.k,
        );
        let k = |name: &str| gpu.kernel(MODULE, name);
        let k = Kernels {
            rows: [
                k("mxfp8_gemv_tc8")?,
                k("mxfp8_gemv_tc16")?,
                k("mxfp8_gemv_tc32")?,
            ],
            swiglu: [
                k("mxfp8_gemv_tc8_swiglu")?,
                k("mxfp8_gemv_tc16_swiglu")?,
                k("mxfp8_gemv_tc32_swiglu")?,
            ],
            quant: gpu.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL)?,
            swiglu_quant: k("mxfp8_swiglu_quant_act")?,
        };
        let stream = gpu.default_stream();
        let w8a8 = match w8a8 {
            Some(m_max) => {
                let gu = MxW8a8::from_twin(gpu, &gate_up.w, [2 * inter, hidden], m_max, stream)?;
                let down = ops::undo_on_err(
                    MxW8a8::from_twin(gpu, &down.w, [hidden, inter], m_max, stream),
                    || gu.free(gpu),
                )?;
                Some((gu, down))
            }
            None => None,
        };
        let side = if split {
            Some(SideLane::get(gpu, [hidden, inter])?)
        } else {
            None
        };
        Ok(Self {
            gate_up,
            down,
            w8a8,
            side,
            split,
            k,
            hidden,
            inter,
        })
    }

    /// `rows` rows of `input` (`[rows, hidden]`) into `out` (`[rows, hidden]`)
    /// on the decode tiers; `act` (`[rows, inter]` BF16) takes SiLU * up.
    pub(crate) fn decode(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: usize,
        act: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (h, i) = (self.hidden, self.inter);
        ops::mxfp8_swiglu_rows(
            gpu,
            &self.k.swiglu,
            &self.gate_up.w,
            input,
            act,
            rows,
            i,
            h,
            stream,
        )?;
        for first in (0..rows).step_by(32) {
            let m = (rows - first).min(32) as u32;
            ops::mxfp8_gemv(
                gpu,
                self.k.rows[ops::mxfp8_tier(m)],
                act.offset(first * i * 2),
                self.down.w.data,
                self.down.w.scales,
                out.offset(first * h * 2),
                m,
                h as u32,
                i as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// A prefill chunk's `rows` rows: W8A8 when installed (`mx` holds the
    /// input's MXFP8 image, then SiLU * up's, so it must fit `[rows,
    /// hidden]`; `gu` takes the `[rows, 2 * inter]` BF16 gate/up), else
    /// [`Self::decode`] (into `act`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        rows: usize,
        mx: MxAct,
        [act, gu]: [DevicePtr; 2],
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some((gate_up, down)) = self.w8a8 else {
            return self.decode(gpu, input, rows, act, out, stream);
        };
        ops::mx_quant_rows(gpu, self.k.quant, input, mx, 0..rows, self.hidden, stream)?;
        gate_up.gemm(mx, gu, rows, stream)?;
        // The gate/up GEMM has read the input's image (stream order).
        let (m, i) = (rows as u32, self.inter as u32);
        ops::mxfp8_swiglu_quant_act(gpu, self.k.swiglu_quant, gu, mx.q, mx.sf, m, i, stream)?;
        down.gemm(mx, out, rows, stream)
    }
}

impl MoeLayer {
    fn native_mxfp8_shared(&self) -> Option<&Mxfp8SharedExpert> {
        match &self.separate_shared_expert {
            Some(SeparateSharedExpert::Mxfp8Native(mx)) => Some(mx),
            _ => None,
        }
    }

    /// Install this layer's [`Mxfp8SharedExpert`]: the fused routed kernels
    /// then run without a shared expert, and this is the only one there is.
    pub(crate) fn set_native_mxfp8_shared_expert(
        &mut self,
        shared: Mxfp8SharedExpert,
    ) -> Result<()> {
        self.btile_storage.require_legacy()?;
        let sh = &self.weights.shared_expert;
        ensure!(
            sh.gate_proj.is_null()
                && sh.up_proj.is_null()
                && sh.down_proj.is_null()
                && self.separate_shared_expert.is_none(),
            "an MXFP8 shared expert replaces the layer's shared expert; it never joins one"
        );
        self.separate_shared_expert = Some(SeparateSharedExpert::Mxfp8Native(shared));
        Ok(())
    }

    /// The shared expert is this rank's slice (module docs): its output is a
    /// partial that joins the routed partial before the EP reduce, and
    /// nothing is added after the reduce.
    pub(super) fn shared_joins_reduce(&self) -> bool {
        self.native_mxfp8_shared().is_some_and(|mx| mx.split)
    }

    /// An EP decode / verify / batch step of `rows` rows: fork the MXFP8
    /// shared expert of `input` onto the side lane (module docs) and return
    /// where its partial lands, for the blend to read after
    /// [`Self::join_mxfp8_shared`]. `None`: nothing forked, the step runs it
    /// in line (`run_separate_shared_expert`).
    pub(super) fn fork_mxfp8_shared(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let Some((mx, lane)) = self
            .native_mxfp8_shared()
            .and_then(|mx| mx.side.map(|l| (mx, l)))
        else {
            return Ok(None);
        };
        if ctx.comm.is_none() || ctx.config.ep_world_size < 2 || rows > SIDE_ROWS {
            return Ok(None);
        }
        let gpu = ctx.gpu;
        gpu.record_event(lane.fork, stream)?;
        gpu.stream_wait_event(lane.stream, lane.fork)?;
        mx.decode(gpu, input, rows, lane.act, lane.out, lane.stream)?;
        gpu.record_event(lane.join, lane.stream)?;
        FORKED.with(|f| f.set(Some((lane.out, rows * mx.hidden * 2))));
        Ok(Some(lane.out))
    }

    /// The step that forked ([`Self::fork_mxfp8_shared`]) is done with the
    /// lane: `stream` waits for it (a no-op past an earlier join) and the
    /// fork is closed. Nothing forked: nothing to do.
    pub(super) fn end_mxfp8_shared_fork(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        if FORKED.with(|f| f.take()).is_some() {
            self.join_mxfp8_shared(ctx, stream)?;
        }
        Ok(())
    }

    /// `stream` waits for the forked shared expert ([`Self::fork_mxfp8_shared`]).
    pub(super) fn join_mxfp8_shared(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let lane = self
            .native_mxfp8_shared()
            .and_then(|mx| mx.side)
            .ok_or_else(|| anyhow::anyhow!("no MXFP8 shared expert side lane to join"))?;
        ctx.gpu.stream_wait_event(stream, lane.join)
    }

    /// `run_separate_shared_expert`'s arm for an [`Mxfp8SharedExpert`]: the
    /// decode tiers for `exact` (decode / verify) rows, else the prefill arm
    /// with the prefill scratch (`act`: `ssm_deinterleaved`, `gu`:
    /// `ssm_qkvz`, the activation image in the FP8 scratch).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_native_mxfp8_shared(
        &self,
        mx: &Mxfp8SharedExpert,
        input: DevicePtr,
        rows: usize,
        act: DevicePtr,
        down_out: DevicePtr,
        exact: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Forked at the step's entry: wait for the lane, whose rows are the
        // ones the caller reads (it was handed `fork_mxfp8_shared`'s output).
        if let Some((base, bytes)) = FORKED.with(|f| f.get()) {
            let end = down_out.0.checked_add((rows * mx.hidden * 2) as u64);
            ensure!(
                down_out.0 >= base.0 && end.is_some_and(|e| e <= base.0 + bytes as u64),
                "MXFP8 shared expert: forked to the side lane, but this call writes outside it"
            );
            return self.join_mxfp8_shared(ctx, stream);
        }
        if exact || mx.w8a8.is_none() {
            return mx.decode(ctx.gpu, input, rows, act, down_out, stream);
        }
        let sizes = ctx.buffers.sizes();
        ensure!(
            sizes.ssm_deinterleaved >= rows * mx.inter * 2
                && sizes.ssm_qkvz >= rows * 2 * mx.inter * 2
                && sizes.attn_output >= rows * mx.hidden * 2,
            "MXFP8 shared expert: {rows} prefill rows overrun its scratch"
        );
        let scratch = ops::mx_act_scratch(ctx.buffers, rows, mx.hidden)?;
        let bufs = [ctx.buffers.ssm_deinterleaved(), ctx.buffers.ssm_qkvz()];
        mx.prefill(ctx.gpu, input, rows, scratch, bufs, down_out, stream)
    }
}

#[cfg(test)]
#[path = "shared_mxfp8_tests.rs"]
mod tests;
