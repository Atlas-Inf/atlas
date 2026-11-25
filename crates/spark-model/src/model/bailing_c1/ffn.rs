// SPDX-License-Identifier: AGPL-3.0-only

//! Ling FFN block: the dense SwiGLU MLP of the two-layer prefix and the
//! 512-expert NVFP4 MoE of layers 2..41 — both on the target's
//! `moe_decode` kernels (the shared-expert GEMV pair is a general
//! limited-SwiGLU block, so the dense MLP is the same kernels with no
//! limit and no routing).

use std::sync::Arc;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

pub(crate) use super::spec::FfnKind;
use super::{LingKernels, LingScratch};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// One module's routed-expert device tables: repacked packed/scale pointers
/// and the folded scale2 values — exactly the `ling_moe_tile.cuh` contract
/// (`scale2_vals[e] = (1/input_global_scale) * (1/weight_global_scale[e])`:
/// the compressed-tensors checkpoint stores both globals as reciprocals of
/// the dequant multipliers).
pub(crate) struct LingExpertTable {
    pub packed_ptrs: DevicePtr,
    pub scale_ptrs: DevicePtr,
    pub scale2_vals: DevicePtr,
}

pub(crate) struct LingMoeWeights {
    /// FP32 router gate `[experts, hidden]` (checkpoint `mlp.gate.weight`).
    pub router: DevicePtr,
    /// FP32 per-expert routing bias `[experts]`.
    pub bias: DevicePtr,
    pub gate: LingExpertTable,
    pub up: LingExpertTable,
    pub down: LingExpertTable,
    /// `a2_gscale[e] = input_global_scale(down)` — the gate/up epilogue's
    /// re-quantizer encodes the checkpoint recipe with `g = ig`, so the
    /// intermediate's dequant global is its reciprocal.
    pub a2_gscale: DevicePtr,
    /// The RECIPROCAL of the gate/up `input_global_scale` (asserted equal
    /// across experts at load) — the dequant global of the quantized
    /// activation the route kernel writes.
    pub a_scale2: f32,
    pub shared_gate: DenseWeight,
    pub shared_up: DenseWeight,
    pub shared_down: DenseWeight,
    /// This layer's SwiGLU clamps (0 = none), from the checkpoint lists.
    pub swiglu_routed: f32,
    pub swiglu_shared: f32,
}

pub(crate) struct LingDenseWeights {
    pub gate: DenseWeight,
    pub up: DenseWeight,
    pub down: DenseWeight,
}

/// One layer's FFN: attention half adds into `hidden`, then this runs on
/// the post-attention-normed rows and writes `ffn_out` — the caller adds.
pub(crate) struct LingFfn {
    kind: FfnKind,
    dense: Option<LingDenseWeights>,
    moe: Option<LingMoeWeights>,
    hidden: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    scaling: f32,
    scratch: Arc<LingScratch>,
    kernels: Arc<LingKernels>,
}

impl LingFfn {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense(
        weights: LingDenseWeights,
        inter: usize,
        hidden: usize,
        scratch: Arc<LingScratch>,
        kernels: Arc<LingKernels>,
    ) -> Result<Self> {
        Ok(Self {
            kind: FfnKind::Dense,
            dense: Some(weights),
            moe: None,
            hidden,
            inter,
            experts: 0,
            top_k: 0,
            n_group: 0,
            topk_group: 0,
            scaling: 1.0,
            scratch,
            kernels,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe(
        weights: LingMoeWeights,
        experts: usize,
        top_k: usize,
        n_group: usize,
        topk_group: usize,
        scaling: f32,
        inter: usize,
        hidden: usize,
        scratch: Arc<LingScratch>,
        kernels: Arc<LingKernels>,
    ) -> Result<Self> {
        ensure!(experts > 0 && top_k > 0, "ling MoE: empty routing geometry");
        Ok(Self {
            kind: FfnKind::Moe,
            dense: None,
            moe: Some(weights),
            hidden,
            inter,
            experts,
            top_k,
            n_group,
            topk_group,
            scaling,
            scratch,
            kernels,
        })
    }

    pub(crate) fn kind(&self) -> FfnKind {
        self.kind
    }

    /// The launch bound of the routed worklist (worst case: one item per
    /// expert plus one 16-row token tile per top-k slot).
    fn max_items(&self, rows: usize) -> usize {
        self.experts + rows * self.top_k / 16
    }

    /// `x[rows, hidden] -> out[rows, hidden]`.
    pub(crate) fn forward(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            rows <= self.scratch.rows,
            "ling FFN: {rows} rows exceeds the scratch row cap {}",
            self.scratch.rows
        );
        match self.kind {
            FfnKind::Dense => self.forward_dense(gpu, x, out, rows, stream),
            FfnKind::Moe => self.forward_moe(gpu, x, out, rows, stream),
        }
    }

    /// The shared-expert / dense-FFN SwiGLU of a wide (>= 16 rows) pass on
    /// the house BF16 tensor cores: `H = limit-swiglu(x @ gateᵀ, x @ upᵀ)`
    /// through `ling_moe_swiglu_epilogue` (the bucket kernel's exact clamp
    /// semantics), then `y = H @ downᵀ`. `gate` stages in `h` in place;
    /// `up` borrows the MLA latent lane, idle in every FFN half.
    #[allow(clippy::too_many_arguments)]
    fn cublas_swiglu(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        gate: &DenseWeight,
        up: &DenseWeight,
        down: &DenseWeight,
        h: DevicePtr,
        y: DevicePtr,
        rows: usize,
        limit: f32,
        stream: u64,
    ) -> Result<()> {
        let k = &self.kernels;
        let s = &self.scratch;
        let m = rows as u32;
        let inter = self.inter as u32;
        let hidden = self.hidden as u32;
        ops::bf16_gemm(x, gate.weight.0, h, m, inter, hidden, stream)?;
        let up_buf = s.mla_latent;
        ops::bf16_gemm(x, up.weight.0, up_buf, m, inter, hidden, stream)?;
        KernelLaunch::new(gpu, k.moe_swiglu_epilogue_k)
            .grid([div_ceil((rows * self.inter) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(h)
            .arg_ptr(up_buf)
            .arg_ptr(h)
            .arg_u64((rows * self.inter) as u64)
            .arg_f32(limit)
            .launch(stream)?;
        ops::bf16_gemm(h, down.weight.0, y, m, hidden, inter, stream)
    }

    fn forward_dense(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let w = self.dense.as_ref().expect("dense weights");
        let k = &self.kernels;
        let s = &self.scratch;
        if rows >= 16 {
            // Prefill shape: the GEMMs read the weights once on tensor
            // cores instead of re-streaming them per 32-row bucket pass.
            self.cublas_swiglu(gpu, x, &w.gate, &w.up, &w.down, s.shared_h, out, rows, 0.0, stream)
        } else {
            for (start, batch) in row_batches(rows, 32) {
                let h = s.shared_h.offset(start as usize * self.inter * 2);
                shared_gate_up(
                    gpu,
                    k,
                    x.offset(start as usize * self.hidden * 2),
                    &w.gate,
                    &w.up,
                    h,
                    batch,
                    self.inter,
                    self.hidden,
                    0.0,
                    stream,
                )?;
                shared_down(
                    gpu,
                    k,
                    h,
                    &w.down,
                    out.offset(start as usize * self.hidden * 2),
                    batch,
                    self.hidden,
                    self.inter,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    #[allow(clippy::too_many_lines)]
    fn forward_moe(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let w = self.moe.as_ref().expect("moe weights");
        let k = &self.kernels;
        let s = &self.scratch;
        // Routed experts over full chunks: the tile kernels stream every
        // live expert's tables per pass, so splitting the chunk re-reads
        // the expert tables per split (at rows=512 the old 32-row splits
        // re-read them 16x). The worklist and all row-indexed staging are
        // sized for `scratch.rows`, which bounds chunks here.
        for (start, batch) in row_batches(rows, self.scratch.rows as u32) {
            let batch = batch as u32;
            let base = start as usize;
            let xb = x.offset(base * self.hidden * 2);
            // 1. route + input quant + group-limited top-8 + sort + worklist.
            // Stream-ordered reset: the capturing decode/verify graphs bake
            // this memset as a graph node (a graph-captured fill goes
            // through `memset_async` per the GpuBackend contract), whereas
            // `memset` synchronizes the stream — illegal inside a capture.
            gpu.memset_async(s.moe_done, 0, 4, stream)?;
            let router_ctas = div_ceil(self.experts as u32, 8);
            let groups = batch * (self.hidden as u32 / 16);
            let quant_ctas = div_ceil(groups, 256);
            KernelLaunch::new(gpu, k.moe_route_k)
                .grid([router_ctas + quant_ctas, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(xb)
                .arg_ptr(w.router)
                .arg_ptr(w.bias)
                .arg_f32(w.a_scale2)
                .arg_f32(1.0 / w.a_scale2)
                .arg_ptr(s.moe_logits.offset(base * self.experts * 4))
                .arg_ptr(s.a_packed.offset(base * (self.hidden / 2)))
                .arg_ptr(s.a_scale.offset(base * (self.hidden / 16)))
                .arg_ptr(s.topk_ids.offset(base * self.top_k * 4))
                .arg_ptr(s.topk_weights.offset(base * self.top_k * 4))
                .arg_ptr(s.sorted_tokens.offset(base * self.top_k * 4))
                .arg_ptr(s.token_to_perm.offset(base * self.top_k * 4))
                .arg_ptr(s.expert_offsets)
                .arg_ptr(s.worklist)
                .arg_u32(self.max_items(rows) as u32)
                .arg_u32(16)
                .arg_u32(batch)
                .arg_u32(self.experts as u32)
                .arg_u32(self.hidden as u32)
                .arg_u32(self.top_k as u32)
                .arg_u32(self.n_group as u32)
                .arg_u32(self.topk_group as u32)
                .arg_f32(self.scaling)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(s.moe_done)
                .launch(stream)?;
            // 2. routed experts: fused gate/up/swiglu/re-quant, then down.
            let items = self.max_items(batch as usize) as u32;
            KernelLaunch::new(gpu, k.moe_gate_up_k)
                .grid([div_ceil(self.inter as u32, 128), items, 1])
                .block([256, 1, 1])
                .arg_ptr(s.a_packed.offset(base * (self.hidden / 2)))
                .arg_ptr(s.a_scale.offset(base * (self.hidden / 16)))
                .arg_ptr(w.gate.packed_ptrs)
                .arg_ptr(w.gate.scale_ptrs)
                .arg_ptr(w.gate.scale2_vals)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(s.expert_offsets)
                .arg_ptr(s.sorted_tokens.offset(base * self.top_k * 4))
                .arg_u32(self.experts as u32)
                .arg_u32(self.inter as u32)
                .arg_u32(self.hidden as u32)
                .arg_ptr(s.worklist)
                .arg_f32(w.swiglu_routed)
                .arg_ptr(w.up.packed_ptrs)
                .arg_ptr(w.up.scale_ptrs)
                .arg_ptr(w.up.scale2_vals)
                .arg_ptr(s.gu_packed.offset(base * self.top_k * (self.inter / 2)))
                .arg_ptr(s.gu_scale.offset(base * self.top_k * (self.inter / 16)))
                .arg_ptr(w.a2_gscale)
                .launch(stream)?;
            KernelLaunch::new(gpu, k.moe_down_k)
                .grid([div_ceil(self.hidden as u32, 256), items, 1])
                .block([256, 1, 1])
                .arg_ptr(s.gu_packed.offset(base * self.top_k * (self.inter / 2)))
                .arg_ptr(s.gu_scale.offset(base * self.top_k * (self.inter / 16)))
                .arg_ptr(w.down.packed_ptrs)
                .arg_ptr(w.down.scale_ptrs)
                .arg_ptr(w.down.scale2_vals)
                .arg_ptr(s.expert_out.offset(base * self.top_k * self.hidden * 2))
                .arg_ptr(s.expert_offsets)
                .arg_ptr(DevicePtr::NULL)
                .arg_u32(self.experts as u32)
                .arg_u32(self.hidden as u32)
                .arg_u32(self.inter as u32)
                .arg_ptr(s.worklist)
                .launch(stream)?;
        }
        // 3. shared expert (BF16, its own clamp list): bucket pass below 16
        // rows; wide chunks run the GEMMs on tensor cores, weights read
        // once. Its down rows land in `out` directly.
        if rows >= 16 {
            self.cublas_swiglu(
                gpu,
                x,
                &w.shared_gate,
                &w.shared_up,
                &w.shared_down,
                s.shared_h,
                out,
                rows,
                w.swiglu_shared,
                stream,
            )?;
        } else {
            for (start, batch) in row_batches(rows, 32) {
                let batch = batch as u32;
                let base = start as usize;
                let xb = x.offset(base * self.hidden * 2);
                let yb = out.offset(base * self.hidden * 2);
                let hb = s.shared_h.offset(base * self.inter * 2);
                shared_gate_up(
                    gpu,
                    k,
                    xb,
                    &w.shared_gate,
                    &w.shared_up,
                    hb,
                    batch,
                    self.inter,
                    self.hidden,
                    w.swiglu_shared,
                    stream,
                )?;
                shared_down(
                    gpu,
                    k,
                    hb,
                    &w.shared_down,
                    yb,
                    batch,
                    self.hidden,
                    self.inter,
                    stream,
                )?;
            }
        }
        // 4. unpermute: routed reduce + shared sum -> y. The shared rows
        // already sit at yb (the shared down wrote them), and the
        // kernel reads each shared element before writing the same
        // element of Y, so Y = shared_out = yb aliases safely.
        KernelLaunch::new(gpu, k.moe_unpermute_k)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.expert_out)
            .arg_ptr(out)
            .arg_ptr(s.token_to_perm)
            .arg_ptr(s.topk_weights)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(out)
            .arg_u32(rows as u32)
            .arg_u32(self.hidden as u32)
            .arg_u32(self.top_k as u32)
            .launch(stream)
    }
}

/// `(start_row, batch_rows)` sub-batches of `rows` at `cap` width.
pub(crate) fn row_batches(rows: usize, cap: u32) -> impl Iterator<Item = (usize, u32)> {
    (0..rows)
        .step_by(cap as usize)
        .map(move |start| (start, (rows - start).min(cap as usize) as u32))
}

/// The shared-expert / dense-FFN bucket index for `batch` rows.
fn bucket(batch: u32) -> usize {
    match batch {
        1 => 0,
        2..=4 => 1,
        5..=8 => 2,
        9..=16 => 3,
        _ => 4,
    }
}

/// `H = swiglu-clamped(x @ gate^T, x @ up^T)` on the bucket kernel.
#[allow(clippy::too_many_arguments)]
fn shared_gate_up(
    gpu: &dyn GpuBackend,
    k: &LingKernels,
    x: DevicePtr,
    gate: &DenseWeight,
    up: &DenseWeight,
    out: DevicePtr,
    batch: u32,
    inter: usize,
    hidden: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.shared_gate_up_k[bucket(batch)])
        .grid([div_ceil(inter as u32, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(gate.weight)
        .arg_ptr(up.weight)
        .arg_ptr(out)
        .arg_u32(batch)
        .arg_u32(inter as u32)
        .arg_u32(hidden as u32)
        .arg_f32(limit)
        .launch(stream)
}

/// `Y = H @ down^T`.
#[allow(clippy::too_many_arguments)]
fn shared_down(
    gpu: &dyn GpuBackend,
    k: &LingKernels,
    h: DevicePtr,
    down: &DenseWeight,
    out: DevicePtr,
    batch: u32,
    hidden: usize,
    inter: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.shared_down_k[bucket(batch)])
        .grid([div_ceil(hidden as u32, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(h)
        .arg_ptr(down.weight)
        .arg_ptr(out)
        .arg_u32(batch)
        .arg_u32(hidden as u32)
        .arg_u32(inter as u32)
        .launch(stream)
}
