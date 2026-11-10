// SPDX-License-Identifier: AGPL-3.0-only

//! DSpine on the DFlash head (Tencent DSpine, arXiv 2609.36173; checkpoint
//! `tencent/Qwen3.8-27B-DSpine-b16`, Apache-2.0).
//!
//! A DSpine drafter is a causal DFlash block drafter (`query_causal`) whose
//! every layer ends in a code bridge that writes each block row a gated,
//! low-rank message from its predecessor's code, and whose proposals come
//! from a serial table over every row's top-16 head candidates instead of a
//! plain argmax. The rest of the propose (context K/V, attention, MLP,
//! graphs, lanes, the batched path) is the DFlash head's. Kernels:
//! `kernels/gb10/common/dspine_bridge.cu`.
//!
//! Proposals only steer which drafts the target verifies; the exact verify
//! (`ATLAS_EXACT_VERIFY`) decides every emitted token, so nothing here can
//! change output.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::weight_loader::dflash_loader::{DspineBridgeWeights, DspineSpec};
use crate::weight_map::DenseWeight;

/// Rows of the table and walk kernels' candidate lists.
const TOPK: usize = 16;
const THREADS: u32 = 256;
const TAIL_THREADS: u32 = 512;
/// Rows of one load-time table chunk.
const TABLE_CHUNK: usize = 8192;

/// The `dspine_bridge` module's entry points.
pub struct DspineKernels {
    read: KernelHandle,
    codes: KernelHandle,
    message: KernelHandle,
    write: KernelHandle,
    table: KernelHandle,
    walk: KernelHandle,
    topk: KernelHandle,
}

impl DspineKernels {
    fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |f: &str| {
            gpu.kernel("dspine_bridge", f)
                .with_context(|| format!("DSpine kernel dspine_bridge::{f}"))
        };
        Ok(Self {
            read: k("dspine_bridge_read")?,
            codes: k("dspine_bridge_codes")?,
            message: k("dspine_bridge_message")?,
            write: k("dspine_bridge_write")?,
            table: k("dspine_serial_table")?,
            walk: k("dspine_serial_walk")?,
            topk: gpu
                .kernel("dflash2_candidate_selector", "dflash2_selector_topk")
                .context("DSpine needs dflash2_candidate_selector::dflash2_selector_topk")?,
        })
    }
}

/// One propose's bridge and tail buffers for up to `rows` block rows. Each
/// propose lane and the batched path own one (captured graphs bake them).
pub struct DspineScratch {
    pub rows: usize,
    normalized: DevicePtr,
    projected: DevicePtr,
    codes: DevicePtr,
    packed: DevicePtr,
    message: DevicePtr,
    delta: DevicePtr,
    rms: DevicePtr,
    /// The last write's pre-write rows (the serial table's raw state).
    raw: DevicePtr,
    lists: DevicePtr,
    topk_scratch: DevicePtr,
    cand: DevicePtr,
    idx: DevicePtr,
}

impl DspineScratch {
    pub fn alloc(
        gpu: &dyn GpuBackend,
        spec: &DspineSpec,
        hidden: usize,
        rows: usize,
    ) -> Result<Self> {
        let (c, r) = (spec.code_size, spec.rank);
        let a = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes.max(16))?;
            gpu.memset(p, 0, bytes.max(16))?;
            Ok(p)
        };
        Ok(Self {
            rows,
            normalized: a(rows * hidden * 2)?,
            projected: a(rows * c * 2)?,
            codes: a(rows * c * 2)?,
            packed: a(rows * 3 * r * 2)?,
            message: a(rows * r * 2)?,
            delta: a(rows * hidden * 2)?,
            rms: a(rows * 4)?,
            raw: a(rows * hidden * 2)?,
            lists: a(ops::dflash2_selector_lists_bytes(rows))?,
            // Zeroed: the top-k kernel's row tickets start (and end) at zero.
            topk_scratch: a(ops::dflash2_selector_scratch_bytes(rows))?,
            cand: a(rows * TOPK * 4)?,
            idx: a(rows * TOPK * 4)?,
        })
    }
}

/// A DSpine head's bridge weights, serial tables and batched scratch.
pub struct Dspine {
    pub spec: DspineSpec,
    read: Vec<DevicePtr>,
    write: Vec<DevicePtr>,
    codebook: DevicePtr,
    packed_w: DevicePtr,
    gate_bias: DevicePtr,
    /// `[V, R]` = (head ⊙ norm) @ write_last.
    tg: DevicePtr,
    /// `[V, 2R]` = (sqrt(C) · codebook) @ [predecessor_gate; prev]ᵀ.
    tsv: DevicePtr,
    k: DspineKernels,
    /// The batched propose's scratch (`batch_capacity × γ` rows).
    pub batch: DspineScratch,
    /// Bytes the tables and the batched scratch hold.
    pub table_bytes: usize,
}

/// Where the anchors (and min_tokens bans) of the bridge's blocks live:
/// block `b` reads `ptr[b * stride]` (stride 0: one anchor for every block).
#[derive(Clone, Copy)]
pub struct DspineBlocks {
    pub anchors: DevicePtr,
    pub anchor_stride: u32,
    pub ban_depth: DevicePtr,
    pub ban_stride: u32,
}

impl Dspine {
    /// Build the bridge state for `head` (constructed without it). The
    /// serial tables are computed once from the shared target head.
    pub fn build(
        head: &BlockDiffusionDraftHead,
        w: DspineBridgeWeights,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            head.gamma >= 2 && w.read.len() == head.layers.len(),
            "DSpine: {} bridge layers for a {}-layer γ={} head",
            w.read.len(),
            head.layers.len(),
            head.gamma
        );
        anyhow::ensure!(
            w.spec.head_topk == TOPK && w.spec.rank <= 512 && w.spec.rank.is_multiple_of(32),
            "DSpine: unsupported top-{} / rank {}",
            w.spec.head_topk,
            w.spec.rank
        );
        let k = DspineKernels::resolve(gpu)?;
        let (h, v) = (head.hidden_size, head.vocab_size);
        let (c, r) = (w.spec.code_size, w.spec.rank);
        let stream = gpu.default_stream();
        let tg = gpu.alloc(v * r * 2)?;
        let tsv = gpu.alloc(v * 2 * r * 2)?;
        let scale = gpu.kernel("dspine_bridge", "dspine_scale")?;
        let chunk = gpu.alloc(TABLE_CHUNK * h.max(c) * 2)?;
        let head_rows = gpu.alloc(TABLE_CHUNK * h * 2)?;
        // write_last is [H, R]; the GEMM wants [R, H] (out = act · Wᵀ).
        let write_t = {
            let mut b = vec![0u8; h * r * 2];
            gpu.copy_d2h(*w.write.last().expect("layers"), &mut b)?;
            let mut t = vec![0u8; h * r * 2];
            for hi in 0..h {
                for ri in 0..r {
                    let (s, d) = ((hi * r + ri) * 2, (ri * h + hi) * 2);
                    t[d..d + 2].copy_from_slice(&b[s..s + 2]);
                }
            }
            let p = gpu.alloc(h * r * 2)?;
            gpu.copy_h2d(&t, p)?;
            p
        };
        let dq = crate::layers::try_kernel(gpu, "dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16");
        let launch_scale =
            |src: DevicePtr, dst: DevicePtr, n: usize, cols: u32, s: f32, col: DevicePtr| {
                KernelLaunch::new(gpu, scale)
                    .grid([1024, 1, 1])
                    .block([THREADS, 1, 1])
                    .arg_ptr(src)
                    .arg_ptr(dst)
                    .arg_u64(n as u64)
                    .arg_u32(cols)
                    .arg_f32(s)
                    .arg_ptr(col)
                    .launch(stream)
            };
        let sqrt_c = (c as f32).sqrt();
        let mut start = 0;
        while start < v {
            let n = (v - start).min(TABLE_CHUNK);
            // TG rows: bf16(head ⊙ norm) @ write_last.
            let src = match head.lm_head_nvfp4.as_ref() {
                Some(q) => {
                    anyhow::ensure!(dq.0 != 0, "DSpine: NVFP4 head needs dequant_nvfp4_to_bf16");
                    anyhow::ensure!(
                        !q.has_per_row_scale2(),
                        "DSpine: per-row NVFP4 head scales are not supported"
                    );
                    let rows = q.rows_from(start, h);
                    KernelLaunch::new(gpu, dq)
                        .grid([n as u32, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(rows.weight)
                        .arg_ptr(rows.weight_scale)
                        .arg_ptr(head_rows)
                        .arg_f32(q.weight_scale_2)
                        .arg_u32(n as u32)
                        .arg_u32(h as u32)
                        .launch(stream)?;
                    head_rows
                }
                None => head.lm_head_shared.offset(start * h * 2),
            };
            launch_scale(src, chunk, n * h, h as u32, 0.0, head.norm.weight)?;
            head.drafter_pipelined_gemm(
                gpu,
                chunk,
                &DenseWeight { weight: write_t },
                tg.offset(start * r * 2),
                n as u32,
                r as u32,
                h as u32,
                stream,
            )?;
            // TSV rows: bf16(sqrt(C) · codebook) @ [predecessor_gate; prev]ᵀ.
            launch_scale(
                w.codebook.weight.offset(start * c * 2),
                chunk,
                n * c,
                c as u32,
                sqrt_c,
                DevicePtr::NULL,
            )?;
            head.drafter_pipelined_gemm(
                gpu,
                chunk,
                &DenseWeight {
                    weight: w.packed.offset(r * c * 2),
                },
                tsv.offset(start * 2 * r * 2),
                n as u32,
                2 * r as u32,
                c as u32,
                stream,
            )?;
            start += n;
        }
        gpu.synchronize(stream)?;
        for p in [chunk, head_rows, write_t] {
            gpu.free(p)?;
        }
        let batch = DspineScratch::alloc(gpu, &w.spec, h, head.batch_capacity * head.gamma)?;
        let table_bytes = v * 3 * r * 2;
        tracing::info!(
            "DSpine serial tables built: TG [{v}, {r}] + TSV [{v}, {}] = {:.0} MiB",
            2 * r,
            table_bytes as f64 / (1 << 20) as f64
        );
        Ok(Self {
            spec: w.spec,
            read: w.read,
            write: w.write,
            codebook: w.codebook.weight,
            packed_w: w.packed,
            gate_bias: w.gate_bias.weight,
            tg,
            tsv,
            k,
            batch,
            table_bytes,
        })
    }
}

impl BlockDiffusionDraftHead {
    /// A single-sequence propose's one block: the anchor and min_tokens ban
    /// `seed_markov_prev` writes before any captured layer or tail.
    pub(super) fn dspine_single_blocks(scratch: &super::DflashScratch) -> DspineBlocks {
        DspineBlocks {
            anchors: scratch.markov_prev_dev,
            anchor_stride: 0,
            ban_depth: scratch
                .markov_prev_dev
                .offset(super::MARKOV_BAN_DEPTH_OFFSET),
            ban_stride: 0,
        }
    }

    /// The batched propose's blocks: one anchor and ban per sequence.
    pub(super) fn dspine_batch_blocks(&self) -> DspineBlocks {
        DspineBlocks {
            anchors: self.batch_markov_prev,
            anchor_stride: 1,
            ban_depth: self.batch_ban_depth(),
            ban_stride: 1,
        }
    }

    /// Layer `layer`'s code bridge on `rows` block rows of `x` (in place):
    /// the predecessor messages land on every non-anchor row. The last
    /// layer also keeps the pre-write rows for the serial table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dspine_bridge(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        x: DevicePtr,
        rows: u32,
        blocks: DspineBlocks,
        sc: &DspineScratch,
        stream: u64,
    ) -> Result<()> {
        let Some(d) = self.dspine.as_ref() else {
            return Ok(());
        };
        anyhow::ensure!(
            rows as usize <= sc.rows,
            "DSpine bridge: {rows} rows > scratch {}",
            sc.rows
        );
        let (h, c, r, g) = (
            self.hidden_size as u32,
            d.spec.code_size as u32,
            d.spec.rank as u32,
            self.gamma as u32,
        );
        let eps = d.spec.eps();
        let launch = |kernel| {
            KernelLaunch::new(gpu, kernel)
                .grid([rows, 1, 1])
                .block([THREADS, 1, 1])
        };
        launch(d.k.read)
            .arg_ptr(x)
            .arg_ptr(sc.normalized)
            .arg_ptr(sc.rms)
            .arg_u32(h)
            .arg_f32(eps)
            .launch(stream)?;
        let gemm = |src, w: DevicePtr, dst, n, k| {
            self.drafter_dense_gemm(
                gpu,
                src,
                &DenseWeight { weight: w },
                dst,
                rows,
                n,
                k,
                stream,
            )
        };
        gemm(sc.normalized, d.read[layer], sc.projected, c, h)?;
        launch(d.k.codes)
            .arg_ptr(sc.projected)
            .arg_ptr(d.codebook)
            .arg_ptr(blocks.anchors)
            .arg_u32(blocks.anchor_stride)
            .arg_ptr(sc.codes)
            .arg_u32(g)
            .arg_u32(c)
            .arg_u32(self.vocab_size as u32)
            .arg_f32(eps)
            .launch(stream)?;
        gemm(sc.codes, d.packed_w, sc.packed, 3 * r, c)?;
        launch(d.k.message)
            .arg_ptr(sc.packed)
            .arg_ptr(d.gate_bias)
            .arg_ptr(sc.message)
            .arg_u32(g)
            .arg_u32(r)
            .launch(stream)?;
        gemm(sc.message, d.write[layer], sc.delta, h, r)?;
        let stash = if layer + 1 == self.layers.len() {
            sc.raw
        } else {
            DevicePtr::NULL
        };
        launch(d.k.write)
            .arg_ptr(x)
            .arg_ptr(sc.rms)
            .arg_ptr(sc.delta)
            .arg_ptr(stash)
            .arg_u32(g)
            .arg_u32(h)
            .launch(stream)
    }

    /// The serial-table proposals of `rows` block rows from their head
    /// `logits` (`vocab` apart) into `out_tokens` (u32, one per row; row 0
    /// of a block takes its top-1, which the scheduler never reads).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dspine_tail(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        rows: u32,
        blocks: DspineBlocks,
        sc: &DspineScratch,
        out_tokens: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let d = self
            .dspine
            .as_ref()
            .context("DSpine tail on a non-DSpine head")?;
        let (h, r, g, v) = (
            self.hidden_size as u32,
            d.spec.rank as u32,
            self.gamma as u32,
            self.vocab_size as u32,
        );
        anyhow::ensure!(
            rows as usize <= sc.rows && rows.is_multiple_of(g),
            "DSpine tail: {rows} rows (scratch {}, γ {g})",
            sc.rows
        );
        ops::dflash2_selector_topk(
            gpu,
            d.k.topk,
            logits,
            v,
            v,
            0,
            rows,
            sc.lists,
            sc.topk_scratch,
            stream,
        )?;
        let (head_bf16, packed, scales, scale2) = match self.lm_head_nvfp4.as_ref() {
            Some(q) => (DevicePtr::NULL, q.weight, q.weight_scale, q.weight_scale_2),
            None => (self.lm_head_shared, DevicePtr::NULL, DevicePtr::NULL, 0.0),
        };
        let end = crate::traits::EosBan::model_end_ids();
        KernelLaunch::new(gpu, d.k.table)
            .grid([rows, 1, 1])
            .block([TAIL_THREADS, 1, 1])
            .arg_ptr(sc.lists)
            .arg_ptr(sc.raw)
            .arg_ptr(sc.rms)
            .arg_ptr(sc.packed)
            .arg_ptr(self.norm.weight)
            .arg_ptr(d.gate_bias)
            .arg_ptr(d.tg)
            .arg_ptr(d.tsv)
            .arg_ptr(blocks.anchors)
            .arg_u32(blocks.anchor_stride)
            .arg_ptr(blocks.ban_depth)
            .arg_u32(blocks.ban_stride)
            .arg_ptr(head_bf16)
            .arg_ptr(packed)
            .arg_ptr(scales)
            .arg_ptr(DevicePtr::NULL)
            .arg_f32(scale2)
            .arg_u32(h)
            .arg_ptr(sc.cand)
            .arg_ptr(sc.idx)
            .arg_u32(rows)
            .arg_u32(g)
            .arg_u32(r)
            .arg_u32(v)
            .arg_u32(end[0])
            .arg_u32(end[1])
            .arg_u32(end[2])
            .arg_u32(end[3])
            .launch(stream)?;
        KernelLaunch::new(gpu, d.k.walk)
            .grid([rows / g, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(sc.lists)
            .arg_ptr(sc.cand)
            .arg_ptr(sc.idx)
            .arg_ptr(out_tokens)
            .arg_u32(rows)
            .arg_u32(g)
            .launch(stream)
    }
}
