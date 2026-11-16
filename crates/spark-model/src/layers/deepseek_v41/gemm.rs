// SPDX-License-Identifier: AGPL-3.0-only

//! The FP8 projections of a batch of rows as GEMMs.
//!
//! A row alone, or a few, go through the kernels of `dsv41_tc.cuh`, which
//! read a projection's weights once for every eight rows: right for a decode
//! step, and nearly the whole cost of a prefill. A batch of [`GEMM_ROWS`]
//! rows or more is rounded once into E4M3 codes (`dsv41_mx.cu`, by the
//! model's own rounding) and multiplied on cuBLASLt's block-scaled tensor
//! cores against the checkpoint's FP8 weights as they are ([`MxW8a8`]).
//!
//! The products are the ones the row-at-a-time kernels form and the sum is
//! FP32; its order is another, so a row's bytes differ from that path's the
//! way the reference's own differ from one batch shape to the next.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::hc::{self, HcSite};
use super::parts::{Held, PARTS};
use super::shape::Shape;
use super::weights::{BlockWeights, Fp8Proj, Slice, row_part};
use crate::layers::ops::{MxAct, MxW8a8, mx_lt_inner_pad, mx_lt_scale_bytes};
use crate::weight_map::Mxfp8Weight;

/// The fewest rows a batch goes through the GEMMs with.
pub const GEMM_ROWS: usize = 64;

/// Whether batches go through the GEMMs at all: `ATLAS_DSV41_NO_GEMM` keeps
/// every batch on the row-at-a-time kernels, whose bytes do not depend on
/// the rows a batch holds, to compare the two.
pub fn gemms() -> bool {
    std::env::var_os("ATLAS_DSV41_NO_GEMM").is_none()
}
/// The batch height a projection's kernel is chosen at.
const REF_ROWS: usize = 512;

/// A batch of vectors for a projection to read: BF16, a row `stride` values
/// after the last. One the ranks gathered is in pieces of `piece` values a
/// row, the next piece `piece_stride` values on; one in a single run has a
/// `piece` of 0.
#[derive(Clone, Copy)]
pub struct Input {
    pub x: DevicePtr,
    pub stride: usize,
    pub piece: usize,
    pub piece_stride: usize,
}

impl Input {
    /// Rows in one run each.
    pub fn rows(x: DevicePtr, stride: usize) -> Self {
        Self {
            x,
            stride,
            piece: 0,
            piece_stride: 0,
        }
    }
}

/// One part of a batch as a rank sends it in codes: `groups` pieces of
/// `piece` values a row, each piece's codes `[rows, piece]` one after
/// another, and then each piece's scales `[rows, piece / 32]`.
#[derive(Clone, Copy)]
pub struct Sent {
    pub groups: usize,
    pub piece: usize,
    /// Bytes of a part, and from one part to the next.
    pub bytes: usize,
    /// Where the scales start in a part.
    pub scales: usize,
}

impl Sent {
    pub fn new(rows: usize, groups: usize, piece: usize) -> Self {
        let scales = groups * rows * piece;
        Self {
            groups,
            piece,
            // On a 16-byte boundary, which the copies want.
            bytes: (scales + groups * rows * piece / 32).next_multiple_of(16),
            scales,
        }
    }
}

/// Where a batch's activation is rounded for its GEMMs: one at a time, in
/// stream order.
#[derive(Clone, Copy)]
pub struct GemmScratch {
    quant: KernelHandle,
    gathered: KernelHandle,
    q: DevicePtr,
    sf: DevicePtr,
    rows: usize,
    k: usize,
}

impl GemmScratch {
    /// Room for `rows` rows of up to `k` values.
    pub fn alloc(gpu: &dyn GpuBackend, rows: usize, k: usize) -> Result<Self> {
        Ok(Self {
            quant: gpu.kernel("dsv41_mx", "dsv41_mx_quant")?,
            gathered: gpu.kernel("dsv41_mx", "dsv41_mx_gathered")?,
            q: gpu.alloc(rows * k)?,
            sf: gpu.alloc(mx_lt_scale_bytes(rows, k))?,
            rows,
            k,
        })
    }

    /// `rows` rows of `k` values of `x`, rounded to FP8, as a GEMM operand.
    /// It holds until the next call.
    pub fn quantize(
        &self,
        gpu: &dyn GpuBackend,
        x: Input,
        rows: usize,
        k: usize,
        stream: u64,
    ) -> Result<MxAct> {
        ensure!(
            rows <= self.rows && k <= self.k && k.is_multiple_of(32) && x.piece.is_multiple_of(32),
            "dsv41_mx_quant: {rows} rows of {k} in pieces of {}, room for {} of {}",
            x.piece,
            self.rows,
            self.k
        );
        KernelLaunch::new(gpu, self.quant)
            .grid([div_ceil((rows * k / 32) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x.x)
            .arg_ptr(self.q)
            .arg_ptr(self.sf)
            .arg_u32(rows as u32)
            .arg_u32(k as u32)
            .arg_u32(x.stride as u32)
            .arg_u32(x.piece as u32)
            .arg_u64(x.piece_stride as u64)
            .arg_u32(mx_lt_inner_pad(k) as u32)
            .launch(stream)?;
        Ok(MxAct {
            q: self.q,
            sf: self.sf,
        })
    }

    /// `rows` rows of `k` values of `x` (in one run a row) as the ranks
    /// send them: `codes` (`[rows, k]` E4M3) and `scales` (`[rows, k / 32]`
    /// E8M0), half the bytes of the values.
    #[allow(clippy::too_many_arguments)]
    pub fn codes(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        codes: DevicePtr,
        scales: DevicePtr,
        rows: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(k.is_multiple_of(32), "dsv41_mx_quant: rows of {k}");
        KernelLaunch::new(gpu, self.quant)
            .grid([div_ceil((rows * k / 32) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(codes)
            .arg_ptr(scales)
            .arg_u32(rows as u32)
            .arg_u32(k as u32)
            .arg_u32(k as u32)
            .arg_u32(0)
            .arg_u64(0)
            .arg_u32(0)
            .launch(stream)
    }

    /// What the ranks gathered of [`Self::codes`] as a GEMM operand: `parts`
    /// is every part's `sent` ([`Sent`]) one after another, `rows` rows of
    /// `sent.groups` pieces of `sent.piece` values a part. It holds until the
    /// next call.
    pub fn gathered(
        &self,
        gpu: &dyn GpuBackend,
        parts: DevicePtr,
        sent: Sent,
        rows: usize,
        stream: u64,
    ) -> Result<MxAct> {
        let k = PARTS * sent.groups * sent.piece;
        ensure!(
            rows <= self.rows && k <= self.k && sent.piece.is_multiple_of(32),
            "dsv41_mx_gathered: {rows} rows of {k} in pieces of {}, room for {} of {}",
            sent.piece,
            self.rows,
            self.k
        );
        KernelLaunch::new(gpu, self.gathered)
            .grid([div_ceil((rows * k / 32) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(parts)
            .arg_ptr(self.q)
            .arg_ptr(self.sf)
            .arg_u32(rows as u32)
            .arg_u32(k as u32)
            .arg_u32(sent.piece as u32)
            .arg_u32(sent.groups as u32)
            .arg_u64(sent.bytes as u64)
            .arg_u32(mx_lt_inner_pad(k) as u32)
            .launch(stream)?;
        Ok(MxAct {
            q: self.q,
            sf: self.sf,
        })
    }

    /// `out` (`[rows, w.n]` BF16) = `x` through `w`.
    pub fn project(
        &self,
        gpu: &dyn GpuBackend,
        x: Input,
        w: &MxW8a8,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let act = self.quantize(gpu, x, rows, w.k, stream)?;
        w.gemm(act, out, rows, stream)
    }
}

/// A block's FP8 projections as GEMM operands: the weights where they are,
/// and their scales laid out again for cuBLASLt (3% of the weights' bytes).
/// The projections in parts are there for the parts a rank holds.
pub struct GemmWeights {
    pub wq_a: MxW8a8,
    pub wkv: MxW8a8,
    /// An index source's query heads, whole.
    pub index_wq_b: Option<MxW8a8>,
    pub wq_b: [Option<MxW8a8>; PARTS],
    /// One group's block of wo_a each.
    pub wo_a: Vec<Option<MxW8a8>>,
    pub wo_b: [Option<MxW8a8>; PARTS],
    /// A part's rows of the shared expert's w1, w3 and w2.
    pub shared: [Option<[MxW8a8; 3]>; PARTS],
    /// The mixing weights of the attention's site and the FFN's, each as
    /// three BF16 matrices that add up to it (`hc::split_fn`).
    pub hc: [DevicePtr; 2],
}

/// The `[n, k]` FP8 projection at `slice` as a GEMM operand.
pub(super) fn operand(
    gpu: &dyn GpuBackend,
    slice: Slice,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<MxW8a8> {
    let twin = Mxfp8Weight {
        data: slice.data,
        scales: slice.scales,
        tiled: false,
    };
    // One kernel a shape, whatever the batch height: a row's bytes do not
    // depend on the rows beside it.
    Ok(MxW8a8::from_twin(gpu, &twin, [n, k], 0, stream)?.pinned_timed(REF_ROWS, stream))
}

/// A site's FP32 mixing weights as the BF16 operand of its GEMM.
fn split(gpu: &dyn GpuBackend, s: &Shape, site: &HcSite) -> Result<DevicePtr> {
    let values = hc::mix_rows(s.hc.copies) * s.hc.copies * s.hidden;
    let mut bytes = vec![0u8; values * 4];
    gpu.copy_d2h(site.fn_, &mut bytes)?;
    let words: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let shares = hc::split_fn(&words)
        .context("a Hyper-Connection weight is not the sum of three BF16 values")?;
    let shares: Vec<u8> = shares.iter().flat_map(|b| b.to_le_bytes()).collect();
    let on_device = gpu.alloc(shares.len())?;
    gpu.copy_h2d(&shares, on_device)?;
    Ok(on_device)
}

impl GemmWeights {
    pub fn load(
        gpu: &dyn GpuBackend,
        s: &Shape,
        w: &BlockWeights,
        held: Held,
        stream: u64,
    ) -> Result<Self> {
        let a = &w.attn;
        let whole = |p: &Fp8Proj| operand(gpu, p.rows_from(0), p.n, p.k, stream);
        let part =
            |p: &Fp8Proj, part: usize| operand(gpu, row_part(p, part), p.n / PARTS, p.k, stream);
        let mut out = Self {
            wq_a: whole(&a.wq_a)?,
            wkv: whole(&a.wkv)?,
            index_wq_b: match &w.indexer {
                Some(indexer) => Some(whole(&indexer.wq_b)?),
                None => None,
            },
            wq_b: [None; PARTS],
            wo_a: vec![None; s.o_groups],
            wo_b: [None; PARTS],
            shared: [None; PARTS],
            hc: [split(gpu, s, &w.hc_attn)?, split(gpu, s, &w.hc_ffn)?],
        };
        let groups = s.o_groups / PARTS;
        for at in held.range() {
            out.wq_b[at] = Some(part(&a.wq_b, at)?);
            for group in at * groups..(at + 1) * groups {
                let rows = a.wo_a.rows_from(group * s.o_rank);
                out.wo_a[group] = Some(operand(gpu, rows, s.o_rank, a.wo_a.k, stream)?);
            }
            out.wo_b[at] = Some(part(&a.wo_b, at)?);
            out.shared[at] = Some([
                part(&w.moe.shared_w1, at)?,
                part(&w.moe.shared_w3, at)?,
                part(&w.moe.shared_w2, at)?,
            ]);
        }
        Ok(out)
    }
}
