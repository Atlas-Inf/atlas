// SPDX-License-Identifier: AGPL-3.0-only

//! The token with the largest logit of each of a step's rows, without the
//! whole head (`dsv41_head.cu`): a rank reads the FP8 copy of its quarter of
//! the head, half the bytes, and the head itself only for the tokens the
//! copy cannot rule out ([`head_bound`](super::head_bound)). The pick is the
//! token `argmax_bf16` finds in the whole head's row, by the head's own
//! bytes; nothing else of the row is computed, so a step that wants more
//! than its largest logit runs the head.

use anyhow::{Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::head_bound::row_bounds;
use super::parts::{self, Held, PARTS, PartKernels};
use crate::weight_map::Mxfp8Weight;

/// Rows a pick takes: the FP8 projection's tile, and more than a verify's.
pub const ROWS: usize = 8;
/// Blocks that add a row's kept tokens, four tokens each at a time: one or
/// two tokens are kept of most rows, and of a flat row the whole part, which
/// this many blocks a row read about as fast as the head is read whole.
const BLOCKS: usize = 48;
/// Threads of a block that scans a part's tokens (`DSV41_HEAD_THREADS`).
const SCAN: u32 = 1024;
/// Bytes of a part's lower bounds, and of its blocks' best, of every row.
const LOW: usize = ROWS * 8;
const FOUND: usize = ROWS * BLOCKS * 8;
/// Head rows whose bounds the host computes at a time.
const LOAD_ROWS: usize = 2048;

#[derive(Clone, Copy)]
struct HeadKernels {
    coarse: KernelHandle,
    low: KernelHandle,
    keep: KernelHandle,
    exact: KernelHandle,
    pick: KernelHandle,
}

pub struct HeadPick {
    kernels: HeadKernels,
    held: Held,
    /// Tokens of a part, and the width of a hidden row.
    n: usize,
    hidden: usize,
    /// The head's rows of the parts held, BF16, and their FP8 copy.
    head: DevicePtr,
    copy: Mxfp8Weight,
    /// `[held, n]` FP32: a token's bound.
    bound: DevicePtr,
    /// `[held, ROWS, n]`: a row's coarse logits (FP32) and kept tokens.
    coarse: DevicePtr,
    list: DevicePtr,
    /// `[held, ROWS]`: how many tokens a row keeps.
    count: DevicePtr,
    /// Every part's [`LOW`] and [`FOUND`], as the ranks gather them.
    low: DevicePtr,
    found: DevicePtr,
    recv: DevicePtr,
    /// `[ROWS]` u32: the picks.
    picked: DevicePtr,
}

impl HeadPick {
    /// `head` and `copy` are the rows of the parts `held`, `n` tokens a
    /// part: the bound of every one of them is computed here, on the host,
    /// from the bytes on the device.
    pub fn load(
        gpu: &dyn GpuBackend,
        head: DevicePtr,
        copy: Mxfp8Weight,
        held: Held,
        n: usize,
        hidden: usize,
    ) -> Result<Self> {
        ensure!(
            !copy.tiled && hidden.is_multiple_of(32) && n.is_multiple_of(16),
            "deepseek_v41: the head's pick over {n} tokens a part of {hidden} values"
        );
        let rows = held.count * n;
        gpu.synchronize(gpu.default_stream())?;
        let mut bounds = Vec::with_capacity(rows);
        let (mut w, mut codes, mut scales) = (Vec::new(), Vec::new(), Vec::new());
        for first in (0..rows).step_by(LOAD_ROWS) {
            let some = (rows - first).min(LOAD_ROWS);
            w.resize(some * hidden * 2, 0u8);
            codes.resize(some * hidden, 0u8);
            scales.resize(some * hidden / 32, 0u8);
            gpu.copy_d2h(head.offset(first * hidden * 2), &mut w)?;
            gpu.copy_d2h(copy.data.offset(first * hidden), &mut codes)?;
            gpu.copy_d2h(copy.scales.offset(first * hidden / 32), &mut scales)?;
            let w: Vec<u16> = w
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            bounds.extend(row_bounds(&w, &codes, &scales, hidden));
        }
        let bound = gpu.alloc(rows * 4)?;
        let bytes: Vec<u8> = bounds.iter().flat_map(|b| b.to_le_bytes()).collect();
        gpu.copy_h2d(&bytes, bound)?;
        let of = |name: &str| gpu.kernel("dsv41_head", name);
        Ok(Self {
            kernels: HeadKernels {
                coarse: PartKernels::load(gpu)?.fp8_f32,
                low: of("dsv41_head_low")?,
                keep: of("dsv41_head_keep")?,
                exact: of("dsv41_head_exact")?,
                pick: of("dsv41_head_pick")?,
            },
            held,
            n,
            hidden,
            head,
            copy,
            bound,
            coarse: gpu.alloc(held.count * ROWS * n * 4)?,
            list: gpu.alloc(held.count * ROWS * n * 4)?,
            count: gpu.alloc(held.count * ROWS * 4)?,
            low: gpu.alloc(PARTS * LOW)?,
            found: gpu.alloc(PARTS * FOUND)?,
            recv: gpu.alloc(PARTS * FOUND)?,
            picked: gpu.alloc(ROWS * 4)?,
        })
    }

    /// The token with the largest logit of each of the `rows` rows of `x`
    /// (`[rows, hidden]` BF16, as the head reads them), queued on `stream`:
    /// `[rows]` u32 on the device, the same on every rank. With `mark`, the
    /// rows of logits the step leaves uncomputed are marked so
    /// ([`Mark::read`]).
    pub fn pick(
        &self,
        gpu: &dyn GpuBackend,
        comm: Option<&dyn CommBackend>,
        x: DevicePtr,
        rows: usize,
        mark: Option<Mark>,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            (1..=ROWS).contains(&rows),
            "deepseek_v41: the head's pick of {rows} rows"
        );
        let (k, n, hidden) = (&self.kernels, self.n, self.hidden);
        let (rows32, n32, hidden32) = (rows as u32, n as u32, hidden as u32);
        let own = |i: usize| {
            (
                self.coarse.offset(i * ROWS * n * 4),
                self.bound.offset(i * n * 4),
                self.list.offset(i * ROWS * n * 4),
                self.count.offset(i * ROWS * 4),
            )
        };
        for (i, part) in self.held.range().enumerate() {
            let (coarse, bound, _, count) = own(i);
            let at = i * n * hidden;
            parts::project_rows(
                gpu,
                k.coarse,
                x,
                self.copy.data.offset(at),
                self.copy.scales.offset(at / 32),
                coarse,
                4,
                n,
                rows,
                n,
                hidden,
                stream,
            )?;
            KernelLaunch::new(gpu, k.low)
                .grid([rows32, 1, 1])
                .block([SCAN, 1, 1])
                .arg_ptr(x)
                .arg_ptr(coarse)
                .arg_ptr(bound)
                .arg_ptr(self.low.offset(part * LOW))
                .arg_ptr(count)
                .arg_u32(n32)
                .arg_u32(hidden32)
                .launch(stream)?;
        }
        parts::gather(gpu, comm, self.held, self.low, self.recv, LOW, LOW, stream)?;
        for (i, part) in self.held.range().enumerate() {
            let (coarse, bound, list, count) = own(i);
            KernelLaunch::new(gpu, k.keep)
                .grid([rows32, 1, 1])
                .block([SCAN, 1, 1])
                .arg_ptr(coarse)
                .arg_ptr(bound)
                .arg_ptr(self.low)
                .arg_ptr(list)
                .arg_ptr(count)
                .arg_u32(n32)
                .arg_u32(PARTS as u32)
                .arg_u32(ROWS as u32)
                .launch(stream)?;
            KernelLaunch::new(gpu, k.exact)
                .grid([BLOCKS as u32, rows32, 1])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(self.head.offset(i * n * hidden * 2))
                .arg_ptr(list)
                .arg_ptr(count)
                .arg_ptr(self.found.offset(part * FOUND))
                .arg_u32(n32)
                .arg_u32(hidden32)
                .arg_u32((part * n) as u32)
                .launch(stream)?;
        }
        parts::gather(
            gpu, comm, self.held, self.found, self.recv, FOUND, FOUND, stream,
        )?;
        KernelLaunch::new(gpu, k.pick)
            .grid([1, 1, 1])
            .block([ROWS as u32, 1, 1])
            .arg_ptr(self.found)
            .arg_ptr(self.picked)
            .arg_ptr(mark.map_or(DevicePtr::NULL, |m| m.logits))
            .arg_u32(rows32)
            .arg_u32(BLOCKS as u32)
            .arg_u32(PARTS as u32)
            .arg_u32(ROWS as u32)
            .arg_u32(mark.map_or(0, |m| m.vocab as u32))
            .arg_u32(mark.map_or(0, |m| m.step))
            .launch(stream)?;
        Ok(self.picked)
    }
}

impl HeadPick {
    /// How many tokens each of the last pick's `rows` rows kept, over the
    /// parts held: what the head itself was read for. Waits for the stream.
    pub fn kept(&self, gpu: &dyn GpuBackend, rows: usize) -> Result<Vec<u32>> {
        let mut bytes = vec![0u8; self.held.count * ROWS * 4];
        gpu.copy_d2h(self.count, &mut bytes)?;
        let count = |i: usize| u32::from_le_bytes([0, 1, 2, 3].map(|b| bytes[4 * i + b]));
        Ok((0..rows)
            .map(|row| (0..self.held.count).map(|i| count(i * ROWS + row)).sum())
            .collect())
    }
}

/// How a pick marks the rows of logits it did not compute: where they are,
/// `vocab` BF16 values a row, and which step picks.
#[derive(Clone, Copy)]
pub struct Mark {
    pub logits: DevicePtr,
    pub vocab: usize,
    pub step: u32,
}

impl Mark {
    /// Bytes of a row's mark, at its start.
    pub const BYTES: usize = 16;

    /// The token a row's first bytes say `step` picked, if they are that
    /// step's mark (`DSV41_HEAD_MARK`) and not logits or another step's.
    pub fn read(bytes: &[u8; Self::BYTES], step: u32) -> Option<u32> {
        let word = |i: usize| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        let pair = |i: usize| u32::from(word(i)) | u32::from(word(i + 1)) << 16;
        ((0..4).all(|i| word(i) == 0x7fc1 + i as u16) && pair(4) == step).then(|| pair(6))
    }
}

#[cfg(test)]
#[path = "head_test_gpu.rs"]
mod test_gpu;
