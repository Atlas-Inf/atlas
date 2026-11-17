// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the draft head's own kernels (`dsv41_draft.cu`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::draft_head::{KEEP, Kept};
use super::parts::PARTS;

#[derive(Clone, Copy)]
pub struct DraftKernels {
    parts_rows: KernelHandle,
    top: KernelHandle,
}

impl DraftKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            parts_rows: gpu.kernel("dsv41_draft", "dsv41_parts_rows")?,
            top: gpu.kernel("dsv41_draft", "dsv41_draft_top")?,
        })
    }
}

/// `out` (`[rows, 4 * part_n]` BF16) = a projection the ranks computed a
/// part each and gathered (`parts`, `[4, rows, part_n]`), a row at a time.
pub fn rows_of_parts(
    gpu: &dyn GpuBackend,
    kernels: &DraftKernels,
    parts: DevicePtr,
    out: DevicePtr,
    rows: usize,
    part_n: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.parts_rows)
        .grid([div_ceil((rows * PARTS * part_n) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(parts)
        .arg_ptr(out)
        .arg_u32(rows as u32)
        .arg_u32(part_n as u32)
        .arg_u32(PARTS as u32)
        .launch(stream)
}

/// `best` (`[rows, KEEP]` of a value and a token, 8 bytes each) = the
/// largest of each row of `logits` (`[rows, n]` FP32), largest first, whose
/// column `c` is token `first + c`.
#[allow(clippy::too_many_arguments)]
pub fn largest(
    gpu: &dyn GpuBackend,
    kernels: &DraftKernels,
    logits: DevicePtr,
    best: DevicePtr,
    rows: usize,
    n: usize,
    first: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.top)
        .grid([rows as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(best)
        .arg_u32(n as u32)
        .arg_u32(first as u32)
        .arg_u32(KEEP as u32)
        .launch(stream)
}

/// What [`largest`] left of `rows` rows for every part of the head
/// (`[4, rows, KEEP]`, as bytes), a row's from every part together.
pub fn kept(best: &[u8], rows: usize) -> Vec<Vec<Kept>> {
    let word = |at: usize| u32::from_le_bytes([0, 1, 2, 3].map(|byte| best[at * 4 + byte]));
    (0..rows)
        .map(|row| {
            (0..PARTS)
                .flat_map(|part| (0..KEEP).map(move |i| ((part * rows + row) * KEEP + i) * 2))
                .map(|at| Kept {
                    logit: f32::from_bits(word(at)),
                    token: word(at + 1),
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[path = "draft_test_gpu.rs"]
mod test_gpu;
