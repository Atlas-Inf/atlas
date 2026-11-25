// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the draft head's own kernels (`dsv41_draft.cu`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::parts::PARTS;

/// Tokens a block of a Markov round takes (`DSV41_MARKOV_THREADS`).
const MARKOV_THREADS: usize = 256;

#[derive(Clone, Copy)]
pub struct DraftKernels {
    parts_rows: KernelHandle,
    markov: KernelHandle,
}

impl DraftKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            parts_rows: gpu.kernel("dsv41_draft", "dsv41_parts_rows")?,
            markov: gpu.kernel("dsv41_draft", "dsv41_markov_round")?,
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

/// The Markov head and the vocabulary it is over: `embed` (`[vocab, rank]`
/// BF16) whole, and `head` the same shape, of which a part reads its rows.
#[derive(Clone, Copy)]
pub struct Markov {
    pub embed: DevicePtr,
    pub head: DevicePtr,
    pub vocab: usize,
    pub rank: usize,
}

impl Markov {
    /// Tokens a part of the vocabulary holds, and the candidates it leaves
    /// of a row: a block's best each.
    pub fn share(&self) -> usize {
        self.vocab / PARTS
    }

    pub fn candidates(&self) -> usize {
        self.share().div_ceil(MARKOV_THREADS)
    }

    /// Bytes of one part's candidates of a row.
    pub fn bytes(&self) -> usize {
        self.candidates() * 8
    }
}

/// Part `part`'s round of one row of a draft block (`dsv41_markov_round`):
/// the row's `logits` (`[share]` FP32, this part's) with the Markov head's
/// gain after the token before, and the part's best of them into `mine`
/// ([`Markov::bytes`]). The token before is `token` on the first row, and
/// otherwise the best of `before`, every part's candidates of the row
/// before, which also goes to `chosen` (a u32) as that row's draft.
#[allow(clippy::too_many_arguments)]
pub fn markov_round(
    gpu: &dyn GpuBackend,
    kernels: &DraftKernels,
    markov: &Markov,
    part: usize,
    logits: DevicePtr,
    before: DevicePtr,
    mine: DevicePtr,
    chosen: DevicePtr,
    token: u32,
    stream: u64,
) -> Result<()> {
    let (share, rank) = (markov.share(), markov.rank);
    KernelLaunch::new(gpu, kernels.markov)
        .grid([markov.candidates() as u32, 1, 1])
        .block([MARKOV_THREADS as u32, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(markov.head.offset(part * share * rank * 2))
        .arg_ptr(markov.embed)
        .arg_ptr(before)
        .arg_ptr(mine)
        .arg_ptr(chosen)
        .arg_u32(share as u32)
        .arg_u32(rank as u32)
        .arg_u32((part * share) as u32)
        .arg_u32((PARTS * markov.candidates()) as u32)
        .arg_u32(token)
        .arg_u32(markov.vocab as u32)
        .launch(stream)
}

/// The last row's draft into `chosen`, from every part's candidates of it:
/// the launch that ends a chain of [`markov_round`]s.
pub fn markov_close(
    gpu: &dyn GpuBackend,
    kernels: &DraftKernels,
    markov: &Markov,
    before: DevicePtr,
    chosen: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.markov)
        .grid([1, 1, 1])
        .block([MARKOV_THREADS as u32, 1, 1])
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(before)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(chosen)
        .arg_u32(0)
        .arg_u32(0)
        .arg_u32(0)
        .arg_u32((PARTS * markov.candidates()) as u32)
        .arg_u32(0)
        .arg_u32(markov.vocab as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "draft_test_gpu.rs"]
mod test_gpu;
