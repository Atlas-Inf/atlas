// SPDX-License-Identifier: AGPL-3.0-only

//! A DFlash2 drafter on an Inkling 2 target (`incoai/Inkling-2-Mini-DFlash2`).
//!
//! Inkling's drafters read four taps: what enters visits 5, 23 and 41, and the
//! final-norm output (`target_layer_ids` `[5, 23, 41, 48]`, as the bundled
//! DSpark). Atlas captures after a visit, so the factory captures visits 4, 22,
//! 40 and 47, and the last tap still lacks the target's final norm. DSpark
//! applies it inside its tap mix; the generic DFlash2 context projection is one
//! `fc` over the concatenated taps, so here the norm is applied to the last tap
//! of a chunk copy of the rows before that `fc`. The accumulator itself is
//! never written, so a row that is projected again is normed again from its
//! raw capture.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::BlockDiffusionDraftHead;

/// Context rows normed and projected per launch (the scratch's height).
const CHUNK_ROWS: usize = 256;

/// The target's final norm, as the checkpoint ships it (`x * w`, Inkling's
/// own `inkling2_rmsnorm`), and a chunk of tap rows to apply it in.
pub struct InklingTapNorm {
    weight: DevicePtr,
    eps: f32,
    kernel: KernelHandle,
    /// `[CHUNK_ROWS, taps * target_hidden]` BF16.
    scratch: DevicePtr,
}

impl InklingTapNorm {
    pub(super) fn build(
        gpu: &dyn GpuBackend,
        weight: DevicePtr,
        eps: f32,
        row_elems: usize,
    ) -> Result<Self> {
        ensure!(!weight.is_null(), "Inkling 2 DFlash2: no target final norm");
        Ok(Self {
            weight,
            eps,
            kernel: gpu.kernel("inkling2_misc", "inkling2_rmsnorm")?,
            scratch: gpu.alloc(CHUNK_ROWS * row_elems * 2)?,
        })
    }
}

/// The `(first row, rows)` chunks of `n` context rows.
fn chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n)
        .step_by(CHUNK_ROWS)
        .map(move |first| (first, (n - first).min(CHUNK_ROWS)))
}

impl BlockDiffusionDraftHead {
    /// `fc` over `n` context rows at `src` (`row_elems` = taps x target
    /// hidden), with the last tap normed by the target's final norm, into
    /// `fc_dst` (`[n, h]`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inkling_tap_fc(
        &self,
        t: &InklingTapNorm,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        n: u32,
        h: u32,
        row_elems: u32,
        fc_dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let tap = self.target_hidden_size as u32;
        ensure!(
            tap > 0 && row_elems.is_multiple_of(tap) && row_elems >= tap,
            "Inkling 2 DFlash2: context row of {row_elems} is not whole taps of {tap}"
        );
        let row_bytes = row_elems as usize * 2;
        let last = (row_elems - tap) as usize * 2;
        for (first, rows) in chunks(n as usize) {
            let rows_src = src.offset(first * row_bytes);
            gpu.copy_d2d_async(rows_src, t.scratch, rows * row_bytes, stream)?;
            KernelLaunch::new(gpu, t.kernel)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(rows_src.offset(last))
                .arg_ptr(t.weight)
                .arg_ptr(t.weight)
                .arg_ptr(t.weight)
                .arg_ptr(t.scratch.offset(last))
                .arg_u32(tap)
                .arg_u32(row_elems)
                .arg_u32(row_elems)
                .arg_u64(0)
                .arg_u64(0)
                .arg_f32(t.eps)
                .launch(stream)?;
            self.ctx_projection(
                gpu,
                0,
                &self.fc,
                t.scratch,
                fc_dst.offset(first * h as usize * 2),
                rows as u32,
                h,
                row_elems,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CHUNK_ROWS, chunks};

    #[test]
    fn context_rows_are_normed_in_whole_chunks() {
        assert_eq!(chunks(1).collect::<Vec<_>>(), [(0, 1)]);
        assert_eq!(chunks(CHUNK_ROWS).collect::<Vec<_>>(), [(0, CHUNK_ROWS)]);
        assert_eq!(
            chunks(CHUNK_ROWS + 3).collect::<Vec<_>>(),
            [(0, CHUNK_ROWS), (CHUNK_ROWS, 3)]
        );
        assert_eq!(chunks(0).count(), 0);
    }
}
