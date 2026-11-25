// SPDX-License-Identifier: AGPL-3.0-only

//! A block's routed experts from a checkpoint: the part of each that a rank
//! computes, in one slab a part (`weights.rs`).

use anyhow::{Context, Result, ensure};
use spark_runtime::fast_weights::{SliceAxis, with_deferred_slice};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::parts::{self, PARTS, PartKernels};
use super::shape::Shape;
use super::weights::ExpertSlab;

/// Part `part` of one routed-expert tensor `name` (`[rows, row_bytes]`
/// bytes) into `dst`: a quarter of its rows or of its columns. The tensor is
/// read from disk where the loader left it there, and from the device
/// otherwise.
#[allow(clippy::too_many_arguments)]
fn expert_part(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    kernels: &PartKernels,
    name: &str,
    rows: usize,
    row_bytes: usize,
    axis: SliceAxis,
    part: usize,
    dst: DevicePtr,
    stream: u64,
) -> Result<()> {
    if !store.contains(name) {
        return with_deferred_slice(store, name, rows, row_bytes, axis, part, PARTS, |bytes| {
            gpu.copy_h2d(bytes, dst)
        });
    }
    let whole = store.get(name)?;
    ensure!(
        whole.shape == [rows, row_bytes] && whole.dtype.byte_size() == 1,
        "{name}: {:?} {:?}, expected [{rows}, {row_bytes}] bytes",
        whole.dtype,
        whole.shape
    );
    match axis {
        SliceAxis::Rows => {
            let bytes = rows / PARTS * row_bytes;
            gpu.synchronize(stream)?;
            gpu.copy_d2d(whole.ptr.offset(part * bytes), dst, bytes)
        }
        SliceAxis::Cols => {
            let width = row_bytes / PARTS;
            parts::slice_columns(
                gpu,
                kernels.slice,
                whole.ptr,
                dst,
                rows,
                row_bytes,
                part * width,
                width,
                stream,
            )
        }
    }
}

/// Part `part` of every routed expert of the block named `layer`.
pub(super) fn expert_slab(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    kernels: &PartKernels,
    shape: &Shape,
    layer: &str,
    part: usize,
    stream: u64,
) -> Result<ExpertSlab> {
    let (inter, hidden) = (shape.inter, shape.hidden);
    ensure!(
        inter.is_multiple_of(PARTS * 32) && hidden.is_multiple_of(32),
        "deepseek_v41: experts [{inter}, {hidden}] do not split in {PARTS} parts"
    );
    // A part of w1 or w3 is inter / 4 rows of hidden values; a part of w2 is
    // hidden / 4 rows of inter values. Both are the same number of bytes.
    let slab = ExpertSlab {
        base: gpu.alloc(shape.experts * 3 * (inter / PARTS) * (hidden / 2 + hidden / 32))?,
        data: inter / PARTS * hidden / 2,
        scales: inter / PARTS * hidden / 32,
    };
    for expert in 0..shape.experts {
        let prefix = format!("{layer}.ffn.experts.{expert}");
        for (projection, name, rows, columns, axis) in [
            (0, "w1", inter, hidden, SliceAxis::Rows),
            (1, "w3", inter, hidden, SliceAxis::Rows),
            (2, "w2", hidden, inter, SliceAxis::Rows),
        ] {
            let slice = slab.slice(expert, projection);
            for (tensor, row_bytes, dst) in [
                ("weight", columns / 2, slice.data),
                ("scale", columns / 32, slice.scales),
            ] {
                expert_part(
                    store,
                    gpu,
                    kernels,
                    &format!("{prefix}.{name}.{tensor}"),
                    rows,
                    row_bytes,
                    axis,
                    part,
                    dst,
                    stream,
                )
                .with_context(|| format!("{prefix}.{name}.{tensor}"))?;
            }
        }
    }
    gpu.synchronize(stream)?;
    Ok(slab)
}

/// Free the store's own copies of a block's routed experts, where it has
/// them on the device: the slabs hold every byte of them.
pub(super) fn release_experts(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    layer: &str,
    experts: usize,
) -> Result<()> {
    for expert in 0..experts {
        for projection in ["w1", "w2", "w3"] {
            for tensor in ["weight", "scale"] {
                let name = format!("{layer}.ffn.experts.{expert}.{projection}.{tensor}");
                if store.contains(&name) {
                    store.reclaim(gpu, &name)?;
                }
            }
        }
    }
    Ok(())
}
