// SPDX-License-Identifier: AGPL-3.0-only

//! A projection held by its input's columns (`parts.rs`): a part is every
//! row's values of a quarter of the input, with their scales. A rank whose
//! own output is that quarter projects it without waiting for the others',
//! and the ranks add what each found.

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::parts::{self, Held, PARTS, PartKernels};
use super::weights::{Fp8Proj, Parted, Slice, fp8};

/// The parts `held` of `whole` by its columns, each `[n, k / PARTS]` with
/// scales `[n, k / PARTS / 32]`, in allocations of their own.
pub(super) fn columns(
    gpu: &dyn GpuBackend,
    kernels: &PartKernels,
    whole: &Fp8Proj,
    held: Held,
    stream: u64,
) -> Result<Parted> {
    let (n, k) = (whole.n, whole.k);
    ensure!(
        k.is_multiple_of(PARTS * 32),
        "{k} columns do not split in {PARTS} parts of whole scale blocks"
    );
    let width = k / PARTS;
    let mut out = [None; PARTS];
    for part in held.range() {
        let (data, scales) = (gpu.alloc(n * width)?, gpu.alloc(n * width / 32)?);
        let slice = |src, dst, stride: usize, of: usize| {
            parts::slice_columns(
                gpu,
                kernels.slice,
                src,
                dst,
                n,
                stride,
                part * of,
                of,
                stream,
            )
        };
        slice(whole.w.data, data, k, width)?;
        slice(whole.w.scales, scales, k / 32, width / 32)?;
        out[part] = Some(Slice { data, scales });
    }
    gpu.synchronize(stream)?;
    Ok(out)
}

/// [`columns`] of the projection named `prefix` in `store`. It reads the
/// store's copy of the whole, so it comes before whatever lets that go
/// ([`fp8_held`](super::weights::fp8_held)).
pub(super) fn fp8_columns(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    kernels: &PartKernels,
    prefix: &str,
    held: Held,
    stream: u64,
) -> Result<Parted> {
    let whole = fp8(store, gpu, prefix)?;
    let out = columns(gpu, kernels, &whole, held, stream);
    // The whole's scales by rows were made for this.
    gpu.free(whole.w.scales)?;
    out
}

/// Let go of what [`columns`] allocated.
pub(super) fn free_columns(gpu: &dyn GpuBackend, parts: &Parted) -> Result<()> {
    for part in parts.iter().flatten() {
        gpu.free(part.data)?;
        gpu.free(part.scales)?;
    }
    Ok(())
}
