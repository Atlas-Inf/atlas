// SPDX-License-Identifier: AGPL-3.0-only

//! Per-stream BF16 rebuild scratch for the EXL3 lazy-dense path (M6f).
//!
//! Dropping the resident BF16 copies of the GDN projections (M6f-b) means every
//! remaining BF16 reader asks the layer to rebuild the weight on demand. The
//! rebuild is five stream-ordered kernels ([`crate::layers::ops::
//! exl3_dense_bf16_nk_with_scratch`]) needing one fp16 temporary plus the
//! BF16 result; allocating and freeing those per call would add a
//! synchronize-and-realloc to every layer, so the buffers are pooled instead.
//!
//! **One scratch per stream, shared by all layers.** That is safe because a
//! stream runs layers in order: every rebuild is stream-ordered before the GEMM
//! that reads it, and a later layer's rebuild on the same stream can only start
//! after the earlier GEMM consumed the previous one. A second stream would get
//! its own entry.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// The four pooled regions of one stream's rebuild scratch.
#[derive(Debug, Clone, Copy)]
pub struct Exl3Bf16Scratch {
    /// BF16 `[qkv.out + z.out, in]` result of the qkvz rebuild.
    pub qkvz: DevicePtr,
    /// BF16 `[out.out, out.in]` result of the out_proj rebuild.
    pub out: DevicePtr,
    /// fp16 `W_inner` temporary of `exl3_dense_bf16_nk_with_scratch`, `tmp_bytes`.
    pub tmp_a: DevicePtr,
}

/// Region sizes requested by one layer (bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exl3Bf16Sizes {
    pub qkvz_bytes: usize,
    pub out_bytes: usize,
    pub tmp_bytes: usize,
}

type Pool = Mutex<HashMap<u64, (Exl3Bf16Sizes, Exl3Bf16Scratch)>>;

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(Default::default)
}

/// This stream's scratch, grown to at least `sizes`.
///
/// The entry is kept for the process lifetime: a stream's scratch is reused by
/// every layer and every rebuild, so it is never freed while the stream lives.
/// A larger request replaces the entry. Work already queued on the stream may
/// still read the old buffers, so the stream is synchronized before they are
/// freed. Growth never happens under CUDA-graph capture in practice: the loader
/// pre-allocates the default stream at the largest size (M6f-b).
pub fn exl3_bf16_scratch(
    gpu: &dyn GpuBackend,
    stream: u64,
    sizes: &Exl3Bf16Sizes,
) -> Result<Exl3Bf16Scratch> {
    let mut map = pool().lock().unwrap_or_else(|e| e.into_inner());
    if let Some((have, scratch)) = map.get(&stream)
        && sizes.qkvz_bytes <= have.qkvz_bytes
        && sizes.out_bytes <= have.out_bytes
        && sizes.tmp_bytes <= have.tmp_bytes
    {
        return Ok(*scratch);
    }
    let old = map.remove(&stream);
    let s = Exl3Bf16Scratch {
        qkvz: gpu.alloc(sizes.qkvz_bytes)?,
        out: gpu.alloc(sizes.out_bytes)?,
        tmp_a: gpu.alloc(sizes.tmp_bytes)?,
    };
    map.insert(stream, (*sizes, s));
    drop(map);
    if let Some((_, old)) = old {
        gpu.synchronize(stream)?;
        for p in [old.qkvz, old.out, old.tmp_a] {
            gpu.free(p)?;
        }
    }
    Ok(s)
}
