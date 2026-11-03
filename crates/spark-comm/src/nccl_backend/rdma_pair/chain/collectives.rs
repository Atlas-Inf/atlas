// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl RdmaChain {
    /// An in-place BF16 all-reduce of any size on `stream`: consecutive
    /// one-shot ops of at most its maximum, each capturable like one.
    /// `false` (nothing enqueued) while the kernel is missing.
    pub(in crate::nccl_backend) fn all_reduce(
        &self,
        ptr: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        let max = self.oneshot().max_bytes();
        if max == 0 {
            return Ok(false);
        }
        ensure!(
            bytes.is_multiple_of(2),
            "RDMA chain: an all-reduce of {bytes} bytes is not a whole number of BF16 values"
        );
        // Large payloads: a reduce-scatter then an all-gather of the summed
        // quarters. Every link then carries 3/4 + 3/4 of the payload instead of
        // three payloads, and the words are the flood's (the reduce-scatter
        // sums in its tree; the gather copies).
        let quarter = bytes / MAX_RANKS;
        if self.oneshot().ranks() == MAX_RANKS
            && bytes >= AR_SPLIT_MIN
            && bytes.is_multiple_of(MAX_RANKS * 128)
            && ptr.is_multiple_of(16)
            && self.reduce_scatter(ptr, quarter as i64, quarter, false, stream)?
        {
            ensure!(
                self.gather(ptr, quarter as i64, quarter, stream)?,
                "RDMA chain: the all-reduce's gather was refused after its reduce-scatter"
            );
            return Ok(true);
        }
        for (at, len) in chunks(bytes, max, 64) {
            let part = ptr + at as u64;
            ensure!(self.oneshot().sum(part, part, len, true, stream)?);
        }
        Ok(true)
    }

    /// An in-place all-gather of every rank's `bytes` at `base + rank *
    /// stride` (`CommBackend::gather_async`) on `stream`: consecutive one-shot
    /// ops of at most its maximum per part, each capturable like one. `false`
    /// (nothing enqueued) while the kernel is missing.
    pub(in crate::nccl_backend) fn gather(
        &self,
        base: u64,
        stride: i64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        let max = self.oneshot().max_bytes();
        if max == 0 {
            return Ok(false);
        }
        ensure!(
            bytes.is_multiple_of(2) && stride.unsigned_abs() as usize >= bytes,
            "RDMA chain: a gather of {bytes}-byte parts {stride} bytes apart"
        );
        for (at, len) in chunks(bytes, max, 64) {
            ensure!(
                self.oneshot()
                    .gather(base + at as u64, stride, len, stream)?
            );
        }
        Ok(true)
    }

    /// A broadcast of rank `root`'s `ptr[..bytes]` into every rank's `ptr`
    /// (`CommBackend::broadcast_async`) on `stream`: consecutive one-shot ops
    /// of at most its maximum, each capturable like one. `false` (nothing
    /// enqueued) while the kernel is missing.
    pub(in crate::nccl_backend) fn broadcast(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        let max = self.oneshot().max_bytes();
        if max == 0 {
            return Ok(false);
        }
        ensure!(
            bytes.is_multiple_of(2),
            "RDMA chain: a broadcast of {bytes} bytes is not a whole number of BF16 values"
        );
        for (at, len) in chunks(bytes, max, 64) {
            let part = ptr + at as u64;
            ensure!(self.oneshot().broadcast(part, len, root, stream)?);
        }
        Ok(true)
    }

    /// The four-rank reduce-scatter (`CommBackend::reduce_scatter_async`):
    /// segment `d` of every rank's partial, `bytes` at `seg0 + d * stride`,
    /// summed as the all-reduce sums, lands in rank `d`'s segment, on
    /// `stream`: consecutive one-shot ops of one slice of every segment, each
    /// capturable like one. `false` (nothing enqueued) when it does not
    /// apply (not four ranks, the kernel missing, `bytes` not whole 128-byte
    /// units, misaligned); every rank sees the same answer. `fp8`: the
    /// lossy FP8 pieces (`CommBackend::reduce_scatter_fp8_async`), ops of
    /// whole 8 KiB of every segment (so each piece is whole KiB).
    pub(in crate::nccl_backend) fn reduce_scatter(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        fp8: bool,
        stream: u64,
    ) -> Result<bool> {
        let unit = if fp8 { 8192 } else { 512 };
        let slice = Self::rs_slice(self.oneshot().max_bytes()) / unit * unit;
        if slice == 0 || bytes == 0 || !bytes.is_multiple_of(if fp8 { unit } else { 128 }) {
            return Ok(false);
        }
        ensure!(
            stride.unsigned_abs() as usize >= bytes,
            "RDMA chain: a reduce-scatter of {bytes}-byte segments {stride} bytes apart"
        );
        for (at, part) in chunks(bytes, slice, unit) {
            if !self
                .oneshot()
                .reduce_scatter(seg0 + at as u64, stride, 4 * part, fp8, stream)?
            {
                // Only the first op can decline (alignment and size do not
                // change along the way).
                ensure!(at == 0, "RDMA chain: reduce-scatter op refused mid-way");
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The four-rank all-to-all (`CommBackend::all_to_all_async`) on
    /// `stream`: consecutive one-shot ops of an equal share of every plane's
    /// rows, each capturable like one. `false` (nothing enqueued) when it does
    /// not apply (not four ranks, the kernel missing, more than two planes, a
    /// shape no split into ops of whole rows and 128-byte segments serves).
    pub(in crate::nccl_backend) fn all_to_all(
        &self,
        planes: &[crate::AllToAllPlane],
        stream: u64,
    ) -> Result<bool> {
        let max = self.oneshot().max_bytes().min(oneshot::RS_BIT - 1);
        if max == 0 || planes.is_empty() || planes.len() > 2 {
            return Ok(false);
        }
        let seg = |n: usize| -> usize { planes.iter().map(|p| p.rows / n * p.row_bytes).sum() };
        let fits = |n: usize| {
            planes.iter().all(|p| p.rows.is_multiple_of(n))
                && seg(n).is_multiple_of(128)
                && MAX_RANKS * seg(n) <= max
        };
        let most = planes.iter().map(|p| p.rows).max().unwrap_or(0);
        let first = (MAX_RANKS * seg(1)).div_ceil(max).max(1);
        let Some(n) = (first..=most).find(|&n| fits(n)) else {
            return Ok(false);
        };
        for i in 0..n {
            let mut a2a = oneshot::A2a::default();
            for (k, p) in planes.iter().enumerate() {
                let rows = p.rows / n;
                let (Ok(row_bytes), Ok(pitch), Ok(bytes)) = (
                    u32::try_from(p.row_bytes),
                    u32::try_from(p.pitch),
                    u32::try_from(rows * p.row_bytes),
                ) else {
                    return Ok(false);
                };
                a2a.planes[k] = oneshot::A2aPlane {
                    src: p.src + (i * rows * p.row_bytes) as u64,
                    src_stride: p.src_stride,
                    dst: p.dst + (i * rows * p.pitch) as u64,
                    dst_stride: p.dst_stride,
                    row_bytes,
                    pitch,
                    bytes,
                    pad: 0,
                };
            }
            if !self.oneshot().all_to_all(a2a, stream)? {
                // Only the first op can decline (every op has the same shape).
                ensure!(i == 0, "RDMA chain: all-to-all op refused mid-way");
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// [`OneShot::rs_fused_begin`] on channel 0, the one with the fused
    /// region (`CommBackend::rs_fused_begin`).
    pub(in crate::nccl_backend) fn rs_fused_begin(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        unit: usize,
    ) -> Result<Option<crate::RsFused>> {
        self.oneshot[0].rs_fused_begin(seg0, stride, bytes, unit)
    }

    /// The chain kernel of a begun fused op, on channel 0.
    pub(in crate::nccl_backend) fn reduce_scatter_fused(
        &self,
        fused: &crate::RsFused,
        stream: u64,
    ) -> Result<bool> {
        self.oneshot[0].reduce_scatter_fused(fused, stream)
    }

    /// `stream` waits for a piece of a fused op's own segment.
    pub(in crate::nccl_backend) fn rs_fused_wait(
        &self,
        fused: &crate::RsFused,
        piece: usize,
        stream: u64,
    ) -> Result<()> {
        self.oneshot[0].rs_fused_wait(fused, piece, stream)
    }

    /// The per-segment slice of one reduce-scatter op under a one-shot
    /// maximum of `max` bytes (0: unavailable): whole 512-byte units (so a
    /// full slice takes the finest pieces, `oneshot::rs_per_segment`) of a
    /// quarter of the op, below `oneshot::RS_BIT`.
    pub(in crate::nccl_backend) fn rs_slice(max: usize) -> usize {
        max.min(oneshot::RS_BIT - 1) / 4 / 512 * 512
    }
}
