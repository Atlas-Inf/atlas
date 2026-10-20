// SPDX-License-Identifier: AGPL-3.0-only

//! A reusable page-locked host buffer for one recurring upload.
//!
//! `ATLAS_QWEN4EXP_VERIFY_BT_PINNED=1` (default off): the batched verify
//! (`trait_impl/verify_e.rs`) built its `[rows, max_blocks]` block-table image
//! in a fresh pageable `Vec` every step and uploaded it with one
//! `cuMemcpyHtoDAsync`. At C8 x K=4 that is 32 x 2049 x 4 = 262,272 B, and
//! from pageable memory of that size the driver stages synchronously: the
//! call held the scheduler thread ~1.25 ms per step, right before the verify
//! graph launch, with the GPU idle (nsys, TP=EP=2 rank 0). From a page-locked
//! buffer the same bytes go as one async DMA of a few microseconds.
//!
//! The device receives the same bytes: the image is built by the same code
//! into this buffer instead of a `Vec`. What changes is the lifetime rule —
//! the DMA engine reads a pinned source after the call returns — so the
//! buffer records an event after each upload and [`PinnedUpload::stage`]
//! waits on it before handing the bytes out again. Every verify step ends
//! with a host sync on its logits, so that wait finds the event complete.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// See the module doc. Freed with [`PinnedUpload::release`] (the model's
/// `Drop`), since it holds no `gpu` handle of its own.
pub(crate) struct PinnedUpload {
    ptr: *mut u8,
    bytes: usize,
    /// Recorded after the last upload; 0 = none yet (or no event support,
    /// where the wait falls back to a stream sync).
    event: u64,
    /// Stream of the last upload: the fallback sync's stream.
    stream: Option<u64>,
}

// SAFETY: the raw pointer is page-locked host memory owned by this struct and
// only touched through `&mut self` (the model keeps it behind a `Mutex`).
unsafe impl Send for PinnedUpload {}

impl PinnedUpload {
    pub(crate) const EMPTY: Self = Self {
        ptr: std::ptr::null_mut(),
        bytes: 0,
        event: 0,
        stream: None,
    };

    /// `len` writable host bytes, ZEROED, once the previous upload from this
    /// buffer has been read. Grows (re-allocates) when `len` exceeds it.
    pub(crate) fn stage(&mut self, gpu: &dyn GpuBackend, len: usize) -> Result<&mut [u8]> {
        self.wait(gpu)?;
        if len == 0 {
            // No allocation yet, so no pointer a slice could be built on.
            return Ok(&mut []);
        }
        if len > self.bytes {
            self.release(gpu)?;
            self.ptr = gpu.alloc_host_pinned(len)?;
            self.bytes = len;
        }
        // SAFETY: `ptr` spans `bytes >= len` page-locked bytes owned by `self`,
        // no DMA reads them any more (`wait` above), and the borrow of `self`
        // keeps them from being freed while the slice lives.
        let buf = unsafe { std::slice::from_raw_parts_mut(self.ptr, len) };
        buf.fill(0);
        Ok(buf)
    }

    /// Upload the first `len` staged bytes to `dst` on `stream`, then mark the
    /// point the next [`Self::stage`] waits for.
    pub(crate) fn send(
        &mut self,
        gpu: &dyn GpuBackend,
        len: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            len <= self.bytes,
            "pinned upload: {len} B past the {} B staged",
            self.bytes
        );
        if len == 0 {
            return Ok(());
        }
        // SAFETY: as in `stage`; the bytes stay untouched until the event below
        // has completed, which is what the retained copy requires.
        let src = unsafe { std::slice::from_raw_parts(self.ptr, len) };
        gpu.copy_h2d_async_retained(src, dst, stream)?;
        if self.event == 0 {
            self.event = gpu.event_create()?;
        }
        gpu.event_record(self.event, stream)?;
        self.stream = Some(stream);
        Ok(())
    }

    fn wait(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            gpu.event_synchronize_on_stream(self.event, stream)?;
        }
        Ok(())
    }

    /// Free the buffer and its event (after the last upload has been read).
    pub(crate) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.wait(gpu)?;
        if !self.ptr.is_null() {
            gpu.free_host_pinned(self.ptr, self.bytes)?;
        }
        if self.event != 0 {
            gpu.event_destroy(self.event)?;
        }
        *self = Self::EMPTY;
        Ok(())
    }
}

/// `ATLAS_QWEN4EXP_VERIFY_BT_PINNED=1`, read once.
pub(crate) fn verify_bt_pinned() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_VERIFY_BT_PINNED").as_deref() == Ok("1"))
}

impl super::types::TransformerModel {
    /// Upload an `entries`-long zero-initialized i32 image that `fill` writes,
    /// to `dst` on `stream`: from the page-locked [`PinnedUpload`] under
    /// [`verify_bt_pinned`], else from a fresh `Vec` (the original form).
    /// `fill` sees zeros either way, so the device gets the same bytes.
    pub(super) fn upload_i32_image(
        &self,
        entries: usize,
        fill: impl FnOnce(&mut [i32]),
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let bytes = entries * 4;
        if verify_bt_pinned() {
            let mut stage = self.verify_bt_stage.lock();
            let buf = stage.stage(self.gpu.as_ref(), bytes)?;
            anyhow::ensure!(
                (buf.as_ptr() as usize).is_multiple_of(4),
                "pinned upload: staging not 4-byte aligned"
            );
            // SAFETY: `buf` is `bytes = entries * 4` zeroed bytes, 4-aligned
            // (checked above; page-locked memory is page-aligned), exclusively
            // borrowed for this scope, and every bit pattern is a valid i32.
            let img =
                unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<i32>(), entries) };
            fill(img);
            return stage.send(self.gpu.as_ref(), bytes, dst, stream);
        }
        let mut img = vec![0i32; entries];
        fill(&mut img);
        // SAFETY: `img` is `entries` initialized i32s; `u8` has no alignment.
        let raw = unsafe { std::slice::from_raw_parts(img.as_ptr().cast::<u8>(), bytes) };
        self.gpu.copy_h2d_async(raw, dst, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn uploads_land_and_the_buffer_is_reused() {
        let gpu = MockGpuBackend::new();
        let dst = gpu.alloc(64).unwrap();
        let mut up = PinnedUpload::EMPTY;
        for round in 0..3u8 {
            let buf = up.stage(&gpu, 64).unwrap();
            assert!(buf.iter().all(|&b| b == 0), "stage hands out zeroed bytes");
            buf[..32].fill(round + 1);
            up.send(&gpu, 64, dst, 0).unwrap();
            let got = gpu.read_alloc(dst).unwrap();
            assert!(got[..32].iter().all(|&b| b == round + 1));
            assert!(got[32..].iter().all(|&b| b == 0));
        }
        assert_eq!(gpu.host_pinned_alloc_count(), 1, "one allocation, reused");
        up.stage(&gpu, 128).unwrap();
        assert_eq!(gpu.host_pinned_alloc_count(), 2, "grows past its size");
        up.release(&gpu).unwrap();
        up.release(&gpu).unwrap();
    }

    #[test]
    fn an_empty_stage_allocates_nothing() {
        let gpu = MockGpuBackend::new();
        let mut up = PinnedUpload::EMPTY;
        assert!(up.stage(&gpu, 0).unwrap().is_empty());
        assert_eq!(gpu.host_pinned_alloc_count(), 0);
        up.send(&gpu, 0, DevicePtr(0), 0).unwrap();
        up.release(&gpu).unwrap();
    }

    #[test]
    fn send_refuses_more_than_staged() {
        let gpu = MockGpuBackend::new();
        let dst = gpu.alloc(64).unwrap();
        let mut up = PinnedUpload::EMPTY;
        up.stage(&gpu, 16).unwrap();
        assert!(up.send(&gpu, 32, dst, 0).is_err());
        up.release(&gpu).unwrap();
    }
}
