// SPDX-License-Identifier: AGPL-3.0-only

//! Reused host buffer for the verify-time dequantised logits.
//!
//! Split out of `verify_pipeline_helper.rs`, which is over the 500 LoC cap.

thread_local! {
    /// Reused host buffer for the per-position dequantised logits.
    ///
    /// Mirrors `decode_logits_step::DECODE_LOGITS_HOST_SCRATCH`, which exists on
    /// the twin decode path "to avoid an mmap/munmap + page-fault cycle every
    /// decoded token". The VERIFY path — the one that actually runs under MTP —
    /// never had it, and was allocating a fresh ~1 MB `Vec<f32>` per K position,
    /// i.e. ~4 MB of first-touch pages per verify step.
    ///
    /// Per-thread: the scheduler drives verify on one thread. Residual contents
    /// are irrelevant — every entry is overwritten before any read.
    pub(super) static DEQUANT_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Returns the dequant buffer to [`DEQUANT_SCRATCH`] on drop.
///
/// `verify_pick_with_pipeline` has three exits — the forced-token short
/// circuit, the temp>0 sample, and the argmax — so a guard is used rather than
/// three hand-placed hand-backs: miss one and the reuse is silently lost with
/// no visible failure, which is the same drift that left `step_done` off the
/// K=4 path entirely.
pub(super) struct ScratchGuard(pub(super) Vec<f32>);

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.0);
        DEQUANT_SCRATCH.with(|s| {
            *s.borrow_mut() = buf;
        });
    }
}

impl std::ops::Deref for ScratchGuard {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        &self.0
    }
}

impl std::ops::DerefMut for ScratchGuard {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        &mut self.0
    }
}

/// The run's reusable host logits buffer (`DecodeScratch::host_bytes`), sized
/// to the rows a pick reads and handed back on drop.
///
/// The host picks used to copy each span into a fresh `vec![0u8; len]`. That
/// is calloc'd from untouched pages, so the D2H itself took the page faults:
/// on the two-rank C8 profile (thinking on) about one ~2 MB row copy a step
/// ran ~7 ms instead of ~35 us. Residual contents are irrelevant — the copy
/// overwrites all `len` bytes before any read.
pub(in crate::scheduler) struct HostRows<'c> {
    buf: Vec<u8>,
    home: &'c std::cell::RefCell<Vec<u8>>,
}

impl<'c> HostRows<'c> {
    pub(in crate::scheduler) fn take(home: &'c std::cell::RefCell<Vec<u8>>, len: usize) -> Self {
        let mut buf = std::mem::take(&mut *home.borrow_mut());
        buf.resize(len, 0);
        Self { buf, home }
    }
}

impl Drop for HostRows<'_> {
    fn drop(&mut self) {
        *self.home.borrow_mut() = std::mem::take(&mut self.buf);
    }
}

impl std::ops::Deref for HostRows<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.buf
    }
}

impl std::ops::DerefMut for HostRows<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::HostRows;

    /// The buffer goes home on drop with its capacity, so the next pick
    /// copies into warm pages instead of a fresh allocation.
    #[test]
    fn host_rows_reuses_one_allocation() {
        let home = std::cell::RefCell::new(Vec::new());
        let ptr = {
            let mut rows = HostRows::take(&home, 1 << 20);
            rows[7] = 1;
            rows.as_ptr()
        };
        assert_eq!(home.borrow().len(), 1 << 20);
        let rows = HostRows::take(&home, 1 << 19);
        assert_eq!(rows.len(), 1 << 19);
        assert_eq!(rows.as_ptr(), ptr, "a smaller span reuses the buffer");
        assert!(home.borrow().is_empty(), "taken while in use");
    }
}
