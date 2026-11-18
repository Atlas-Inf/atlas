// SPDX-License-Identifier: AGPL-3.0-only

//! Timing-enabled CUDA events, separate from synchronization-only events.

use anyhow::{Result, ensure};

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

pub struct Event(u64);

impl Event {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut event = 0;
            // Flags zero enables timestamps; context is bound by the caller.
            ensure!(
                unsafe { cuEventCreate(&mut event, 0) } == 0,
                "timing event create"
            );
            Ok(Self(event))
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("CUDA timing events unavailable")
    }

    pub fn record(&self, stream: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        ensure!(
            unsafe { cuEventRecord(self.0, stream) } == 0,
            "timing event record"
        );
        #[cfg(not(feature = "cuda"))]
        let _ = stream;
        Ok(())
    }

    /// Both events must have completed on the caller's stream.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f64> {
        #[cfg(feature = "cuda")]
        {
            let mut ms = 0.;
            ensure!(
                unsafe { cuEventElapsedTime(&mut ms, self.0, end.0) } == 0,
                "timing event elapsed"
            );
            Ok(ms as f64)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = end;
            anyhow::bail!("CUDA timing events unavailable")
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if unsafe { cuEventDestroy_v2(self.0) } != 0 {
            tracing::warn!("timing event destroy failed");
        }
    }
}
