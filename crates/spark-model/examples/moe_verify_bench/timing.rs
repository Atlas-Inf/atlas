// SPDX-License-Identifier: AGPL-3.0-only
//! CUDA-event rep timer and occupancy query for `moe_verify_bench`.

use super::*;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks: *mut i32,
        func: u64,
        block: i32,
        smem: usize,
    ) -> i32;
}

/// Per-rep CUDA-event timer on one stream. `flush` memsets the scratch
/// before each rep; all reps are queued before one sync. Returns median us.
pub(super) struct Timer {
    pub(super) stream: u64,
    pub(super) scratch: DevicePtr,
    pub(super) reps: usize,
}

impl Timer {
    pub(super) fn time(
        &self,
        g: &dyn GpuBackend,
        flush: bool,
        f: &dyn Fn(usize) -> Result<()>,
    ) -> Result<f64> {
        f(0)?;
        g.synchronize(self.stream)?;
        let mut ev = vec![0u64; 2 * self.reps];
        for e in ev.iter_mut() {
            if unsafe { cuEventCreate(e, 0) } != 0 {
                bail!("cuEventCreate failed");
            }
        }
        for i in 0..self.reps {
            if flush {
                g.memset_async(self.scratch, i as u8, FLUSH_BYTES, self.stream)?;
            }
            if unsafe { cuEventRecord(ev[2 * i], self.stream) } != 0 {
                bail!("cuEventRecord failed");
            }
            f(i)?;
            if unsafe { cuEventRecord(ev[2 * i + 1], self.stream) } != 0 {
                bail!("cuEventRecord failed");
            }
        }
        if unsafe { cuEventSynchronize(ev[2 * self.reps - 1]) } != 0 {
            bail!("cuEventSynchronize failed");
        }
        let mut us = Vec::with_capacity(self.reps);
        for i in 0..self.reps {
            let mut ms = 0f32;
            if unsafe { cuEventElapsedTime(&mut ms, ev[2 * i], ev[2 * i + 1]) } != 0 {
                bail!("cuEventElapsedTime failed");
            }
            us.push(ms as f64 * 1e3);
        }
        for e in ev {
            unsafe { cuEventDestroy_v2(e) };
        }
        us.sort_by(f64::total_cmp);
        Ok(us[us.len() / 2])
    }
}

/// Resident CTAs per SM of `kernel` at `threads` (static shared memory only).
pub(super) fn ctas_per_sm(kernel: KernelHandle, threads: u32) -> Result<u32> {
    let mut n = 0i32;
    if unsafe { cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut n, kernel.0, threads as i32, 0) }
        != 0
    {
        bail!("cuOccupancyMaxActiveBlocksPerMultiprocessor failed");
    }
    Ok(n as u32)
}
