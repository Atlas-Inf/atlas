// SPDX-License-Identifier: AGPL-3.0-only

//! Device memory the host reads without a copy.
//!
//! A copy from the device into ordinary host memory takes one to three
//! milliseconds on this hardware, whatever its size, with the GPU idle: a
//! verify step paid for three of them around every draft pass. Where the
//! backend can, such a buffer is page-locked host memory that the device
//! writes straight into, and the host reads it once the stream's work is
//! done. Elsewhere it is the device's own and is copied as before.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// `bytes` bytes that kernels write at `at` and the host reads.
#[derive(Clone, Copy)]
pub(super) struct Seen {
    pub at: DevicePtr,
    /// The address of the host memory `at` is an alias of.
    host: Option<usize>,
    bytes: usize,
}

impl Seen {
    pub fn alloc(gpu: &dyn GpuBackend, bytes: usize) -> Result<Self> {
        let host = gpu.alloc_host_pinned(bytes)?;
        match gpu.host_ptr_to_device(host) {
            Ok(at) => Ok(Self {
                at,
                host: Some(host as usize),
                bytes,
            }),
            // A backend that maps nothing: the device's own memory.
            Err(_) => {
                gpu.free_host_pinned(host, bytes)?;
                Ok(Self {
                    at: gpu.alloc(bytes)?,
                    host: None,
                    bytes,
                })
            }
        }
    }

    /// The bytes, once everything `stream` was given has run.
    pub fn read(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<Vec<u8>> {
        gpu.synchronize(stream)?;
        match self.host {
            // SAFETY: `host` is the page-locked region `alloc` got with
            // `bytes` bytes, zeroed, and never frees. Only kernels write
            // it, through its device alias `at`, and the `synchronize`
            // above put their writes before this read.
            Some(host) => {
                Ok(unsafe { std::slice::from_raw_parts(host as *const u8, self.bytes) }.to_vec())
            }
            None => {
                let mut bytes = vec![0u8; self.bytes];
                gpu.copy_d2h(self.at, &mut bytes)?;
                Ok(bytes)
            }
        }
    }
}
