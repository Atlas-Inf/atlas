// SPDX-License-Identifier: AGPL-3.0-only

//! `BufferArena` strict-verify row masks (`spark_model`'s `glm_verify_masks`).

use super::BufferArena;
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;

impl BufferArena {
    /// Allocate `bytes` for the per-row grammar masks of a strict verify.
    /// Done at load, before KV sizing, on every rank, so a verify never
    /// allocates after the ranks have committed to the masked protocol.
    pub fn attach_verify_masks(&mut self, bytes: usize, gpu: &dyn GpuBackend) -> Result<()> {
        anyhow::ensure!(
            self.verify_masks.is_null() && bytes > 0,
            "verify masks need one attach of a positive size"
        );
        self.verify_masks = gpu.alloc(bytes)?;
        self.verify_masks_bytes = bytes;
        Ok(())
    }

    /// `(masks, bytes)` when attached.
    pub fn verify_masks(&self) -> Option<(DevicePtr, usize)> {
        (!self.verify_masks.is_null()).then_some((self.verify_masks, self.verify_masks_bytes))
    }
}
