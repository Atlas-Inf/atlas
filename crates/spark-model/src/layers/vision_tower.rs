// SPDX-License-Identifier: AGPL-3.0-only

//! `VisionTower` — the model-side handle on either vision encoder family:
//! the Qwen-VL tower (`VisionEncoder`) or the omni RADIO tower
//! (`RadioVisionTower`). The prefill splice sites stay one-liners via
//! `forward_batched` / `out_row` / `out_hidden_size` on the enum.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::radio_tower::RadioVisionTower;

use super::VisionEncoder;

pub enum VisionTower {
    Qwen(Box<VisionEncoder>),
    Radio(Box<RadioVisionTower>),
}

impl VisionTower {
    /// Packed merged-token output rows (`[total_merged, out_hidden]` BF16).
    pub fn out_row(&self) -> DevicePtr {
        match self {
            VisionTower::Qwen(v) => v.buf_out,
            VisionTower::Radio(r) => r.buf_out,
        }
    }

    pub fn out_hidden_size(&self) -> usize {
        match self {
            VisionTower::Qwen(v) => v.out_hidden_size,
            VisionTower::Radio(r) => r.out_hidden_size,
        }
    }

    /// Encode every `(pixels, grid_h, grid_w)` item into the packed
    /// `out_row` buffer; returns per-item `(h', w', merged_p)`.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        match self {
            VisionTower::Qwen(v) => v.forward_batched(images, gpu, stream),
            VisionTower::Radio(r) => r.forward_batched(images, gpu, stream),
        }
    }
}
