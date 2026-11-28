// SPDX-License-Identifier: AGPL-3.0-only

//! The router's top-k, the visit's KV page write and the window layers' KV
//! slot map (split from `glue.rs` for the 500-line cap; code moved as is).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::GlueKernels;

impl GlueKernels {
    /// Sigmoid top-k over fp32 logits `[rows, experts]`, normalised, x `scale`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn route(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        logits: DevicePtr,
        bias: DevicePtr,
        ids: DevicePtr,
        w: DevicePtr,
        experts: usize,
        top_k: usize,
        scale: f32,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.topk)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(bias)
            .arg_ptr(ids)
            .arg_ptr(w)
            .arg_u32(experts as u32)
            .arg_u32(top_k as u32)
            .arg_u32(1)
            .arg_f32(scale)
            .launch(s)
    }

    /// Write `rows` tokens' K and V into this visit's BF16 pages at `slots`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kv_write(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        k: DevicePtr,
        v: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        slots: DevicePtr,
        kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        s: u64,
    ) -> Result<()> {
        let stride = (kv_heads * head_dim) as u32;
        KernelLaunch::new(gpu, self.kv_write)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(k)
            .arg_ptr(v)
            .arg_ptr(k_pool)
            .arg_ptr(v_pool)
            .arg_ptr(slots)
            .arg_u32(kv_heads as u32)
            .arg_u32(head_dim as u32)
            .arg_u32(block_size as u32)
            .arg_u32(stride)
            .arg_u32(stride)
            .launch(s)
    }

    /// The window layers' KV slots and block tables for `rows` rows
    /// (`inkling2_window_meta`): `meta`'s, through the window `map`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::inkling2) fn window_meta(
        &self,
        gpu: &dyn GpuBackend,
        meta: &crate::layers::inkling2::batch::Meta,
        map: DevicePtr,
        (local_slots, local_tables): (DevicePtr, DevicePtr),
        rows: usize,
        block_size: usize,
        window: usize,
        s: u64,
    ) -> Result<()> {
        let tables = if meta.table_stride == 0 { 1 } else { rows };
        KernelLaunch::new(gpu, self.window_meta)
            .grid([tables as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(meta.slots)
            .arg_ptr(meta.positions)
            .arg_ptr(meta.block_table)
            .arg_ptr(map)
            .arg_ptr(local_slots)
            .arg_ptr(local_tables)
            .arg_u32(rows as u32)
            .arg_u32(meta.table_stride as u32)
            .arg_u32(block_size as u32)
            .arg_u32(window as u32)
            .launch(s)
    }
}
