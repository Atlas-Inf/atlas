// SPDX-License-Identifier: AGPL-3.0-only

use super::loader::Hy4Attention;
use anyhow::Result;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};

pub(super) struct Kernels {
    pub linear: KernelHandle,
    pub fp8: KernelHandle,
    pub join: KernelHandle,
    pub norm: KernelHandle,
    pub gemv: KernelHandle,
    pub rotary: KernelHandle,
    pub gate: KernelHandle,
    pub attention: KernelHandle,
    pub merge: KernelHandle,
    pub pool: KernelHandle,
    pub fwht: KernelHandle,
    pub score: KernelHandle,
    pub select: KernelHandle,
    pub fwht_batch: KernelHandle,
    pub score_batch: KernelHandle,
    pub select_batch: KernelHandle,
}
impl Kernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            linear: gpu.kernel("hy4_ffn", "hy4_linear")?,
            fp8: gpu.kernel("hy4_mla", "hy4_fp8_gemv")?,
            join: gpu.kernel("hy4_mla", "hy4_join_qkv")?,
            norm: gpu.kernel("hy4_ffn", "hy4_norm_add")?,
            gemv: gpu.kernel("hy4_mla", "hy4_head_gemv")?,
            rotary: gpu.kernel("hy4_mla", "hy4_rotary")?,
            gate: gpu.kernel("hy4_mla", "hy4_gate")?,
            attention: gpu.kernel("hy4_mla", "hy4_mla_partial")?,
            merge: gpu.kernel("hy4_mla", "hy4_mla_merge")?,
            pool: gpu.kernel("hy4_indexpool", "hy4_pool")?,
            fwht: gpu.kernel("hy4_indexpool", "hy4_fwht")?,
            score: gpu.kernel("hy4_indexpool", "hy4_index_scores")?,
            select: gpu.kernel("hy4_indexpool", "hy4_index_select")?,
            fwht_batch: gpu.kernel("hy4_indexpool", "hy4_fwht_batch")?,
            score_batch: gpu.kernel("hy4_indexpool", "hy4_index_scores_batch")?,
            select_batch: gpu.kernel("hy4_indexpool", "hy4_index_select_batch")?,
        })
    }
}
impl Hy4Attention<'_> {
    pub(super) fn linear(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        if self.scales.contains_key(&w) {
            return self.fp8(x, w, y, [n, k, 0, 0], 1, stream);
        }
        KernelLaunch::new(self.memory.gpu, self.kernels.linear)
            .grid([n.div_ceil(8) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(0)
            .arg_u32(0)
            .launch(stream)
    }
    pub(super) fn norm(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let _phase = crate::layers::hy4::timing::span("attn_norm", stream)?;
        KernelLaunch::new(self.memory.gpu, self.kernels.norm)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rotary(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        heads: usize,
        layout: [usize; 4],
        position: [i64; 3],
        split: bool,
        stream: u64,
    ) -> Result<()> {
        let _phase = crate::layers::hy4::timing::span("attn_rope", stream)?;
        let [input_stride, output_stride, input_offset, output_offset] = layout;
        KernelLaunch::new(self.memory.gpu, self.kernels.rotary)
            .grid([heads as u32, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(y)
            .arg_u32(input_stride as u32)
            .arg_u32(output_stride as u32)
            .arg_u32(input_offset as u32)
            .arg_u32(output_offset as u32)
            .arg_u64(position[0] as u64)
            .arg_u64(position[1] as u64)
            .arg_u64(position[2] as u64)
            .arg_f32(self.rope.rope_theta as f32)
            .arg_u32(self.rope.mrope_section[1] as u32)
            .arg_u32(self.rope.mrope_section[2] as u32)
            .arg_u32(u32::from(self.rope.mrope_interleaved))
            .arg_u32(u32::from(split))
            .launch(stream)
    }
    pub(super) fn gemv(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        dimensions: [usize; 4],
        stream: u64,
    ) -> Result<()> {
        let [n, k, input_stride, output_stride] = dimensions;
        if self.scales.contains_key(&w) {
            return self.fp8(x, w, y, dimensions, self.heads, stream);
        }
        KernelLaunch::new(self.memory.gpu, self.kernels.gemv)
            .grid([n.div_ceil(8) as u32, self.heads as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(input_stride as u32)
            .arg_u32(output_stride as u32)
            .launch(stream)
    }
    #[allow(clippy::too_many_arguments)]
    fn fp8(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        dimensions: [usize; 4],
        heads: usize,
        stream: u64,
    ) -> Result<()> {
        let [n, k, input_stride, output_stride] = dimensions;
        KernelLaunch::new(self.memory.gpu, self.kernels.fp8)
            .grid([n.div_ceil(8) as u32, heads as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(self.scales[&w])
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(input_stride as u32)
            .arg_u32(output_stride as u32)
            .launch(stream)
    }
}
