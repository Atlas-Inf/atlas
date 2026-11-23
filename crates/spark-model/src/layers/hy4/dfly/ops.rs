// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::{gpu::KernelHandle, kernel_args::KernelLaunch};

#[derive(Clone, Copy)]
pub(super) enum Op {
    Embed,
    Norm,
    Add,
    Fuse,
    Silu,
    Gate,
    Pack,
    Attention,
    Heads,
    Head,
    Pick,
}
pub(super) struct Ops<'a> {
    pub gpu: &'a dyn GpuBackend,
    kernels: [KernelHandle; 11],
    config: Config,
}
impl<'a> Ops<'a> {
    pub fn load(gpu: &'a dyn GpuBackend, config: Config) -> Result<Self> {
        let names = [
            "embed",
            "norm",
            "add",
            "fuse",
            "silu",
            "gate",
            "pack",
            "attention",
        ];
        let mut handles = Vec::new();
        for name in names {
            handles.push(gpu.kernel("hy4_dfly", &format!("hy4_dfly_{name}"))?);
        }
        handles.push(gpu.kernel("hy4_mla", "hy4_head_gemv")?);
        handles.push(gpu.kernel("hy4_model", "hy4_model_head")?);
        handles.push(gpu.kernel("hy4_model", "hy4_model_argmax")?);
        Ok(Self {
            gpu,
            kernels: handles.try_into().unwrap(),
            config,
        })
    }
    fn kernel(&self, op: Op) -> KernelHandle {
        self.kernels[op as usize]
    }
    pub fn linear(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        dims: [usize; 3],
        stream: u64,
    ) -> Result<()> {
        let [m, n, k] = dims;
        spark_runtime::cublaslt::bf16_gemm_act_weight_t(
            x.0, w.0, out.0, m as u32, n as u32, k as u32, stream,
        )
    }
    pub fn norm(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        layout: [usize; 4],
        stream: u64,
    ) -> Result<()> {
        let [rows, h, xs, ys] = layout;
        KernelLaunch::new(self.gpu, self.kernel(Op::Norm))
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(h as u32)
            .arg_u32(xs as u32)
            .arg_u32(ys as u32)
            .arg_f32(self.config.eps)
            .launch(stream)
    }
    pub fn element(
        &self,
        op: Op,
        a: DevicePtr,
        b: DevicePtr,
        out: DevicePtr,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.kernel(op))
            .grid([(count as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(out)
            .arg_u32(count as u32)
            .launch(stream)
    }
    pub fn embed(
        &self,
        table: DevicePtr,
        ids: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.kernel(Op::Embed))
            .grid([(H as u32).div_ceil(256), rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(table)
            .arg_ptr(ids)
            .arg_ptr(out)
            .arg_u32(H as u32)
            .launch(stream)
    }
    pub fn fuse(
        &self,
        anchor: DevicePtr,
        projection: DevicePtr,
        gate: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.kernel(Op::Fuse))
            .grid([(H as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(anchor)
            .arg_ptr(projection)
            .arg_ptr(gate)
            .arg_ptr(out)
            .arg_u32(H as u32)
            .launch(stream)
    }
    pub fn gate(
        &self,
        values: DevicePtr,
        gates: DevicePtr,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.kernel(Op::Gate))
            .grid([(count as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(values)
            .arg_ptr(gates)
            .arg_u32(count as u32)
            .launch(stream)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn heads(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        layout: [usize; 4],
        stream: u64,
    ) -> Result<()> {
        let [n, k, xs, ys] = layout;
        let heads = 32 / self.config.world;
        for row in 0..rows {
            KernelLaunch::new(self.gpu, self.kernel(Op::Heads))
                .grid([(n as u32).div_ceil(8), heads as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(x.offset(row * heads * xs * 2))
                .arg_ptr(w)
                .arg_ptr(out.offset(row * heads * ys * 2))
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(xs as u32)
                .arg_u32(ys as u32)
                .launch(stream)?;
        }
        Ok(())
    }
    pub fn attention(
        &self,
        buffers: [DevicePtr; 7],
        rows: usize,
        anchor: usize,
        stream: u64,
    ) -> Result<()> {
        let [q, kv, absorbed, cache, indices, sink, output] = buffers;
        let heads = (32 / self.config.world) as u32;
        KernelLaunch::new(self.gpu, self.kernel(Op::Pack))
            .grid([heads, rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(q)
            .arg_ptr(kv)
            .arg_ptr(absorbed)
            .arg_ptr(cache)
            .arg_u32(heads)
            .arg_u32(anchor as u32)
            .arg_f32(self.config.theta)
            .launch(stream)?;
        KernelLaunch::new(self.gpu, self.kernel(Op::Attention))
            .grid([heads, rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(absorbed)
            .arg_ptr(cache)
            .arg_ptr(indices)
            .arg_ptr(sink)
            .arg_ptr(output)
            .arg_u32(heads)
            .arg_u32(anchor as u32)
            .arg_u32(2048)
            .launch(stream)
    }
    pub fn reduce(
        &self,
        p: DevicePtr,
        rows: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        if let Some(comm) = comm.filter(|c| c.world_size() > 1) {
            comm.all_reduce_async(p.0, rows * H * 2, stream)?;
        }
        Ok(())
    }
    pub fn head(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        logits: DevicePtr,
        id: DevicePtr,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let c = self.config;
        let local = c.vocab / c.world;
        let start = local * c.rank;
        KernelLaunch::new(self.gpu, self.kernel(Op::Head))
            .grid([(local as u32).div_ceil(8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(weight)
            .arg_ptr(logits.offset(start * 4))
            .arg_u32(local as u32)
            .arg_u32(H as u32)
            .arg_u32(0)
            .launch(stream)?;
        if c.world > 1 {
            crate::layers::glm_kv_shard::pair_exchange(
                comm.unwrap(),
                logits.offset(start * 4),
                logits.offset((local - start) * 4),
                local * 4,
                stream,
            )?;
        }
        KernelLaunch::new(self.gpu, self.kernel(Op::Pick))
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(id)
            .arg_u32(c.vocab as u32)
            .launch(stream)
    }
}
