// SPDX-License-Identifier: AGPL-3.0-only
//! Air-VL routed matrices use checkpoint BF16 or OCP MXFP4/32. On GB10 the
//! device can read immutable shard mappings directly in either format.
use super::{LingProjection, LingProjectionScratch};
use crate::layer::ForwardContext;
use crate::layers::{ling_moe, ops};
use crate::weight_map::DenseWeight;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::ling_paging::{LingPage, LingPagingReader};
use spark_runtime::weights::{WeightDtype, WeightStore};
use std::sync::{Arc, Mutex};

const H: usize = 2560;
const I: usize = 768;
const E: usize = 512;
const TOP: u32 = 8;

pub(super) struct SanteFfn {
    layer: usize,
    mx: bool,
    tables: DevicePtr,
    _pages: Vec<LingPage>,
    router: DenseWeight,
    bias: DevicePtr,
    shared: [LingProjection; 3],
    scratch: LingProjectionScratch,
    shared_gate: DevicePtr,
    shared_up: DevicePtr,
    ids: DevicePtr,
    weights: DevicePtr,
    route_logits: KernelHandle,
    route_select: KernelHandle,
    project: KernelHandle,
    silu: KernelHandle,
    sum: KernelHandle,
    add: KernelHandle,
}

impl SanteFfn {
    pub(super) fn load(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        reader: &Arc<Mutex<LingPagingReader>>,
        layer: usize,
        scratch: LingProjectionScratch,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer}.mlp");
        let weight = store.get(&format!("{prefix}.gate.weight"))?;
        let bias = store.get(&format!("{prefix}.gate.expert_bias"))?;
        ensure!(weight.shape == [E, H] && weight.dtype == WeightDtype::BF16);
        ensure!(bias.shape == [E] && bias.dtype == WeightDtype::FP32);
        let shared = |suffix, n, k| {
            LingProjection::load(
                gpu,
                store,
                &format!("{prefix}.shared_experts.{suffix}"),
                n,
                k,
            )
        };
        let mx = store
            .deferred(&format!("{prefix}.experts.0.gate_proj.weight_scale_inv"))
            .is_some();
        ensure!(
            gpu.pageable_memory_access_uses_host_page_tables()?,
            "Air-VL on one Spark requires coherent host memory"
        );
        // MXFP4 fits on one GB10. A single allocation per layer keeps decode
        // on device memory and avoids a CUDA allocation for each expert tensor.
        let resident = if mx {
            Some(gpu.alloc(E * 3 * I * H / 32 * 17)?)
        } else {
            None
        };
        let mut resident_offset = 0;
        let mut pages = Vec::new();
        let mut ptrs = Vec::with_capacity(E * 6 * 8);
        let mut reader = reader
            .lock()
            .map_err(|_| anyhow::anyhow!("Air-VL paging reader poisoned"))?;
        for expert in 0..E {
            for (proj, n, k) in [("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)] {
                let base = format!("{prefix}.experts.{expert}.{proj}");
                let mut append = |suffix: &str, dtype, shape: [usize; 2]| -> Result<()> {
                    let name = format!("{base}.{suffix}");
                    let tensor = store
                        .deferred(&name)
                        .ok_or_else(|| anyhow::anyhow!("missing deferred {name}"))?;
                    ensure!(
                        tensor.dtype == dtype && tensor.shape == shape,
                        "unexpected dtype or shape for {name}"
                    );
                    let page = reader.page(tensor)?;
                    let ptr = if let Some(slab) = resident {
                        let ptr = slab.offset(resident_offset);
                        gpu.copy_h2d(page.bytes(), ptr)?;
                        resident_offset += page.len();
                        ptr
                    } else {
                        let ptr = gpu.pageable_host_ptr_to_device(page.host_ptr())?;
                        pages.push(page);
                        ptr
                    };
                    ensure!(!ptr.is_null());
                    ptrs.extend_from_slice(&ptr.0.to_le_bytes());
                    Ok(())
                };
                if mx {
                    append("weight", WeightDtype::UInt8, [n, k / 2])?;
                    append("weight_scale_inv", WeightDtype::FP8E8M0, [n, k / 32])?;
                } else {
                    append("weight", WeightDtype::BF16, [n, k])?;
                    ptrs.extend_from_slice(&0u64.to_le_bytes());
                }
            }
        }
        let tables = gpu.alloc(ptrs.len())?;
        gpu.copy_h2d(&ptrs, tables)?;
        Ok(Self {
            layer,
            mx,
            tables,
            _pages: pages,
            router: DenseWeight { weight: weight.ptr },
            bias: bias.ptr,
            shared: [
                shared("gate_proj", I, H)?,
                shared("up_proj", I, H)?,
                shared("down_proj", H, I)?,
            ],
            scratch,
            shared_gate: gpu.alloc(2 * I)?,
            shared_up: gpu.alloc(2 * I)?,
            ids: gpu.alloc(32)?,
            weights: gpu.alloc(32)?,
            route_logits: gpu.kernel("gemv", "dense_gemv_bf16_fp32out")?,
            route_select: gpu.kernel("ling_router", "ling_grouped_top8")?,
            project: gpu.kernel("ling_ffn", "sante_expert_gemv")?,
            silu: gpu.kernel("ling_ffn", "ling_silu_mul_bf16")?,
            sum: gpu.kernel("ling_ffn", "ling_weighted_sum")?,
            add: gpu.kernel("residual_add", "bf16_residual_add")?,
        })
    }

    pub(super) fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<DevicePtr> {
        let gpu = ctx.gpu;
        let b = ctx.buffers;
        ops::dense_gemv_bf16_fp32out(
            gpu,
            self.route_logits,
            input,
            &self.router,
            b.gate_logits_f32(),
            E as u32,
            H as u32,
            stream,
        )?;
        ops::ling_grouped_top8(
            gpu,
            self.route_select,
            b.gate_logits_f32(),
            self.bias,
            self.ids,
            self.weights,
            stream,
        )?;
        let project = |which: u32, x, out, n: usize, k: usize| {
            KernelLaunch::new(gpu, self.project)
                .grid([n.div_ceil(4) as u32, TOP, 1])
                .block([128, 1, 1])
                .arg_ptr(x)
                .arg_ptr(out)
                .arg_ptr(self.tables)
                .arg_ptr(self.ids)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(which)
                .arg_u32(self.mx as u32)
                .launch(stream)
        };
        project(0, input, b.expert_gate_out(), I, H)?;
        project(1, input, b.expert_up_out(), I, H)?;
        ops::ling_silu_mul_bf16(
            gpu,
            self.silu,
            b.expert_gate_out(),
            b.expert_up_out(),
            b.expert_gate_out(),
            TOP,
            I as u32,
            ling_moe::routed_swiglu_limit(self.layer),
            stream,
        )?;
        project(2, b.expert_gate_out(), b.expert_down_out(), H, I)?;
        ops::ling_weighted_sum(
            gpu,
            self.sum,
            b.expert_down_out(),
            self.weights,
            b.moe_output(),
            H as u32,
            TOP,
            stream,
        )?;
        self.shared[0].forward(gpu, input, self.shared_gate, 1, self.scratch, stream)?;
        self.shared[1].forward(gpu, input, self.shared_up, 1, self.scratch, stream)?;
        ops::ling_silu_mul_bf16(
            gpu,
            self.silu,
            self.shared_gate,
            self.shared_up,
            self.shared_gate,
            1,
            I as u32,
            ling_moe::shared_swiglu_limit(self.layer),
            stream,
        )?;
        self.shared[2].forward(
            gpu,
            self.shared_gate,
            b.attn_output(),
            1,
            self.scratch,
            stream,
        )?;
        ops::residual_add(
            gpu,
            self.add,
            b.moe_output(),
            b.attn_output(),
            H as u32,
            stream,
        )?;
        Ok(b.moe_output())
    }
}
