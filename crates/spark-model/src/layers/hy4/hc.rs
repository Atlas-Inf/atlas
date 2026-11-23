// SPDX-License-Identifier: AGPL-3.0-only

use crate::weight_map::dense_keep_f32;
use anyhow::{Result, ensure};
use atlas_core::{config::ModelConfig, hy4_reference::hc::HcParams};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
    weights::{WeightDtype, WeightStore},
};

/// Borrows FP32 checkpoint weights; owns any BF16-to-FP32 promotions.
pub struct IndependentHc<'a> {
    _store: &'a WeightStore,
    gpu: &'a dyn GpuBackend,
    promoted: Vec<DevicePtr>,
    params: HcParams,
    weights: [DevicePtr; 3],
    pre_kernel: KernelHandle,
    post_kernel: KernelHandle,
    norm_kernel: KernelHandle,
    scatter_norm_kernel: KernelHandle,
    is_head: bool,
}

impl<'a> IndependentHc<'a> {
    /// `prefix` is `model.layers.N.hc_{attn,mlp}_layer.hc_pre` or `model.hc_head`.
    pub fn load(
        store: &'a WeightStore,
        prefix: &str,
        head: bool,
        config: &ModelConfig,
        gpu: &'a dyn GpuBackend,
    ) -> Result<Self> {
        let hy = config
            .hy4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Hy4 metadata required"))?;
        let params = HcParams {
            hidden: config.hidden_size,
            mult: config.hc_mult,
            norm_eps: config.rms_norm_eps as f32,
            gate_eps: config.hc_eps,
            magnitude: hy.hc_magnitude as f32,
        };
        params.validate()?;
        let rows = if head { params.mult } else { 2 * params.mult };
        let stem = if head { "hc_head" } else { "hc" };
        let specs = [
            (
                format!("{prefix}.{stem}_fn"),
                vec![rows, params.mult * params.hidden],
            ),
            (
                format!("{prefix}.{stem}_scale"),
                vec![if head { 1 } else { 2 }],
            ),
            (format!("{prefix}.{stem}_base"), vec![rows]),
        ];
        for (name, shape) in &specs {
            let tensor = store.get(name)?;
            ensure!(
                matches!(tensor.dtype, WeightDtype::FP32 | WeightDtype::BF16)
                    && tensor.shape == *shape,
                "Hy4 iHC {name}: expected FP32/BF16 {shape:?}"
            );
        }
        let mut block = Self {
            _store: store,
            gpu,
            promoted: Vec::new(),
            params,
            weights: [DevicePtr::NULL; 3],
            pre_kernel: gpu.kernel("hy4_hc", if head { "hy4_hc_head" } else { "hy4_hc_pre" })?,
            post_kernel: gpu.kernel("hy4_hc", "hy4_hc_post")?,
            norm_kernel: gpu.kernel(
                "hy4_hc",
                if head {
                    "hy4_hc_head_norm"
                } else {
                    "hy4_hc_pre_norm"
                },
            )?,
            scatter_norm_kernel: gpu.kernel("hy4_hc", "hy4_hc_post_pre_norm")?,
            is_head: head,
        };
        for (i, (name, _)) in specs.iter().enumerate() {
            let weight = dense_keep_f32(store, name, gpu)?.weight;
            block.weights[i] = weight;
            if store.get(name)?.dtype == WeightDtype::BF16 {
                block.promoted.push(weight);
            }
        }
        Ok(block)
    }

    pub fn release(&mut self) -> Result<()> {
        self.weights = [DevicePtr::NULL; 3];
        while let Some(ptr) = self.promoted.last() {
            self.gpu.free(*ptr)?;
            self.promoted.pop();
        }
        Ok(())
    }

    /// `[T,4,H]` BF16 -> `[T,H]` BF16; post `[T,4]` FP32.
    /// Head needs no post buffer. No Sinkhorn comb matrix is produced.
    pub fn collapse(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        y: DevicePtr,
        post: DevicePtr,
        tokens: u32,
        stream: u64,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        ensure!(
            self.weights.iter().all(|p| !p.is_null()),
            "Hy4 iHC already released"
        );
        ensure!(
            !x.is_null() && !y.is_null() && (self.is_head || !post.is_null()),
            "Hy4 iHC null buffer"
        );
        let p = self.params;
        let launch = KernelLaunch::new(gpu, self.pre_kernel)
            .grid([tokens, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.weights[0])
            .arg_ptr(self.weights[1])
            .arg_ptr(self.weights[2])
            .arg_ptr(y);
        let launch = if self.is_head {
            launch
        } else {
            launch.arg_ptr(post)
        };
        let launch = launch
            .arg_u32(u32::try_from(p.hidden)?)
            .arg_f32(p.norm_eps)
            .arg_f32(p.gate_eps);
        let launch = if self.is_head {
            launch
        } else {
            launch.arg_f32(p.magnitude)
        };
        launch.launch(stream)
    }

    /// `[T,4,H]` BF16 -> RMSNorm(`[T,H]`); post `[T,4]` FP32 unless head.
    pub fn collapse_norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        post: DevicePtr,
        tokens: u32,
        stream: u64,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        ensure!(
            self.weights.iter().all(|p| !p.is_null()),
            "Hy4 iHC already released"
        );
        ensure!(
            !x.is_null()
                && !weight.is_null()
                && !out.is_null()
                && (self.is_head || !post.is_null()),
            "Hy4 iHC null buffer"
        );
        let p = self.params;
        ensure!(p.hidden <= 8192, "Hy4 iHC fused norm hidden bound");
        let launch = KernelLaunch::new(gpu, self.norm_kernel)
            .grid([tokens, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.weights[0])
            .arg_ptr(self.weights[1])
            .arg_ptr(self.weights[2])
            .arg_ptr(out)
            .arg_ptr(weight);
        let launch = if self.is_head {
            launch
        } else {
            launch.arg_ptr(post)
        };
        let launch = launch
            .arg_u32(u32::try_from(p.hidden)?)
            .arg_f32(p.norm_eps)
            .arg_f32(p.gate_eps);
        let launch = if self.is_head {
            launch
        } else {
            launch.arg_f32(p.magnitude)
        };
        launch.arg_f32(p.norm_eps).launch(stream)
    }

    /// Fuses the preceding block scatter with this HC's collapse+norm for small batches.
    /// `post` carries the scatter gates in and this HC's post gates out; `highway` is updated in place.
    pub fn scatter_collapse_norm(
        &self,
        gpu: &dyn GpuBackend,
        block: DevicePtr,
        highway: DevicePtr,
        post: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        tokens: u32,
        stream: u64,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        ensure!(!self.is_head, "Hy4 head has no scatter collapse");
        ensure!(
            self.weights.iter().all(|p| !p.is_null()),
            "Hy4 iHC already released"
        );
        ensure!(
            !block.is_null()
                && !highway.is_null()
                && !post.is_null()
                && !weight.is_null()
                && !out.is_null(),
            "Hy4 iHC null buffer"
        );
        let p = self.params;
        ensure!(p.hidden <= 4096, "Hy4 iHC fused scatter norm hidden bound");
        let shared = u32::try_from(4 * p.hidden * 2)?;
        KernelLaunch::new(gpu, self.scatter_norm_kernel)
            .grid([tokens, 1, 1])
            .block([256, 1, 1])
            .shared_mem(shared)
            .arg_ptr(block)
            .arg_ptr(highway)
            .arg_ptr(post)
            .arg_ptr(self.weights[0])
            .arg_ptr(self.weights[1])
            .arg_ptr(self.weights[2])
            .arg_ptr(out)
            .arg_ptr(weight)
            .arg_u32(u32::try_from(p.hidden)?)
            .arg_f32(p.norm_eps)
            .arg_f32(p.gate_eps)
            .arg_f32(p.magnitude)
            .arg_f32(p.norm_eps)
            .launch(stream)
    }

    /// Residual and out may alias. Scatter rounds the BF16 highway each site.
    pub fn scatter(
        &self,
        gpu: &dyn GpuBackend,
        buffers: [DevicePtr; 4],
        tokens: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(!self.is_head, "Hy4 head has no scatter");
        if tokens == 0 {
            return Ok(());
        }
        ensure!(buffers.iter().all(|p| !p.is_null()), "Hy4 iHC null buffer");
        KernelLaunch::new(gpu, self.post_kernel)
            .grid([tokens, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(buffers[0])
            .arg_ptr(buffers[1])
            .arg_ptr(buffers[2])
            .arg_ptr(buffers[3])
            .arg_u32(u32::try_from(self.params.hidden)?)
            .launch(stream)
    }
}

impl Drop for IndependentHc<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            tracing::error!("Hy4 iHC cleanup: {e:#}");
        }
    }
}
