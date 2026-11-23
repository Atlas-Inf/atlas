// SPDX-License-Identifier: AGPL-3.0-only

//! Ling 3.2's Qwen3-VL visual tower.
//!
//! The checkpoint keeps the Qwen ViT backbone but sets
//! `vision_config.disable_merger_proj=true`.  The 2×2 spatial merge therefore
//! emits 4608-wide rows directly after the merger's per-patch LayerNorm; the
//! text model's `linear_proj` then applies
//! `Linear(4608, 2560) -> GELU -> Linear(2560, 2560)`.  This module reuses the
//! existing BF16 Qwen [`VisionEncoder`] and only owns that final projector.

use anyhow::{Context, Result, ensure};
use atlas_core::config::{ModelConfig, VisionConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::vision_encoder::{ViTBlock, VisionEncoder, VitWeight};

const TEXT_HIDDEN: usize = 2560;
const RAW_VISION_HIDDEN: usize = 4608;
const PROJECTOR_HIDDEN: usize = TEXT_HIDDEN;

fn bf16_ptr(store: &WeightStore, name: &str, shape: &[usize]) -> Result<DevicePtr> {
    let tensor = store.get(name)?;
    ensure!(
        tensor.dtype == WeightDtype::BF16,
        "{name} must be BF16, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape == shape,
        "{name} must have shape {shape:?}, got {:?}",
        tensor.shape
    );
    Ok(tensor.ptr)
}

/// The checkpoint stores Conv3d patch weights in their native five-dimensional
/// layout. The existing ViT patch GEMM consumes the same contiguous values as
/// `[out, in * temporal * height * width]`, so no device reshape is needed.
fn patch_embed_ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let tensor = store.get(name)?;
    ensure!(
        tensor.dtype == WeightDtype::BF16,
        "{name} must be BF16, got {:?}",
        tensor.dtype
    );
    ensure!(
        tensor.shape == [1152, 3, 2, 16, 16] || tensor.shape == [1152, 1536],
        "{name} must have Conv3d shape [1152, 3, 2, 16, 16] (or its contiguous flattened form), got {:?}",
        tensor.shape
    );
    Ok(tensor.ptr)
}

fn block(store: &WeightStore, prefix: &str) -> Result<ViTBlock> {
    let h = 1152;
    let qkv = 3 * h;
    let m = 4304;
    Ok(ViTBlock {
        norm1_w: bf16_ptr(store, &format!("{prefix}.norm1.weight"), &[h])?,
        norm1_b: bf16_ptr(store, &format!("{prefix}.norm1.bias"), &[h])?,
        qkv_w: VitWeight::Bf16(bf16_ptr(
            store,
            &format!("{prefix}.attn.qkv.weight"),
            &[qkv, h],
        )?),
        qkv_b: bf16_ptr(store, &format!("{prefix}.attn.qkv.bias"), &[qkv])?,
        proj_w: VitWeight::Bf16(bf16_ptr(
            store,
            &format!("{prefix}.attn.proj.weight"),
            &[h, h],
        )?),
        proj_b: bf16_ptr(store, &format!("{prefix}.attn.proj.bias"), &[h])?,
        norm2_w: bf16_ptr(store, &format!("{prefix}.norm2.weight"), &[h])?,
        norm2_b: bf16_ptr(store, &format!("{prefix}.norm2.bias"), &[h])?,
        fc1_w: VitWeight::Bf16(bf16_ptr(
            store,
            &format!("{prefix}.mlp.linear_fc1.weight"),
            &[m, h],
        )?),
        fc1_b: bf16_ptr(store, &format!("{prefix}.mlp.linear_fc1.bias"), &[m])?,
        fc2_w: VitWeight::Bf16(bf16_ptr(
            store,
            &format!("{prefix}.mlp.linear_fc2.weight"),
            &[h, m],
        )?),
        fc2_b: bf16_ptr(store, &format!("{prefix}.mlp.linear_fc2.bias"), &[h])?,
    })
}

/// The two BF16 text-side projector matrices and their biases.
pub struct LingVisionProjector {
    fc1_w: DevicePtr,
    fc1_b: DevicePtr,
    fc2_w: DevicePtr,
    fc2_b: DevicePtr,
    k_gemm: KernelHandle,
    k_gelu: KernelHandle,
    buf_fc1: DevicePtr,
    buf_out: DevicePtr,
    max_rows: usize,
}

impl LingVisionProjector {
    fn load(store: &WeightStore, gpu: &dyn GpuBackend, max_rows: usize) -> Result<Self> {
        Ok(Self {
            fc1_w: bf16_ptr(
                store,
                "linear_proj.0.weight",
                &[TEXT_HIDDEN, RAW_VISION_HIDDEN],
            )?,
            fc1_b: bf16_ptr(store, "linear_proj.0.bias", &[TEXT_HIDDEN])?,
            fc2_w: bf16_ptr(store, "linear_proj.2.weight", &[TEXT_HIDDEN, TEXT_HIDDEN])?,
            fc2_b: bf16_ptr(store, "linear_proj.2.bias", &[TEXT_HIDDEN])?,
            k_gemm: gpu.kernel("vision_encoder", "vision_gemm_bias")?,
            k_gelu: gpu.kernel("vision_encoder", "vision_gelu")?,
            buf_fc1: gpu.alloc(max_rows * PROJECTOR_HIDDEN * 2)?,
            buf_out: gpu.alloc(max_rows * PROJECTOR_HIDDEN * 2)?,
            max_rows,
        })
    }

    fn forward(
        &self,
        input: DevicePtr,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        ensure!(
            rows <= self.max_rows,
            "Ling vision projector rows {rows} exceed capacity {}",
            self.max_rows
        );
        let n = PROJECTOR_HIDDEN as u32;
        let k = RAW_VISION_HIDDEN as u32;
        let m = rows as u32;
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(n, 32), div_ceil(m, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(input)
            .arg_ptr(self.fc1_w)
            .arg_ptr(self.fc1_b)
            .arg_ptr(self.buf_fc1)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil((rows * PROJECTOR_HIDDEN) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_fc1)
            .arg_u32((rows * PROJECTOR_HIDDEN) as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(n, 32), div_ceil(m, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.buf_fc1)
            .arg_ptr(self.fc2_w)
            .arg_ptr(self.fc2_b)
            .arg_ptr(self.buf_out)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(n)
            .launch(stream)
    }
}

/// Ling's visual tower, exposed through the same splice contract as Qwen and
/// RADIO towers. `raw.buf_out` is the 4608-wide spatial merge; `projector`
/// writes the final 2560-wide rows to `projector.buf_out`.
pub struct LingVisionTower {
    pub(crate) raw: Box<VisionEncoder>,
    projector: LingVisionProjector,
}

impl LingVisionTower {
    pub(crate) fn load(
        store: &WeightStore,
        config: &ModelConfig,
        vcfg: &VisionConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            config.hidden_size == TEXT_HIDDEN,
            "Ling text hidden size must be {TEXT_HIDDEN}, got {}",
            config.hidden_size
        );
        ensure!(
            vcfg.depth == 27
                && vcfg.hidden_size == 1152
                && vcfg.num_heads == 16
                && vcfg.patch_size == 16
                && vcfg.temporal_patch_size == 2
                && vcfg.spatial_merge_size == 2
                && vcfg.intermediate_size == 4304,
            "unsupported Ling vision geometry: depth={}, hidden={}, heads={}, patch={}, temporal={}, merge={}, intermediate={}",
            vcfg.depth,
            vcfg.hidden_size,
            vcfg.num_heads,
            vcfg.patch_size,
            vcfg.temporal_patch_size,
            vcfg.spatial_merge_size,
            vcfg.intermediate_size
        );
        ensure!(
            vcfg.deepstack_visual_indexes.is_empty(),
            "Ling vision has no deepstack mergers"
        );

        let vp = "model.visual";
        let patch_embed_w = patch_embed_ptr(store, &format!("{vp}.patch_embed.proj.weight"))?;
        let patch_embed_b = bf16_ptr(store, &format!("{vp}.patch_embed.proj.bias"), &[1152])?;
        let pos_name = format!("{vp}.pos_embed.weight");
        let pos_embed = bf16_ptr(store, &pos_name, &[2304, 1152])?;
        let merger_norm_w = bf16_ptr(store, &format!("{vp}.merger.norm.weight"), &[1152])?;
        let merger_norm_b = bf16_ptr(store, &format!("{vp}.merger.norm.bias"), &[1152])?;
        let mut blocks = Vec::with_capacity(vcfg.depth);
        for i in 0..vcfg.depth {
            blocks.push(block(store, &format!("{vp}.blocks.{i}"))?);
        }

        let raw = Box::new(
            VisionEncoder::new_raw(
                patch_embed_w,
                patch_embed_b,
                pos_embed,
                2304,
                blocks,
                merger_norm_w,
                merger_norm_b,
                vcfg.hidden_size,
                vcfg.num_heads,
                vcfg.spatial_merge_size,
                vcfg.intermediate_size,
                vcfg.patch_size,
                vcfg.max_pixels,
                gpu,
            )
            .context("construct Ling raw Qwen vision encoder")?,
        );
        let max_rows = raw.p_max / (vcfg.spatial_merge_size * vcfg.spatial_merge_size);
        let projector = LingVisionProjector::load(store, gpu, max_rows)?;
        tracing::info!(
            "Ling vision tower loaded: Qwen ViT depth={}, raw_width={}, out_width={}, max_rows={}",
            vcfg.depth,
            RAW_VISION_HIDDEN,
            TEXT_HIDDEN,
            max_rows
        );
        Ok(Self { raw, projector })
    }

    pub fn out_row(&self) -> DevicePtr {
        self.projector.buf_out
    }

    pub fn out_hidden_size(&self) -> usize {
        TEXT_HIDDEN
    }

    pub fn output_rows(&self) -> usize {
        self.projector.max_rows
    }

    pub fn forward_items(
        &self,
        items: &[&crate::VisionItem],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let result = self.raw.forward_items(items, gpu, stream)?;
        let rows = result.iter().map(|(_, _, merged)| *merged).sum();
        self.projector
            .forward(self.raw.buf_out, rows, gpu, stream)?;
        Ok(result)
    }

    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let result = self.raw.forward_batched(images, gpu, stream)?;
        let rows = result.iter().map(|(_, _, merged)| *merged).sum();
        self.projector
            .forward(self.raw.buf_out, rows, gpu, stream)?;
        Ok(result)
    }
}

/// Loader hook used by the `bailing_moe_v3_vl` checkpoint boundary.
pub(crate) fn load_ling_vision(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<super::VisionTower>> {
    let Some(vcfg) = config.vision.as_ref() else {
        return Ok(None);
    };
    Ok(Some(super::VisionTower::Ling(Box::new(
        LingVisionTower::load(store, config, vcfg, gpu)?,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_geometry_has_expected_projector_widths() {
        assert_eq!(RAW_VISION_HIDDEN, 1152 * 2 * 2);
        assert_eq!(PROJECTOR_HIDDEN, 2560);
    }
}
