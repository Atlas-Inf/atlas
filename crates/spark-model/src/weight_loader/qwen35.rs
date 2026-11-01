// SPDX-License-Identifier: AGPL-3.0-only

pub(crate) mod load_layers;

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense_auto, detect_nvfp4_variant, load_mtp};

pub struct Qwen35WeightLoader;

/// A ViT projection in its native precision: BF16 stays a raw pointer;
/// FP4-Local's MXFP8 (`attn.qkv`/`attn.proj`) and NVFP4 (`linear_fc{1,2}`)
/// tensors upload as shipped and expand on-GPU per GEMM — bit-identical to
/// the host expansion other checkpoints get.
/// A quantized tensor's resident pointer: resident if materialize uploaded
/// it natively, else the deferred shard bytes uploaded verbatim (the vision
/// NVFP4 tensors stay deferred — nothing else consumes them).
#[cfg(feature = "cuda")]
fn vision_tensor_ptr(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<(
    spark_runtime::gpu::DevicePtr,
    spark_runtime::weights::WeightDtype,
    Vec<usize>,
)> {
    if let Ok(t) = store.get(name) {
        return Ok((t.ptr, t.dtype, t.shape.clone()));
    }
    let d = store
        .deferred(name)
        .with_context(|| format!("vision: `{name}` neither resident nor deferred"))?;
    let len = d.shape.iter().product::<usize>();
    let bytes = len
        * match d.dtype {
            WeightDtype::BF16 | WeightDtype::FP16 => 2,
            WeightDtype::FP32 => 4,
            _ => 1,
        };
    use std::io::Read as _;
    let mut buf = vec![0u8; bytes];
    let mut f = std::fs::File::open(&d.path)?;
    use std::io::Seek as _;
    f.seek(std::io::SeekFrom::Start(d.offset))?;
    f.read_exact(&mut buf)?;
    let p = gpu.alloc(bytes)?;
    gpu.copy_h2d(&buf, p)?;
    Ok((p, d.dtype, d.shape.clone()))
}

#[cfg(feature = "cuda")]
fn vision_weight(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<crate::layers::VitWeight> {
    let name = format!("{prefix}.weight");
    let (wptr, wdtype, wshape) = vision_tensor_ptr(store, &name, gpu)?;
    match wdtype {
        WeightDtype::BF16 => Ok(crate::layers::VitWeight::Bf16(wptr)),
        WeightDtype::FP8E4M3 => {
            // MXFP8: E4M3 [n,k] + U8 E8M0 [n,k/32].
            let (sptr, sdtype, sshape) =
                vision_tensor_ptr(store, &format!("{prefix}.weight_scale"), gpu)?;
            anyhow::ensure!(
                wshape.len() == 2
                    && sdtype == WeightDtype::UInt8
                    && sshape == [wshape[0], wshape[1] / 32],
                "vision MXFP8 {prefix}: weight {wshape:?} + scale {sshape:?}"
            );
            Ok(crate::layers::VitWeight::Mxfp8(
                crate::weight_map::Mxfp8Proj {
                    w: crate::weight_map::Mxfp8Weight {
                        data: wptr,
                        scales: sptr,
                    },
                    n: wshape[0] as u32,
                    k: wshape[1] as u32,
                },
            ))
        }
        WeightDtype::UInt8 => {
            // NVFP4: packed U8 [n,k/2] + E4M3 [n,k/16] + F32 weight_scale_2.
            let (sptr, sdtype, sshape) =
                vision_tensor_ptr(store, &format!("{prefix}.weight_scale"), gpu)?;
            anyhow::ensure!(
                wshape.len() == 2 && sdtype == WeightDtype::FP8E4M3 && sshape[0] == wshape[0],
                "vision NVFP4 {prefix}: weight {wshape:?} + scale {sshape:?}"
            );
            let gname = format!("{prefix}.weight_scale_2");
            let gt = store.get(&gname)?;
            let mut gb = [0u8; 4];
            gpu.copy_d2h(gt.ptr, &mut gb)?;
            store.reclaim(gpu, &gname)?;
            Ok(crate::layers::VitWeight::Nvfp4 {
                packed: wptr,
                scales: sptr,
                global: f32::from_le_bytes(gb),
                n: wshape[0] as u32,
                k: (wshape[1] * 2) as u32,
            })
        }
        other => anyhow::bail!("vision_weight: unsupported dtype {other:?} for {name}"),
    }
}

fn vision_tensor_dense_auto(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP32 => crate::weight_map::dense_f32_safe(store, name, gpu),
        other => anyhow::bail!("vision_tensor_dense_auto: unsupported dtype {other:?} for {name}"),
    }
}

impl ModelWeightLoader for Qwen35WeightLoader {
    fn supports_tp(&self) -> bool {
        // FullAttention layers are TP-sharded across all 3 quant paths
        // (FP8 native, NVFP4-from-disk, BF16 → NVFP4). LinearAttention
        // (GDN SSM) layers are now TP-sharded head-parallel for the BF16
        // and NVFP4 paths (GDN HeadParallel): linear_num_key/value_heads
        // are divided per rank in topology.rs, each rank owns a contiguous
        // head range, and out_proj is row-parallel with one all-reduce.
        // Native block-scaled FP8 SSM still requires TP=1 (per-128-row
        // scale slicing deferred) — build_linear_attention_fp8 errors
        // clearly when tp_size > 1.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(self, store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense_auto(store, &format!("{prefix}.embed_tokens.weight"), gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense_auto(store, &format!("{prefix}.norm.weight"), gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // lm_head location varies by quantizer:
        //   Sehyo: "lm_head.weight"
        //   Kbenkhaled: "language_model.lm_head.weight"
        //
        // Dequant FP8 ONLY; hand every other dtype through untouched.
        // `dense` does no dtype check, which is correct for a BF16 head and for
        // a Standard-NVFP4 head (`weight` U8-packed + weight_scale_2, e.g.
        // nvidia/Qwen3.6-*-NVFP4) that the consumer unpacks itself — routing
        // those through `dense_auto_fp8_or_bf16` hard-errors on `unsupported
        // dtype UInt8`. But it is WRONG for FP8: mixed-precision checkpoints
        // (unsloth Qwen3.6-*-NVFP4, 2026-07-10) keep lm_head as FP8 E4M3 +
        // per-row `weight_scale`, and feeding those bytes to a BF16 GEMM reads
        // 2x the allocation on the largest tensor in the model →
        // CUDA_ERROR_ILLEGAL_ADDRESS at the first sync after build.
        for prefix in ["lm_head", "language_model.lm_head", "model.lm_head"] {
            let key = format!("{prefix}.weight");
            if !store.contains(&key) {
                continue;
            }
            let is_fp8 = store
                .get(&key)
                .map(|w| w.dtype == WeightDtype::FP8E4M3)
                .unwrap_or(false);
            return if is_fp8 {
                crate::weight_map::dense_auto_fp8_or_bf16(store, prefix, gpu)
            } else {
                crate::weight_map::dense(store, &key)
            };
        }
        // Tied embeddings: the head IS the embedding table.
        let prefix = &config.weight_prefix;
        crate::weight_map::dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("No MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading MTP weights ({} experts, variant={:?})...",
            config.num_experts,
            variant
        );
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        tracing::info!(
            "MTP weights loaded: fc=[2048,4096], {} experts, attn layer",
            mtp.experts.len(),
        );
        Ok(Some(mtp))
    }

    /// Load the Qwen3.6 ViT tower. Returns `None` when `config.vision` is
    /// `None` (Qwen3.5 text-only). Otherwise matches the Qwen3-VL shape
    /// exactly (27 blocks, `model.visual.*` prefix, optional deepstack
    /// merger list + final merger) but auto-dequants FP8 per-channel
    /// weights to BF16 for blocks 4+. Blocks 0-3 are exempted in the
    /// checkpoint's `modules_to_not_convert` list and stay BF16 on disk.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionTower>> {
        let vcfg = match &config.vision {
            Some(v) => v.clone(),
            None => return Ok(None),
        };
        // AEON-7's v2 NVFP4 re-quant (and other multimodal-preserved
        // checkpoints quantized via AutoModelForImageTextToText) keeps
        // the canonical nested layout `model.language_model.visual.*`
        // instead of the flat `model.visual.*` form. Probe the canonical
        // tensor under both prefixes; first hit wins.
        let vp = if store.contains("model.visual.patch_embed.proj.weight") {
            "model.visual"
        } else if store.contains("model.language_model.visual.patch_embed.proj.weight") {
            "model.language_model.visual"
        } else {
            tracing::warn!(
                "Vision encoder tensors absent under both `model.visual.*` and \
                 `model.language_model.visual.*`; skipping vision tower (text-only mode)"
            );
            return Ok(None);
        };

        let patch_embed_w =
            vision_tensor_dense_auto(store, &format!("{vp}.patch_embed.proj.weight"), gpu)?;
        let patch_embed_b =
            vision_tensor_dense_auto(store, &format!("{vp}.patch_embed.proj.bias"), gpu)?;
        let pos_embed = vision_tensor_dense_auto(store, &format!("{vp}.pos_embed.weight"), gpu)?;
        let pos_embed_shape = store.get(&format!("{vp}.pos_embed.weight"))?.shape.clone();
        let num_position_embeddings = pos_embed_shape
            .first()
            .copied()
            .context("pos_embed shape missing rows")?;

        let mut blocks = Vec::with_capacity(vcfg.depth);
        for i in 0..vcfg.depth {
            let bp = format!("{vp}.blocks.{i}");
            blocks.push(crate::layers::ViTBlock {
                norm1_w: vision_tensor_dense_auto(store, &format!("{bp}.norm1.weight"), gpu)?
                    .weight,
                norm1_b: vision_tensor_dense_auto(store, &format!("{bp}.norm1.bias"), gpu)?.weight,
                qkv_w: vision_weight(store, &format!("{bp}.attn.qkv"), gpu)?,
                qkv_b: vision_tensor_dense_auto(store, &format!("{bp}.attn.qkv.bias"), gpu)?.weight,
                proj_w: vision_weight(store, &format!("{bp}.attn.proj"), gpu)?,
                proj_b: vision_tensor_dense_auto(store, &format!("{bp}.attn.proj.bias"), gpu)?
                    .weight,
                norm2_w: vision_tensor_dense_auto(store, &format!("{bp}.norm2.weight"), gpu)?
                    .weight,
                norm2_b: vision_tensor_dense_auto(store, &format!("{bp}.norm2.bias"), gpu)?.weight,
                fc1_w: vision_weight(store, &format!("{bp}.mlp.linear_fc1"), gpu)?,
                fc1_b: vision_tensor_dense_auto(store, &format!("{bp}.mlp.linear_fc1.bias"), gpu)?
                    .weight,
                fc2_w: vision_weight(store, &format!("{bp}.mlp.linear_fc2"), gpu)?,
                fc2_b: vision_tensor_dense_auto(store, &format!("{bp}.mlp.linear_fc2.bias"), gpu)?
                    .weight,
            });
        }

        let mut deepstack = Vec::with_capacity(vcfg.deepstack_visual_indexes.len());
        for i in 0..vcfg.deepstack_visual_indexes.len() {
            let mp = format!("{vp}.deepstack_merger_list.{i}");
            deepstack.push(crate::layers::MergerLayer {
                norm_w: vision_tensor_dense_auto(store, &format!("{mp}.norm.weight"), gpu)?.weight,
                norm_b: vision_tensor_dense_auto(store, &format!("{mp}.norm.bias"), gpu)?.weight,
                fc1_w: vision_weight(store, &format!("{mp}.linear_fc1"), gpu)?,
                fc1_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc1.bias"), gpu)?
                    .weight,
                fc2_w: vision_weight(store, &format!("{mp}.linear_fc2"), gpu)?,
                fc2_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc2.bias"), gpu)?
                    .weight,
            });
        }

        let mp = format!("{vp}.merger");
        let merger = crate::layers::MergerLayer {
            norm_w: vision_tensor_dense_auto(store, &format!("{mp}.norm.weight"), gpu)?.weight,
            norm_b: vision_tensor_dense_auto(store, &format!("{mp}.norm.bias"), gpu)?.weight,
            fc1_w: vision_weight(store, &format!("{mp}.linear_fc1"), gpu)?,
            fc1_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc1.bias"), gpu)?.weight,
            fc2_w: vision_weight(store, &format!("{mp}.linear_fc2"), gpu)?,
            fc2_b: vision_tensor_dense_auto(store, &format!("{mp}.linear_fc2.bias"), gpu)?.weight,
        };

        let deepstack_indexes = vcfg.deepstack_visual_indexes.clone();
        let ve = crate::layers::VisionEncoder::new(
            patch_embed_w.weight,
            patch_embed_b.weight,
            pos_embed.weight,
            num_position_embeddings,
            blocks,
            deepstack,
            deepstack_indexes,
            merger,
            vcfg.hidden_size,
            vcfg.num_heads,
            vcfg.spatial_merge_size,
            vcfg.out_hidden_size,
            vcfg.intermediate_size,
            vcfg.patch_size,
            vcfg.max_pixels,
            gpu,
        )?;
        tracing::info!(
            "Qwen3.6 vision encoder loaded: depth={}, hidden={}, heads={}, FP8-blocks>=4",
            vcfg.depth,
            vcfg.hidden_size,
            vcfg.num_heads,
        );
        Ok(Some(crate::layers::VisionTower::Qwen(Box::new(ve))))
    }
}
