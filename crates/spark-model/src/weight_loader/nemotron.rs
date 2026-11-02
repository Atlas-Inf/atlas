// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

mod lm_head;
mod policy;
mod ssm_layer;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, NemotronMoeLayer, Qwen3AttentionLayer};
use crate::tp_shard::{TpAttentionDims, TpShardKind, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizedWeight, dense,
    load_fp8_block_scaled_as_fp8weight, load_kv_scales, load_nemotron_attention, load_nemotron_moe,
    quantize_to_nvfp4,
};

pub struct NemotronHWeightLoader;

impl ModelWeightLoader for NemotronHWeightLoader {
    /// nemotron_h_omni: the RADIO vision tower. Loads
    /// `vision_model.radio_model.*` + `vision_projector.*` + `mlp1.*`
    /// into `RadioVisionTower`; returns `None` for text-only nemotron_h
    /// or when `config.vision` is unset.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionTower>> {
        if config.vision.as_ref().is_none_or(|v| v.omni.is_none()) {
            return Ok(None);
        }
        let tower = crate::radio_tower::RadioVisionTower::new(store, config, gpu)?;
        Ok(Some(crate::layers::VisionTower::Radio(Box::new(tower))))
    }

    fn dead_namespaces(&self, config: &ModelConfig) -> Vec<String> {
        policy::dead_namespaces(config)
    }

    fn slab_pack_weights(&self, config: &ModelConfig) -> bool {
        policy::slab_pack_weights(config)
    }

    fn supports_tp(&self) -> bool {
        // FullAttention layers TP-sharded across both quant paths
        // (NVFP4-from-disk and BF16/FP8 → NVFP4). LinearAttention
        // (Mamba-2 SSM) and MoE layers run full-replica per rank —
        // SSM stays correct because hidden in/out is the same on
        // every rank; MoE under EP+TP composition only uses EP.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = &config.layer_types;
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;
        let h = config.hidden_size;

        // Runtime quantization kernels for BF16→NVFP4 conversion of unquantized layers.
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        // Pre-allocate a reusable scratch buffer for FP8→BF16 dequant intermediates.
        // On GB10 UVM, gpu.free() posts in-band TLB invalidations that corrupt
        // nearby allocations (BUG #29). Using a scratch buffer avoids all frees
        // during loading. Size = max(in_proj, out_proj, shared_up, shared_down) in BF16 bytes.
        let moe_input = config.moe_input_size();
        // Puzzle: intermediate size varies per MoE layer — size scratch to max.
        let max_moe_inter = config.max_moe_intermediate_size();
        let scratch_elems = (config.mamba2_in_proj_size() * h)
            .max(h * config.mamba2_d_inner())
            .max(config.shared_expert_intermediate_size * h)
            .max(h * config.shared_expert_intermediate_size)
            .max(max_moe_inter * moe_input)
            .max(moe_input * max_moe_inter);
        let scratch_bytes = scratch_elems * 2; // BF16 = 2 bytes
        let scratch = gpu.alloc(scratch_bytes)?;

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let norm = dense(store, &format!("{lp}.norm.weight"))?;

            match lt {
                atlas_core::config::LayerType::LinearAttention => {
                    let layer = Self::build_ssm_layer(
                        gpu, store, config, i, h, &lp, norm, quantize_k, absmax_k, scratch, stream,
                    )?;
                    layers.push(Box::new(layer));
                }
                atlas_core::config::LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                atlas_core::config::LayerType::Moe => {
                    // Standalone MoE FFN layer (uniform Super/Nano or Puzzle per-block)
                    let moe_inter = config.moe_intermediate_size_for(i);
                    let top_k = config.num_experts_per_tok_for(i);
                    let moe = load_nemotron_moe(
                        store,
                        i,
                        config.num_experts,
                        gpu,
                        config,
                        Some(absmax_k),
                        Some(quantize_k),
                        stream,
                        Some(scratch),
                        &lp,
                    )?;
                    if i < 4 || moe_inter != config.moe_intermediate_size {
                        tracing::info!(
                            "L{i} MoE: inter={moe_inter} top_k={top_k} latent={} has_fc1={} has_fc2={} shared_up_s2={:.6e} experts[0].up_s2={:.6e}",
                            config.moe_latent_size,
                            moe.fc1_latent_proj.is_some(),
                            moe.fc2_latent_proj.is_some(),
                            moe.shared_up.weight_scale_2,
                            moe.experts
                                .first()
                                .map(|e| e.up_proj.weight_scale_2)
                                .unwrap_or(0.0),
                        );
                    }
                    let mut moe_layer =
                        NemotronMoeLayer::new(moe, norm, config, gpu, moe_inter, top_k, i)?;
                    // Builds the transposed shared-expert weights (2 small matrices
                    // per layer) so prefill can use `w4a16_gemm_t` instead of the base
                    // `w4a16_gemm`. Routed-expert transposition stays disabled inside
                    // for LatentMoE — 512 experts x 40 layers would not fit.
                    moe_layer.prepare_prefill_weights(gpu, config);
                    layers.push(Box::new(moe_layer));
                }
                atlas_core::config::LayerType::FullAttention => {
                    let mp = format!("{lp}.mixer");
                    // Native-FP8 attention: the omni checkpoint ships q/k/v/o as
                    // F8_E4M3 with a per-tensor scalar `weight_scale`. When on,
                    // the FP8 bytes feed `QuantWeight::Fp8` directly — no BF16
                    // dequant at all (so nothing to free, per BUG #29) — and
                    // `load_fp8_block_scaled_as_fp8weight` expands the scalar
                    // into the FP32 [N/128, K/128] block-scale buffer the w8a16
                    // kernels expect (multiply convention, shared with
                    // quantize_fp8_bs). The weight pointer aliases the
                    // slab-resident safetensors bytes — same convention as the
                    // FP8 MoE tensors and safe for the same reason: the slab is
                    // read-only weight storage for the process lifetime.
                    // Gate: on by default for `nemotron_h_omni` by model type
                    // (survives the text-only `config.vision = None` drop on
                    // targets without a vision encoder).
                    // ATLAS_NEMOTRON_NATIVE_FP8_ATTN=0 opts out (old dequant
                    // path), =1 opts in on non-omni.
                    let is_fp8_attn = !store.contains(&format!("{mp}.q_proj.weight_scale_2"))
                        && store.contains(&format!("{mp}.q_proj.weight_scale"));
                    let native_fp8_attn = policy::native_fp8_attn(
                        config,
                        std::env::var("ATLAS_NEMOTRON_NATIVE_FP8_ATTN")
                            .ok()
                            .as_deref(),
                    );
                    let tp_size0 = config.tp_world_size.max(1);
                    if is_fp8_attn && native_fp8_attn && tp_size0 == 1 {
                        let q_dim = config.num_attention_heads * config.head_dim;
                        let kv_dim = config.num_key_value_heads * config.head_dim;
                        let mut q = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{mp}.q_proj"),
                            gpu,
                        )?;
                        let mut k = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{mp}.k_proj"),
                            gpu,
                        )?;
                        let mut v = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{mp}.v_proj"),
                            gpu,
                        )?;
                        let mut o = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{mp}.o_proj"),
                            gpu,
                        )?;
                        // Shape + alignment: q/o are [h,h], k/v are [kv_dim,h],
                        // and w8a16_gemv issues uint4 loads with no tail so K
                        // must be a multiple of 16 (K=4096 on this checkpoint).
                        for (w, want_n) in [(&q, q_dim), (&k, kv_dim), (&v, kv_dim), (&o, h)] {
                            anyhow::ensure!(
                                w.n as usize == want_n && w.k as usize == h,
                                "L{i} attention FP8 shape [{},{}] != expected [{want_n},{h}]",
                                w.n,
                                w.k
                            );
                        }
                        anyhow::ensure!(
                            h.is_multiple_of(16),
                            "L{i} attention FP8 K={h} is not a multiple of 16"
                        );
                        // Slab policy (same as the SSM FP8 arm, ssm_layer.rs):
                        // alias slab-resident pointers — interior frees are
                        // no-ops and offsets are never reused, so the alias can
                        // never read stale bytes. Anything else gets an owned
                        // copy: the w8a16_gemv_numeric history shows a non-slab
                        // alias reading stale bytes while the model stays
                        // fluent and loses the prompt.
                        let mut slab_all = true;
                        for w in [&mut q, &mut k, &mut v, &mut o] {
                            if gpu.is_slab_interior(w.weight) {
                                continue;
                            }
                            slab_all = false;
                            let bytes = (w.n as usize) * (w.k as usize);
                            let owned = gpu.alloc(bytes)?;
                            gpu.copy_d2d(w.weight, owned, bytes)?;
                            w.weight = owned;
                        }
                        if slab_all {
                            tracing::info!(
                                "L{i} attention: FP8 q/k/v/o are slab-resident — aliased, not copied"
                            );
                        }
                        let (k_scale, v_scale) = load_kv_scales(store, &mp, gpu);
                        let dummy = DenseWeight {
                            weight: spark_runtime::gpu::DevicePtr::NULL,
                        };
                        let attn = AttentionWeights {
                            q_proj: dummy,
                            k_proj: dummy,
                            v_proj: dummy,
                            o_proj: QuantizedWeight::null(),
                            q_norm: dummy,
                            k_norm: dummy,
                            q_norm_full: None,
                            k_norm_full: None,
                            k_scale,
                            v_scale,
                        };
                        let mut attn_layer = Qwen3AttentionLayer::new_ungated(
                            norm,
                            attn,
                            dummy,
                            FfnComponent::None,
                            attn_idx,
                            None,
                            None,
                            None,
                            gpu,
                            layer_kv_dtypes[attn_idx],
                            config.fp8_kv_calibration_tokens,
                            config,
                        )?;
                        attn_layer.set_fp8_weights(Some(q), Some(k), Some(v), Some(o));
                        if let Err(e) = attn_layer.transpose_fp8_for_prefill(gpu, stream) {
                            tracing::warn!("L{i} attention FP8 prefill transpose failed: {e}");
                        }
                        // No rotary on Nemotron-H attention (see below).
                        attn_layer.set_rope_disabled(true);
                        if i < 2 {
                            tracing::info!(
                                "L{i} attention: native FP8 q/k/v/o (w8a16, no dequant)"
                            );
                        }
                        layers.push(Box::new(attn_layer));
                        attn_idx += 1;
                        continue;
                    }
                    // Attention layer — quantize BF16 Q/K/V/O directly from
                    // WeightStore pointers (no intermediate alloc/free needed).
                    let (mut attn, mut q_nvfp4, mut k_nvfp4, mut v_nvfp4, mut o_dense, is_nvfp4) =
                        load_nemotron_attention(store, i, gpu, &lp)?;
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let dims = TpAttentionDims::from_config(config);
                    if is_nvfp4 && tp_size > 1 {
                        // NVFP4-from-disk: shard packed weight + FP8 scales.
                        let group_size = 16usize;
                        if let Some(q) = q_nvfp4.as_ref() {
                            let s = shard_quantized_nvfp4(
                                q,
                                dims.full_q_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                group_size,
                                gpu,
                            )?;
                            gpu.free(q.weight)?;
                            gpu.free(q.weight_scale)?;
                            q_nvfp4 = Some(s);
                        }
                        if let Some(k) = k_nvfp4.as_ref() {
                            let s = shard_quantized_nvfp4(
                                k,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                group_size,
                                gpu,
                            )?;
                            gpu.free(k.weight)?;
                            gpu.free(k.weight_scale)?;
                            k_nvfp4 = Some(s);
                        }
                        if let Some(v) = v_nvfp4.as_ref() {
                            let s = shard_quantized_nvfp4(
                                v,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                group_size,
                                gpu,
                            )?;
                            gpu.free(v.weight)?;
                            gpu.free(v.weight_scale)?;
                            v_nvfp4 = Some(s);
                        }
                        // O proj is stored on attn.o_proj as QuantizedWeight in NVFP4-disk path.
                        let o_old = attn.o_proj;
                        let o_sharded = shard_quantized_nvfp4(
                            &o_old,
                            dims.h,
                            dims.full_o_in,
                            TpShardKind::RowParallel,
                            tp_rank,
                            tp_size,
                            group_size,
                            gpu,
                        )?;
                        gpu.free(o_old.weight)?;
                        gpu.free(o_old.weight_scale)?;
                        attn.o_proj = o_sharded;
                    }
                    let mut bf16_o_dense: Option<DenseWeight> = None;
                    let (q_nv, k_nv, v_nv) = if is_nvfp4 {
                        (q_nvfp4, k_nvfp4, v_nvfp4)
                    } else {
                        let num_heads = config.num_attention_heads;
                        let kv_heads = config.num_key_value_heads;
                        let hd = config.head_dim;
                        // BF16 / FP8-dequant fallback: shard the dense BF16
                        // before quantization. Dims here are TP-LOCAL after
                        // sharding (config head counts already TP-divided).
                        if tp_size > 1 {
                            let (qp, _, _) = shard_dense_bf16(
                                attn.q_proj.weight,
                                dims.full_q_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if qp != attn.q_proj.weight {
                                gpu.free(attn.q_proj.weight)?;
                            }
                            attn.q_proj.weight = qp;
                            let (kp, _, _) = shard_dense_bf16(
                                attn.k_proj.weight,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if kp != attn.k_proj.weight {
                                gpu.free(attn.k_proj.weight)?;
                            }
                            attn.k_proj.weight = kp;
                            let (vp, _, _) = shard_dense_bf16(
                                attn.v_proj.weight,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if vp != attn.v_proj.weight {
                                gpu.free(attn.v_proj.weight)?;
                            }
                            attn.v_proj.weight = vp;
                            let (op, _, _) = shard_dense_bf16(
                                o_dense.weight,
                                dims.h,
                                dims.full_o_in,
                                TpShardKind::RowParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if op != o_dense.weight {
                                gpu.free(o_dense.weight)?;
                            }
                            o_dense.weight = op;
                        }
                        // Keep attention in BF16 when the checkpoint ships it that
                        // way. ModelOpt left Q/K/V/O unquantized ON PURPOSE here:
                        // Puzzle is Mamba-dominant with only 9 full-attention
                        // layers, so those layers carry the long-range retrieval
                        // and are the last place to spend precision — and at
                        // ~1.2 GB BF16 they are cheap to keep. Crushing them
                        // 16-bit -> 4-bit saved ~0.9 GB and degraded exactly what
                        // they exist for. `ATLAS_NEMOTRON_BF16_ATTN=0` restores
                        // the old quantize-everything behaviour for an A/B.
                        let keep_bf16_attn =
                            std::env::var("ATLAS_NEMOTRON_BF16_ATTN").as_deref() != Ok("0");
                        if keep_bf16_attn {
                            tracing::info!(
                                "L{i} attention: keeping checkpoint BF16 Q/K/V/O (no NVFP4 requant)"
                            );
                            bf16_o_dense = Some(o_dense);
                            (None, None, None)
                        } else {
                            let q = quantize_to_nvfp4(
                                &attn.q_proj,
                                num_heads * hd,
                                h,
                                gpu,
                                absmax_k,
                                quantize_k,
                                stream,
                            )?;
                            let k = quantize_to_nvfp4(
                                &attn.k_proj,
                                kv_heads * hd,
                                h,
                                gpu,
                                absmax_k,
                                quantize_k,
                                stream,
                            )?;
                            let v = quantize_to_nvfp4(
                                &attn.v_proj,
                                kv_heads * hd,
                                h,
                                gpu,
                                absmax_k,
                                quantize_k,
                                stream,
                            )?;
                            let o = quantize_to_nvfp4(
                                &o_dense,
                                h,
                                num_heads * hd,
                                gpu,
                                absmax_k,
                                quantize_k,
                                stream,
                            )?;
                            attn.o_proj = o;
                            (Some(q), Some(k), Some(v))
                        }
                    };
                    // Transposed Q/K/V/O so prefill uses `w4a16_gemm_t` (FP8 MMA,
                    // N128/K32, cp.async) instead of the base `w4a16_gemm`. Same
                    // gap as the SSM/MoE layers: the setter existed but was never
                    // called for Nemotron. Q/K/V/O are small (h=4096, kv=2 heads),
                    // so the extra copies cost ~0.3 GB.
                    let q_dim = config.num_attention_heads * config.head_dim;
                    let kv_dim = config.num_key_value_heads * config.head_dim;
                    let qt = q_nv
                        .as_ref()
                        .and_then(|w| w.transpose_for_gemm(gpu, q_dim, h).ok());
                    let kt = k_nv
                        .as_ref()
                        .and_then(|w| w.transpose_for_gemm(gpu, kv_dim, h).ok());
                    let vt = v_nv
                        .as_ref()
                        .and_then(|w| w.transpose_for_gemm(gpu, kv_dim, h).ok());
                    // Keep-BF16 attention leaves `attn.o_proj` as the NULL
                    // placeholder (`load_nemotron_attention` returns the real
                    // weight via `o_dense`); transposing it would launch
                    // `transpose_u8` on a NULL pointer, and the swallowed `.ok()`
                    // error left the CUDA context poisoned (700 at the next
                    // module load). Only transpose a real quantized o_proj.
                    let ot = if attn.o_proj.is_null() {
                        None
                    } else {
                        attn.o_proj.transpose_for_gemm(gpu, h, q_dim).ok()
                    };

                    let mut attn_layer = Qwen3AttentionLayer::new_ungated(
                        norm,
                        attn,
                        DenseWeight {
                            weight: spark_runtime::gpu::DevicePtr::NULL,
                        },
                        FfnComponent::None,
                        attn_idx,
                        q_nv,
                        k_nv,
                        v_nv,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    // Nemotron-H attention has NO rotary embeddings in the HF
                    // reference — position is carried by the Mamba layers.
                    // Applying Qwen3 RoPE here corrupts long-range attention
                    // (the G0 distance-decay retrieval bug).
                    attn_layer.set_rope_disabled(true);
                    if let Some(od) = bf16_o_dense {
                        // Dispatch checks `o_dense_bf16` first (see gemma4 loader).
                        attn_layer.set_o_dense_bf16(od);
                    }
                    attn_layer.set_prefill_weights(qt, kt, vt, ot);
                    layers.push(Box::new(attn_layer));
                    attn_idx += 1;
                }
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Nemotron-H weight loader: {} layers ({} SSM, {} MoE, {} attention)",
            layers.len(),
            config.num_ssm_layers(),
            config.num_moe_layers(),
            attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(
            store,
            &format!("{}.embeddings.weight", config.weight_prefix),
        )
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, &format!("{}.norm_f.weight", config.weight_prefix))
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Self::load_lm_head_impl(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if store.contains("mtp.layers.0.eh_proj.weight") {
            tracing::info!(
                "Loading Lightning MTP (eh_proj + layer1 ReLU2 MoE, {} experts)",
                config.num_experts
            );
            let mtp = crate::weight_map::load_mtp_lightning(
                store,
                config.num_experts,
                gpu,
                config.hidden_size,
            )?;
            tracing::info!(
                "Lightning MTP loaded: eh_proj, {} experts (gate_proj=up_proj stand-in)",
                mtp.experts.len()
            );
            return Ok(Some(mtp));
        }
        Ok(None) // Nano / Super have no MTP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nemotron_h_loader_exists() {
        let _loader = NemotronHWeightLoader;
    }

    #[test]
    fn omni_policies_skip_dead_towers_and_enable_slab_packing() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "nemotron_h_omni".to_string();
        config.tp_world_size = 1;
        config.ep_world_size = 1;
        let loader = NemotronHWeightLoader;
        let dead = loader.dead_namespaces(&config);
        assert!(dead.iter().any(|p| p == "language_model.mtp."));
        assert!(dead.iter().any(|p| p == "vision_model."));
        assert!(dead.iter().all(|p| p.ends_with('.')));
        assert!(loader.slab_pack_weights(&config));

        config.tp_world_size = 2;
        assert!(
            !loader.slab_pack_weights(&config),
            "pack mode must not arm under TP — its frees reach store ptrs"
        );

        config.model_type = "nemotron_h".to_string();
        config.tp_world_size = 1;
        assert!(loader.dead_namespaces(&config).is_empty());
        assert!(!loader.slab_pack_weights(&config));
    }

    #[test]
    fn native_fp8_attn_lever_keys_on_model_type_not_vision() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "nemotron_h_omni".to_string();
        // The strix-hip text-only drop sets vision to None — the lever must
        // still be on by default.
        config.vision = None;
        assert!(policy::native_fp8_attn(&config, None));
        assert!(!policy::native_fp8_attn(&config, Some("0")));
        config.model_type = "nemotron_h".to_string();
        assert!(!policy::native_fp8_attn(&config, None));
        assert!(policy::native_fp8_attn(&config, Some("1")));
        // Unknown env values fall back to the default.
        config.model_type = "nemotron_h_omni".to_string();
        assert!(policy::native_fp8_attn(&config, Some("2")));
    }
}
