// SPDX-License-Identifier: AGPL-3.0-only

//! `load_gemma_audio_encoder_impl` tests (split from `loader_e` for the
//! ≤500-LoC cap; the module is `#[cfg(test)]`-gated in `loader_e.rs`).

#[cfg(test)]
mod tests {
    use super::super::{AUDIO_PREFIX, EMBED_AUDIO_PREFIX, load_gemma_audio_encoder_impl};
    use atlas_core::config::parse_config;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    /// Small synthetic audio tower (hidden 16, 2 layers) whose geometry
    /// passes `GemmaAudioEncoder::new` validation: two subsample conv
    /// stages, mel_bins % 4 == 0, heads divide hidden, `output_proj_dims ==
    /// OUT_HIDDEN_SIZE`, `activation == "silu"`.
    const E2B_AUDIO_JSON: &str = r#"{
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 6144,
            "vocab_size": 262144,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6
        },
        "audio_config": {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "subsampling_conv_channels": [8, 4],
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 3,
            "attention_context_right": 0,
            "output_proj_dims": 1536,
            "residual_weight": 0.5,
            "use_clipped_linears": true,
            "mel_bins": 16,
            "audio_seq_length": 8,
            "activation": "silu"
        },
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "eoa_token_id": 258883
    }"#;

    /// Text-only Gemma-4 (no `audio_config`) — the 26B/31B shape.
    const TEXT_ONLY_JSON: &str = r#"{
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 5376,
            "num_hidden_layers": 12,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "intermediate_size": 21504,
            "vocab_size": 262144,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6
        }
    }"#;

    /// `GemmaAudioEncoder` lacks `Debug` — `Option::expect` won't compile.
    fn expect_loaded(
        r: anyhow::Result<Option<crate::layers::GemmaAudioEncoder>>,
    ) -> crate::layers::GemmaAudioEncoder {
        match r {
            Ok(Some(enc)) => enc,
            Ok(None) => panic!("expected Some(GemmaAudioEncoder), got None"),
            Err(e) => panic!("audio tower load failed: {e:#}"),
        }
    }

    /// Fake pointer tensor (never read from the mock — pointer-extracted).
    fn tensor(base: u64, shape: &[usize]) -> WeightTensor {
        WeightTensor {
            ptr: DevicePtr(base),
            shape: shape.to_vec(),
            dtype: WeightDtype::BF16,
        }
    }

    /// Clip-bound scalar backed by a REAL mock allocation so `copy_d2h` can
    /// read it back; `bf16` picks the storage dtype (both are legal).
    fn clip_scalar(gpu: &MockGpuBackend, v: f32, bf16: bool) -> WeightTensor {
        if bf16 {
            let bits = ((v.to_bits() >> 16) as u16).to_le_bytes();
            let ptr = gpu.alloc(2).unwrap();
            gpu.copy_h2d(&bits, ptr).unwrap();
            WeightTensor {
                ptr,
                shape: vec![],
                dtype: WeightDtype::BF16,
            }
        } else {
            let ptr = gpu.alloc(4).unwrap();
            gpu.copy_h2d(&v.to_le_bytes(), ptr).unwrap();
            WeightTensor {
                ptr,
                shape: vec![],
                dtype: WeightDtype::FP32,
            }
        }
    }

    /// Per-layer scalar index → distinct bounds (BF16 on even j, FP32 on odd
    /// j — both storage dtypes must parse to the right f32).
    fn bounds(gpu: &MockGpuBackend, j: usize) -> [(String, WeightTensor); 4] {
        [
            (
                "input_min".into(),
                clip_scalar(gpu, -1.0 - j as f32, j.is_multiple_of(2)),
            ),
            (
                "input_max".into(),
                clip_scalar(gpu, 2.0 + j as f32, j.is_multiple_of(2)),
            ),
            (
                "output_min".into(),
                clip_scalar(gpu, -3.0 - j as f32, j.is_multiple_of(2)),
            ),
            (
                "output_max".into(),
                clip_scalar(gpu, 4.0 + j as f32, j.is_multiple_of(2)),
            ),
        ]
    }

    /// Full synthetic audio-tower store; `relative_k_proj` per layer is a
    /// REAL allocation (the constructor downloads it via `copy_d2h` to
    /// precompute the relative position keys).
    fn build_store(gpu: &MockGpuBackend) -> WeightStore {
        let mut w = std::collections::HashMap::new();
        let sub = |s: &str| format!("{AUDIO_PREFIX}.subsample_conv_projection.{s}");
        w.insert(sub("layer0.conv.weight"), tensor(0x1000, &[8, 1, 3, 3]));
        w.insert(sub("layer0.norm.weight"), tensor(0x1010, &[8]));
        w.insert(sub("layer1.conv.weight"), tensor(0x1020, &[4, 8, 3, 3]));
        w.insert(sub("layer1.norm.weight"), tensor(0x1030, &[4]));
        w.insert(sub("input_proj_linear.weight"), tensor(0x1040, &[16, 16]));

        for i in 0..2 {
            let lp = format!("{AUDIO_PREFIX}.layers.{i}");
            let base = 0x4000 + i as u64 * 0x8000;
            let ff1 = format!("{lp}.feed_forward1");
            let ff2 = format!("{lp}.feed_forward2");
            let lc = format!("{lp}.lconv1d");
            let sa = format!("{lp}.self_attn");
            // 10 clipped linears per layer: scalar index j = 10*i .. 10*i+9.
            let mut j = 10 * i;
            // ── feed_forward1 / feed_forward2 ──
            for (ff, off) in [(&ff1, 0x000u64), (&ff2, 0x300)] {
                w.insert(
                    format!("{ff}.ffw_layer_1.linear.weight"),
                    tensor(base + off, &[64, 16]),
                );
                w.insert(
                    format!("{ff}.ffw_layer_2.linear.weight"),
                    tensor(base + off + 0x100, &[16, 64]),
                );
                w.insert(
                    format!("{ff}.pre_layer_norm.weight"),
                    tensor(base + off + 0x200, &[16]),
                );
                w.insert(
                    format!("{ff}.post_layer_norm.weight"),
                    tensor(base + off + 0x210, &[16]),
                );
                for (proj, _o2) in [("ffw_layer_1", 0x0u64), ("ffw_layer_2", 0x100)] {
                    let p = format!("{ff}.{proj}");
                    for (suffix, t) in bounds(gpu, j) {
                        w.insert(format!("{p}.{suffix}"), t);
                    }
                    j += 1;
                }
            }
            // ── lconv1d ──
            w.insert(
                format!("{lc}.linear_start.linear.weight"),
                tensor(base + 0x600, &[32, 16]),
            );
            w.insert(
                format!("{lc}.linear_end.linear.weight"),
                tensor(base + 0x700, &[16, 16]),
            );
            w.insert(
                format!("{lc}.depthwise_conv1d.weight"),
                tensor(base + 0x800, &[16, 1, 5]),
            );
            w.insert(
                format!("{lc}.pre_layer_norm.weight"),
                tensor(base + 0x810, &[16]),
            );
            w.insert(
                format!("{lc}.conv_norm.weight"),
                tensor(base + 0x820, &[16]),
            );
            for (proj, _o2) in [("linear_start", 0x600u64), ("linear_end", 0x700)] {
                let p = format!("{lc}.{proj}");
                for (suffix, t) in bounds(gpu, j) {
                    w.insert(format!("{p}.{suffix}"), t);
                }
                j += 1;
            }
            // ── self_attn: q/k/v/post clipped, relative_k_proj + per_dim_scale plain ──
            for (proj, o2) in [
                ("q_proj", 0x900u64),
                ("k_proj", 0xA00),
                ("v_proj", 0xB00),
                ("post", 0xC00),
            ] {
                let p = format!("{sa}.{proj}");
                w.insert(format!("{p}.linear.weight"), tensor(base + o2, &[16, 16]));
                for (suffix, t) in bounds(gpu, j) {
                    w.insert(format!("{p}.{suffix}"), t);
                }
                j += 1;
            }
            // REAL allocation — `GemmaAudioEncoder::new` downloads it.
            let rel_k_bytes = 16 * 16 * 2;
            let rel_k_ptr = gpu.alloc(rel_k_bytes).unwrap();
            gpu.copy_h2d(&vec![0u8; rel_k_bytes], rel_k_ptr).unwrap();
            w.insert(
                format!("{sa}.relative_k_proj.weight"),
                WeightTensor {
                    ptr: rel_k_ptr,
                    shape: vec![16, 16],
                    dtype: WeightDtype::BF16,
                },
            );
            // REAL allocation — `GemmaAudioEncoder::new` downloads it too
            // (host-side softplus → spd for the chunked attention).
            let spd_bytes = 4 * 2;
            let spd_ptr = gpu.alloc(spd_bytes).unwrap();
            gpu.copy_h2d(&vec![0u8; spd_bytes], spd_ptr).unwrap();
            w.insert(
                format!("{sa}.per_dim_scale"),
                WeightTensor {
                    ptr: spd_ptr,
                    shape: vec![4],
                    dtype: WeightDtype::BF16,
                },
            );
            // ── layer norms ──
            w.insert(
                format!("{lp}.norm_pre_attn.weight"),
                tensor(base + 0xF00, &[16]),
            );
            w.insert(
                format!("{lp}.norm_post_attn.weight"),
                tensor(base + 0xF10, &[16]),
            );
            w.insert(format!("{lp}.norm_out.weight"), tensor(base + 0xF20, &[16]));
        }

        w.insert(
            format!("{AUDIO_PREFIX}.output_proj.weight"),
            tensor(0x2000, &[1536, 16]),
        );
        w.insert(
            format!("{AUDIO_PREFIX}.output_proj.bias"),
            tensor(0x3000, &[1536]),
        );
        w.insert(
            format!("{EMBED_AUDIO_PREFIX}.embedding_projection.weight"),
            tensor(0x5000, &[1536, 1536]),
        );
        WeightStore::from_map(w)
    }

    /// Every checkpoint tensor must land in the exact field it names —
    /// including `output_proj.bias` in its dedicated [`GemmaAudioOutputProj`]
    /// slot and the UNclipped `relative_k_proj` / `per_dim_scale`.
    #[test]
    fn audio_loader_maps_all_fields() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(E2B_AUDIO_JSON).unwrap();
        let store = build_store(&gpu);
        let enc = expect_loaded(load_gemma_audio_encoder_impl(&store, &cfg, &gpu));

        // Subsample conv projection.
        assert_eq!(enc.subsample_conv0_w.0, 0x1000, "layer0 conv");
        assert_eq!(enc.subsample_ln0_w.0, 0x1010, "layer0 norm");
        assert_eq!(enc.subsample_conv1_w.0, 0x1020, "layer1 conv");
        assert_eq!(enc.subsample_ln1_w.0, 0x1030, "layer1 norm");
        assert_eq!(enc.subsample_proj_w.0, 0x1040, "input_proj_linear");

        // output_proj weight AND bias (the dedicated bias field).
        assert_eq!(enc.output_proj_w.0, 0x2000, "output_proj.weight");
        assert_eq!(enc.output_proj_b.0, 0x3000, "output_proj.bias");
        assert_eq!(enc.embed_audio_proj_w.0, 0x5000, "embed_audio projection");

        // Layer 0: FFNs + lconv + attn + norms.
        let l0 = &enc.layers[0];
        assert_eq!(
            l0.feed_forward1.ffw_layer_1.weight.weight.0, 0x4000,
            "L0 ff1.ffw1"
        );
        assert_eq!(
            l0.feed_forward1.ffw_layer_2.weight.weight.0, 0x4100,
            "L0 ff1.ffw2"
        );
        assert_eq!(
            l0.feed_forward1.pre_layer_norm.weight.0, 0x4200,
            "L0 ff1.pre_norm"
        );
        assert_eq!(
            l0.feed_forward1.post_layer_norm.weight.0, 0x4210,
            "L0 ff1.post_norm"
        );
        assert_eq!(
            l0.feed_forward2.ffw_layer_1.weight.weight.0, 0x4300,
            "L0 ff2.ffw1"
        );
        assert_eq!(
            l0.feed_forward2.ffw_layer_2.weight.weight.0, 0x4400,
            "L0 ff2.ffw2"
        );
        assert_eq!(
            l0.lconv1d.linear_start.weight.weight.0, 0x4600,
            "L0 lconv start"
        );
        assert_eq!(
            l0.lconv1d.linear_end.weight.weight.0, 0x4700,
            "L0 lconv end"
        );
        assert_eq!(
            l0.lconv1d.depthwise_conv1d.weight.0, 0x4800,
            "L0 depthwise conv1d"
        );
        assert_eq!(
            l0.lconv1d.pre_layer_norm.weight.0, 0x4810,
            "L0 lconv pre_norm"
        );
        assert_eq!(l0.lconv1d.conv_norm.weight.0, 0x4820, "L0 lconv conv_norm");
        assert_eq!(l0.self_attn.q_proj.weight.weight.0, 0x4900, "L0 q_proj");
        assert_eq!(l0.self_attn.k_proj.weight.weight.0, 0x4A00, "L0 k_proj");
        assert_eq!(l0.self_attn.v_proj.weight.weight.0, 0x4B00, "L0 v_proj");
        assert_eq!(l0.self_attn.post.weight.weight.0, 0x4C00, "L0 post");
        // per_dim_scale is a REAL allocation (the constructor reads it back
        // to precompute spd) — assert non-zero rather than a fixed slot.
        assert_ne!(
            l0.self_attn.per_dim_scale.weight.0, 0,
            "L0 per_dim_scale real alloc"
        );
        assert_eq!(l0.norm_pre_attn.weight.0, 0x4F00, "L0 norm_pre_attn");
        assert_eq!(l0.norm_post_attn.weight.0, 0x4F10, "L0 norm_post_attn");
        assert_eq!(l0.norm_out.weight.0, 0x4F20, "L0 norm_out");

        // Layer 1 spot-checks (offset per layer).
        let l1 = &enc.layers[1];
        assert_eq!(
            l1.feed_forward1.ffw_layer_1.weight.weight.0, 0xC000,
            "L1 ff1.ffw1"
        );
        assert_eq!(
            l1.feed_forward2.ffw_layer_2.weight.weight.0, 0xC400,
            "L1 ff2.ffw2"
        );
        assert_eq!(
            l1.lconv1d.linear_start.weight.weight.0, 0xC600,
            "L1 lconv start"
        );
        assert_eq!(l1.self_attn.v_proj.weight.weight.0, 0xCB00, "L1 v_proj");
        assert_eq!(l1.self_attn.post.weight.weight.0, 0xCC00, "L1 post");
        assert_eq!(l1.norm_out.weight.0, 0xCF20, "L1 norm_out");
        assert_eq!(enc.layers.len(), 2, "layer count");
    }

    /// Clip bounds parse to the exact f32 from BOTH storage dtypes (even
    /// scalar index BF16, odd FP32 — see `bounds`).
    #[test]
    fn audio_loader_parses_clip_bounds() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(E2B_AUDIO_JSON).unwrap();
        let store = build_store(&gpu);
        let enc = expect_loaded(load_gemma_audio_encoder_impl(&store, &cfg, &gpu));

        // Layer 0: ff1.ffw_layer_1 j=0 (BF16), ff1.ffw_layer_2 j=1 (FP32).
        let f1 = &enc.layers[0].feed_forward1;
        assert_eq!(f1.ffw_layer_1.input_min, -1.0, "j=0 input_min");
        assert_eq!(f1.ffw_layer_1.input_max, 2.0, "j=0 input_max");
        assert_eq!(f1.ffw_layer_1.output_min, -3.0, "j=0 output_min");
        assert_eq!(f1.ffw_layer_1.output_max, 4.0, "j=0 output_max");
        assert_eq!(f1.ffw_layer_2.input_min, -2.0, "j=1 input_min (FP32)");
        assert_eq!(f1.ffw_layer_2.output_max, 5.0, "j=1 output_max (FP32)");
        // Layer 0: lconv.linear_end j=5 (BF16), self_attn.q_proj j=6 (BF16).
        assert_eq!(enc.layers[0].lconv1d.linear_end.input_min, -6.0, "j=5");
        assert_eq!(enc.layers[0].lconv1d.linear_end.output_max, 9.0, "j=5");
        assert_eq!(enc.layers[0].self_attn.q_proj.output_max, 10.0, "j=6");
        // Layer 1: scalar indices restart at j=10 — ff2.ffw_layer_1 j=12
        // (BF16), self_attn.post j=19 (FP32).
        let l1 = &enc.layers[1];
        assert_eq!(
            l1.feed_forward2.ffw_layer_1.output_min, -15.0,
            "L1 ff2.ffw1 j=12"
        );
        assert_eq!(l1.self_attn.post.input_min, -20.0, "L1 post j=19 (FP32)");
        assert_eq!(l1.self_attn.post.output_max, 23.0, "L1 post j=19");
    }

    /// Text-only config returns `Ok(None)` against an empty store — never
    /// errors, never loads.
    #[test]
    fn text_only_config_returns_none() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(TEXT_ONLY_JSON).unwrap();
        assert!(cfg.gemma_audio.is_none());
        let store = WeightStore::empty();
        let enc = load_gemma_audio_encoder_impl(&store, &cfg, &gpu).unwrap();
        assert!(enc.is_none());
    }

    /// A missing critical tensor fails with a NAMED error, never a silent
    /// partial tower — drop `output_proj.bias`.
    #[test]
    fn missing_critical_tensor_fails_named() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(E2B_AUDIO_JSON).unwrap();
        let missing = format!("{AUDIO_PREFIX}.output_proj.bias");
        let full = build_store(&gpu);
        // WeightStore has no removal API — rebuild the map minus the key.
        let mut map = full
            .names()
            .map(|n| {
                let t = full.get(n).unwrap();
                (
                    n.to_string(),
                    WeightTensor {
                        ptr: t.ptr,
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                    },
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        map.remove(&missing);
        let reduced = WeightStore::from_map(map);
        // `unwrap_err()` needs `T: Debug` on the Ok payload — match instead.
        let err = match load_gemma_audio_encoder_impl(&reduced, &cfg, &gpu) {
            Ok(_) => panic!("expected an error for missing {missing}"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&missing),
            "error should name the missing tensor, got: {msg}"
        );
    }
}
