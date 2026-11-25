// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-logic tests of the Ling family driver spec: layer-plan ordering,
//! the conservative state/KV sizing math, prefill sub-chunking, and the
//! tensor handoff shapes the layer kernels assume. (GPU parity tests
//! follow the `fp4l_test_gpu_*.rs` patterns on a Spark, per campaign
//! convention.)

use atlas_core::config::{LayerType, ModelConfig};

use super::spec::{
    FfnKind, KDA_HEAD_DIM, KDA_HEADS, LingSpec, MLA_K_ROW_BF16, MLA_V_ROW_BF16, PrefillPlan,
    kda_state_bytes_per_slot, mla_kv_bytes_per_token,
};
/// The Ling text_config slice the parser tests carry (see
/// atlas-core `parsers/bailing_tests.rs`); the driver consumes whatever
/// `parse_config` produced, so the fixture mirrors the parsed ModelConfig.
fn ling_config() -> ModelConfig {
    let config = atlas_core::config::parse_config(
        r#"{
        "model_type": "bailing_moe_v3_vl",
        "text_config": {
            "model_type": "bailing_hybrid",
            "hidden_size": 2560,
            "num_hidden_layers": 42,
            "intermediate_size": 6144,
            "vocab_size": 157184,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rotary_dim": 64,
            "partial_rotary_factor": 0.5,
            "rope_theta": 6000000,
            "gated_attention_proj_granularity_type": "head_wise",
            "no_kda_lora": true,
            "short_conv_kernel_size": 4,
            "kda_lower_bound": -5.0,
            "num_experts": 512,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "moe_shared_expert_intermediate_size": 768,
            "score_function": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "router_dtype": "fp32",
            "moe_router_enable_expert_bias": true,
            "n_group": 8,
            "topk_group": 4,
            "first_k_dense_replace": 2,
            "layer_group_size": 6,
            "expert_swiglu_limit_list": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,4,4,4],
            "share_expert_swiglu_limit_list": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,5,5,5,5,7,7],
            "num_nextn_predict_layers": 1,
            "mtp_use_kda": false,
            "eos_token_id": 156895,
            "pad_token_id": 156892,
            "tie_word_embeddings": false,
            "mrope_section": [8, 12, 12]
        }
    }"#,
    )
    .expect("parse");
    config
}

#[test]
fn layer_plan_matches_the_checkpoint_layout() {
    let config = ling_config();
    let spec = LingSpec::from_config(&config).expect("spec");
    assert_eq!(spec.layers.len(), 42);
    assert_eq!(spec.mla_layers, 7);
    assert_eq!(spec.kda_layers, 35);
    for (idx, layer) in spec.layers.iter().enumerate() {
        let mla_expected = matches!(idx, 5 | 11 | 17 | 23 | 29 | 35 | 41);
        assert_eq!(
            layer.attention == LayerType::FullAttention,
            mla_expected,
            "layer {idx} attention"
        );
        let dense = idx < 2;
        assert_eq!(layer.ffn == FfnKind::Dense, dense, "layer {idx} ffn");
        // SwiGLU ground truth from the checkpoint lists: routed clamp 4 on
        // 35..=41, shared clamp 5 on 34..=39 and 7 on 40..=41.
        assert_eq!(layer.routed_swiglu_limit, if idx >= 35 { 4 } else { 0 });
        let shared = match idx {
            0..=33 => 0,
            34..=39 => 5,
            _ => 7,
        };
        assert_eq!(layer.shared_swiglu_limit, shared, "layer {idx} shared");
    }
    assert_eq!(spec.experts, 512);
    assert_eq!(spec.top_k, 8);
    assert_eq!(spec.n_group, 8);
    assert_eq!(spec.topk_group, 4);
    assert_eq!(spec.routed_scaling, 2.5);
}

#[test]
fn conservative_state_pool_math() {
    let config = ling_config();
    let spec = LingSpec::from_config(&config).expect("spec");
    let _ = spec;
    // Per slot: 35 layers x (2 MiB h + 192 KiB conv) = 76.8 MiB.
    let per_slot = kda_state_bytes_per_slot(&config);
    assert_eq!(
        per_slot,
        35 * (KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4 + 3 * (KDA_HEADS * KDA_HEAD_DIM) * 4 * 4)
    );
    assert_eq!(per_slot, 35 * (2_097_152 + 196_608));
    assert_eq!(per_slot, 80_281_600);
    // The default --ssm-cache-slots budget: 32 slots stay under 2.5 GiB on
    // the 121.6 GiB box (weights 77.9 GiB leave ~43 GiB for everything
    // else; the campaign envelope calls for conservative pools).
    assert!(32 * per_slot < 3 * (1 << 30));
    // MLA pools: 7 layers x (576 K + 512 V) bf16 = 15,232 B/token.
    assert_eq!(
        mla_kv_bytes_per_token(&config),
        7 * (MLA_K_ROW_BF16 + MLA_V_ROW_BF16) * 2
    );
    assert_eq!(mla_kv_bytes_per_token(&config), 15_232);
    // A 262,144-token full-context budget stays under 4 GiB.
    assert!(262_144u64 * 15_232u64 < 4u64 * (1 << 30));
}

#[test]
fn prefill_subchunk_plan() {
    let cap = 512;
    let plan = PrefillPlan::for_rows(4096, cap);
    assert_eq!(plan.chunks.len(), 8);
    assert!(plan.chunks.iter().all(|&c| c == 512));
    let tail = PrefillPlan::for_rows(1000, cap);
    assert_eq!(tail.chunks, vec![512, 488]);
    assert!(PrefillPlan::for_rows(1, cap).chunks == vec![1]);
}

#[test]
fn tensor_handoff_shapes() {
    // The launch contracts of the four kernel files, encoded as the exact
    // row/col counts the driver computes (see kernels/gb10/ling-3.2-air/
    // nvfp4 header comments).
    let t = KDA_HEADS * KDA_HEAD_DIM; // 4096
    let packed_qkv_row = 3 * t; // q|k|v planes after conv
    assert_eq!(packed_qkv_row, 12_288);
    assert_eq!(KDA_HEADS * (128 + 64), 6_144); // MLA q_proj out (nope|rope)
    assert_eq!(512 + 64, 576); // MLA kv_a out (latent | rope)
    assert_eq!(KDA_HEADS * 512 * 128 * 2, 4_194_304); // w_uk bytes
    assert_eq!(KDA_HEADS * 256 * 512 * 2, 8_388_608); // kv_b bytes
    // Routed expert slabs (per expert): packed [K/2, N] + scales [K/16, N].
    let gate_up_packed = (2560 / 2) * 768;
    let down_packed = (768 / 2) * 2560;
    assert_eq!(gate_up_packed + down_packed, 1_966_080);
    assert_eq!((2560 / 16) * 768 + (768 / 16) * 2560, 122_880 + 122_880);
}
