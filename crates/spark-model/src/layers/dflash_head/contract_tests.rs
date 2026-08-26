// SPDX-License-Identifier: AGPL-3.0-only

use super::contract::*;

#[test]
fn valid_lightning_profile_is_exact_contract() {
    let profile = LightningDsparkProfile::lightning();

    profile.validate().expect("canonical Lightning profile");
    assert_eq!(profile.algorithm, "DSpark");
    assert_eq!(profile.model_identity, "Qwen3DSparkModel");
    assert_eq!(profile.checkpoint.block_size, 8);
    assert_eq!(profile.checkpoint.physical_kv_page_size, 16);
    assert_eq!(profile.served_gamma, 4);
    assert_eq!(profile.num_drafts, 3);
    assert_eq!(profile.taps, [1, 5, 19, 29, 41, 51]);
    assert!(profile.attention.causal);
    assert!(profile.attention.use_swa);
    assert_eq!(profile.attention.swa_window, 1024);
    assert!(profile.attention.attention_sink_bias);
    assert_eq!(profile.markov.rank, 512);
    assert!(profile.markov.w1_present);
    assert!(profile.markov.w2_present);
    assert!(profile.bonus.bonus_anchor);
    assert!(!profile.bonus.sample_from_anchor);
    assert!(!profile.confidence.head_present);
    assert!(!profile.confidence.adaptive);
    assert_eq!(profile.kv.target, KvDtype::Fp8);
    assert_eq!(profile.kv.drafter, KvDtype::Bf16);
    assert_eq!(profile.parallelism.tensor_parallel, 1);
    assert_eq!(profile.parallelism.expert_parallel, 1);
}

#[test]
fn rejects_block_page_size_confusion() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.checkpoint.block_size = profile.checkpoint.physical_kv_page_size;

    let error = profile
        .validate()
        .expect_err("block/page confusion must fail");
    assert!(error.to_string().contains("checkpoint.block_size"));
    assert!(error.to_string().contains("expected 8"));
}

#[test]
fn rejects_gamma_or_k_drift() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.num_drafts = 4;

    let error = profile.validate().expect_err("K drift must fail");
    assert!(error.to_string().contains("num_drafts"));
    assert!(error.to_string().contains("expected 3"));
}

#[test]
fn rejects_tap_order_or_membership_drift() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.taps.swap(1, 2);

    let error = profile.validate().expect_err("tap drift must fail");
    assert!(error.to_string().contains("taps"));
    assert!(error.to_string().contains("ordered"));
}

#[test]
fn rejects_attention_swa_or_sink_drift() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.attention.use_swa = false;

    let error = profile.validate().expect_err("SWA drift must fail");
    assert!(error.to_string().contains("attention.use_swa"));
    assert!(error.to_string().contains("expected true"));
}

#[test]
fn rejects_markov_rank_or_missing_weights() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.markov.w2_present = false;

    let error = profile
        .validate()
        .expect_err("missing Markov weight must fail");
    assert!(error.to_string().contains("markov.w2_present"));
    assert!(error.to_string().contains("expected present"));
}

#[test]
fn rejects_bonus_anchor_sampling_drift() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.bonus.sample_from_anchor = true;

    let error = profile
        .validate()
        .expect_err("anchor sampling drift must fail");
    assert!(error.to_string().contains("bonus.sample_from_anchor"));
    assert!(error.to_string().contains("expected false"));
}

#[test]
fn rejects_confidence_head_or_adaptive_drift() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.confidence.head_present = true;

    let error = profile
        .validate()
        .expect_err("confidence head must be absent");
    assert!(error.to_string().contains("confidence.head_present"));
    assert!(error.to_string().contains("expected absent"));
}

#[test]
fn rejects_kv_dtype_ambiguity() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.kv.drafter = KvDtype::Fp8;

    let error = profile.validate().expect_err("drafter KV dtype must fail");
    assert!(error.to_string().contains("kv.drafter"));
    assert!(error.to_string().contains("expected BF16"));
}

#[test]
fn rejects_tensor_or_expert_parallelism_drift() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.parallelism.expert_parallel = 2;

    let error = profile
        .validate()
        .expect_err("EP drift must fail for one-GB10 profile");
    assert!(error.to_string().contains("parallelism.expert_parallel"));
    assert!(error.to_string().contains("expected 1"));
}
