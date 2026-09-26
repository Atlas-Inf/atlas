// SPDX-License-Identifier: AGPL-3.0-only

//! Layout-detection tests for the qwen4_exp MTP loader (`mtp.rs`).

use super::*;
use std::collections::HashMap;

fn store_with(names: &[&str]) -> WeightStore {
    let map: HashMap<String, spark_runtime::weights::WeightTensor> = names
        .iter()
        .map(|n| {
            (
                n.to_string(),
                spark_runtime::weights::WeightTensor {
                    ptr: spark_runtime::gpu::DevicePtr::NULL,
                    shape: vec![1],
                    dtype: spark_runtime::weights::WeightDtype::BF16,
                },
            )
        })
        .collect();
    WeightStore::from_map(map)
}

/// RadixArk pack: the fused pair selects the stacked path.
#[test]
fn stacked_markers_select_stacked_layout() {
    let store = store_with(&[
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
    ]);
    assert_eq!(
        mtp_expert_layout(&store, "mtp.layers.0.mlp"),
        MtpExpertLayout::StackedBf16
    );
}

/// nvidia pack: per-expert weight_scale_inv selects the FP8 block path —
/// and must win even when a stray fused name is absent.
#[test]
fn per_expert_scale_inv_selects_fp8_layout() {
    let store = store_with(&[
        "mtp.layers.0.mlp.experts.0.gate_proj.weight",
        "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
    ]);
    assert_eq!(
        mtp_expert_layout(&store, "mtp.layers.0.mlp"),
        MtpExpertLayout::PerExpertFp8BlockScaled
    );
}

/// Neither marker → stacked default, whose loader errors with the names
/// it probed (better than guessing the other format).
#[test]
fn no_markers_defaults_to_stacked() {
    let store = store_with(&[]);
    assert_eq!(
        mtp_expert_layout(&store, "mtp.layers.0.mlp"),
        MtpExpertLayout::StackedBf16
    );
}

/// turboderp EXL3 build: a per-expert `trellis` selects the EXL3 path.
#[test]
fn per_expert_trellis_selects_exl3_layout() {
    let store = store_with(&["mtp.layers.0.mlp.experts.0.gate_proj.trellis"]);
    assert_eq!(
        mtp_expert_layout(&store, "mtp.layers.0.mlp"),
        MtpExpertLayout::PerExpertExl3
    );
}

/// The stacked check stays first: a pack with BOTH the fused pair and
/// per-expert trellis names takes the stacked path.
#[test]
fn stacked_wins_over_trellis() {
    let store = store_with(&[
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.experts.0.gate_proj.trellis",
    ]);
    assert_eq!(
        mtp_expert_layout(&store, "mtp.layers.0.mlp"),
        MtpExpertLayout::StackedBf16
    );
}
