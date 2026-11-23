// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightTensor};

fn put_projection(store: &mut WeightStore, name: &str, shape: &[usize], ptr: u64) {
    store
        .insert(
            format!("{name}.weight"),
            WeightTensor {
                ptr: DevicePtr(ptr),
                shape: shape.to_vec(),
                dtype: WeightDtype::BF16,
            },
        )
        .unwrap();
}

fn put_dense(store: &mut WeightStore, name: &str, shape: &[usize], dtype: WeightDtype, ptr: u64) {
    store
        .insert(
            name.to_owned(),
            WeightTensor {
                ptr: DevicePtr(ptr),
                shape: shape.to_vec(),
                dtype,
            },
        )
        .unwrap();
}

fn test_store() -> WeightStore {
    let mut store = WeightStore::empty();
    let proj = [LING_KDA_PROJECTION_SIZE, LING_KDA_HIDDEN_SIZE];
    for (name, ptr) in [
        ("q_proj", 0x1000),
        ("k_proj", 0x2000),
        ("v_proj", 0x3000),
        ("f_proj", 0x4000),
        ("g_proj", 0x5000),
        ("o_proj", 0x6000),
    ] {
        put_projection(&mut store, name, &proj, ptr);
    }
    put_projection(
        &mut store,
        "b_proj",
        &[LING_KDA_HEADS, LING_KDA_HIDDEN_SIZE],
        0x7000,
    );
    put_dense(
        &mut store,
        "q_conv1d.weight",
        &[LING_KDA_PROJECTION_SIZE, 1, LING_KDA_CONV_WIDTH],
        WeightDtype::BF16,
        0x8000,
    );
    for (name, ptr) in [("k_conv1d.weight", 0x9000), ("v_conv1d.weight", 0xa000)] {
        put_dense(
            &mut store,
            name,
            &[LING_KDA_PROJECTION_SIZE, LING_KDA_CONV_WIDTH],
            WeightDtype::BF16,
            ptr,
        );
    }
    put_dense(
        &mut store,
        "A_log",
        &[LING_KDA_HEADS],
        WeightDtype::FP32,
        0xb000,
    );
    put_dense(
        &mut store,
        "dt_bias",
        &[LING_KDA_HEADS, LING_KDA_HEAD_DIM],
        WeightDtype::FP32,
        0xc000,
    );
    put_dense(
        &mut store,
        "o_norm.weight",
        &[LING_KDA_HEAD_DIM],
        WeightDtype::BF16,
        0xd000,
    );
    store
}

fn test_scratch() -> LingKdaScratch {
    LingKdaScratch::new(
        DevicePtr(0x10_0000),
        DevicePtr(0x11_0000),
        DevicePtr(0x12_0000),
        DevicePtr(0x13_0000),
        DevicePtr(0x14_0000),
        DevicePtr(0x15_0000),
        DevicePtr(0x16_0000),
        DevicePtr(0x17_0000),
    )
}

#[test]
fn pinned_geometry_matches_kda_kernel_contract() {
    assert_eq!(LING_KDA_PROJECTION_SIZE, 4096);
    assert_eq!(LingKdaState::conv_state_bytes(), 4096 * 4 * 4);
    assert_eq!(LingKdaState::h_state_bytes(), 32 * 128 * 128 * 4);
    assert_eq!(
        std::mem::size_of::<LingKdaScratch>(),
        8 * std::mem::size_of::<DevicePtr>()
    );
}

#[test]
fn mock_decode_dispatches_direct_kda_chain_and_separate_residual() {
    let gpu = MockGpuBackend::new();
    let weights = LingKdaWeights::load(&test_store(), "").unwrap();
    let layer = LingKdaLayer::new(weights, LingKdaConfig::pinned(), &gpu).unwrap();
    let state = LingKdaState {
        q_conv: DevicePtr(0x20_0000),
        k_conv: DevicePtr(0x21_0000),
        v_conv: DevicePtr(0x22_0000),
        h: DevicePtr(0x23_0000),
    };
    let scratch = test_scratch();
    let projection_scratch = LingProjectionScratch::new(DevicePtr::NULL, 0, DevicePtr::NULL, 0);

    let output = layer
        .forward_attention(
            &gpu,
            DevicePtr(0x24_0000),
            state,
            scratch,
            projection_scratch,
            projection_scratch,
            7,
        )
        .unwrap();
    assert_eq!(output, scratch.output);
    // q/k/v + f/b/g + o, then three convs, recurrence, gated norm.
    assert_eq!(gpu.launch_count(), 12);

    layer
        .add_residual(&gpu, DevicePtr(0x25_0000), output, 7)
        .unwrap();
    assert_eq!(gpu.launch_count(), 13);
}

#[test]
fn config_rejects_non_pinned_decay_bound() {
    let gpu = MockGpuBackend::new();
    let config = LingKdaConfig {
        lower_bound: -4.0,
        ..LingKdaConfig::pinned()
    };
    let weights = LingKdaWeights::load(&test_store(), "").unwrap();
    let err = match LingKdaLayer::new(weights, config, &gpu) {
        Ok(_) => panic!("non-pinned KDA config unexpectedly accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("lower_bound"));
}
