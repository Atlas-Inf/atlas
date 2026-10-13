// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn runtime_scope_is_exact_independent_local_geometry() {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.tp_world_size = 2;
    config.ep_world_size = 2;
    config.linear_num_key_heads = 32;
    config.linear_num_value_heads = 32;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_conv_kernel_dim = 4;
    for rows in 2..=4 {
        assert!(runtime_eligible(&config, 2, true, true, rows, rows));
    }
    for (world, independent, enabled, rows, padded) in [
        (1, true, true, 2, 2),
        (2, false, true, 2, 2),
        (2, true, false, 2, 2),
        (2, true, true, 1, 1),
        (2, true, true, 5, 5),
        (2, true, true, 3, 4),
    ] {
        assert!(!runtime_eligible(
            &config,
            world,
            independent,
            enabled,
            rows,
            padded
        ));
    }
    let unsupported: [fn(&mut ModelConfig); 7] = [
        |c| c.tp_world_size = 1,
        |c| c.ep_world_size = 1,
        |c| c.linear_num_key_heads = 64,
        |c| c.linear_num_value_heads = 64,
        |c| c.linear_key_head_dim = 64,
        |c| c.linear_value_head_dim = 64,
        |c| c.linear_conv_kernel_dim = 3,
    ];
    for change in unsupported {
        let mut wrong = config.clone();
        change(&mut wrong);
        assert!(!runtime_eligible(&wrong, 2, true, true, 2, 2));
    }
    config.model_type = "qwen3_next".into();
    assert!(!runtime_eligible(&config, 2, true, true, 2, 2));
}

#[test]
fn rank_local_pools_refresh_ids_without_assuming_head_rank_slots() {
    let kinds = [LayerType::LinearAttention];
    let layout = DecodeMetaLayout::for_max_batch_size(4);
    for (h, c, slots) in [(0x1000, 0x3000, [6usize, 1]), (0x5000, 0x7000, [2usize, 5])] {
        let hb = [DevicePtr(h)];
        let cb = [DevicePtr(c)];
        let pool = SsmPoolView::new(&hb, &cb, 64, 64, 32, 8).unwrap();
        let states: Vec<_> = slots
            .iter()
            .map(|&s| vec![state(h + s as u64 * 64, c + s as u64 * 32)])
            .collect();
        for order in [[0, 1], [1, 0]] {
            let rows: Vec<_> = order
                .iter()
                .map(|&i| SsmRow {
                    slot: Some(slots[i]),
                    states: &states[i],
                })
                .collect();
            let prepared =
                prepare_rows(pool, &rows, &kinds, DevicePtr(0xa000), 4096, layout).unwrap();
            let expected: Vec<u8> = order
                .iter()
                .flat_map(|&i| (slots[i] as i32).to_le_bytes())
                .collect();
            assert_eq!(&prepared.slot_bytes[..8], expected.as_slice());
            assert_eq!(prepared.view.layer(0).unwrap().h_base(), DevicePtr(h));
        }
    }
}

fn state(h: u64, conv: u64) -> Box<dyn LayerState> {
    Box::new(SsmLayerState {
        h_state: DevicePtr(h),
        conv_state: DevicePtr(conv),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: vec![],
        kda_records: spark_runtime::gpu::DevicePtr::NULL,
        gdn_commit_qkv: DevicePtr(0),
        gdn_commit_gb: DevicePtr(0),
        gdn_commit_pending: false,
        conv_state_intermediates: vec![],
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    })
}

#[test]
fn live_pointers_follow_nonprefix_slots_and_global_layer_ordinals() {
    let h = [DevicePtr(0x1000), DevicePtr(0x3000)];
    let c = [DevicePtr(0x5000), DevicePtr(0x7000)];
    let pool = SsmPoolView::new(&h, &c, 64, 64, 32, 8).unwrap();
    let kinds = [
        LayerType::LinearAttention,
        LayerType::FullAttention,
        LayerType::LinearAttention,
    ];
    let a = vec![
        state(0x1180, 0x50c0),
        Box::new(crate::layer::EmptyLayerState) as Box<dyn LayerState>,
        state(0x3180, 0x70c0),
    ];
    let b = vec![
        state(0x1040, 0x5020),
        Box::new(crate::layer::EmptyLayerState) as Box<dyn LayerState>,
        state(0x3040, 0x7020),
    ];
    let rows = [
        SsmRow {
            slot: Some(6),
            states: &a,
        },
        SsmRow {
            slot: Some(1),
            states: &b,
        },
    ];
    let layout = DecodeMetaLayout::for_max_batch_size(4);
    let prepared = prepare_rows(pool, &rows, &kinds, DevicePtr(0x9000), 4096, layout).unwrap();
    assert_eq!(prepared.view.layer(1).unwrap().h_base(), h[1]);
    assert_eq!(&prepared.slot_bytes[..8], &[6, 0, 0, 0, 1, 0, 0, 0]);
    // Reordered and drained active rows must rewrite all padded IDs every step.
    let drained = prepare_rows(pool, &rows[1..], &kinds, DevicePtr(0x9000), 4096, layout).unwrap();
    assert_eq!(&drained.slot_bytes[..4], &[1, 0, 0, 0]);
    assert!(drained.slot_bytes[4..].iter().all(|&v| v == 255));
    let swapped = [
        SsmRow {
            slot: Some(1),
            states: &b,
        },
        SsmRow {
            slot: Some(6),
            states: &a,
        },
    ];
    let reordered = prepare_rows(pool, &swapped, &kinds, DevicePtr(0xa000), 4096, layout).unwrap();
    assert_eq!(&reordered.slot_bytes[..8], &[1, 0, 0, 0, 6, 0, 0, 0]);
    assert_eq!(reordered.view.layer(0).unwrap().slots(), DevicePtr(0xa280));
    for slot in [None, Some(1), Some(8), Some(usize::MAX)] {
        assert!(
            prepare_rows(
                pool,
                &[SsmRow { slot, states: &a }],
                &kinds,
                DevicePtr(0x9000),
                4096,
                layout
            )
            .is_err()
        );
    }
    assert!(prepare_rows(pool, &rows, &kinds[..2], DevicePtr(0x9000), 4096, layout).is_err());
}

#[test]
fn present_non_fp32_missing_and_stale_layer_states_fail_closed() {
    let h = [DevicePtr(0x1000)];
    let c = [DevicePtr(0x3000)];
    let pool = SsmPoolView::new(&h, &c, 64, 64, 32, 8).unwrap();
    let kinds = [LayerType::LinearAttention];
    let layout = DecodeMetaLayout::for_max_batch_size(4);
    let mut states = [state(0x1000, 0x3000)];
    for mode in 0..5 {
        let st = states[0]
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .unwrap();
        st.h_is_f16 = mode == 0;
        st.h_prefill_stage = (mode == 1).then_some(DevicePtr(0x9000));
        st.h_state = DevicePtr(if mode == 2 { 0x1040 } else { 0x1000 });
        st.conv_state = DevicePtr(if mode == 3 { 0x3020 } else { 0x3000 });
        let live = if mode == 4 { &states[..0] } else { &states[..] };
        assert!(
            prepare_rows(
                pool,
                &[SsmRow {
                    slot: Some(0),
                    states: live
                }],
                &kinds,
                DevicePtr(0x9000),
                4096,
                layout
            )
            .is_err()
        );
    }
    let empty: Vec<Box<dyn LayerState>> = vec![Box::new(crate::layer::EmptyLayerState)];
    assert!(
        prepare_rows(
            pool,
            &[SsmRow {
                slot: Some(0),
                states: &empty
            }],
            &kinds,
            DevicePtr(0x9000),
            4096,
            layout
        )
        .is_err()
    );
}

#[test]
fn absent_pool_is_fallback_not_malformed_metadata() {
    assert!(
        prepare_indexed_decode(
            None,
            &[],
            &[],
            DevicePtr::NULL,
            0,
            DecodeMetaLayout::for_max_batch_size(4)
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn padded_slot_payload_cannot_overlap_live_state() {
    let h = [DevicePtr(0x1000)];
    let c = [DevicePtr(0x3000)];
    let pool = SsmPoolView::new(&h, &c, 64, 64, 32, 8).unwrap();
    let states = vec![state(0x1000, 0x3000)];
    let row = [SsmRow {
        slot: Some(0),
        states: &states,
    }];
    let kinds = [LayerType::LinearAttention];
    let layout = DecodeMetaLayout::for_max_batch_size(4);
    // Active ID ends before H, but the complete 32-row padded upload overlaps it.
    assert!(prepare_rows(pool, &row, &kinds, DevicePtr(0xd04), 4096, layout).is_err());
    // Moving four bytes earlier makes the padded region exactly adjacent.
    assert!(prepare_rows(pool, &row, &kinds, DevicePtr(0xd00), 4096, layout).is_ok());
}

#[test]
fn layer_ordinals_skip_attention_and_reject_wrong_layer_kinds() {
    let kinds = [
        LayerType::LinearAttention,
        LayerType::FullAttention,
        LayerType::LinearAttention,
    ];
    assert_eq!(ssm_ordinal(&kinds, 0).unwrap(), 0);
    assert_eq!(ssm_ordinal(&kinds, 2).unwrap(), 1);
    assert!(ssm_ordinal(&kinds, 1).is_err());
    assert!(ssm_ordinal(&kinds, 3).is_err());
}

#[test]
fn slot_region_is_bounded_and_preserves_legacy_metadata() {
    for rows in [32, 64, 128] {
        let layout = DecodeMetaLayout::for_max_batch_size(rows);
        let (ptr, bytes) =
            slot_upload(DevicePtr(0x1000), layout.meta_bytes(2), layout, &[7, 2]).unwrap();
        assert_eq!(ptr, DevicePtr(0x1000 + (20 * rows) as u64));
        assert_eq!(bytes.len(), rows * 4);
        let ids: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(&ids[..2], &[7, 2]);
        assert!(ids[2..].iter().all(|&id| id == -1));
        assert!(slot_upload(DevicePtr(0x1000), 24 * rows - 1, layout, &[0]).is_err());
    }
    let layout = DecodeMetaLayout::for_max_batch_size(4);
    assert!(slot_upload(DevicePtr::NULL, 4096, layout, &[0]).is_err());
    assert!(slot_upload(DevicePtr(u64::MAX - 3), 4096, layout, &[0]).is_err());
    assert!(
        slot_upload(
            DevicePtr(0x1000),
            4096,
            DecodeMetaLayout::for_max_batch_size(usize::MAX),
            &[0]
        )
        .is_err()
    );
}
