// SPDX-License-Identifier: AGPL-3.0-only
//! Unified layout keeps routed allocations owned by the weight store.
use super::*;

#[test]
fn unified_tables_keep_routed_allocations_and_report_only_released_shared_weights() {
    for gs in [16, 32] {
        for local in [false, true] {
            for (shared, inter) in [(false, 0), (false, 32), (true, 0), (true, 32)] {
                for keep_shared in [false, true] {
                    let gpu = RecordingGpu::new(false);
                    let (mut layer, config, originals) = fixture(&gpu, gs, local, shared, inter);
                    let freed = if keep_shared {
                        layer.transpose_for_prefill_unified_keep_shared(&gpu, &config)
                    } else {
                        layer.transpose_for_prefill_unified(&gpu, &config)
                    }
                    .unwrap();
                    let live = gpu.live();
                    for (projection, table) in
                        [&layer.gate_ptrs_t, &layer.up_ptrs_t, &layer.down_ptrs_t]
                            .into_iter()
                            .enumerate()
                    {
                        let table = table.as_ref().unwrap();
                        for scales in [false, true] {
                            let ptrs = decode_ptrs(&gpu.read(
                                if scales {
                                    table.scale_ptrs
                                } else {
                                    table.packed_ptrs
                                },
                                32,
                            ));
                            for (index, expert) in originals[..4].iter().enumerate() {
                                let w = projections(expert)[projection];
                                let original = if scales { w.weight_scale } else { w.weight };
                                assert_eq!(ptrs[index], original);
                                if !original.is_null() {
                                    assert!(live.contains_key(&original.0));
                                    assert!(!freed.contains(&original));
                                    assert!(gpu.trace().iter().any(|event| matches!(event, Event::D2d(_, dst, _) if *dst == original)));
                                }
                            }
                        }
                    }
                    let expected: Vec<_> = if shared && inter > 0 && !keep_shared {
                        projections(&originals[4])
                            .into_iter()
                            .flat_map(|w| [w.weight, w.weight_scale])
                            .collect()
                    } else {
                        vec![]
                    };
                    assert_eq!(freed, expected);
                    for w in projections(&originals[4]) {
                        if !w.is_null() {
                            assert_eq!(live.contains_key(&w.weight.0), keep_shared || inter == 0);
                            assert_eq!(
                                live.contains_key(&w.weight_scale.0),
                                keep_shared || inter == 0
                            );
                        }
                    }
                }
            }
        }
    }
}
