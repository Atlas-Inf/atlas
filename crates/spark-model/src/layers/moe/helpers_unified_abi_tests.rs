// SPDX-License-Identifier: AGPL-3.0-only
//! Typed transpose ABI, copyback extents, and published tables.
use super::*;

pub(super) fn assert_abi_and_tables(
    layer: &MoeLayer,
    gpu: &RecordingGpu,
    originals: &[ExpertWeight],
    gs: usize,
    local: bool,
    shared: bool,
    host: bool,
    mode: usize,
) {
    let trace = gpu.trace();
    let mut routed = 0;
    let mut singles = 0;
    let mut expected_copies = Vec::new();
    for event in &trace {
        if let Event::Launch(k, grid, block, smem, stream, args) = event {
            assert_eq!(*block, [32, 8, 1]);
            assert_eq!(*smem, 0);
            assert_eq!(args.len(), 4);
            let (rows, cols) = (argument_u32(&args[2]), argument_u32(&args[3]));
            assert_eq!(
                *grid,
                [
                    cols.div_ceil(32),
                    rows.div_ceil(32),
                    if *k == 102 { 4 } else { 1 }
                ]
            );
            if *k == 102 {
                assert_eq!(*stream, 77);
                let p = routed / 2;
                let scales = routed % 2 == 1;
                assert_eq!(
                    (rows, cols),
                    if p < 2 {
                        (64, if scales { 128 / gs as u32 } else { 64 })
                    } else {
                        (128, if scales { 64 / gs as u32 } else { 32 })
                    }
                );
                let src = decode_ptrs(&bytes_at(&trace, argument_ptr(&args[0])));
                let dst = decode_ptrs(&bytes_at(&trace, argument_ptr(&args[1])));
                for e in 0..4 {
                    let w = projections(&originals[e])[p];
                    assert_eq!(src[e], if scales { w.weight_scale } else { w.weight });
                    assert_eq!(dst[e].is_null(), !local || ![1, 3].contains(&e));
                }
                if mode < 2 {
                    for e in [1, 3] {
                        if local {
                            expected_copies.push((
                                p,
                                e,
                                scales,
                                Event::D2d(dst[e], src[e], (rows * cols) as usize),
                            ));
                        }
                    }
                }
                if local {
                    assert_eq!(
                        dst[3].0 - dst[1].0,
                        u64::from(rows * cols) * if mode < 2 { 2 } else { 1 }
                    );
                }
                routed += 1;
            } else {
                assert_eq!(*k, 101);
                assert_eq!(*stream, 0);
                let p = singles / 2;
                let scales = singles % 2 == 1;
                assert_eq!(
                    (rows, cols),
                    if p < 2 {
                        (32, if scales { 8 } else { 64 })
                    } else {
                        (128, if scales { 2 } else { 16 })
                    }
                );
                let w = projections(&originals[4])[p];
                assert_eq!(
                    argument_ptr(&args[0]),
                    if scales { w.weight_scale } else { w.weight }
                );
                singles += 1;
            }
        }
    }
    expected_copies.sort_by_key(|(p, e, scales, _)| (*p, *e, *scales));
    assert_eq!(
        trace
            .iter()
            .filter(|e| matches!(e, Event::D2d(..)))
            .cloned()
            .collect::<Vec<_>>(),
        expected_copies
            .into_iter()
            .map(|(_, _, _, e)| e)
            .collect::<Vec<_>>()
    );
    assert_eq!(routed, if local || mode < 2 { 6 } else { 0 });
    assert_eq!(singles, if shared && !host { 6 } else { 0 });
    for (p, t) in [&layer.gate_ptrs_t, &layer.up_ptrs_t, &layer.down_ptrs_t]
        .iter()
        .enumerate()
    {
        let t = t.as_ref().unwrap();
        for (scales, ptr) in [(false, t.packed_ptrs), (true, t.scale_ptrs)] {
            let got = decode_ptrs(&gpu.read(ptr, 32));
            if local {
                let launch = trace
                    .iter()
                    .filter_map(|e| {
                        if let Event::Launch(102, _, _, _, _, a) = e {
                            Some(a)
                        } else {
                            None
                        }
                    })
                    .nth(p * 2 + usize::from(scales))
                    .unwrap();
                assert_eq!(
                    got,
                    decode_ptrs(&bytes_at(
                        &trace,
                        argument_ptr(&launch[usize::from(mode >= 2)])
                    ))
                );
            } else {
                assert_eq!(got, vec![DevicePtr::NULL; 4]);
            }
        }
        let expected: Vec<_> = originals[..4]
            .iter()
            .flat_map(|e| {
                let w = projections(e)[p];
                // Legacy transpose canonicalizes remote slots through QW::null,
                // whose scalar is zero (MoeWeights::empty uses scalar one).
                if w.is_null() {
                    0.0f32
                } else {
                    w.weight_scale_2
                }
                .to_le_bytes()
            })
            .collect();
        assert_eq!(gpu.read(t.scale2_vals, 16), expected);
    }
    for (p, t) in [layer.shared_gate_t, layer.shared_up_t, layer.shared_down_t]
        .into_iter()
        .enumerate()
    {
        assert_eq!(t.is_some(), shared);
        if let Some(t) = t {
            let src = projections(&originals[4])[p];
            assert_eq!(
                (
                    t.weight_scale_2.to_bits(),
                    t.input_scale,
                    t.weight_scale_2_vec
                ),
                (
                    src.weight_scale_2.to_bits(),
                    src.input_scale,
                    src.weight_scale_2_vec
                )
            );
        }
    }
}
