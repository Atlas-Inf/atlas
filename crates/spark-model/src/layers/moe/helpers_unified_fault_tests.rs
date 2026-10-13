// SPDX-License-Identifier: AGPL-3.0-only
//! Isolated phases, byte oracle and legacy partial-construction semantics.
use super::*;

#[test]
fn actual_shared_host_fallback_compares_every_transposed_byte() {
    let gpu = RecordingGpu::new(true);
    let (mut layer, config, originals) = fixture(&gpu, 16, true, true, 32);
    run(&mut layer, &gpu, &config, 2).unwrap();
    for (p, dst) in [layer.shared_gate_t, layer.shared_up_t, layer.shared_down_t]
        .into_iter()
        .enumerate()
    {
        let src = projections(&originals[4])[p];
        let dst = dst.unwrap();
        let rows = if p < 2 { 32 } else { 128 };
        for (a, b, cols) in [
            (src.weight, dst.weight, if p < 2 { 64 } else { 16 }),
            (
                src.weight_scale,
                dst.weight_scale,
                if p < 2 { 8 } else { 2 },
            ),
        ] {
            let input = gpu.read(a, rows * cols);
            let output = gpu.read(b, rows * cols);
            let expected: Vec<_> = (0..rows * cols)
                .map(|i| input[(i % rows) * cols + i / rows])
                .collect();
            assert_eq!(output, expected);
        }
    }
}

#[test]
fn isolated_shared_down_phases_preserve_native_routed_gate_up_and_no_ready() {
    for gs in [16, 32] {
        for keep in [false, true] {
            for keep_shared in [false, true] {
                let gpu = RecordingGpu::new(false);
                let (mut layer, config, originals) = fixture(&gpu, gs, true, true, 32);
                let native = [table_ptrs(&layer.gate_ptrs), table_ptrs(&layer.up_ptrs)];
                layer
                    .transpose_unified_shared_gate_up(&gpu, &config)
                    .unwrap();
                layer
                    .transpose_unified_down_owned(&gpu, &config, gs, keep, keep_shared, None)
                    .unwrap();
                assert_eq!(
                    [table_ptrs(&layer.gate_ptrs), table_ptrs(&layer.up_ptrs)],
                    native
                );
                assert!(layer.gate_ptrs_t.is_none() && layer.up_ptrs_t.is_none());
                assert!(layer.down_ptrs_t.is_some());
                assert!(!layer.use_t_layout_for_decode() && !layer.use_t_layout_for_prefill());
                for (e, original) in layer.weights.experts.iter().zip(&originals) {
                    for (a, b) in [
                        (e.gate_proj, original.gate_proj),
                        (e.up_proj, original.up_proj),
                    ] {
                        assert_eq!((a.weight, a.weight_scale), (b.weight, b.weight_scale));
                        if !a.is_null() {
                            assert!(gpu.live().contains_key(&a.weight.0));
                        }
                    }
                }
                let forbidden: Vec<_> = originals[..4]
                    .iter()
                    .flat_map(|e| {
                        [
                            e.gate_proj.weight,
                            e.gate_proj.weight_scale,
                            e.up_proj.weight,
                            e.up_proj.weight_scale,
                        ]
                    })
                    .filter(|p| !p.is_null())
                    .collect();
                for event in gpu.trace() {
                    match event {
                        Event::Free(p) | Event::D2h(p, _) => assert!(!forbidden.contains(&p)),
                        Event::H2d(_, bytes) if bytes.len() == 32 => {
                            for p in decode_ptrs(&bytes) {
                                assert!(!forbidden.contains(&p));
                            }
                        }
                        Event::Launch(_, _, _, _, _, a) => {
                            for arg in a {
                                if let Arg::Ptr(p) = arg {
                                    assert!(!forbidden.contains(&p));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

#[test]
fn isolated_down_preserves_gate_up_without_shared_gate_presence() {
    let gpu = RecordingGpu::new(false);
    let (mut layer, config, originals) = fixture(&gpu, 16, true, true, 32);
    layer.weights.shared_expert.gate_proj = QuantizedWeight::null();
    layer
        .transpose_unified_down_owned(&gpu, &config, 16, false, false, None)
        .unwrap();
    assert!(layer.shared_down_t.is_some());
    assert!(layer.weights.shared_expert.down_proj.is_null());
    assert_eq!(
        layer.weights.shared_expert.up_proj.weight,
        originals[4].up_proj.weight
    );
    assert!(layer.gate_ptrs_t.is_none() && layer.up_ptrs_t.is_none());
    assert!(!layer.use_t_layout_for_decode() && !layer.use_t_layout_for_prefill());
}

#[test]
fn every_legacy_io_failure_stops_at_exact_prefix_and_preserves_nulling_order() {
    for host in [false, true] {
        for mode in 2..4 {
            let gpu = RecordingGpu::new(host);
            let (mut layer, config, _) = fixture(&gpu, 16, true, true, 32);
            run(&mut layer, &gpu, &config, mode).unwrap();
            let good = gpu.trace();
            let tables = [&layer.gate_ptrs_t, &layer.up_ptrs_t, &layer.down_ptrs_t]
                .map(|t| table_ptrs(t.as_ref().unwrap()));
            let shared = [layer.shared_gate_t, layer.shared_up_t, layer.shared_down_t];
            let mut publication = Vec::new();
            for t in tables {
                publication.push(
                    good.iter()
                        .position(|e| matches!(e, Event::H2d(p, _) if *p == t[2]))
                        .unwrap(),
                );
            }
            for w in shared {
                let w = w.unwrap();
                let last_write = good
                    .iter()
                    .position(|e| match e {
                        Event::H2d(p, _) => *p == w.weight_scale,
                        Event::Launch(101, _, _, _, _, a) => argument_ptr(&a[1]) == w.weight_scale,
                        _ => false,
                    })
                    .unwrap();
                publication.push(last_write + usize::from(!host)); // shared GPU sync
            }
            for at in 1..=good.len() {
                let gpu = RecordingGpu::new(host);
                let (mut layer, config, originals) = fixture(&gpu, 16, true, true, 32);
                let mut live = gpu.live();
                gpu.fail(at);
                let error = run(&mut layer, &gpu, &config, mode).unwrap_err();
                assert!(error.to_string().contains("injected op"), "{error:#}");
                assert_eq!(gpu.trace(), good[..at], "host={host} mode={mode} op={at}");
                for (i, event) in good[..at].iter().enumerate() {
                    match event {
                        Event::Alloc(p, n) if i + 1 < at => {
                            live.insert(p.0, *n);
                        }
                        // Actual CUDA ledger removes even the failed free.
                        Event::Free(p) => {
                            live.remove(&p.0).unwrap();
                        }
                        _ => {}
                    }
                }
                assert_eq!(gpu.live(), live);
                let present = [
                    layer.gate_ptrs_t.is_some(),
                    layer.up_ptrs_t.is_some(),
                    layer.down_ptrs_t.is_some(),
                    layer.shared_gate_t.is_some(),
                    layer.shared_up_t.is_some(),
                    layer.shared_down_t.is_some(),
                ];
                for (i, &published) in present.iter().enumerate() {
                    assert_eq!(published, publication[i] + 1 < at, "field={i} failure={at}");
                }
                for (e, current) in layer
                    .weights
                    .experts
                    .iter()
                    .chain(std::iter::once(&layer.weights.shared_expert))
                    .enumerate()
                {
                    for (p, w) in projections(current).iter().enumerate() {
                        let src = projections(&originals[e])[p];
                        let paired_free_succeeded =
                            good[..at - 1].contains(&Event::Free(src.weight_scale));
                        assert_eq!(
                            w.weight,
                            if paired_free_succeeded {
                                DevicePtr::NULL
                            } else {
                                src.weight
                            }
                        );
                        assert_eq!(
                            w.weight_scale,
                            if paired_free_succeeded {
                                DevicePtr::NULL
                            } else {
                                src.weight_scale
                            }
                        );
                    }
                }
                // No cleanup/fallback or completion after an error is introduced.
                // The caller must abandon this partially constructed layer.
            }
        }
    }
}

#[test]
fn exact_six_temporary_table_frees_and_scalar_metadata() {
    for gs in [16, 32] {
        let gpu = RecordingGpu::new(false);
        let (mut layer, config, originals) = fixture(&gpu, gs, true, true, 32);
        run(&mut layer, &gpu, &config, 2).unwrap();
        let trace = gpu.trace();
        let slabs: Vec<_> = trace
            .windows(23)
            .filter(|w| matches!(w[0], Event::Alloc(_, 8192)))
            .collect();
        assert_eq!(slabs.len(), 3);
        for (projection, window) in slabs.into_iter().enumerate() {
            let mut allocations = Vec::new();
            for i in [2, 4, 6, 8, 10, 12] {
                let Event::Alloc(p, _) = window[i] else {
                    panic!("table allocation")
                };
                allocations.push(p);
            }
            assert_eq!(
                &window[17..],
                &allocations
                    .iter()
                    .copied()
                    .map(Event::Free)
                    .collect::<Vec<_>>()
            );
            for i in [6, 12] {
                let Event::Alloc(p, 16) = window[i] else {
                    panic!("scalar table")
                };
                let expected: Vec<_> = originals[..4]
                    .iter()
                    .flat_map(|e| {
                        let w = projections(e)[projection];
                        if w.is_null() {
                            0.0f32
                        } else {
                            w.weight_scale_2
                        }
                        .to_le_bytes()
                    })
                    .collect();
                assert_eq!(bytes_at(&trace, p), expected);
            }
        }
    }
}
