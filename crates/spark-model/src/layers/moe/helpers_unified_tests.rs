// SPDX-License-Identifier: AGPL-3.0-only
//! Characterize real legacy construction before extracting its private phases.

use super::*;
use crate::weight_map::{ExpertWeight, MoeWeights, WeightQuantFormat};
#[path = "helpers_unified_test_gpu.rs"]
mod recording;
use recording::{Arg, Event, RecordingGpu};
#[path = "helpers_checkpoint_down_tests.rs"]
mod checkpoint_down;
#[path = "helpers_unified_fault_tests.rs"]
mod faults;
#[path = "helpers_unified_inplace_tests.rs"]
mod inplace;

fn make_weight(gpu: &RecordingGpu, n: usize, k: usize, gs: usize, seed: u8) -> QuantizedWeight {
    let weight = gpu.alloc(n * k / 2).unwrap();
    let weight_scale = gpu.alloc(n * k / gs).unwrap();
    for (ptr, len) in [(weight, n * k / 2), (weight_scale, n * k / gs)] {
        let bytes: Vec<_> = (0..len)
            .map(|i| (i.wrapping_mul(71) ^ (i >> 8)) as u8 ^ seed)
            .collect();
        gpu.copy_h2d(&bytes, ptr).unwrap();
    }
    QuantizedWeight {
        weight,
        weight_scale,
        weight_scale_2: f32::from(seed) + 0.25,
        input_scale: DevicePtr(0xf000_0000 + u64::from(seed) * 16),
        weight_scale_2_vec: DevicePtr(0xe000_0000 + u64::from(seed) * 16),
    }
}
fn fixture(
    gpu: &RecordingGpu,
    gs: usize,
    local: bool,
    shared: bool,
    shared_inter: usize,
) -> (MoeLayer, atlas_core::config::ModelConfig, Vec<ExpertWeight>) {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.moe_intermediate_size = 64;
    config.shared_expert_intermediate_size = shared_inter;
    config.num_experts = 4;
    config.num_experts_per_tok = 2;
    let mut weights = MoeWeights::empty(4);
    if local {
        for e in [1, 3] {
            weights.experts[e] = ExpertWeight {
                gate_proj: make_weight(gpu, 64, 128, gs, 1 + e as u8 * 3),
                up_proj: make_weight(gpu, 64, 128, gs, 2 + e as u8 * 3),
                down_proj: make_weight(gpu, 128, 64, gs, 3 + e as u8 * 3),
            };
        }
    }
    if shared {
        weights.shared_expert = ExpertWeight {
            gate_proj: make_weight(gpu, 32, 128, 16, 31),
            up_proj: make_weight(gpu, 32, 128, 16, 32),
            down_proj: make_weight(gpu, 128, 32, 16, 33),
        };
    }
    let mut originals = weights.experts.clone();
    originals.push(weights.shared_expert);
    let mut layer = MoeLayer::new(weights, 4, None, gpu, &config).unwrap();
    layer.experts_scale_kind = if gs == 32 {
        WeightQuantFormat::Mxfp4E8m0
    } else {
        WeightQuantFormat::Nvfp4
    };
    layer.unified_layout = true;
    layer.hybrid_layout = false;
    gpu.clear();
    (layer, config, originals)
}
fn run(
    layer: &mut MoeLayer,
    gpu: &RecordingGpu,
    config: &atlas_core::config::ModelConfig,
    mode: usize,
) -> Result<()> {
    match mode {
        0 => layer.transpose_for_prefill_unified(gpu, config).map(|_| ()),
        1 => layer
            .transpose_for_prefill_unified_keep_shared(gpu, config)
            .map(|_| ()),
        2 => layer.transpose_for_prefill_hybrid(gpu, config),
        3 => layer
            .transpose_for_prefill_unified_inner(gpu, config, true, false)
            .map(|_| ()),
        _ => unreachable!(),
    }
}
fn projections(e: &ExpertWeight) -> [QuantizedWeight; 3] {
    [e.gate_proj, e.up_proj, e.down_proj]
}
fn table_ptrs(t: &ExpertPtrTable) -> [DevicePtr; 3] {
    [t.packed_ptrs, t.scale_ptrs, t.scale2_vals]
}
fn decode_ptrs(b: &[u8]) -> Vec<DevicePtr> {
    b.chunks_exact(8)
        .map(|v| DevicePtr(u64::from_le_bytes(v.try_into().unwrap())))
        .collect()
}
fn bytes_at(trace: &[Event], p: DevicePtr) -> Vec<u8> {
    trace
        .iter()
        .find_map(|e| match e {
            Event::H2d(dst, b) if *dst == p => Some(b.clone()),
            _ => None,
        })
        .expect("uploaded table")
}
fn argument_ptr(a: &Arg) -> DevicePtr {
    if let Arg::Ptr(p) = a {
        *p
    } else {
        panic!("pointer ABI")
    }
}
fn argument_u32(a: &Arg) -> u32 {
    if let Arg::Bytes(b) = a {
        u32::from_ne_bytes(b.as_slice().try_into().unwrap())
    } else {
        panic!("u32 ABI")
    }
}
fn kinds(trace: &[Event]) -> String {
    trace
        .iter()
        .map(|e| match e {
            Event::Alloc(..) => 'A',
            Event::H2d(..) => 'H',
            Event::D2h(..) => 'D',
            Event::D2d(..) => 'C',
            Event::Free(..) => 'F',
            Event::Sync(..) => 'S',
            Event::Launch(..) => 'L',
        })
        .collect()
}

// Declarative operation/size contract, independent of the implementation's
// control flow. This is run unchanged on the pre-extraction legacy wrappers.
fn expected_profile(
    gs: usize,
    local: bool,
    shared: bool,
    host: bool,
    mode: usize,
) -> (String, Vec<usize>) {
    let slab = "AAAHAHAHAHAHAHLLSFFFFFF";
    let slab_sizes = [8192, 16384 / gs, 32, 32, 16, 32, 32, 16];
    let table = "AHAHAH";
    let single = if host { "DAHDAH" } else { "AALLS" };
    let mut ops = String::new();
    let mut sizes = Vec::new();
    for _ in 0..2 {
        if local {
            ops.push_str(slab);
            sizes.extend(slab_sizes);
        }
    }
    ops.push_str(&table.repeat(2));
    sizes.extend([32, 32, 16, 32, 32, 16]);
    if shared {
        ops.push_str(&single.repeat(2));
        sizes.extend([2048, 256, 2048, 256]);
    }
    if mode < 2 {
        if local {
            ops.push_str("FFFFFFFF");
        }
        if shared && mode == 0 {
            ops.push_str("FFFF");
        }
    }
    if local {
        ops.push_str(slab);
        sizes.extend(slab_sizes);
    }
    ops.push_str(table);
    sizes.extend([32, 32, 16]);
    if shared {
        ops.push_str(single);
        sizes.extend([2048, 256]);
    }
    if mode < 2 {
        if local {
            ops.push_str("FFFF");
        }
        if shared && mode == 0 {
            ops.push_str("FF");
        }
    }
    (ops, sizes)
}
fn assert_abi_and_tables(
    layer: &MoeLayer,
    gpu: &RecordingGpu,
    originals: &[ExpertWeight],
    gs: usize,
    local: bool,
    shared: bool,
    host: bool,
) {
    let trace = gpu.trace();
    let mut routed = 0;
    let mut singles = 0;
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
                    assert_eq!(dst[e].is_null(), ![1, 3].contains(&e));
                }
                assert_eq!(dst[3].0 - dst[1].0, u64::from(rows * cols));
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
    assert_eq!(routed, if local { 6 } else { 0 });
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
                    decode_ptrs(&bytes_at(&trace, argument_ptr(&launch[1])))
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

#[test]
fn actual_legacy_wrappers_exact_order_abi_ownership_and_peak() {
    // Upstream MoeLayer::new now owns a permanent zero-accumulator slab.
    const ZERO_ACCUM_BYTES: usize = 65536;
    for gs in [16, 32] {
        for (local, shared, inter) in [
            (true, true, 32),
            (false, true, 32),
            (true, false, 32),
            (true, true, 0),
            (false, false, 32),
        ] {
            for host in [false, true] {
                for mode in 2..4 {
                    let gpu = RecordingGpu::new(host);
                    let (mut layer, config, originals) = fixture(&gpu, gs, local, shared, inter);
                    let initial = gpu.live();
                    let initial_bytes = gpu.profile().0;
                    if gs == 16 && local && shared {
                        assert_eq!(initial_bytes, 34560 + 240 + ZERO_ACCUM_BYTES);
                    }
                    run(&mut layer, &gpu, &config, mode).unwrap();
                    let trace = gpu.trace();
                    let active_shared = shared && inter > 0;
                    let (ops, allocs) = expected_profile(gs, local, active_shared, host, mode);
                    assert_eq!(
                        kinds(&trace),
                        ops,
                        "gs={gs} local={local} shared={shared} inter={inter} host={host} mode={mode}"
                    );
                    assert_eq!(
                        trace
                            .iter()
                            .filter_map(|e| if let Event::Alloc(_, n) = e {
                                Some(*n)
                            } else {
                                None
                            })
                            .collect::<Vec<_>>(),
                        allocs
                    );
                    assert_abi_and_tables(&layer, &gpu, &originals, gs, local, active_shared, host);
                    let mut expected_frees = Vec::new();
                    if mode < 2 {
                        for e in &originals[..4] {
                            for w in [e.gate_proj, e.up_proj] {
                                if !w.is_null() {
                                    expected_frees.extend([w.weight, w.weight_scale]);
                                }
                            }
                        }
                        if active_shared && mode == 0 {
                            for w in [originals[4].gate_proj, originals[4].up_proj] {
                                expected_frees.extend([w.weight, w.weight_scale]);
                            }
                        }
                        for e in &originals[..4] {
                            if !e.down_proj.is_null() {
                                expected_frees
                                    .extend([e.down_proj.weight, e.down_proj.weight_scale]);
                            }
                        }
                        if active_shared && mode == 0 {
                            expected_frees.extend([
                                originals[4].down_proj.weight,
                                originals[4].down_proj.weight_scale,
                            ]);
                        }
                    }
                    let actual_frees: Vec<_> = trace
                        .iter()
                        .filter_map(|e| {
                            if let Event::Free(p) = e {
                                initial.contains_key(&p.0).then_some(*p)
                            } else {
                                None
                            }
                        })
                        .collect();
                    assert_eq!(actual_frees, expected_frees);
                    let mut live = initial;
                    let mut peak = initial_bytes;
                    for event in &trace {
                        match event {
                            Event::Alloc(p, n) => {
                                assert!(live.insert(p.0, *n).is_none());
                                peak = peak.max(live.values().sum());
                            }
                            Event::Free(p) => {
                                assert!(live.remove(&p.0).is_some());
                            }
                            _ => {}
                        }
                    }
                    assert_eq!(gpu.live(), live);
                    assert_eq!(gpu.profile(), (live.values().sum(), peak));
                    if gs == 16 && local && active_shared && !host {
                        assert_eq!(
                            gpu.profile(),
                            match mode {
                                0 => (35040 + ZERO_ACCUM_BYTES, 58000 + ZERO_ACCUM_BYTES),
                                1 => (41952 + ZERO_ACCUM_BYTES, 58000 + ZERO_ACCUM_BYTES),
                                _ => (69600 + ZERO_ACCUM_BYTES, 69600 + ZERO_ACCUM_BYTES),
                            }
                        );
                    }
                    for (e, got) in layer
                        .weights
                        .experts
                        .iter()
                        .chain(std::iter::once(&layer.weights.shared_expert))
                        .enumerate()
                    {
                        for (p, w) in projections(got).iter().enumerate() {
                            let original = projections(&originals[e])[p];
                            let freed = expected_frees.contains(&original.weight);
                            assert_eq!(
                                w.weight,
                                if freed {
                                    DevicePtr::NULL
                                } else {
                                    original.weight
                                }
                            );
                            assert_eq!(
                                w.weight_scale,
                                if freed {
                                    DevicePtr::NULL
                                } else {
                                    original.weight_scale
                                }
                            );
                            assert_eq!(
                                (
                                    w.weight_scale_2.to_bits(),
                                    w.input_scale,
                                    w.weight_scale_2_vec
                                ),
                                (
                                    original.weight_scale_2.to_bits(),
                                    original.input_scale,
                                    original.weight_scale_2_vec
                                )
                            );
                        }
                    }
                }
            }
        }
    }
}
