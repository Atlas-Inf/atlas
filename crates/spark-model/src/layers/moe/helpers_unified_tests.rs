// SPDX-License-Identifier: AGPL-3.0-only
//! Characterize real legacy construction before extracting its private phases.

use super::*;
use crate::weight_map::{ExpertWeight, MoeWeights, WeightQuantFormat};
#[path = "helpers_unified_test_gpu.rs"]
mod recording;
use recording::{Arg, Event, RecordingGpu};
#[path = "helpers_unified_abi_tests.rs"]
mod abi;
#[path = "helpers_checkpoint_down_tests.rs"]
mod checkpoint_down;
#[path = "helpers_unified_fault_tests.rs"]
mod faults;
use abi::assert_abi_and_tables;

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

// Declarative operation/size contract for upstream in-place unified transforms
// and the branch's compact hybrid transforms.
fn expected_profile(
    gs: usize,
    local: bool,
    shared: bool,
    host: bool,
    mode: usize,
) -> (String, Vec<usize>) {
    let compact = "AAAHAHAHAHAHAHLLSFFFFFF";
    let table = "AHAHAH";
    let single = if host { "DAHDAH" } else { "AALLS" };
    let mut ops = String::new();
    let mut sizes = Vec::new();
    let inplace = mode < 2;
    for p in 0..3 {
        if inplace {
            if p != 1 {
                ops.push_str("AA");
                sizes.extend([16384, 32768 / gs]);
            }
            ops.push_str("AHAHAHAHAHAHLLSFFFFFF");
            sizes.extend([32, 32, 16, 32, 32, 16]);
            if local {
                ops.push_str("CCCC");
            }
            ops.push('S');
            if p == 2 {
                ops.push_str("FF");
            }
        } else if local {
            ops.push_str(compact);
            sizes.extend([8192, 16384 / gs, 32, 32, 16, 32, 32, 16]);
        }
        // Gate/up tables and shared transforms publish together after both
        // routed projections; down publishes after its scratch is released.
        if p == 0 {
            continue;
        }
        let count = if p == 1 { 2 } else { 1 };
        ops.push_str(&table.repeat(count));
        for _ in 0..count {
            sizes.extend([32, 32, 16]);
        }
        if shared {
            ops.push_str(&single.repeat(count));
            for _ in 0..count {
                sizes.extend([2048, 256]);
            }
            if mode == 0 {
                ops.push_str(&"FF".repeat(count));
            }
        }
        if inplace && p == 1 {
            ops.push_str("FF");
        }
    }
    (ops, sizes)
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
                for mode in 0..4 {
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
                    assert_abi_and_tables(
                        &layer,
                        &gpu,
                        &originals,
                        gs,
                        local,
                        active_shared,
                        host,
                        mode,
                    );
                    let mut expected_frees = Vec::new();
                    if active_shared && mode == 0 {
                        for w in projections(&originals[4]) {
                            expected_frees.extend([w.weight, w.weight_scale]);
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
                                1 => (41952 + ZERO_ACCUM_BYTES, 58160 + ZERO_ACCUM_BYTES),
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
                            let nulled =
                                (e < 4 && mode < 2) || expected_frees.contains(&original.weight);
                            assert_eq!(
                                w.weight,
                                if nulled {
                                    DevicePtr::NULL
                                } else {
                                    original.weight
                                }
                            );
                            assert_eq!(
                                w.weight_scale,
                                if nulled {
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
