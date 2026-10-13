// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::layers::moe::build_ptr_table_from_qw;
use crate::weight_map::QuantizedWeight;
use spark_runtime::gpu::mock::MockGpuBackend;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.moe_intermediate_size = 2048;
    c.shared_expert_intermediate_size = 2048;
    c.num_experts = 288;
    c.num_experts_per_tok = 8;
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.tp_rank = 0;
    c.ep_rank = 0;
    c
}
#[test]
fn actual_complete_family_resolves_all_handles() {
    let gpu = MockGpuBackend::new();
    let family = KernelFamily::resolve(&gpu, &config(), 77).unwrap();
    assert!(family.handles.iter().all(|h| h.0 != 0));
}
#[test]
fn actual_existing_builder_supplies_capacity_authority() {
    let gpu = MockGpuBackend::new();
    let mut table = build_ptr_table_from_qw(&vec![QuantizedWeight::null(); 288], &gpu).unwrap();
    let regions = table.owned_regions(&gpu, 288).unwrap();
    assert_eq!(regions.map(|r| r.1), [2304, 2304, 1152]);
    assert!(table.owned_regions(&gpu, 287).is_err());
    table.packed_ptrs.0 += 8;
    assert!(table.owned_regions(&gpu, 288).is_err());
}

#[test]
fn complete_family_fails_closed_at_each_export_before_publication() {
    use super::super::recording::{Event, Gpu};
    use std::sync::atomic::Ordering;
    for zero in [false, true] {
        for index in 1..=16 {
            let gpu = Gpu::new();
            gpu.lookup_failure.store(index, Ordering::Relaxed);
            gpu.lookup_zero.store(zero, Ordering::Relaxed);
            assert!(KernelFamily::resolve(&gpu, &config(), 77).is_err());
            assert_eq!(gpu.trace().len(), index);
            assert!(gpu.trace().iter().all(|e| matches!(e, Event::Lookup(..))));
        }
    }
    let gpu = Gpu::new();
    gpu.capture.store(true, Ordering::Relaxed);
    assert!(KernelFamily::resolve(&gpu, &config(), 77).is_err());
    assert!(gpu.trace().is_empty());
    gpu.capture.store(false, Ordering::Relaxed);
    let mut bad = config();
    bad.hidden_size = 8192;
    assert!(KernelFamily::resolve(&gpu, &bad, 77).is_err());
    assert!(gpu.trace().is_empty());
    KernelFamily::resolve(&gpu, &config(), 77).unwrap();
    assert_eq!(
        gpu.trace(),
        [
            ("glm_moe_btile_native_repack", "glm_native_to_btile_u8"),
            ("transpose_u8", "transpose_u8"),
            ("glm_moe_btile_decode", "glm_btile_decode_word1"),
            ("glm_moe_btile_decode", "glm_btile_decode_word2"),
            ("glm_moe_btile_decode", "glm_btile_decode_word3"),
            ("glm_moe_btile_decode", "glm_btile_decode_vec1"),
            ("glm_moe_btile_decode", "glm_btile_decode_vec2"),
            ("glm_moe_btile_decode", "glm_btile_decode_vec3"),
            ("moe_w4a16", "glm_moe_gate_up_btile"),
            ("moe_w4a16", "glm_moe_gate_up_btile_vecscale"),
            ("moe_w4a16", "glm_moe_gate_up_btile_m64"),
            ("moe_w4a16", "glm_moe_gate_up_btile_m64_vecscale"),
            ("moe_w4a16", "glm_moe_btile_m64_dense"),
            ("moe_w4a16", "glm_moe_btile_m64_vecscale_dense"),
            ("moe_w4a16", "glm_moe_btile_m64_compact"),
            ("moe_w4a16", "glm_moe_btile_m64_vecscale_compact"),
        ]
        .iter()
        .map(|(m, n)| Event::Lookup((*m).into(), (*n).into()))
        .collect::<Vec<_>>()
    );
}

#[test]
fn table_receipts_preserve_exact_legacy_events_and_reject_foreign_backend() {
    use super::super::recording::{Event, Gpu};
    use spark_runtime::gpu::DevicePtr;
    let gpu = Gpu::new();
    let table = build_ptr_table_from_qw(&vec![QuantizedWeight::null(); 288], &gpu).unwrap();
    let regions = table.owned_regions(&gpu, 288).unwrap();
    let expected: Vec<_> = regions
        .iter()
        .flat_map(|&(p, n)| [Event::Alloc(p, n), Event::H2d(p, n)])
        .collect();
    assert_eq!(gpu.trace(), expected);
    assert!(table.owned_regions(&Gpu::new(), 288).is_err());
    let borrowed = crate::layers::moe::ExpertPtrTable {
        allocation: None,
        packed_ptrs: DevicePtr(1),
        scale_ptrs: DevicePtr(2),
        scale2_vals: DevicePtr(3),
    };
    assert!(borrowed.owned_regions(&gpu, 288).is_err());
}
