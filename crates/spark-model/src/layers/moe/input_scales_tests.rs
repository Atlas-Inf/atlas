// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_GLM_FP4L_INPUT_SCALES`: the scale folding, and the kernels and
//! launch ABI of the per-expert SiLU quantization.
use super::fold_scales;

#[test]
fn gate_up_take_the_largest_scale_and_down_each_experts_own() {
    let (a1, a2, present) = fold_scales(&[
        [Some(0.5), Some(2.0), Some(3.0)],
        [None, None, None],
        [Some(4.0), Some(1.0), Some(0.25)],
    ])
    .unwrap()
    .unwrap();
    assert_eq!(a1, 4.0);
    // A non-local expert (no scales) keeps 1: no row reaches it.
    assert_eq!(a2, vec![3.0, 1.0, 0.25]);
    assert_eq!(present, 2);
}

#[test]
fn no_scales_keep_global_scale_one_and_partial_ones_are_refused() {
    assert!(fold_scales(&[[None, None, None]; 4]).unwrap().is_none());
    assert!(fold_scales(&[]).unwrap().is_none());
    for partial in [
        [Some(1.0), None, None],
        [Some(1.0), Some(1.0), None],
        [None, None, Some(1.0)],
    ] {
        assert!(
            fold_scales(&[[Some(1.0); 3], partial]).is_err(),
            "{partial:?}"
        );
    }
}

#[test]
fn by_expert_silu_quant_passes_the_offsets_scales_and_expert_count() {
    use crate::layers::moe::gate_up_repack::recording;
    use spark_runtime::gpu::{DevicePtr, KernelHandle};
    let gpu = recording::Gpu::new();
    let p = |v: u64| DevicePtr(v);
    crate::layers::ops::silu_mul_quant_nvfp4_by_expert(
        &gpu,
        KernelHandle(7),
        [p(1), p(2)],
        [p(3), p(4), DevicePtr::NULL],
        [40, 1024],
        [p(5), p(6)],
        256,
        0,
    )
    .unwrap();
    let launches: Vec<_> = gpu
        .trace()
        .into_iter()
        .filter_map(|e| match e {
            recording::Event::Launch(f, grid, block, _, _, args) => Some((f, grid, block, args)),
            _ => None,
        })
        .collect();
    assert_eq!(launches.len(), 1);
    let (f, grid, block, args) = &launches[0];
    assert_eq!((*f, *grid, *block), (7, [40, 1, 1], [128, 1, 1]));
    let u = |v: u32| recording::Arg::Bytes(v.to_le_bytes().to_vec());
    let ptr = |v: u64| recording::Arg::Ptr(DevicePtr(v));
    assert_eq!(
        args,
        &vec![
            ptr(1),
            ptr(2),
            ptr(3),
            ptr(4),
            ptr(0),
            u(40),
            u(1024),
            ptr(5),
            ptr(6),
            u(256)
        ]
    );
}

/// The kernel `build_input_scales` resolves is declared in the GLM target.
#[test]
fn the_by_expert_silu_quant_kernel_is_in_the_glm_target() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/gb10/glm-5.3-flash/nvfp4/moe_silu_mul.cu"),
    )
    .unwrap();
    assert!(src.contains("extern \"C\" __global__ void silu_mul_quant_nvfp4_by_expert("));
}
