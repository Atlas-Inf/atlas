// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_QWEN4EXP_MXFP8_DECODE=1`: `mxfp8_proj`'s 1..=32-row passes on the
//! `mxfp8_gemv_tc{8,16,32}` tiers. Host tests pin the dispatch; the GPU test
//! proves every row is the 1-row launch's bytes at any m, the contract
//! `mxfp8_proj_exact_rows` gives decode and verify.
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::{Mxfp8ProjKernels, Mxfp8Scratch, mxfp8_decode_tc_requested, mxfp8_proj_exact_rows};
use crate::weight_map::{Mxfp8Proj, Mxfp8Weight};

// The MoE tests load the same recording backend under their own module.
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../moe/gate_up_btile_test_gpu.rs"]
mod recording;

const GEMV: [KernelHandle; 4] = [
    KernelHandle(1),
    KernelHandle(2),
    KernelHandle(3),
    KernelHandle(4),
];
const TC: [KernelHandle; 3] = [KernelHandle(8), KernelHandle(16), KernelHandle(32)];

fn kernels(tc: [KernelHandle; 3]) -> Mxfp8ProjKernels {
    Mxfp8ProjKernels {
        gemv: GEMV[0],
        batchm: GEMV[1],
        dequant: GEMV[2],
        rows32: GEMV[3],
        tc,
    }
}

fn proj(n: u32, k: u32) -> Mxfp8Proj {
    Mxfp8Proj {
        w: Mxfp8Weight {
            data: DevicePtr(0x200),
            scales: DevicePtr(0x300),
        },
        n,
        k,
    }
}

#[test]
fn only_one_requests_the_tensor_core_tiers() {
    assert!(mxfp8_decode_tc_requested(Some("1")));
    for v in [None, Some("0"), Some(""), Some("true"), Some(" 1")] {
        assert!(!mxfp8_decode_tc_requested(v), "{v:?}");
    }
}

#[test]
fn tc_tiers_serve_every_row_count_in_32_row_launches() {
    let p = proj(8192, 2560);
    // (m, tier handle per launch)
    for (m, want) in [
        (1usize, vec![8u64]),
        (8, vec![8]),
        (9, vec![16]),
        (16, vec![16]),
        (17, vec![32]),
        (32, vec![32]),
        (33, vec![32, 8]),
        (64, vec![32, 32]),
        (65, vec![32, 32, 8]),
    ] {
        let gpu = MockGpuBackend::new();
        mxfp8_proj_exact_rows(
            &gpu,
            &kernels(TC),
            &p,
            DevicePtr(0x100),
            DevicePtr(0x400),
            m,
            0,
        )
        .unwrap();
        let launches = gpu.launches_snapshot();
        let got: Vec<u64> = launches.iter().map(|l| l.func).collect();
        assert_eq!(got, want, "m={m}");
        for l in &launches {
            // A CTA is 16 weight rows: ceil(n / 16).
            assert_eq!(l.grid, [8192 / 16, 1, 1], "m={m}");
            assert_eq!(l.block, [256, 1, 1], "m={m}");
        }
    }
}

#[test]
fn without_tc_tiers_the_gemv_tiers_serve_as_before() {
    let p = proj(2560, 3072);
    for m in [1usize, 2, 16, 17, 32] {
        let gpu = MockGpuBackend::new();
        mxfp8_proj_exact_rows(
            &gpu,
            &kernels([KernelHandle(0); 3]),
            &p,
            DevicePtr(0x100),
            DevicePtr(0x400),
            m,
            0,
        )
        .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1, "m={m}");
        assert!(GEMV.iter().any(|h| h.0 == launches[0].func), "m={m}");
        assert!(!TC.iter().any(|h| h.0 == launches[0].func), "m={m}");
    }
}

#[test]
fn tc_tiers_pass_the_kernel_abi_in_order() {
    use recording::{Arg, Event};
    let gpu = recording::Gpu::new();
    let k = 3072u32;
    let n = 2560u32;
    mxfp8_proj_exact_rows(
        &gpu,
        &kernels(TC),
        &proj(n, k),
        DevicePtr(0x100),
        DevicePtr(0x400),
        40,
        0,
    )
    .unwrap();
    let ptr = |p| Arg::Ptr(DevicePtr(p));
    let u32 = |v: u32| Arg::Bytes(v.to_le_bytes().to_vec());
    // (A, W, S, C, M, N, K, out_stride); the second launch starts at row 32.
    let abi = |a: u64, c: u64, m: u32| {
        vec![
            ptr(a),
            ptr(0x200),
            ptr(0x300),
            ptr(c),
            u32(m),
            u32(n),
            u32(k),
            u32(n),
        ]
    };
    assert_eq!(
        gpu.trace(),
        [
            Event::Launch(32, [n / 16, 1, 1], [256, 1, 1], 0, 0, abi(0x100, 0x400, 32)),
            Event::Launch(
                8,
                [n / 16, 1, 1],
                [256, 1, 1],
                0,
                0,
                abi(
                    0x100 + 32 * u64::from(k) * 2,
                    0x400 + 32 * u64::from(n) * 2,
                    8
                )
            ),
        ]
    );
}

#[test]
fn a_prefill_scratch_is_not_needed_by_the_tc_tiers() {
    // The non-ROWINV prefill split serves m <= 32 on the same tiers.
    let gpu = MockGpuBackend::new();
    let none = Mxfp8Scratch {
        ptr: DevicePtr(0),
        bytes: 0,
    };
    super::mxfp8_proj(
        &gpu,
        &kernels(TC),
        &proj(1280, 2560),
        DevicePtr(0x100),
        DevicePtr(0x400),
        5,
        none,
        0,
    )
    .unwrap();
    assert_eq!(gpu.launches_snapshot()[0].func, 8);
}

/// Every row of an `m`-row tc launch equals the same row launched alone, at
/// the real FP4-Local TP2 shapes, on hostile and realistic bytes, for every
/// m in 1..=65 (two full launches and a tail).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU"]
fn tc_rows_match_the_one_row_launch_bit_for_bit() {
    use spark_runtime::gpu::GpuBackend;
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next nvfp4 target set");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let mut ks = Mxfp8ProjKernels::resolve(g).expect("mxfp8 kernels");
    ks.tc = Mxfp8ProjKernels::tc_tiers(g).expect("mxfp8_gemv tc tiers");
    let mut seed = 0x5EED_u64;
    let mut byte = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u8
    };
    let upload = |bytes: &[u8]| {
        let p = g.alloc(bytes.len()).unwrap();
        g.copy_h2d(bytes, p).unwrap();
        p
    };
    let down = |p: DevicePtr, n: usize| {
        let mut b = vec![0u8; n];
        g.copy_d2h(p, &mut b).unwrap();
        b
    };
    const M: usize = 65;
    for (n, k, label) in [
        (8192u32, 2560u32, "gdn-qkvz"),
        (2560, 3072, "out/o_proj"),
        (1280, 2560, "shared gate|up"),
        (2560, 640, "shared down"),
    ] {
        for hostile in [false, true] {
            let w: Vec<u8> = (0..n as usize * k as usize)
                .map(|_| if hostile { byte() } else { byte() & 0x7E })
                .collect();
            let s: Vec<u8> = (0..n as usize * (k / 32) as usize)
                .map(|_| if hostile { byte() } else { 115 + byte() % 13 })
                .collect();
            let a: Vec<u8> = (0..M * k as usize)
                .flat_map(|_| {
                    let v = if hostile {
                        half::bf16::from_bits(u16::from(byte()) | (u16::from(byte()) << 8))
                    } else {
                        half::bf16::from_f32(f32::from(byte()) / 128.0 - 1.0)
                    };
                    v.to_bits().to_le_bytes()
                })
                .collect();
            let p = Mxfp8Proj {
                w: Mxfp8Weight {
                    data: upload(&w),
                    scales: upload(&s),
                },
                n,
                k,
            };
            let act = upload(&a);
            let row = n as usize * 2;
            let want = g.alloc(M * row).unwrap();
            for r in 0..M {
                let a_r = act.offset(r * k as usize * 2);
                mxfp8_proj_exact_rows(g, &ks, &p, a_r, want.offset(r * row), 1, stream).unwrap();
            }
            let want = down(want, M * row);
            let got = g.alloc(M * row).unwrap();
            for m in 1..=M {
                mxfp8_proj_exact_rows(g, &ks, &p, act, got, m, stream).unwrap();
                assert!(
                    down(got, m * row) == want[..m * row],
                    "{label} hostile={hostile} m={m}: rows differ from the 1-row launch"
                );
            }
        }
    }
}
