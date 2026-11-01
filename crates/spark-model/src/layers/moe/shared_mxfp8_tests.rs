// SPDX-License-Identifier: AGPL-3.0-only
//! Host tests of `shared_mxfp8.rs` on the mock backend (every kernel name
//! resolves to one handle, so launches are told apart by their grids).

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Mxfp8SharedExpert;
use crate::weight_map::{Mxfp8Proj, Mxfp8Weight};

const H: usize = 2560;
/// A TP2 rank's slice of FP4-Local's 640-wide shared intermediate.
const I: usize = 320;

fn proj(gpu: &MockGpuBackend, n: usize, k: usize) -> Mxfp8Proj {
    Mxfp8Proj {
        w: Mxfp8Weight {
            data: gpu.alloc(n * k).unwrap(),
            scales: gpu.alloc(n * k / 32).unwrap(),
        },
        n: n as u32,
        k: k as u32,
    }
}

fn expert(gpu: &MockGpuBackend, split: bool) -> Mxfp8SharedExpert {
    let (gu, down) = (proj(gpu, 2 * I, H), proj(gpu, H, I));
    Mxfp8SharedExpert::new(gpu, gu, down, split, None).unwrap()
}

/// Decode: the SwiGLU tier writes `[rows, inter]` (a CTA per 8 outputs),
/// then the plain tier projects down (a CTA per 16 weight rows), each in
/// launches of up to 32 rows.
#[test]
fn decode_is_the_swiglu_tier_then_the_down_tier() {
    let gpu = MockGpuBackend::new();
    let mx = expert(&gpu, false);
    let from = gpu.launches_snapshot().len();
    let (input, act, out) = (DevicePtr(0x1000), DevicePtr(0x2000), DevicePtr(0x3000));
    mx.decode(&gpu, input, 40, act, out, 0).unwrap();
    let grids: Vec<[u32; 3]> = gpu.launches_snapshot()[from..]
        .iter()
        .map(|l| l.grid)
        .collect();
    let (swiglu, down) = ([(I / 8) as u32, 1, 1], [(H / 16) as u32, 1, 1]);
    assert_eq!(grids, [swiglu, swiglu, down, down]);
}

/// Without W8A8 operands a prefill chunk runs the decode tiers (the same
/// bytes), never a GEMM.
#[test]
fn prefill_without_w8a8_runs_the_decode_tiers() {
    let gpu = MockGpuBackend::new();
    let mx = expert(&gpu, false);
    let from = gpu.launches_snapshot().len();
    let act = crate::layers::ops::MxAct {
        q: DevicePtr(0x4000),
        sf: DevicePtr(0x5000),
    };
    let bufs = [DevicePtr(0x2000), DevicePtr(0x6000)];
    mx.prefill(&gpu, DevicePtr(0x1000), 8, act, bufs, DevicePtr(0x3000), 0)
        .unwrap();
    let grids: Vec<[u32; 3]> = gpu.launches_snapshot()[from..]
        .iter()
        .map(|l| l.grid)
        .collect();
    assert_eq!(grids, [[(I / 8) as u32, 1, 1], [(H / 16) as u32, 1, 1]]);
}

/// The twins must pair: `[2 * inter, hidden]` gate/up against `[hidden,
/// inter]` down, inter whole 32-value blocks.
#[test]
fn mismatched_twins_are_refused() {
    let gpu = MockGpuBackend::new();
    for (gu, down) in [
        (proj(&gpu, 2 * I, H), proj(&gpu, H, 2 * I)),
        (proj(&gpu, 2 * I, H + 32), proj(&gpu, H, I)),
        (proj(&gpu, 2 * 48, H), proj(&gpu, H, 48)),
    ] {
        assert!(Mxfp8SharedExpert::new(&gpu, gu, down, false, None).is_err());
    }
}
