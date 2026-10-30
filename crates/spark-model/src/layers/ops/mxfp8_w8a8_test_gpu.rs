// SPDX-License-Identifier: AGPL-3.0-only
//! GPU tests of the W8A8 prefill at FP4-Local's TP2 shapes: the activation
//! quant and the scale relayout write the host references' bytes, every row
//! of a pass equals the same row in a pass of another height, and the
//! output is within the activation rounding of the BF16 expansion's.
#![cfg(feature = "cuda")]

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::mxfp8_act::tests::{bf16, quant_rows, tile_scales};
use super::mxfp8_proj::mxfp8_proj_rowinv;
use super::{
    MX_QUANT_KERNEL, MX_TILE_KERNEL, MXFP8_ACT_MODULE, MxAct, MxW8a8, Mxfp8ProjKernels,
    Mxfp8Scratch, mx_lt_scale_bytes, mxfp8_lt_tile_scales, mxfp8_quant_act,
};
use crate::weight_map::{Mxfp8Proj, Mxfp8Weight};

const SHAPES: &[(usize, usize, &str)] = &[
    (8192, 2560, "gdn-qkvz"),
    (2560, 3072, "out/o_proj"),
    (2560, 640, "shared down"),
];

struct Rng(u64);
impl Rng {
    fn u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

fn gpu() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next nvfp4 target set");
    spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len().max(1)).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b).unwrap();
    b
}

/// Realistic MXFP8 weight bytes (finite E4M3, scales near 2^-10) and BF16
/// activations in [-1, 1).
fn operands(rng: &mut Rng, n: usize, k: usize, m: usize) -> (Vec<u8>, Vec<u8>, Vec<u16>) {
    let w = (0..n * k).map(|_| rng.u8() & 0x7E).collect();
    let s = (0..n * k / 32).map(|_| 115 + rng.u8() % 13).collect();
    let a = (0..m * k)
        .map(|_| half::bf16::from_f32(f32::from(rng.u8()) / 128.0 - 1.0).to_bits())
        .collect();
    (w, s, a)
}

#[test]
#[ignore = "GPU"]
fn quant_and_relayout_write_the_host_reference_bytes() {
    let g = gpu();
    let g: &dyn GpuBackend = &g;
    let s = g.default_stream();
    let quant = g.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL).unwrap();
    let tile = g.kernel(MXFP8_ACT_MODULE, MX_TILE_KERNEL).unwrap();
    let mut rng = Rng(0xAC7);
    for &(n, k, label) in SHAPES {
        for m in [1usize, 127, 129, 300] {
            let (_, sc, a) = operands(&mut rng, n, k, m);
            let ab: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
            let (q, sf) = (
                g.alloc(m * k).unwrap(),
                g.alloc(mx_lt_scale_bytes(m, k)).unwrap(),
            );
            g.memset(sf, 0, mx_lt_scale_bytes(m, k)).unwrap();
            let x = upload(g, &ab);
            mxfp8_quant_act(g, quant, x, k as u32, q, sf, m as u32, k as u32, s).unwrap();
            let (hq, hsf) = quant_rows(&a, m, k);
            assert!(down(g, q, m * k) == hq, "{label} m={m}: E4M3 bytes");
            assert!(down(g, sf, hsf.len()) == hsf, "{label} m={m}: tiled scales");
            let wsf = g.alloc(mx_lt_scale_bytes(n, k)).unwrap();
            let src = upload(g, &sc);
            mxfp8_lt_tile_scales(g, tile, src, (k / 32) as u32, wsf, n as u32, k as u32, s)
                .unwrap();
            let want = tile_scales(&sc, n, k / 32, k / 32);
            assert!(
                down(g, wsf, want.len()) == want,
                "{label}: weight scale relayout"
            );
        }
    }
}

#[test]
#[ignore = "GPU"]
fn w8a8_rows_do_not_depend_on_the_pass_and_track_the_bf16_expansion() {
    let g = gpu();
    let g: &dyn GpuBackend = &g;
    let s = g.default_stream();
    let quant = g.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL).unwrap();
    let kernels = Mxfp8ProjKernels::resolve(g).unwrap();
    let mut rng = Rng(0xF00D);
    const TALL: usize = 2048;
    for &(n, k, label) in SHAPES {
        let (w, sc, a) = operands(&mut rng, n, k, TALL);
        let proj = Mxfp8Proj {
            w: Mxfp8Weight {
                data: upload(g, &w),
                scales: upload(g, &sc),
            },
            n: n as u32,
            k: k as u32,
        };
        let op = MxW8a8::from_twin(g, &proj.w, [n, k], TALL, s).unwrap();
        let ab: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
        let x = upload(g, &ab);
        let act = MxAct {
            q: g.alloc(TALL * k).unwrap(),
            sf: g.alloc(mx_lt_scale_bytes(TALL, k)).unwrap(),
        };
        let run = |m: usize, out: DevicePtr| {
            mxfp8_quant_act(g, quant, x, k as u32, act.q, act.sf, m as u32, k as u32, s).unwrap();
            op.gemm(act, out, m, s).unwrap();
        };
        let row = n * 2;
        let tall = g.alloc(TALL * row).unwrap();
        run(TALL, tall);
        let tall = down(g, tall, TALL * row);
        for m in [1usize, 31, 128, 300, 1000] {
            let out = g.alloc(m * row).unwrap();
            run(m, out);
            assert!(
                down(g, out, m * row) == tall[..m * row],
                "{label} m={m}: rows differ from the {TALL}-row pass"
            );
        }
        // Against the exact BF16 expansion + GEMM: only the activations'
        // E4M3 rounding separates them (~2-3e-2 relative).
        let scratch = Mxfp8Scratch {
            ptr: g.alloc(n * k * 2).unwrap(),
            bytes: n * k * 2,
        };
        let refo = g.alloc(TALL * row).unwrap();
        mxfp8_proj_rowinv(g, &kernels, &proj, x, refo, TALL, scratch, s).unwrap();
        let (got, want) = (tall, down(g, refo, TALL * row));
        let val =
            |b: &[u8], i: usize| f64::from(bf16(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])));
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..TALL * n {
            num += (val(&got, i) - val(&want, i)).powi(2);
            den += val(&want, i).powi(2);
        }
        let rel = (num / den).sqrt();
        assert!(
            rel < 5e-2,
            "{label}: W8A8 is {rel:.3e} (rel L2) from the BF16 expansion"
        );
        op.free(g).unwrap();
    }
}
