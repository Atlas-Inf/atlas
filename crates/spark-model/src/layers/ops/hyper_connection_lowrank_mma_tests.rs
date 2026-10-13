// SPDX-License-Identifier: AGPL-3.0-only

//! GPU checks of the tensor-core mHC collapse (`ATLAS_QWEN4EXP_HC_MMA`,
//! [`hc_mma_launch`]) on the real site shape (hidden 2560, hc 4, rank 320)
//! with synthetic weights:
//!
//! * contract (b), row invariance: every row of a batched launch (T up to 32,
//!   at two offsets, with and without the injection rows) is byte-equal to
//!   that row launched alone;
//! * accuracy: `low`, `inj` and `y` against an FP64 host reference of the same
//!   `normed`.
//!
//! `#[ignore]` per repo convention (CI is CPU-only):
//! ```text
//! cargo test -p spark-model --release hc_mma -- --ignored --nocapture
//! ```
//! scripts/dev/qwen4exp_hc_mma_bench.cu runs the exhaustive version (every T,
//! three offsets, three scales) and the timings.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::hyper_connection_lowrank_mma::hc_mma_launch;
use crate::layers::qwen3_attention::HcLowRank;

const H: usize = 2560;
const HC: usize = 4;
const RANK: usize = 320;
const K: usize = HC * H;
const ROWS: usize = 64;

struct Lcg(u64);
impl Lcg {
    /// Roughly N(0, sd): a sum of four uniforms, enough for these checks.
    fn normal(&mut self, sd: f32) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s += (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        }
        s * sd * 1.732
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16 // round to nearest even (finite x)
}
fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn upload<T: Copy>(g: &dyn GpuBackend, v: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(v);
    let p = g.alloc(bytes).unwrap();
    // SAFETY: plain-old-data slice viewed as bytes.
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, bytes) };
    g.copy_h2d_async(raw, p, g.default_stream()).unwrap();
    p
}
fn download(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    g.synchronize(g.default_stream()).unwrap();
    let mut raw = vec![0u8; bytes];
    g.copy_d2h(p, &mut raw).unwrap();
    raw
}

struct Outs {
    low: DevicePtr,
    inj: DevicePtr,
    y: DevicePtr,
}
impl Outs {
    fn new(g: &dyn GpuBackend) -> Self {
        Self {
            low: g.alloc(ROWS * RANK * 4).unwrap(),
            inj: g.alloc(ROWS * HC * 4).unwrap(),
            y: g.alloc(ROWS * H * 2).unwrap(),
        }
    }
    /// (low, inj, y) bytes of the first `t` rows.
    fn get(&self, g: &dyn GpuBackend, t: usize) -> [Vec<u8>; 3] {
        [
            download(g, self.low, t * RANK * 4),
            download(g, self.inj, t * HC * 4),
            download(g, self.y, t * H * 2),
        ]
    }
}

#[test]
#[ignore]
fn hc_mma_rows_are_invariant_and_accurate() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 is not in this build");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let mut rng = Lcg(7);
    let mut bf =
        |n: usize, sd: f32| -> Vec<u16> { (0..n).map(|_| bf16_bits(rng.normal(sd))).collect() };
    let down = bf(RANK * K, 0.02);
    let up = bf(RANK * K, 0.05);
    let inject = bf(HC * K, 0.02);
    let normed: Vec<f32> = (0..ROWS * K).map(|_| rng.normal(1.0)).collect();
    let w = HcLowRank {
        norm_w: DevicePtr::NULL, // the stage is not run here
        down_w: upload(g, &down),
        up_w: upload(g, &up),
        inject_w: upload(g, &inject),
        rank: RANK,
    };
    let nx = upload(g, &normed);
    let (solo, got) = (Outs::new(g), Outs::new(g));

    for with_inj in [true, false] {
        let launch = |o: &Outs, row0: usize, t: u32| {
            let ok = hc_mma_launch(
                g,
                &w,
                nx.offset(row0 * K * 4),
                o.low,
                o.y,
                if with_inj { o.inj } else { DevicePtr::NULL },
                t,
                H as u32,
                HC as u32,
                stream,
            )
            .unwrap();
            assert!(ok, "qwen4exp_hc_mma kernels missing from this build");
        };
        // Every row alone.
        let mut alone: Vec<[Vec<u8>; 3]> = Vec::new();
        for r in 0..ROWS {
            launch(&solo, r, 1);
            alone.push(solo.get(g, 1));
        }
        for t in [2usize, 5, 8, 9, 16, 17, 24, 25, 31, 32] {
            for row0 in [0, ROWS - t] {
                launch(&got, row0, t as u32);
                let batch = got.get(g, t);
                for i in 0..t {
                    for (k, per) in [RANK * 4, HC * 4, H * 2].into_iter().enumerate() {
                        if k == 1 && !with_inj {
                            continue;
                        }
                        assert_eq!(
                            &batch[k][i * per..(i + 1) * per],
                            &alone[row0 + i][k][..],
                            "row {} of a T={t} batch at row {row0} (inject {with_inj}): output {k} differs from the row alone",
                            row0 + i
                        );
                    }
                }
            }
        }
        println!("  row invariance ok (inject {with_inj})");

        // FP64 reference on the T = 32 batch at row 0.
        launch(&got, 0, 32);
        let [low_b, inj_b, y_b] = got.get(g, 32);
        let f32s = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let (low_g, inj_g) = (f32s(&low_b), f32s(&inj_b));
        let (mut e_low, mut e_inj, mut e_y) = (0f64, 0f64, 0f64);
        for t in 0..32 {
            let n = &normed[t * K..(t + 1) * K];
            let dot = |row: &[u16]| -> f64 {
                row.iter()
                    .zip(n)
                    .map(|(&a, &b)| bf16_val(a) as f64 * b as f64)
                    .sum()
            };
            let lo: Vec<f64> = (0..RANK)
                .map(|r| {
                    let v = dot(&down[r * K..(r + 1) * K]) / HC as f64;
                    v / (1.0 + (-v).exp())
                })
                .collect();
            for r in 0..RANK {
                e_low = e_low.max((lo[r] - low_g[t * RANK + r] as f64).abs());
            }
            if with_inj {
                for s in 0..HC {
                    let v = dot(&inject[s * K..(s + 1) * K]) / HC as f64;
                    e_inj = e_inj.max((2.0 / (1.0 + (-v).exp()) - inj_g[t * HC + s] as f64).abs());
                }
            }
            for d in 0..H {
                let (mut mix, mut scale) = (0f64, 0f64);
                for s in 0..HC {
                    let u: f64 = (0..RANK)
                        .map(|r| bf16_val(up[r * K + s * H + d]) as f64 * lo[r])
                        .sum();
                    let p = n[s * H + d] as f64 / (1.0 + (-u).exp());
                    mix += p;
                    scale += p.abs();
                }
                mix /= HC as f64;
                let i = (t * H + d) * 2;
                let got_y = bf16_val(u16::from_le_bytes([y_b[i], y_b[i + 1]])) as f64;
                // In bf16 ulps of the larger of |y| and the mean |term|: where
                // the stream mean cancels, the error scale is the terms'.
                let mag = mix.abs().max(scale / HC as f64).max(1e-30);
                e_y = e_y.max((got_y - mix).abs() / (mag.log2().floor().exp2() / 128.0));
            }
        }
        println!(
            "  vs fp64 (inject {with_inj}): low max|err| {e_low:.3e}  inj {e_inj:.3e}  y {e_y:.2} ulp"
        );
        assert!(e_low < 5e-5, "low error {e_low:.3e}");
        assert!(e_inj < 5e-5, "inj error {e_inj:.3e}");
        // Rounding to bf16 is 0.5 of these ulps; the FP32 kernels' worst
        // against the correctly rounded value is ~1.8 on the bench's data.
        assert!(e_y <= 2.0, "y error {e_y:.2} bf16 ulp");
    }
}
