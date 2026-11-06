// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a qwen3.6-35b-a3b
//! kernel build): `moe_bf16mma_prefill.cu` against an FP64 CPU reference of
//! the exact NVFP4 dequant, for experts of 0, 1, 9, 64, 65 and 135 rows (one,
//! two and three row chunks, partial token tiles), zero and subnormal block
//! scales, and every e2m1 code. Rows outside the experts' range stay
//! untouched, and a one-row grid that strides the chunks writes the same bits
//! as the full grid.
//!
//!   cargo test --release -p spark-model --lib bf16mma -- --ignored --nocapture

use super::*;
use crate::layers::bf16mma_test_ref::*;
use anyhow::ensure;
use spark_runtime::cuda_backend::AtlasCudaBackend;

const H: usize = 1024;
const I: usize = 128;
const COUNTS: [usize; 6] = [0, 1, 9, 64, 65, 135];
const TOKENS: usize = 140;
const SPARE_ROWS: usize = 3;
const POISON: u16 = 0xa5a5;

/// One projection of every expert in the transposed layout: `[K/2, N]`
/// packed (low nibble = even k) and `[K/16, N]` scales, plus `scale2`.
struct Proj {
    k: usize,
    n: usize,
    packed: Vec<Vec<u8>>,
    scales: Vec<Vec<u8>>,
    scale2: Vec<f32>,
}

impl Proj {
    fn new(rng: &mut Rng, k: usize, n: usize, scale2: f32) -> Self {
        let e = COUNTS.len();
        Self {
            k,
            n,
            packed: (0..e)
                .map(|_| (0..k / 2 * n).map(|_| rng.next() as u8).collect())
                .collect(),
            scales: (0..e)
                .map(|_| (0..k / 16 * n).map(|_| rng.scale()).collect())
                .collect(),
            scale2: (0..e).map(|_| scale2 * (1.0 + rng.unit().abs())).collect(),
        }
    }
    fn w(&self, e: usize, k: usize, n: usize) -> f64 {
        let b = self.packed[e][k / 2 * self.n + n];
        let code = if k & 1 == 0 { b & 15 } else { b >> 4 };
        e2m1(code) * e4m3(self.scales[e][k / 16 * self.n + n])
    }
    /// FP64 `x · W[:, n]` times scale2.
    fn dot(&self, e: usize, x: &[u16], n: usize) -> f64 {
        let s: f64 = (0..self.k)
            .map(|k| f32_of(x[k]) as f64 * self.w(e, k, n))
            .sum();
        s * self.scale2[e] as f64
    }
    fn upload(&self, gpu: &dyn GpuBackend) -> Result<Table> {
        let ptrs = |v: &Vec<Vec<u8>>| -> Result<Vec<u64>> {
            v.iter().map(|b| Ok(upload(gpu, b)?.0)).collect()
        };
        Ok([
            upload(gpu, &bytes_of(&ptrs(&self.packed)?))?,
            upload(gpu, &bytes_of(&ptrs(&self.scales)?))?,
            upload(gpu, &bytes_of(&self.scale2))?,
        ])
    }
}

#[test]
#[ignore = "needs a GB10 and a qwen3.6-35b-a3b kernel build"]
fn bf16mma_moe_matches_exact_dequant_reference() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let [k_gu, k_down] = resolve(&gpu);
    ensure!(
        k_gu.0 != 0 && k_down.0 != 0,
        "moe_bf16mma_prefill not in this build"
    );
    let stream = gpu.default_stream();
    let mut rng = Rng(0x0b16_f4a3);

    let (gate, up, down) = (
        Proj::new(&mut rng, H, I, 0.002),
        Proj::new(&mut rng, H, I, 0.002),
        Proj::new(&mut rng, I, H, 0.004),
    );
    let x: Vec<u16> = (0..TOKENS * H).map(|_| bf16(rng.unit() * 2.0)).collect();
    let mut offsets = vec![0i32];
    for c in COUNTS {
        offsets.push(offsets.last().unwrap() + c as i32);
    }
    let total = *offsets.last().unwrap() as usize;
    let sorted: Vec<i32> = (0..total)
        .map(|_| (rng.next() % TOKENS as u64) as i32)
        .collect();
    let tiles = COUNTS.iter().map(|c| c.div_ceil(64)).max().unwrap() as u32;

    let poison = vec![POISON; (total + SPARE_ROWS) * H];
    let (x_d, act, out) = (
        upload(&gpu, &bytes_of(&x))?,
        upload(&gpu, &bytes_of(&poison))?,
        upload(&gpu, &bytes_of(&poison))?,
    );
    let rows = |sorted_token_ids, grid_y| Rows {
        offsets: upload(&gpu, &bytes_of(&offsets)).expect("offsets"),
        sorted_token_ids,
        num_experts: COUNTS.len() as u32,
        grid_y,
    };
    let sorted_d = upload(&gpu, &bytes_of(&sorted))?;
    let tables = [gate.upload(&gpu)?, up.upload(&gpu)?, down.upload(&gpu)?];
    let gu_rows = rows(sorted_d, tiles);
    launch_gate_up(
        &gpu,
        k_gu,
        x_d,
        [tables[0], tables[1]],
        act,
        gu_rows,
        [H as u32, I as u32],
        stream,
    )?;
    let dn_rows = rows(DevicePtr::NULL, tiles);
    launch_down(
        &gpu,
        k_down,
        act,
        tables[2],
        out,
        dn_rows,
        [H as u32, I as u32],
        stream,
    )?;
    gpu.synchronize(stream)?;
    let act_h = read_u16(&gpu, act, (total + SPARE_ROWS) * I)?;
    let out_h = read_u16(&gpu, out, (total + SPARE_ROWS) * H)?;

    let (mut w_gu, mut w_dn) = (0.0f64, 0.0f64);
    for (e, win) in offsets.windows(2).enumerate() {
        for r in win[0] as usize..win[1] as usize {
            let xr = &x[sorted[r] as usize * H..][..H];
            let want: Vec<f64> = (0..I)
                .map(|n| {
                    // The SiLU reads BF16 gate/up, as the unfused chain does.
                    let g = f32_of(bf16(gate.dot(e, xr, n) as f32)) as f64;
                    let u = f32_of(bf16(up.dot(e, xr, n) as f32)) as f64;
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            w_gu = w_gu.max(worst(&act_h[r * I..][..I], &want, 2e-2, 2e-3));
            let ar = &act_h[r * I..][..I];
            let want: Vec<f64> = (0..H).map(|n| down.dot(e, ar, n)).collect();
            w_dn = w_dn.max(worst(&out_h[r * H..][..H], &want, 1e-2, 1e-3));
        }
    }
    println!("worst error / tolerance: gate_up+silu {w_gu:.3}, down {w_dn:.3}");
    ensure!(
        w_gu <= 1.0,
        "gate_up+silu off the reference ({w_gu:.3} x tolerance)"
    );
    ensure!(
        w_dn <= 1.0,
        "down off the reference ({w_dn:.3} x tolerance)"
    );
    ensure!(
        act_h[total * I..]
            .iter()
            .chain(&out_h[total * H..])
            .all(|&v| v == POISON),
        "rows past the experts' range were written"
    );

    // A one-row grid strides the chunks: the same bits.
    let (act1, out1) = (
        upload(&gpu, &bytes_of(&poison))?,
        upload(&gpu, &bytes_of(&poison))?,
    );
    launch_gate_up(
        &gpu,
        k_gu,
        x_d,
        [tables[0], tables[1]],
        act1,
        rows(sorted_d, 1),
        [H as u32, I as u32],
        stream,
    )?;
    launch_down(
        &gpu,
        k_down,
        act1,
        tables[2],
        out1,
        rows(DevicePtr::NULL, 1),
        [H as u32, I as u32],
        stream,
    )?;
    gpu.synchronize(stream)?;
    ensure!(
        read_u16(&gpu, act1, (total + SPARE_ROWS) * I)? == act_h,
        "strided gate_up differs"
    );
    ensure!(
        read_u16(&gpu, out1, (total + SPARE_ROWS) * H)? == out_h,
        "strided down differs"
    );
    Ok(())
}
