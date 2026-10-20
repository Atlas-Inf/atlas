// SPDX-License-Identifier: AGPL-3.0-only

//! Accuracy of each verify kernel against an FP64 reference (GPU, ignored):
//! the share of outputs that are not the correctly rounded BF16 of the exact
//! dot product, the RMS error relative to the outputs' RMS, and for the MoE
//! router the rows whose top-8 expert set differs from the exact logits'.
//! Rows run in launches of the kernel's natural width.
//!
//!   ATLAS_W4A16_TC=1 cargo test --release -p spark-model --lib canonical_verify_accuracy -- --ignored --nocapture

use super::*;

struct Score {
    off: usize,
    total: usize,
    err2: f64,
    ref2: f64,
}

impl Score {
    fn new() -> Self {
        Self {
            off: 0,
            total: 0,
            err2: 0.0,
            ref2: 0.0,
        }
    }
    fn add(&mut self, got: u16, exact: f64) {
        assert!(
            from_bf16(got).is_finite() && exact.is_finite(),
            "nonfinite accuracy sample"
        );
        self.off += (got != bf16(exact as f32)) as usize;
        self.total += 1;
        self.err2 += (from_bf16(got) as f64 - exact).powi(2);
        self.ref2 += exact * exact;
    }
    fn line(&self) -> String {
        format!(
            "off-RNE {:.3}%  rel-RMS-err {:.3e}",
            100.0 * self.off as f64 / self.total as f64,
            (self.err2 / self.ref2).sqrt()
        )
    }
}

fn exact(rows: &[Vec<u16>], w: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows.len() * n];
    for (r, row) in rows.iter().enumerate() {
        let a: Vec<f64> = row.iter().map(|&h| from_bf16(h) as f64).collect();
        for j in 0..n {
            out[r * n + j] = a
                .iter()
                .zip(&w[j * k..(j + 1) * k])
                .map(|(x, y)| x * y)
                .sum();
        }
    }
    out
}

/// Every row of `rows` through `launch` in launches of `width` rows.
fn chunked(
    gpu: &dyn GpuBackend,
    rows: &[Vec<u16>],
    n: u32,
    width: usize,
    launch: impl Fn(DevicePtr, DevicePtr, u32) -> Result<()>,
) -> Result<Vec<u16>> {
    let mut out = vec![];
    for chunk in rows.chunks(width) {
        out.extend(run_rows(gpu, chunk, n, &launch)?);
    }
    Ok(out)
}

fn top8(logits: impl Iterator<Item = f64>) -> Vec<usize> {
    let mut idx: Vec<(usize, f64)> = logits.enumerate().collect();
    // Highest first; a tie goes to the lower expert id.
    idx.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut set: Vec<usize> = idx[..8].iter().map(|&(i, _)| i).collect();
    set.sort();
    set
}

#[test]
#[ignore = "requires a GB10, a glm-5.3-flash kernel build and ATLAS_W4A16_TC=1"]
fn canonical_verify_accuracy() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = gpu;
    crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu, true);
    let mut rng = Rng(0x5eed_1234_abcd_0003);

    // W4A16 (shared-expert gate shape): 48 rows.
    let (n, k) = (1024u32, 4096u32);
    let w = w4(gpu, &mut rng, n, k)?;
    let rows: Vec<Vec<u16>> = (0..48).map(|_| random_row(&mut rng, k)).collect();
    let want = exact(&rows, &w.dense, n as usize, k as usize);
    for (name, width) in [
        ("w4a16_gemv", 1),
        ("w4a16_gemv_batch2", 2),
        ("w4a16_gemv_batch3", 3),
        ("w4a16_gemv_tc8", 8),
        ("w4a16_gemv_tc8", 32),
        ("w4a16_gemv_tc16", 16),
        ("w4a16_gemv_tc32", 32),
    ] {
        let got = chunked(gpu, &rows, n, width, |a, c, m| {
            launch_w4(gpu, name, &w, a, c, m)
        })?;
        let mut score = Score::new();
        got.iter().zip(&want).for_each(|(&g, &e)| score.add(g, e));
        println!("ACCURACY w4a16 {n}x{k} {name:<22} {}", score.line());
    }

    // Router (288 x 4096): 512 rows, logits ~N(0, 1).
    let (n, k) = (288u32, 4096u32);
    let (gate, dense_w) = bf16_weight(gpu, &mut rng, n, k, 0.05)?;
    let rows: Vec<Vec<u16>> = (0..512).map(|_| random_row(&mut rng, k)).collect();
    let want = exact(&rows, &dense_w, n as usize, k as usize);
    for ((module, name), width) in [
        (("gemm", "dense_gemm_bf16_router"), 8),
        (("gemm", "dense_gemm_bf16_router_rows"), 8),
        (("gemm", "dense_gemm_bf16_router_m5"), 5),
        (("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"), 8),
        (("dense_gemv_bf16_batchm", "dense_gemv_bf16_tc8"), 8),
        (("dense_gemv_bf16_batchm", "dense_gemv_bf16_tc16"), 16),
        (("dense_gemv_bf16_batchm", "dense_gemv_bf16_tc32"), 32),
    ] {
        let mut sub = rows.clone();
        // The M5 kernel only runs exactly five rows.
        sub.truncate(rows.len() / width * width);
        let got = chunked(gpu, &sub, n, width, |a, c, m| {
            launch_dense(gpu, (module, name), &gate, a, c, m, n, k)
        })?;
        let mut score = Score::new();
        got.iter().zip(&want).for_each(|(&g, &e)| score.add(g, e));
        let flips = (0..sub.len())
            .filter(|&r| {
                let row = |v: usize| r * n as usize + v;
                top8((0..n as usize).map(|v| from_bf16(got[row(v)]) as f64))
                    != top8((0..n as usize).map(|v| want[row(v)]))
            })
            .count();
        println!(
            "ACCURACY router {name:<28} {}  top8-set-flips {flips}/{}",
            score.line(),
            sub.len()
        );
    }
    Ok(())
}
