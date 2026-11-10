// SPDX-License-Identifier: AGPL-3.0-only
//! GPU check of the DFlash2 grouped dynamic causal conv (`dflash2_conv`)
//! through `Dflash2Conv::apply_input` / `finish` at the checkpoints' tap
//! counts: 4 (Ripple-65B's DFlash2, `base_kernel [2, 4, 5120]`) and 2
//! (Qwen3.8 / GLM-5.3).
//!
//! Host reference: SGLang 0.5.19 `srt/models/dflash.py` `_grouped_conv` /
//! `DFlashGroupedConv` — the projection row is `[2 (input/output), taps,
//! num_groups]`, `base_kernel[side]` is `[taps, hidden]`, and
//!   y[t, c] = sum_{j < taps, j <= t} (base[side, j, c] + delta[t, side, j, c / group]) * x[t - j, c]
//! with `t` the row's position in the block. The replica sums in f32, tap 0
//! first, as the kernel does (built `--fmad=false`), so it must match the
//! kernel bit for bit; at 2 taps that is the two-term expression the
//! pre-generic kernel computed.
//!
//! Run on the GB10 (all tests are `#[ignore]`d), with a real kernel build:
//!   cargo test --release -p spark-model --test dflash2_conv_taps -- --ignored --nocapture
#[path = "arm2_common/support.rs"]
mod support;

use anyhow::{Result, ensure};
use spark_model::layers::dflash_head::Dflash2Conv;
use spark_model::weight_map::DenseWeight;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use support::{Rng, bf16_bits_to_f32, f32_to_bf16_bits, rd_u16, setup, up_u16};

struct Shape {
    rows: usize,
    hidden: usize,
    group: usize,
    taps: usize,
}

/// One side's host conv over `rows` block rows.
fn host_conv(x: &[u16], delta: &[u16], base: &[u16], s: &Shape, side: usize) -> Vec<u16> {
    let groups = s.hidden / s.group;
    let f = |v: u16| bf16_bits_to_f32(v);
    let coef = |t: usize, j: usize, c: usize| {
        f(base[(side * s.taps + j) * s.hidden + c])
            + f(delta[t * 2 * s.taps * groups + (side * s.taps + j) * groups + c / s.group])
    };
    let mut y = vec![0u16; s.rows * s.hidden];
    for t in 0..s.rows {
        for c in 0..s.hidden {
            let mut v = coef(t, 0, c) * f(x[t * s.hidden + c]);
            for j in 1..s.taps.min(t + 1) {
                v += coef(t, j, c) * f(x[(t - j) * s.hidden + c]);
            }
            y[t * s.hidden + c] = f32_to_bf16_bits(v);
        }
    }
    y
}

fn run(s: &Shape, seed: u64) -> Result<()> {
    let (gpu, stream) = setup()?;
    let kernel = gpu.kernel("dflash2_conv", "dflash2_grouped_dynamic_causal_conv")?;
    let groups = s.hidden / s.group;
    let mut rng = Rng(seed);
    let mut draw = |n: usize, scale: f32| -> Vec<u16> {
        (0..n)
            .map(|_| f32_to_bf16_bits((rng.unit() * 2.0 - 1.0) * scale))
            .collect()
    };
    // Base kernels near SGLang's init (tap 0 = 1, others 0) plus noise.
    let mut base = draw(2 * s.taps * s.hidden, 0.25);
    for side in 0..2 {
        for c in 0..s.hidden {
            let at = side * s.taps * s.hidden + c;
            base[at] = f32_to_bf16_bits(1.0 + bf16_bits_to_f32(base[at]));
        }
    }
    let delta = draw(s.rows * 2 * s.taps * groups, 0.5);
    let x = draw(s.rows * s.hidden, 2.0);
    let sub = draw(s.rows * s.hidden, 2.0);

    let d_base = up_u16(&gpu, &base)?;
    let d_delta = up_u16(&gpu, &delta)?;
    let d_x = up_u16(&gpu, &x)?;
    let d_sub = up_u16(&gpu, &sub)?;
    let bytes = s.rows * s.hidden * 2;
    let (d_in_out, d_fin_out) = (gpu.alloc(bytes)?, gpu.alloc(bytes)?);
    let conv = Dflash2Conv::new(
        DenseWeight { weight: d_base },
        DenseWeight {
            weight: DevicePtr::NULL,
        },
        s.hidden,
        s.group,
        s.taps,
    );
    assert_eq!(conv.delta_row_elems(), 2 * s.taps * groups);
    let rows = s.rows as u32;
    let out_delta = conv.apply_input(&gpu, Some(kernel), d_x, d_delta, d_in_out, rows, stream)?;
    conv.finish(
        &gpu,
        Some(kernel),
        d_sub,
        out_delta,
        d_fin_out,
        rows,
        stream,
    )?;
    gpu.synchronize(stream)?;

    for (side, input, out) in [(0, &x, d_in_out), (1, &sub, d_fin_out)] {
        let got = rd_u16(&gpu, out, s.rows * s.hidden)?;
        let want = host_conv(input, &delta, &base, s, side);
        let mismatches = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        eprintln!(
            "taps {} rows {} side {side}: {mismatches} of {} values differ",
            s.taps,
            s.rows,
            got.len()
        );
        ensure!(
            mismatches == 0,
            "side {side}: kernel != SGLang-math replica"
        );
    }
    for p in [d_base, d_delta, d_x, d_sub, d_in_out, d_fin_out] {
        gpu.free(p)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn conv_ripple_four_taps_gamma32() -> Result<()> {
    run(
        &Shape {
            rows: 32,
            hidden: 5120,
            group: 16,
            taps: 4,
        },
        0xC04A_0001,
    )
}

#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn conv_ripple_four_taps_gamma16() -> Result<()> {
    run(
        &Shape {
            rows: 16,
            hidden: 5120,
            group: 16,
            taps: 4,
        },
        0xC04A_0002,
    )
}

#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn conv_qwen38_two_taps_gamma8() -> Result<()> {
    run(
        &Shape {
            rows: 8,
            hidden: 5120,
            group: 16,
            taps: 2,
        },
        0xC04A_0003,
    )
}
