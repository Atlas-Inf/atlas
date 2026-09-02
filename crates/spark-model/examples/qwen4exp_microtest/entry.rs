// SPDX-License-Identifier: AGPL-3.0-only

//! The trunk entry (`q4e_hc_expand`), sentinel-checked.

use super::*;

/// `q4e_hc_expand` runs once after the embedding lookup, not per layer. It is
/// three lines of kernel, and it is here because getting it wrong is silent:
/// the streams must start IDENTICAL. Zero-initialising them instead makes the
/// first hyper-connection collapse read a zero mean, and the model does not
/// recover -- it still emits tokens.
pub(super) fn hc_expand_entry(g: &dyn GpuBackend) -> Result<()> {
    const HIDDEN: usize = 2560;
    const HC: usize = 4;
    const TOKENS: usize = 3;

    let mut state = 0x51DE_4A17_C0DE_0001u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let hidden: Vec<f32> = (0..TOKENS * HIDDEN).map(|_| next()).collect();

    let d_hidden = up_bf16(g, &hidden)?;
    // Pre-fill the destination with a sentinel so an unwritten slot is visible
    // rather than reading as a plausible zero.
    let sentinel = vec![-7.0f32; TOKENS * HC * HIDDEN];
    let d_streams = up_bf16(g, &sentinel)?;

    let kernel = g.kernel("qwen4exp_hc", "q4e_hc_expand")?;
    KernelLaunch::new(g, kernel)
        .grid([TOKENS as u32, HC as u32, 1])
        .block([HIDDEN.min(1024) as u32, 1, 1])
        .arg_ptr(d_hidden)
        .arg_ptr(d_streams)
        .arg_u32(HIDDEN as u32)
        .arg_u32(HC as u32)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_streams, TOKENS * HC * HIDDEN)?;

    let round = |v: &f32| bf16::from_f32(*v).to_f32();
    let want: Vec<f32> = (0..TOKENS)
        .flat_map(|t| {
            let row: Vec<f32> = hidden[t * HIDDEN..(t + 1) * HIDDEN]
                .iter()
                .map(round)
                .collect();
            std::iter::repeat_n(row, HC).flatten()
        })
        .collect();

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("hc expand: max|diff| {worst:.3e} over {} values", got.len());

    // The sentinel must be gone everywhere -- a kernel that wrote only stream 0
    // would leave -7.0 in the other three and still match on the part it wrote.
    let untouched = got.iter().filter(|v| (**v + 7.0).abs() < 1e-6).count();
    println!("slots left at the sentinel: {untouched} (must be 0)");
    anyhow::ensure!(untouched == 0, "q4e_hc_expand did not write every stream");

    anyhow::ensure!(worst < 1e-6, "q4e_hc_expand does not tile the hidden state");
    println!("HC EXPAND MATCHES THE ORACLE\n");
    hc_sandwich_roundtrip(g)
}

// ── The sandwich, composed: collapse -> block -> scatter ───────────────────
