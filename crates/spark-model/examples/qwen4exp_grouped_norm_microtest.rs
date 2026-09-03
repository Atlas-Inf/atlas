// SPDX-License-Identifier: AGPL-3.0-only

//! Check the `rms_norm_grouped` kernel against the CPU oracle.
//!
//! ```text
//! cargo run --release -p spark-model --example qwen4exp_grouped_norm_microtest
//! ```
//!
//! The oracle (`atlas_core::qwen4exp_reference::grouped_rms_norm`) is itself
//! checked against HuggingFace at real weights, so agreement here chains the
//! kernel to the reference implementation rather than to another guess.
//!
//! Grouping is the whole point: qwen4_exp's residual stream is `hc_count`
//! streams of `hidden_size` concatenated and each normalises over ITSELF. A
//! single reduction over the full row is a different function that still
//! produces fluent output, so this also asserts the two disagree — a kernel
//! that silently ignored `group_size` would otherwise look correct.

use anyhow::Result;
use atlas_core::qwen4exp_reference::grouped_rms_norm;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const HIDDEN: usize = 2560; // published hidden_size
const GROUPS: usize = 4; // hc_count
const TOKENS: usize = 3;
const EPS: f32 = 1e-6;

fn up_bf16(g: &dyn GpuBackend, values: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    let ptr = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

fn down_bf16(g: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; len * 2];
    g.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let wide = HIDDEN * GROUPS;

    // Deterministic inputs; the streams are given different scales so a
    // whole-row reduction cannot coincidentally agree with a per-group one.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0
    };
    let input: Vec<f32> = (0..TOKENS * wide)
        .map(|i| next() * (1.0 + (i / HIDDEN % GROUPS) as f32))
        .collect();
    let weight: Vec<f32> = (0..wide).map(|_| next() * 0.1).collect();

    // The kernel reads BF16. Round the oracle's inputs the same way, so the
    // comparison measures the kernel rather than the upload -- otherwise BF16's
    // ~2^-8 input rounding dominates and the test says nothing.
    let round = |v: &f32| bf16::from_f32(*v).to_f32();
    let input_bf16: Vec<f32> = input.iter().map(round).collect();
    let weight_bf16: Vec<f32> = weight.iter().map(round).collect();

    let d_in = up_bf16(g, &input)?;
    let d_w = up_bf16(g, &weight)?;
    let d_out = g.alloc(TOKENS * wide * 2)?;

    let kernel = g.kernel("norm", "rms_norm_grouped")?;
    KernelLaunch::new(g, kernel)
        .grid([TOKENS as u32, GROUPS as u32, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_in)
        .arg_ptr(d_w)
        .arg_ptr(d_out)
        .arg_u32(HIDDEN as u32)
        .arg_u32(GROUPS as u32)
        .arg_f32(EPS)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_out, TOKENS * wide)?;

    // Oracle, per token.
    let mut want = Vec::with_capacity(TOKENS * wide);
    for t in 0..TOKENS {
        want.extend(grouped_rms_norm(
            &input_bf16[t * wide..(t + 1) * wide],
            HIDDEN,
            &weight_bf16,
            EPS,
        ));
    }

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "grouped: max|diff| {worst:.3e} over {} values (up to {scale:.3e}), relative {:.3e}",
        got.len(),
        worst / scale.max(1e-9)
    );

    // A whole-row normalisation must NOT match — otherwise the kernel could be
    // ignoring group_size and this test would pass for the wrong reason.
    let mut ungrouped = Vec::with_capacity(TOKENS * wide);
    for t in 0..TOKENS {
        ungrouped.extend(grouped_rms_norm(
            &input_bf16[t * wide..(t + 1) * wide],
            wide,
            &weight_bf16,
            EPS,
        ));
    }
    let ungrouped_gap = got
        .iter()
        .zip(&ungrouped)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("ungrouped control: max|diff| {ungrouped_gap:.3e} (must be large)");

    // Inputs now match; only the BF16 OUTPUT store separates the two, so this
    // should sit near a single ulp (~2^-8 relative) rather than a few.
    anyhow::ensure!(
        worst / scale.max(1e-9) < 5e-3,
        "rms_norm_grouped disagrees with the oracle"
    );
    anyhow::ensure!(
        ungrouped_gap > worst * 100.0,
        "grouping made no difference — the kernel may be ignoring group_size"
    );
    println!("GROUPED RMS NORM MATCHES THE ORACLE\n");

    hyper_connection_block(g)
}

// ── The whole hyper-connection block on GPU ─────────────────────────────────

// The five block checks live in their own files: the repo caps a .rs at 500
// LoC and this example was 1093. `use super::*` gives each one the device
// helpers and the shared imports above, so the harness cannot drift between
// them.
#[path = "qwen4exp_microtest/attn.rs"]
mod attn;
#[path = "qwen4exp_microtest/entry.rs"]
mod entry;
#[path = "qwen4exp_microtest/gdn.rs"]
mod gdn;
#[path = "qwen4exp_microtest/hyper_connection.rs"]
mod hyper_connection;
#[path = "qwen4exp_microtest/ple.rs"]
mod ple;
#[path = "qwen4exp_microtest/sandwich.rs"]
mod sandwich;

use attn::attn_decode_step;
use entry::hc_expand_entry;
use gdn::gdn_decode_step;
use hyper_connection::hyper_connection_block;
use ple::ple_block;
use sandwich::hc_sandwich_roundtrip;
