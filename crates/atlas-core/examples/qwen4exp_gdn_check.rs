// SPDX-License-Identifier: AGPL-3.0-only

//! Check the gated-delta-net CPU reference against HuggingFace's own
//! `Qwen4ExpTextGatedDeltaNet`.
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_gdn_check -- <ckpt> <fixture.json>
//! ```

use anyhow::{Context, Result};
use atlas_core::config::parse_config;
use atlas_core::qwen4exp_reference::{GdnDims, GdnWeights, gdn_forward};
use atlas_core::weight_manifest::{TensorLocation, locate_checkpoint};
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;

fn tensor_f32(located: &BTreeMap<String, TensorLocation>, name: &str) -> Result<Vec<f32>> {
    let loc = located
        .get(name)
        .with_context(|| format!("missing {name}"))?;
    let file = std::fs::File::open(&loc.path)?;
    let mut raw = vec![0u8; loc.span.len as usize];
    file.read_exact_at(&mut raw, loc.span.abs_offset)?;
    Ok(match loc.dtype.as_str() {
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "BF16" => raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        other => anyhow::bail!("{name}: no f32 path for dtype {other}"),
    })
}

#[derive(serde::Deserialize)]
struct Fixture {
    layer: usize,
    sigmoid_gate: bool,
    hidden: Vec<f32>,
    output: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: <ckpt> <fixture.json>")?);
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(args.next().context("fixture")?)?)?;

    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let located = locate_checkpoint(&dir)?;
    let p = format!("model.language_model.layers.{}.linear_attn", fixture.layer);
    let get = |n: &str| tensor_f32(&located, &format!("{p}.{n}"));

    let (qkv, z, a, b) = (
        get("in_proj_qkv.weight")?,
        get("in_proj_z.weight")?,
        get("in_proj_a.weight")?,
        get("in_proj_b.weight")?,
    );
    let (conv, a_log, dt_bias) = (get("conv1d.weight")?, get("A_log")?, get("dt_bias")?);
    let (norm, out_proj) = (get("norm.weight")?, get("out_proj.weight")?);

    let dims = GdnDims {
        hidden: config.hidden_size,
        num_k_heads: config.linear_num_key_heads,
        key_head_dim: config.linear_key_head_dim,
        num_v_heads: config.linear_num_value_heads,
        value_head_dim: config.linear_value_head_dim,
        conv_kernel: config.linear_conv_kernel_dim,
        eps: config.rms_norm_eps as f32,
        sigmoid_gate: fixture.sigmoid_gate,
    };
    let weights = GdnWeights {
        in_proj_qkv: &qkv,
        in_proj_z: &z,
        in_proj_a: &a,
        in_proj_b: &b,
        conv1d: &conv,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm: &norm,
        out_proj: &out_proj,
    };

    let got = gdn_forward(&dims, &weights, &fixture.hidden);
    anyhow::ensure!(got.len() == fixture.output.len(), "shape");
    let worst = got
        .iter()
        .zip(&fixture.output)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = fixture.output.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "  {} positions  max|diff| {worst:.3e}  (values up to {scale:.3e})",
        fixture.hidden.len() / dims.hidden
    );
    let relative = worst / scale.max(1e-9);
    println!("\nworst relative error: {relative:.3e}");
    anyhow::ensure!(relative < 1e-4, "GDN reference disagrees with HuggingFace");
    println!("GDN MATCHES THE REFERENCE");
    Ok(())
}
