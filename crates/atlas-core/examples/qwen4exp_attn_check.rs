// SPDX-License-Identifier: AGPL-3.0-only

//! Check the attention CPU reference against HuggingFace's own
//! `Qwen4ExpTextAttention`, with the indexer neutralised (it is a verified
//! no-op below its budget).
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_attn_check -- <ckpt> <fixture.json>
//! ```

use anyhow::{Context, Result};
use atlas_core::config::parse_config;
use atlas_core::qwen4exp_reference::{AttnDims, AttnWeights, attention_forward};
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
    rotary_dim: usize,
    hidden: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    output: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: <ckpt> <fixture.json>")?);
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(args.next().context("fixture")?)?)?;

    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let located = locate_checkpoint(&dir)?;
    let p = format!("model.language_model.layers.{}.self_attn", fixture.layer);
    let get = |n: &str| tensor_f32(&located, &format!("{p}.{n}"));

    let (q, k, v, o) = (
        get("q_proj.weight")?,
        get("k_proj.weight")?,
        get("v_proj.weight")?,
        get("o_proj.weight")?,
    );
    let (qn, kn) = (get("q_norm.weight")?, get("k_norm.weight")?);

    let dims = AttnDims {
        hidden: config.hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        rotary_dim: fixture.rotary_dim,
        eps: config.rms_norm_eps as f32,
    };
    let weights = AttnWeights {
        q_proj: &q,
        k_proj: &k,
        v_proj: &v,
        o_proj: &o,
        q_norm: &qn,
        k_norm: &kn,
    };

    let got = attention_forward(&dims, &weights, &fixture.hidden, &fixture.cos, &fixture.sin);
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
    anyhow::ensure!(
        relative < 1e-4,
        "attention reference disagrees with HuggingFace"
    );
    println!("ATTENTION MATCHES THE REFERENCE");
    Ok(())
}
