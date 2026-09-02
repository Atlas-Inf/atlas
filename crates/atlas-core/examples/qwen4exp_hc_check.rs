// SPDX-License-Identifier: AGPL-3.0-only

//! Check the hyper-connection CPU reference against fixtures from
//! HuggingFace's own `Qwen4ExpTextGatedResidual`.
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_hc_check -- <ckpt> <fixture.json>
//! ```

use anyhow::{Context, Result};
use atlas_core::config::parse_config;
use atlas_core::qwen4exp_reference::{HyperConnectionWeights, PleDims, hyper_connection_forward};
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
struct Case {
    name: String,
    combine: bool,
    input: Vec<f32>,
    mixed: Vec<f32>,
    injection: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    hc_lowrank: usize,
    cases: Vec<Case>,
}

/// Fixture names map to the checkpoint prefixes they were taken from.
fn prefix_for(name: &str) -> Result<String> {
    Ok(match name {
        "layer0_attn" => "model.language_model.layers.0.attn_hyper_connection".into(),
        "layer3_mlp" => "model.language_model.layers.3.mlp_hyper_connection".into(),
        "trunk_mixer" => "model.language_model.hyper_connection_mixer".into(),
        other => anyhow::bail!("unknown fixture case {other}"),
    })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: <ckpt> <fixture.json>")?);
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(args.next().context("fixture")?)?)?;

    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let located = locate_checkpoint(&dir)?;
    let dims = PleDims {
        hidden: config.hidden_size,
        hc_count: config.hc_count,
        ple_embed_dim: config.ple_embed_dim,
        kernel: 0,
        dilation: 0,
        eps: config.rms_norm_eps as f32,
    };

    let mut worst_overall = 0f32;
    for case in &fixture.cases {
        let p = prefix_for(&case.name)?;
        let hc_norm = tensor_f32(&located, &format!("{p}.hc_norm.weight"))?;
        let mix_down = tensor_f32(&located, &format!("{p}.input_mix_weight_down.weight"))?;
        let mix_up = tensor_f32(&located, &format!("{p}.input_mix_weight_up.weight"))?;
        let inject = case
            .combine
            .then(|| tensor_f32(&located, &format!("{p}.block_inject_weight.weight")))
            .transpose()?;
        let weights = HyperConnectionWeights {
            hc_norm: &hc_norm,
            mix_down: &mix_down,
            mix_up: &mix_up,
            block_inject: inject.as_deref(),
        };

        let wide = dims.wide();
        let positions = case.input.len() / wide;
        let mut worst = 0f32;
        for t in 0..positions {
            let out = hyper_connection_forward(
                &dims,
                &weights,
                fixture.hc_lowrank,
                &case.input[t * wide..(t + 1) * wide],
            );
            for (got, want) in out
                .mixed
                .iter()
                .zip(&case.mixed[t * dims.hidden..(t + 1) * dims.hidden])
            {
                worst = worst.max((got - want).abs());
            }
            // The trunk and MTP mixers have no block_inject_weight, so both
            // sides are empty here rather than zero-length-but-present.
            if !out.injection.is_empty() {
                let want = &case.injection[t * dims.hc_count..(t + 1) * dims.hc_count];
                for (got, want) in out.injection.iter().zip(want) {
                    worst = worst.max((got - want).abs());
                }
            }
        }
        let scale = case.mixed.iter().map(|v| v.abs()).fold(0f32, f32::max);
        worst_overall = worst_overall.max(worst / scale.max(1e-9));
        println!(
            "  {:12}: {positions} positions  max|diff| {worst:.3e}  (values up to {scale:.3e})",
            case.name
        );
    }

    println!("\nworst relative error: {worst_overall:.3e}");
    anyhow::ensure!(
        worst_overall < 1e-4,
        "hyper-connection reference disagrees with HuggingFace"
    );
    println!("HYPER-CONNECTION MATCHES THE REFERENCE");
    Ok(())
}
