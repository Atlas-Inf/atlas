// SPDX-License-Identifier: AGPL-3.0-only

//! Check the MoE CPU reference against HuggingFace's own
//! `Qwen4ExpTextSparseMoeBlock`.
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_moe_check -- <ckpt> <fixture.json>
//! ```

use anyhow::{Context, Result};
use atlas_core::config::parse_config;
use atlas_core::qwen4exp_reference::{MoeDims, MoeWeights, moe_forward};
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
    num_experts: usize,
    top_k: usize,
    intermediate: usize,
    shared_intermediate: usize,
    norm_topk_prob: bool,
    layer: usize,
    input: Vec<f32>,
    output: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: <ckpt> <fixture.json>")?);
    let fixture: Fixture =
        serde_json::from_str(&std::fs::read_to_string(args.next().context("fixture")?)?)?;

    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let located = locate_checkpoint(&dir)?;
    let p = format!("model.language_model.layers.{}.mlp", fixture.layer);

    let router = tensor_f32(&located, &format!("{p}.gate.weight"))?;
    let shared_gate = tensor_f32(&located, &format!("{p}.shared_expert_gate.weight"))?;
    let shared: Vec<Vec<f32>> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|proj| tensor_f32(&located, &format!("{p}.shared_expert.{proj}.weight")))
        .collect::<Result<_>>()?;

    // On disk the experts are split per nn.Linear; the reference wants the
    // fused gate_up. Build it once here — a real loader would do the same.
    let dims = MoeDims {
        hidden: config.hidden_size,
        num_experts: fixture.num_experts,
        top_k: fixture.top_k,
        intermediate: fixture.intermediate,
        shared_intermediate: fixture.shared_intermediate,
        norm_topk_prob: fixture.norm_topk_prob,
    };
    let mut fused: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(dims.num_experts);
    for e in 0..dims.num_experts {
        let gate = tensor_f32(&located, &format!("{p}.experts.{e}.gate_proj.weight"))?;
        let up = tensor_f32(&located, &format!("{p}.experts.{e}.up_proj.weight"))?;
        let down = tensor_f32(&located, &format!("{p}.experts.{e}.down_proj.weight"))?;
        let mut gate_up = gate;
        gate_up.extend(up);
        fused.push((gate_up, down));
    }

    let weights = MoeWeights {
        router: &router,
        shared_gate: &shared_gate,
        shared_expert: [&shared[0], &shared[1], &shared[2]],
    };

    let positions = fixture.input.len() / dims.hidden;
    let mut worst = 0f32;
    for t in 0..positions {
        let x = &fixture.input[t * dims.hidden..(t + 1) * dims.hidden];
        let got = moe_forward(&dims, &weights, x, |e| {
            fused.get(e).map(|(gu, d)| (gu.as_slice(), d.as_slice()))
        });
        for (a, b) in got
            .iter()
            .zip(&fixture.output[t * dims.hidden..(t + 1) * dims.hidden])
        {
            worst = worst.max((a - b).abs());
        }
    }
    let scale = fixture.output.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!("  {positions} positions  max|diff| {worst:.3e}  (values up to {scale:.3e})");
    let relative = worst / scale.max(1e-9);
    println!("\nworst relative error: {relative:.3e}");
    anyhow::ensure!(relative < 1e-4, "MoE reference disagrees with HuggingFace");
    println!("MoE MATCHES THE REFERENCE");
    Ok(())
}
