// SPDX-License-Identifier: AGPL-3.0-only

//! Check the PLE CPU reference against fixtures produced by HuggingFace's own
//! `Qwen4ExpTextPLELayer`.
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_ple_check -- <ckpt> <fixture.json>
//! ```

use anyhow::{Context, Result};
use atlas_core::config::parse_config;
use atlas_core::ngram_table::NgramTable;
use atlas_core::qwen4exp_reference::{PleDims, PleWeights, ple_forward};
use atlas_core::weight_manifest::{TensorLocation, locate_checkpoint};
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Read a tensor out of a checkpoint as f32, whatever it is stored as.
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
        "F8_E4M3" => raw
            .iter()
            .map(|b| atlas_core::numeric::fp8_e4m3_to_f32(*b))
            .collect(),
        other => anyhow::bail!("{name}: no f32 path for dtype {other}"),
    })
}

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    input_ids: Vec<u32>,
    hidden_states: Vec<f32>,
    output: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    ple_conv_kernel_size: usize,
    ngram_size: usize,
    cases: Vec<Case>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: <ckpt> <fixture.json>")?);
    let fixture_path = args.next().context("usage: <ckpt> <fixture.json>")?;
    let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(&fixture_path)?)?;

    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let ngram = config.qwen4exp_ngram(0)?.context("no PLE tower")?;
    let table = NgramTable::open(&dir, &config, 0)?;
    let located = locate_checkpoint(Path::new(&dir))?;
    let layer = config.ple_decoder_layer(0).context("no PLE layer")?;
    let p = format!("model.language_model.layers.{layer}.ple");

    let (key_proj, value_proj) = (
        tensor_f32(&located, &format!("{p}.key_proj.weight"))?,
        tensor_f32(&located, &format!("{p}.value_proj.weight"))?,
    );
    let (norm_key, norm_query, norm_conv) = (
        tensor_f32(&located, &format!("{p}.norm_key.weight"))?,
        tensor_f32(&located, &format!("{p}.norm_query.weight"))?,
        tensor_f32(&located, &format!("{p}.norm_conv.weight"))?,
    );
    let conv1d = tensor_f32(&located, &format!("{p}.conv1d.weight"))?;

    let dims = PleDims {
        hidden: config.hidden_size,
        hc_count: config.hc_count,
        ple_embed_dim: config.ple_embed_dim,
        kernel: fixture.ple_conv_kernel_size,
        dilation: fixture.ngram_size,
        eps: config.rms_norm_eps as f32,
    };
    let weights = PleWeights {
        key_proj: &key_proj,
        value_proj: &value_proj,
        norm_key: &norm_key,
        norm_query: &norm_query,
        norm_conv: &norm_conv,
        conv1d: &conv1d,
    };

    let mut worst_overall = 0f32;
    for case in &fixture.cases {
        // The tower's input: gathered n-gram rows, head-major per position.
        //
        // With no cache the reference seeds `ngram_size - 1` EOS tokens as the
        // carried context and slices them back off, so the ids for position 0
        // are NOT the ids of a bare stream starting there.
        let carry = config.ngram_size - 1;
        let mut stream = vec![config.eos_token_id; carry];
        stream.extend_from_slice(&case.input_ids);
        let ids = ngram.ngram_ids(config.ngram_vocab_size_base, &stream);
        let heads = ids.len();
        let mut embeddings = vec![0f32; case.input_ids.len() * heads * table.head_dim()];
        for t in 0..case.input_ids.len() {
            let row_ids: Vec<u32> = ids.iter().map(|head| head[t + carry]).collect();
            let span = t * heads * table.head_dim()..(t + 1) * heads * table.head_dim();
            table.gather_dequant(&row_ids, 1.0, &mut embeddings[span])?;
        }

        if std::env::var("PLE_DEBUG").is_ok() && case.name == "short" {
            let first: Vec<u32> = ids.iter().map(|h| h[carry]).collect();
            eprintln!("  DEBUG ids[t=0] = {first:?}");
            eprintln!(
                "  DEBUG emb[t=0][..6] = {:?}",
                &embeddings[..6]
                    .iter()
                    .map(|v| (v * 1e6).round() / 1e6)
                    .collect::<Vec<_>>()
            );
        }
        let got = ple_forward(&dims, &weights, &embeddings, &case.hidden_states);
        anyhow::ensure!(got.len() == case.output.len(), "{}: shape", case.name);
        let worst = got
            .iter()
            .zip(&case.output)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = case.output.iter().map(|v| v.abs()).fold(0f32, f32::max);
        worst_overall = worst_overall.max(worst / scale.max(1e-9));
        println!(
            "  {:8}: seq {:>3}  max|diff| {worst:.3e}  (values up to {scale:.3e})",
            case.name,
            case.input_ids.len()
        );
    }

    println!("\nworst relative error: {worst_overall:.3e}");
    // f32 CPU vs torch f32: a few ulps of accumulated difference is expected.
    anyhow::ensure!(
        worst_overall < 1e-4,
        "PLE reference disagrees with HuggingFace"
    );
    println!("PLE MATCHES THE REFERENCE");
    Ok(())
}
