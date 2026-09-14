// SPDX-License-Identifier: AGPL-3.0-only

//! Print the `qwen4_exp` weight manifest as JSON, for diffing against a
//! published checkpoint's `model.safetensors.index.json` without downloading
//! the weights.
//!
//! ```text
//! cargo run -p atlas-core --example qwen4exp_manifest -- [config.json]
//! ```

use atlas_core::config::parse_config;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .filter(|a| !a.starts_with("--"))
        .unwrap_or_else(|| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../test_data/qwen4_exp_flash_next_config.json"
            )
            .to_string()
        });
    // Some releases store the MTP block's experts in HF's native stacked form
    // rather than split per expert; pass --stacked-mtp for those.
    let layout = atlas_core::weight_manifest::Qwen4ExpLayout {
        mtp_experts: if std::env::args().any(|a| a == "--stacked-mtp") {
            atlas_core::weight_manifest::ExpertLayout::Stacked
        } else {
            atlas_core::weight_manifest::ExpertLayout::PerExpert
        },
        ..Default::default()
    };
    let config = parse_config(&std::fs::read_to_string(&path)?)?;
    let mut manifest = atlas_core::weight_manifest::manifest_for_with(&config, layout)?
        .ok_or_else(|| anyhow::anyhow!("no manifest for model_type {}", config.model_type))?;

    // Apply the checkpoint's declared quantization, when it is one this
    // manifest describes. NVFP4 rewrites weight shapes as well as adding
    // siblings, so this replaces the list rather than extending it.
    let base = manifest.len();
    if let Some(quantized) = atlas_core::weight_manifest::quantized_manifest(&config, &manifest)? {
        manifest = quantized;
    }
    eprintln!(
        "{} tensors expected ({base} logical weights; excludes model.visual.*)",
        manifest.len()
    );
    let rows: Vec<_> = manifest
        .iter()
        .map(|t| serde_json::json!({ "name": t.name, "shape": t.shape }))
        .collect();
    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}
