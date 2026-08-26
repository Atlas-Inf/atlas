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
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_data/qwen4_exp_flash_next_config.json"
        )
        .to_string()
    });
    let config = parse_config(&std::fs::read_to_string(&path)?)?;
    let manifest = atlas_core::weight_manifest::manifest_for(&config)?
        .ok_or_else(|| anyhow::anyhow!("no manifest for model_type {}", config.model_type))?;

    eprintln!(
        "{} tensors expected (excludes model.visual.*)",
        manifest.len()
    );
    let rows: Vec<_> = manifest
        .iter()
        .map(|t| serde_json::json!({ "name": t.name, "shape": t.shape }))
        .collect();
    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}
