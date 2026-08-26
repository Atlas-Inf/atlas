// SPDX-License-Identifier: AGPL-3.0-only

//! Check a `qwen4_exp` checkpoint on disk against its manifest.
//!
//! ```text
//! cargo run -p atlas-core --example qwen4exp_preflight -- /path/to/checkpoint
//! ```
//!
//! Reads only the safetensors headers — it does not load weights, so it runs in
//! seconds against a 135 GB checkpoint and needs no GPU.

use atlas_core::config::parse_config;
use atlas_core::weight_manifest::verify_checkpoint;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: qwen4exp_preflight <checkpoint dir>"))?,
    );
    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let (diff, layout) = verify_checkpoint(&dir, &config)?;

    println!("checkpoint : {}", dir.display());
    println!("model_type : {}", config.model_type);
    println!(
        "layout     : trunk {:?}, mtp {:?}",
        layout.trunk_experts, layout.mtp_experts
    );
    if let Some(q) = config.quantization_config.as_ref() {
        println!(
            "quant      : {} {} (block {:?}, group {})",
            q.quant_method, q.quant_algo, q.weight_block_size, q.group_size
        );
    }

    for (label, rows) in [("missing", &diff.missing), ("unexpected", &diff.unexpected)] {
        println!("{label:11}: {}", rows.len());
        for row in rows.iter().take(8) {
            println!("    {row}");
        }
    }
    println!("mismatched : {}", diff.mismatched.len());
    for (name, want, got) in diff.mismatched.iter().take(8) {
        println!("    {name}: manifest {want:?} vs checkpoint {got:?}");
    }

    if diff.is_clean() {
        println!("\nPREFLIGHT OK");
        Ok(())
    } else {
        anyhow::bail!("{} discrepancies", diff.count())
    }
}
