// SPDX-License-Identifier: AGPL-3.0-only

//! Dump specific n-gram rows, dequantized, so an independent reader can check
//! the offset arithmetic and the FP8 decode against the same checkpoint.
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_ngram_rows -- <dir> <row>...
//! ```

use atlas_core::config::parse_config;
use atlas_core::ngram_table::NgramTable;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("usage: qwen4exp_ngram_rows <dir> <row>..."))?,
    );
    let rows: Vec<u32> = args.filter_map(|a| a.parse().ok()).collect();
    anyhow::ensure!(!rows.is_empty(), "give at least one row index");

    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let table = NgramTable::open(&dir, &config, 0)?;
    eprintln!(
        "table {} rows, dtype {}, head_dim {}",
        table.total_rows(),
        table.dtype(),
        table.head_dim()
    );

    // Scale of 1.0: the checkpoint's own weight_scale is applied by the caller,
    // and leaving it out here keeps this comparable to a raw decode.
    let mut out = vec![0f32; rows.len() * table.head_dim()];
    table.gather_dequant(&rows, 1.0, &mut out)?;
    let payload: Vec<_> = rows
        .iter()
        .zip(out.chunks_exact(table.head_dim()))
        .map(|(row, values)| serde_json::json!({ "row": row, "values": values }))
        .collect();
    println!("{}", serde_json::to_string(&payload)?);
    Ok(())
}
