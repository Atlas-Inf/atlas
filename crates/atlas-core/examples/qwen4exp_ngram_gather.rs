// SPDX-License-Identifier: AGPL-3.0-only

//! Measure n-gram row gather against a real checkpoint.
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_ngram_gather -- <dir> [tokens]
//! ```
//!
//! The question this answers: the table is ~52 GB and cannot be held on a GB10,
//! so rows are read on demand. Is that fast enough to sit under decode?
//!
//! Reports cold (first touch, off NVMe) and warm (page-cached) rates. Decode on
//! this class of model runs tens of tokens/s per sequence, and each token needs
//! `K*(N-1)` rows, so the number to compare against is per-token latency.

use atlas_core::config::parse_config;
use atlas_core::ngram_table::NgramTable;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: qwen4exp_ngram_gather <dir> [tokens]"))?,
    );
    let tokens: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2048);

    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let ngram = config
        .qwen4exp_ngram(0)?
        .ok_or_else(|| anyhow::anyhow!("no PLE tower"))?;
    let open_start = Instant::now();
    let table = NgramTable::open(&dir, &config, 0)?;
    println!(
        "table      : {} rows x {} B  ({:.1} GB), opened in {:?}",
        table.total_rows(),
        table.row_bytes(),
        table.total_rows() as f64 * table.row_bytes() as f64 / 1e9,
        open_start.elapsed()
    );

    // A pseudo-random token stream, so the gather hits the table the way real
    // text does rather than walking it sequentially.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let stream: Vec<u32> = (0..tokens)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % config.vocab_size as u64) as u32
        })
        .collect();

    let ids = ngram.ngram_ids(config.ngram_vocab_size_base, &stream);
    let heads = ids.len();
    // Per token: one row from each head.
    let per_token: Vec<Vec<u32>> = (0..stream.len())
        .map(|t| ids.iter().map(|head| head[t]).collect())
        .collect();

    let mut buf = vec![0u8; heads * table.row_bytes()];
    let mut run = |label: &str| -> anyhow::Result<()> {
        let start = Instant::now();
        let mut checksum = 0u64;
        for row_ids in &per_token {
            table.gather(row_ids, &mut buf)?;
            checksum = checksum.wrapping_add(buf[0] as u64);
        }
        let elapsed = start.elapsed();
        let per = elapsed / per_token.len() as u32;
        println!(
            "{label:11}: {:?} for {} tokens  ->  {per:?}/token, \
             {:.0} rows/s, {:.0} tok/s ceiling  (checksum {checksum})",
            elapsed,
            per_token.len(),
            (per_token.len() * heads) as f64 / elapsed.as_secs_f64(),
            1.0 / per.as_secs_f64(),
        );
        Ok(())
    };

    println!(
        "heads/token: {heads}  ({} B per token)",
        heads * table.row_bytes()
    );
    run("cold")?;
    run("warm")?;
    Ok(())
}
