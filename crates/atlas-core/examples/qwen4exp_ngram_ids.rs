// SPDX-License-Identifier: AGPL-3.0-only

//! Dump Atlas's `qwen4_exp` n-gram row ids as JSON, so an independent run of
//! HuggingFace's own module can be diffed against them.
//!
//! Used by `bench/ngram_embed/qwen4exp_xcheck.py`; see that file for why the
//! comparison exists at all.
//!
//! ```text
//! cargo run -p atlas-core --example qwen4exp_ngram_ids -- streams.json
//! ```
//!
//! `streams.json` is a JSON array of token-id arrays. Output is
//! `[stream][head][position]`.

use atlas_core::config::parse_config;

fn main() -> anyhow::Result<()> {
    let streams_path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: qwen4exp_ngram_ids <streams.json>"))?;

    let config_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen4_exp_flash_next_config.json"
    );
    let config = parse_config(&std::fs::read_to_string(config_path)?)?;
    let ngram = config
        .qwen4exp_ngram(0)?
        .ok_or_else(|| anyhow::anyhow!("fixture declares no PLE tower"))?;

    let streams: Vec<Vec<u32>> = serde_json::from_str(&std::fs::read_to_string(&streams_path)?)?;
    let ids: Vec<Vec<Vec<u32>>> = streams
        .iter()
        .map(|s| ngram.ngram_ids(config.ngram_vocab_size_base, s))
        .collect();

    println!("{}", serde_json::to_string(&ids)?);
    Ok(())
}
