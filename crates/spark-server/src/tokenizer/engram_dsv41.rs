// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 Engram's compressed token map.
//!
//! Engram hashes n-grams over a smaller id space in which tokens that
//! normalize alike share an id, so " The", "the" and "THE" hash the same way.
//! This is `build_compressed_token_map` of the checkpoint's
//! `inference/engram.py`; the hashing itself is
//! `atlas_core::config::DeepseekV41Engram`. Every hash multiplier derives from
//! the size of the compressed space, so a map one entry off rehashes both
//! tables without any error.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use atlas_core::config::{DeepseekV41Engram, ModelConfig};
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{Lowercase, NFD, NFKC, Replace, Sequence, Strip, StripAccents};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

/// A private-use character that stands in for a token that is exactly one
/// space, so `Strip` does not empty it and merge it with unrelated tokens.
const LONE_SPACE: &str = "\u{e000}";

/// What the Engram layers need from the tokenizer.
pub struct EngramTokens {
    /// The compressed id of every token id.
    pub map: Vec<u32>,
    /// The hash geometry. Its `pad_id` is already a compressed id.
    pub geometry: DeepseekV41Engram,
}

/// The Engram token map and geometry for `config`, or `None` for a model with
/// no Engram layers. Refuses a tokenizer that does not compress to the size
/// the checkpoint's tables were hashed over, as the reference asserts.
pub fn resolve(
    tokenizer: &Tokenizer,
    vocab_size: u32,
    config: &ModelConfig,
) -> Result<Option<EngramTokens>> {
    if config.engram_layer_ids.is_empty() {
        return Ok(None);
    }
    let (map, size) = compress_vocab(tokenizer, vocab_size)?;
    ensure!(
        u64::from(size) == config.engram_compressed_vocab_size,
        "the tokenizer compresses to {size} ids, but the checkpoint's Engram tables were hashed \
         over {}: this is not the checkpoint's tokenizer",
        config.engram_compressed_vocab_size
    );
    let pad_id = *map
        .get(config.engram_pad_token_id as usize)
        .context("engram_pad_token_id is outside the tokenizer's vocabulary")?;
    Ok(config
        .deepseek_v41_engram(pad_id)?
        .map(|geometry| EngramTokens { map, geometry }))
}

/// Map every token id below `vocab_size` to its compressed id, and return the
/// number of compressed ids. Ids are handed out in token-id order.
pub fn compress_vocab(tokenizer: &Tokenizer, vocab_size: u32) -> Result<(Vec<u32>, u32)> {
    let normalizer = normalizer()?;
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut map = Vec::with_capacity(vocab_size as usize);
    for token_id in 0..vocab_size {
        // The raw tokenizer, as training decoded: specials kept, no clean-up.
        let text = tokenizer
            .decode(&[token_id], false)
            .map_err(anyhow::Error::msg)?;
        let key = if text.contains('\u{fffd}') {
            // A partial UTF-8 byte: nothing to normalize, so key it by its raw form.
            tokenizer
                .id_to_token(token_id)
                .with_context(|| format!("token id {token_id} has no token"))?
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(anyhow::Error::msg)?;
            match normalized.get() {
                "" => text,
                normalized => normalized.to_string(),
            }
        };
        let next = ids.len() as u32;
        map.push(*ids.entry(key).or_insert(next));
    }
    Ok((map, ids.len() as u32))
}

fn normalizer() -> Result<Sequence> {
    let replace = |pattern, with: &str| Replace::new(pattern, with).map_err(anyhow::Error::msg);
    Ok(Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        replace(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ")?.into(),
        replace(ReplacePattern::Regex("^ $".into()), LONE_SPACE)?.into(),
        Strip::new(true, true).into(),
        replace(ReplacePattern::String(LONE_SPACE.into()), " ")?.into(),
    ]))
}

#[cfg(test)]
#[path = "engram_dsv41_tests.rs"]
mod tests;
