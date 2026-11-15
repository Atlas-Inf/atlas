// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::parse_config;
use sha2::{Digest, Sha256};

/// deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json, revision 1bae0e10, unedited.
const PUBLISHED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/deepseek_v41_flash_1017_config.json"
));

/// A byte-level tokenizer whose ids are the positions in `tokens`.
fn byte_level(tokens: &[&str]) -> Tokenizer {
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            tokens
                .iter()
                .enumerate()
                .map(|(id, text)| (text.to_string(), id as u32))
                .collect(),
        )
        .unk_token(tokens[0].into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
}

/// `Ġ` is a space and `Ċ` a newline in byte-level spelling; `Ã©` is the two
/// bytes of `é`, and `Ã` / `Â` alone are the first byte of a two-byte char.
const TOKENS: [&str; 11] = [
    "<unk>", "The", "Ġthe", "THE", "Ġ", "ĠĠ", "Ċ", "Ã©", "e", "Ã", "Â",
];

#[test]
fn tokens_that_normalize_alike_share_an_id() {
    let (map, size) = compress_vocab(&byte_level(&TOKENS), TOKENS.len() as u32).unwrap();
    // Case and the leading space fold together; so do accents.
    assert_eq!(map[1..4], [1, 1, 1]);
    assert_eq!(map[7..9], [3, 3]);
    // Whitespace-only tokens collapse to one id of their own, not to "".
    assert_eq!(map[4..7], [2, 2, 2]);
    // A partial UTF-8 byte has nothing to normalize: each keeps its own id.
    assert_eq!(map[9..11], [4, 5]);
    assert_eq!((map[0], size), (0, 6));
}

#[test]
fn a_model_without_engram_needs_no_map() {
    let mut config = parse_config(PUBLISHED).unwrap();
    config.engram_layer_ids.clear();
    let tokenizer = byte_level(&TOKENS);
    assert!(
        resolve(&tokenizer, TOKENS.len() as u32, &config)
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_tokenizer_that_compresses_differently_is_refused() {
    let config = parse_config(PUBLISHED).unwrap();
    let error = resolve(&byte_level(&TOKENS), TOKENS.len() as u32, &config)
        .err()
        .expect("six compressed ids cannot serve tables hashed over 99,092")
        .to_string();
    assert!(
        error.contains("6 ids") && error.contains("99092"),
        "{error}"
    );
}

/// The whole map against the reference's. Needs the 1017
/// `tokenizer.json` (6 MB, not in the repo):
/// `ATLAS_DSV41_TOKENIZER=/path/tokenizer.json cargo test ... -- --ignored`.
/// The digest is SHA-256 over the map as little-endian i32, from the
/// checkpoint's `inference/engram.py` with tokenizers 0.22.2.
#[test]
#[ignore = "needs ATLAS_DSV41_TOKENIZER=<1017 tokenizer.json>"]
fn the_1017_tokenizer_compresses_as_the_reference_does() {
    let path = std::env::var("ATLAS_DSV41_TOKENIZER").expect("ATLAS_DSV41_TOKENIZER");
    let tokenizer = Tokenizer::from_file(path).expect("tokenizer.json");
    let vocab_size = tokenizer.get_vocab_size(true) as u32;
    let config = parse_config(PUBLISHED).unwrap();

    let tokens = resolve(&tokenizer, vocab_size, &config)
        .expect("the published pair must agree")
        .expect("1017 declares Engram layers");
    assert_eq!(tokens.map.len(), 129_280);
    assert_eq!(tokens.geometry.compressed_vocab_size, 99_092);
    assert_eq!(tokens.geometry.pad_id, 2);

    let mut digest = Sha256::new();
    for &id in &tokens.map {
        digest.update((id as i32).to_le_bytes());
    }
    let digest: String = digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest,
        "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
    );
}
