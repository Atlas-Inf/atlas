// SPDX-License-Identifier: AGPL-3.0-only

//! Serving consumes IDs; optional offset-free encoding avoids unused metadata.
use tokenizers::{Encoding, Tokenizer};

pub(super) fn configured(raw: Option<&str>) -> Result<bool, String> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_TOKENIZER_IDS_ONLY must be unset, 0, or 1".to_string()),
    }
}

pub(super) fn encode(
    tokenizer: &Tokenizer,
    text: &str,
    ids_only: bool,
) -> tokenizers::Result<Encoding> {
    if ids_only {
        tokenizer.encode_fast(text, false)
    } else {
        tokenizer.encode(text, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_only_preserves_special_and_unicode_ids_without_unused_metadata() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                ["[UNK]", "hello", "café", "🙂", "العربية"]
                    .into_iter()
                    .enumerate()
                    .map(|(id, text)| (text.to_string(), id as u32))
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        tokenizer
            .add_special_tokens([tokenizers::AddedToken::from("<special>", true)])
            .unwrap();
        for text in ["hello café 🙂 العربية", "hello<special>café", " \t\n", ""] {
            let ordinary = encode(&tokenizer, text, false).unwrap();
            let ids_only = encode(&tokenizer, text, true).unwrap();
            assert_eq!(ordinary.get_ids(), ids_only.get_ids(), "{text:?}");
            assert!(ids_only.get_tokens().iter().all(String::is_empty));
            assert!(
                ids_only
                    .get_offsets()
                    .iter()
                    .all(|&offset| offset == (0, 0))
            );
        }
        assert!(
            encode(&tokenizer, "hello café", false)
                .unwrap()
                .get_tokens()
                .iter()
                .any(|s| !s.is_empty()),
            "unset mode retains ordinary encoding behavior"
        );
    }

    #[test]
    fn ids_only_requires_an_explicit_valid_switch() {
        assert!(!configured(None).unwrap());
        assert!(!configured(Some("0")).unwrap());
        assert!(configured(Some("1")).unwrap());
        for raw in ["", "2", "true"] {
            assert!(configured(Some(raw)).is_err());
        }
    }
}
