// SPDX-License-Identifier: AGPL-3.0-only

use super::{TokenizerInputs, TokenizerPrep, run_tokenizer_phase};
use atlas_core::config::ModelConfig;
use clap::Parser;

const TOKENS: [&str; 6] = [
    "[UNK]",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "x",
];

/// A model directory holding only a word-level `tokenizer.json`.
fn model_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let vocab = TOKENS
        .iter()
        .enumerate()
        .map(|(i, t)| (t.to_string(), i as u32));
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab.collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    tokenizers::Tokenizer::new(model)
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    dir
}

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.eos_token_id = 0;
    c
}

fn args() -> crate::cli::ServeArgs {
    crate::cli::ServeArgs::parse_from(["spark", "m"])
}

fn spawn(dir: &tempfile::TempDir, config: &ModelConfig) -> TokenizerPrep {
    TokenizerPrep::spawn(dir.path(), &args(), config, "", false).expect("thread started")
}

fn now(config: &ModelConfig) -> TokenizerInputs {
    TokenizerInputs::new(&args(), config, &[0], "", false)
}

/// The early phase hands over what the in-place phase would have built.
#[test]
fn prepared_phase_matches_the_in_place_phase() {
    let dir = model_dir();
    let early = spawn(&dir, &config())
        .take(&now(&config()))
        .expect("inputs agree");
    let mut cfg = config();
    let serial = run_tokenizer_phase(dir.path(), &now(&cfg), &mut cfg).unwrap();
    assert_eq!(early.vocab_size, serial.vocab_size);
    assert_eq!(early.vocab_size, TOKENS.len(), "capped to the tokenizer");
    assert_eq!(early.eos_tokens, serial.eos_tokens);
    let (e, s) = (&early.runtime, &serial.runtime);
    assert_eq!(e.think_start_token, s.think_start_token);
    assert_eq!(e.think_end_token, s.think_end_token);
    assert_eq!(e.tool_call_start_token, s.tool_call_start_token);
    assert_eq!(e.tool_call_end_token, s.tool_call_end_token);
    assert_eq!(
        e.vocab_masks.mid_word.as_deref(),
        s.vocab_masks.mid_word.as_deref()
    );
    assert_eq!(e.think_start_token, Some(1));
}

/// The serial path caps `vocab_size` to the tokenizer before this phase; a
/// raw size and its capped size resolve to the same vocabulary.
#[test]
fn a_vocab_size_capped_in_between_still_agrees() {
    let dir = model_dir();
    let mut capped = config();
    capped.vocab_size = TOKENS.len();
    assert!(spawn(&dir, &config()).take(&now(&capped)).is_some());
}

/// Any input that changed after the thread started sends the phase back to
/// the serial path.
#[test]
fn changed_inputs_resolve_in_place() {
    let dir = model_dir();
    let mut other_eos = now(&config());
    other_eos.eos_tokens = vec![5];
    assert!(spawn(&dir, &config()).take(&other_eos).is_none());

    let mut smaller = config();
    smaller.vocab_size = 3;
    assert!(spawn(&dir, &config()).take(&now(&smaller)).is_none());

    let mut parser = now(&config());
    parser.args.tool_call_parser = Some("hermes".into());
    assert!(spawn(&dir, &config()).take(&parser).is_none());
}

/// A failed early phase is not an error here: the serial path reports it.
#[test]
fn a_failed_phase_resolves_in_place() {
    let empty = tempfile::tempdir().unwrap();
    let prep = TokenizerPrep::spawn(empty.path(), &args(), &config(), "", false).unwrap();
    assert!(prep.take(&now(&config())).is_none());
}

/// The vocab cap applied before the KV config, from the size parsed off the
/// critical path, agrees with the cap the tokenizer phase applies.
#[test]
fn the_early_vocab_cap_is_the_tokenizer_phase_cap() {
    use super::super::{cap_vocab_size_to_tokenizer, tokenizer_vocab_size};
    let dir = model_dir();
    let vocab = tokenizer_vocab_size(dir.path());
    assert_eq!(vocab, Some(TOKENS.len()));
    let mut capped = config();
    cap_vocab_size_to_tokenizer(vocab, &mut capped);
    assert_eq!(capped.vocab_size, TOKENS.len());
    let mut smaller = config();
    smaller.vocab_size = 3;
    cap_vocab_size_to_tokenizer(vocab, &mut smaller);
    assert_eq!(smaller.vocab_size, 3, "never raised");

    let empty = tempfile::tempdir().unwrap();
    assert_eq!(tokenizer_vocab_size(empty.path()), None);
    let mut untouched = config();
    cap_vocab_size_to_tokenizer(None, &mut untouched);
    assert_eq!(untouched.vocab_size, config().vocab_size);
}
