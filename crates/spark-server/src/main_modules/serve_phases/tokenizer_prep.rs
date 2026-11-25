// SPDX-License-Identifier: AGPL-3.0-only

//! The tokenizer phase, started on its own thread while the weights load.
//!
//! Loading the chat tokenizer and resolving its runtime (four decode passes
//! over the whole vocabulary, the tool markers, the XGrammar engine) is CPU
//! work: ~1.2 s on Qwen3.8's 248K vocabulary, and it used to run after the
//! model was built, adding straight to startup. It reads only the values in
//! [`TokenizerInputs`], so it can run under the weight read.
//! [`TokenizerPrep::take`] hands the result over only when the serial path's
//! inputs, taken where the serial path takes them, are identical to the
//! thread's; otherwise the caller resolves the phase in place, as before.

use super::{TokenizerRuntime, load_eos_tokens, resolve_tokenizer_runtime};
use crate::{cli, tokenizer::ChatTokenizer};
use atlas_core::config::ModelConfig;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

/// Everything the tokenizer phase reads besides the tokenizer files.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TokenizerInputs {
    pub(crate) eos_tokens: Vec<u32>,
    /// `config.vocab_size` before the phase caps it to the tokenizer's.
    pub(crate) vocab_size: usize,
    pub(crate) model_type: String,
    pub(crate) supports_thinking: bool,
    pub(crate) jinja_template: &'static str,
    pub(crate) verbatim_prompt: bool,
    pub(crate) args: cli::ServeArgs,
}

impl TokenizerInputs {
    pub(crate) fn new(
        args: &cli::ServeArgs,
        config: &ModelConfig,
        eos_tokens: &[u32],
        jinja_template: &'static str,
        verbatim_prompt: bool,
    ) -> Self {
        Self {
            eos_tokens: eos_tokens.to_vec(),
            vocab_size: config.vocab_size,
            model_type: config.model_type.clone(),
            // Thinking support is derived from model capabilities, not model
            // names: SSM / Qwen3.5-style architectures support <think> tokens.
            supports_thinking: config.capabilities().supports_thinking,
            jinja_template,
            verbatim_prompt,
            args: args.clone(),
        }
    }

    /// The vocabulary the phase resolves to: the cap
    /// `resolve_tokenizer_runtime` applies, so two inputs that differ only
    /// in a vocab size the tokenizer caps to the same value agree.
    fn effective_vocab(&self, tokenizer_vocab: usize) -> usize {
        if tokenizer_vocab > 0 && tokenizer_vocab < self.vocab_size {
            tokenizer_vocab
        } else {
            self.vocab_size
        }
    }

    fn agrees_with(&self, other: &Self, tokenizer_vocab: usize) -> bool {
        let (mut a, mut b) = (self.clone(), other.clone());
        a.vocab_size = self.effective_vocab(tokenizer_vocab);
        b.vocab_size = other.effective_vocab(tokenizer_vocab);
        a == b
    }
}

/// The phase's outputs, and the two values it changes in place.
pub(crate) struct TokenizerPhase {
    pub(crate) tokenizer: ChatTokenizer,
    pub(crate) runtime: TokenizerRuntime,
    pub(crate) vocab_size: usize,
    pub(crate) eos_tokens: Vec<u32>,
}

/// Run the tokenizer phase exactly as the serial path does.
pub(crate) fn run_tokenizer_phase(
    model_dir: &Path,
    inputs: &TokenizerInputs,
    config: &mut ModelConfig,
) -> anyhow::Result<TokenizerPhase> {
    let mut eos_tokens = inputs.eos_tokens.clone();
    let tokenizer = ChatTokenizer::from_model_dir(
        model_dir,
        eos_tokens[0],
        inputs.supports_thinking,
        &inputs.model_type,
        inputs.jinja_template,
        Some(Path::new(".")), // repo root for override templates
        inputs.args.disable_template_overrides || inputs.args.reference_mode,
        inputs.verbatim_prompt,
    )?;
    let runtime = resolve_tokenizer_runtime(
        &inputs.args,
        config,
        &tokenizer,
        &mut eos_tokens,
        inputs.supports_thinking,
    );
    Ok(TokenizerPhase {
        tokenizer,
        runtime,
        vocab_size: config.vocab_size,
        eos_tokens,
    })
}

/// The tokenizer phase running ahead on its own thread.
pub(crate) struct TokenizerPrep {
    inputs: TokenizerInputs,
    handle: JoinHandle<anyhow::Result<TokenizerPhase>>,
}

impl TokenizerPrep {
    /// Start the phase from `config` as it stands now. The EOS list is read
    /// the way the serial path reads it. `None` when the thread cannot be
    /// started: the phase then runs in place.
    pub(crate) fn spawn(
        model_dir: &Path,
        args: &cli::ServeArgs,
        config: &ModelConfig,
        jinja_template: &'static str,
        verbatim_prompt: bool,
    ) -> Option<Self> {
        let eos_tokens = load_eos_tokens(model_dir, config);
        let inputs =
            TokenizerInputs::new(args, config, &eos_tokens, jinja_template, verbatim_prompt);
        let (dir, thread_inputs, mut cfg): (PathBuf, _, _) =
            (model_dir.to_path_buf(), inputs.clone(), config.clone());
        let handle = std::thread::Builder::new()
            .name("tokenizer-prep".into())
            .spawn(move || run_tokenizer_phase(&dir, &thread_inputs, &mut cfg))
            .inspect_err(|e| tracing::warn!("tokenizer-prep thread not started ({e}); in place"))
            .ok()?;
        Some(Self { inputs, handle })
    }

    /// The prepared phase, if it succeeded and every input it read equals
    /// `current`. `None` means: resolve the phase in place.
    pub(crate) fn take(self, current: &TokenizerInputs) -> Option<TokenizerPhase> {
        let phase = match self.handle.join() {
            Ok(Ok(phase)) => phase,
            Ok(Err(e)) => {
                tracing::warn!("tokenizer prepared early failed ({e:#}); resolving in place");
                return None;
            }
            Err(_) => {
                tracing::warn!("tokenizer-prep thread panicked; resolving in place");
                return None;
            }
        };
        let tokenizer_vocab = phase.tokenizer.inner().get_vocab_size(true);
        if self.inputs.agrees_with(current, tokenizer_vocab) {
            Some(phase)
        } else {
            tracing::warn!(
                "tokenizer inputs changed during weight load (eos {:?} -> {:?}, vocab {} -> {}, \
                 model_type {} -> {}); resolving in place",
                self.inputs.eos_tokens,
                current.eos_tokens,
                self.inputs.vocab_size,
                current.vocab_size,
                self.inputs.model_type,
                current.model_type,
            );
            None
        }
    }
}

#[cfg(test)]
#[path = "tokenizer_prep_tests.rs"]
mod tests;
