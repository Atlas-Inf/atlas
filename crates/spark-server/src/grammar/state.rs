// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request grammar matching state.

use xgrammar::{CompiledGrammar, GrammarMatcher, allocate_token_bitmask, reset_token_bitmask};

use super::engine::GrammarError;

// ── GrammarState ───────────────────────────────────────────────────────

/// How many of the costliest token-masks to pre-warm when a grammar
/// state is created (xgrammar Tier 2, overlapped mask generation).
///
/// `CompiledGrammar::compile_top_k_masks` ranks reachable scanable
/// parser states by first-character scan breadth and eagerly populates
/// the JIT mask cache for the top `k`. Called from [`GrammarState::new`]
/// — which runs during the prefill phase of a grammar-constrained
/// request, while the GPU is busy with the prompt — so the first decode
/// steps never pay a cold mask-computation stall.
///
/// `512`: measured on Qwen3.6-35B-A3B (248K vocab, qwen3_coder structural-tag
/// grammar, GB10), the old `8` left the decode loop's resident states —
/// preamble/trigger-tracking and the JSON body scanners — COLD, and every
/// grammar-armed decode token paid a ~41 ms JIT mask computation: 26 → 12.6
/// tok/s from merely ARMING tools (2.05x), which also inflated the MTP verify
/// step past its net-negative gate. The warm set must cover the states the
/// loop actually lives in, not just the 8 broadest scanners. The call is
/// bounded by `ranked.len()` (over-provisioning stays harmless), runs during
/// prefill overlap, and the cache is shared per-grammar across requests.
const FORCED_TOKEN_TOP_K: usize = 512;

/// Per-request grammar matching state.
///
/// Wraps a [`GrammarMatcher`] with its own bitmask buffer. The bitmask
/// is reused across decode steps to avoid re-allocation.
pub struct GrammarState {
    /// The run's verify-timing sink.
    timing: std::sync::Arc<crate::scheduler::mtp_timing::RunTiming>,
    matcher: GrammarMatcher,
    /// Tokenizer bytes paired with the compiled matcher. The scheduler uses
    /// this only when a candidate native opener is emitted, to distinguish a
    /// literal XML mention in earlier reasoning from a real tool boundary.
    tokenizer_info: xgrammar::tokenizer::TokenizerInfo,
    /// Bitmask buffer: `Box<[i32]>` of shape `(1, ceil(vocab_size / 32))`.
    bitmask_data: Box<[i32]>,
    vocab_size: usize,
    /// Model stop/EOS token IDs (e.g. `<|im_end|>`). These are control
    /// tokens that terminate generation, NOT part of the grammar's content
    /// language — [`Self::accept_token`] accepts them unconditionally rather
    /// than feeding them to the xgrammar matcher (which refuses a stop token
    /// in a non-accepting state, desyncing the NPDA and aborting the turn).
    /// Empty unless set via [`Self::with_stop_tokens`] — production wires the
    /// request's `eos_tokens` in; unit tests that exercise only grammar
    /// structure leave it empty.
    stop_tokens: Box<[u32]>,
    /// Per-position fill cache. `fill_next_token_bitmask` over a 248K vocab is
    /// ~30ms; it is re-run redundantly within a SINGLE token position (once to
    /// constrain sampling, again by `stop_legal`/`grammar_blocks_stop` to test
    /// EOS-legality against the same matcher state). The mask only changes when
    /// the matcher advances, so cache it and skip the refill until then. When
    /// `bitmask_valid` is true, `bitmask_data` holds the fill for the current
    /// matcher position and `bitmask_fill_result` its return value. Invalidated
    /// (set false) at every matcher-advancing site: `accept_token`, `rollback`,
    /// `reset`. Over-invalidation only costs a refill; under-invalidation would
    /// serve a stale mask, so the invalidations are deliberately generous.
    bitmask_valid: bool,
    bitmask_fill_result: bool,
    /// A `response_format` grammar (json_object / json_schema): the response
    /// must conform, so a refused token ends it with an error instead of
    /// disengaging the grammar as tool-call grammars do. Set via
    /// [`Self::strict_output`].
    strict: bool,
    /// The strict grammar's response opens inside `<think>`, so the first
    /// sampled token is reasoning and must not be masked or consumed.
    opens_in_thinking: bool,
    /// GLM's native tool opener, which ends reasoning without a `</think>`
    /// (`first_token_thinking::apply_native_tool_boundary`) and is itself the
    /// first output token. Set for a grammar with a tool branch beside its
    /// response_format (`GrammarSpec::ToolCallOrJson`), whose matcher must
    /// see it; `None` elsewhere.
    native_opener: Option<u32>,
    /// A tool-call-or-JSON union's answer (JSON) branch first-token mask: the
    /// branch is chosen by mass at the first output token (`grammar::branch`).
    answer_start: Option<Box<[i32]>>,
}

impl GrammarState {
    /// Create a new per-request grammar state from a compiled grammar.
    ///
    /// `vocab_size` must match the tokenizer vocabulary used during compilation.
    ///
    /// As part of construction this pre-warms the [`FORCED_TOKEN_TOP_K`]
    /// costliest token-masks via [`CompiledGrammar::compile_top_k_masks`]
    /// (xgrammar Tier 2). Construction happens during prefill, so the
    /// warm-up overlaps the prompt forward pass and the first decode
    /// steps never stall on a cold mask computation. The warm-up only
    /// populates a cache — it cannot change matcher behavior — so it is
    /// safe unconditionally.
    pub fn new(compiled: &CompiledGrammar, vocab_size: usize) -> Result<Self, GrammarError> {
        Self::with_timing(compiled, vocab_size, std::sync::Arc::default())
    }

    /// Same, with the run's timing sink. The grammar engine records two verify
    /// phases and has no scheduler context of its own — it is handed the sink
    /// rather than reaching for a global one.
    pub fn with_timing(
        compiled: &CompiledGrammar,
        vocab_size: usize,
        timing: std::sync::Arc<crate::scheduler::mtp_timing::RunTiming>,
    ) -> Result<Self, GrammarError> {
        let mut state = Self::build(compiled, vocab_size)?;
        state.timing = timing;
        Ok(state)
    }

    fn build(compiled: &CompiledGrammar, vocab_size: usize) -> Result<Self, GrammarError> {
        let matcher = GrammarMatcher::new(
            compiled, None,  // use stop tokens from compiled grammar
            false, // require stop token for proper termination
            -1,    // unlimited rollback
        )
        .map_err(GrammarError::Compilation)?;

        // Tier 2 (overlapped mask generation): eagerly compute the
        // costliest masks so they are warm before the first decode
        // step. Pure cache population — no behavioral effect.
        let warmed = compiled.compile_top_k_masks(FORCED_TOKEN_TOP_K);
        tracing::debug!(
            warmed,
            requested = FORCED_TOKEN_TOP_K,
            "Grammar: pre-warmed top-k token masks during prefill"
        );

        let bitmask_data = allocate_token_bitmask(1, vocab_size);

        Ok(Self {
            timing: std::sync::Arc::default(),
            matcher,
            tokenizer_info: compiled.tokenizer_info().clone(),
            bitmask_data,
            vocab_size,
            stop_tokens: Box::new([]),
            bitmask_valid: false,
            bitmask_fill_result: false,
            strict: false,
            opens_in_thinking: false,
            native_opener: None,
            answer_start: None,
        })
    }

    /// Register the model's stop/EOS token IDs so [`Self::accept_token`]
    /// exempts them from grammar refusal. See the `stop_tokens` field doc.
    ///
    /// Builder form so the existing two-arg [`Self::new`] (used widely by
    /// unit tests of pure grammar structure) is unchanged; the single
    /// production creation site chains this with the request's eos tokens.
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: &[u32]) -> Self {
        self.stop_tokens = stop_tokens.to_vec().into_boxed_slice();
        self
    }

    /// Mark this as a `response_format` grammar (see the `strict` field).
    /// `opens_in_thinking`: the prompt leaves the model inside `<think>`.
    #[must_use]
    pub fn strict_output(mut self, opens_in_thinking: bool) -> Self {
        self.strict = true;
        self.opens_in_thinking = opens_in_thinking;
        self
    }
}
#[path = "state/literal_ops.rs"]
mod literal_ops;
pub use literal_ops::grammar_blocks_stop;
