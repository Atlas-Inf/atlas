// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request grammar matching state.

use xgrammar::{CompiledGrammar, GrammarMatcher, allocate_token_bitmask, reset_token_bitmask};

use super::engine::GrammarError;
use crate::literal_xml::LiteralXmlState;

mod literal_dispatch;

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
    /// The response opens inside `<think>`, so the first
    /// sampled token is reasoning and must not be masked or consumed.
    opens_in_thinking: bool,
    /// Native tool token forbidden during a deferred required-tool preamble.
    first_token_suppress: Option<u32>,
    /// GLM's native tool opener, which ends reasoning without a `</think>`
    /// (`first_token_thinking::apply_native_tool_boundary`) and is itself the
    /// first output token. Set for a grammar with a tool branch beside its
    /// response_format (`GrammarSpec::ToolCallOrJson`), whose matcher must
    /// see it; `None` elsewhere.
    native_opener: Option<u32>,
    /// A tool-call-or-JSON union's answer (JSON) branch first-token mask: the
    /// branch is chosen by mass at the first output token (`grammar::branch`).
    answer_start: Option<Box<[i32]>>,
    /// Whether this is an automatic tool-call grammar whose root has a
    /// free-text dispatch state. Required tool grammars and response-format
    /// grammars stay on the exact token matcher path.
    auto_literal_dispatch: bool,
    /// Lexical state of the emitted free-text prefix. Quotes inside an actual
    /// tool body must never make the body unconstrained.
    literal_xml_state: LiteralXmlState,
    /// One lexical state per xgrammar matcher history step. `accept_bytes`
    /// records one rollback step for a whole decoded token.
    literal_xml_history: Vec<LiteralXmlState>,
    /// Vocabulary tokens that contain a possible markup delimiter. This keeps
    /// the literal-candidate repair out of the full-vocabulary hot loop.
    literal_lt_tokens: Box<[u32]>,
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
            first_token_suppress: None,
            native_opener: None,
            answer_start: None,
            auto_literal_dispatch: false,
            literal_xml_state: LiteralXmlState::default(),
            literal_xml_history: Vec::new(),
            literal_lt_tokens: Box::new([]),
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

    /// Register the native tool opener (see the `native_opener` field).
    #[must_use]
    pub fn with_native_opener(mut self, native_opener: Option<u32>) -> Self {
        self.native_opener = native_opener;
        self
    }

    /// Whether `tok`, emitted inside `<think>`, ends the reasoning and is the
    /// grammar's first output token (the native tool opener).
    pub fn opens_output(&self, tok: u32) -> bool {
        self.native_opener == Some(tok)
    }

    /// Register a union's answer-branch first-token mask (`answer_start`).
    #[must_use]
    pub fn with_answer_start(mut self, answer_start: Box<[i32]>) -> Self {
        self.answer_start = Some(answer_start);
        self
    }

    /// A union's first output token is still to choose its branch.
    pub fn branch_pending(&self) -> bool {
        self.answer_start.is_some()
            && self.matcher.num_history_steps() == 0
            && !self.matcher.is_terminated()
    }

    /// At a pending branch, restrict this position's mask to the branch the
    /// model prefers by mass over `logits`, the position's logits before the
    /// mask (`grammar::branch`). Call before the mask is applied; a no-op
    /// anywhere else.
    pub fn choose_branch(&mut self, logits: &[f32]) {
        if !self.branch_pending() || !self.fill_bitmask() {
            return;
        }
        let answer = self.answer_start.as_deref().unwrap_or_default();
        let tool = super::branch::prefers_tool(logits, &self.bitmask_data, answer);
        super::branch::restrict(&mut self.bitmask_data, answer, tool);
    }

    /// Whether this is a `response_format` grammar whose output must conform.
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Fill the allowed-token bitmask for the next decode step.
    ///
    /// Returns `true` if the bitmask constrains at least one token (i.e., is not
    /// all-ones). When `false`, the caller can skip bitmask application.
    ///
    /// Optimized for structural-tag grammars: in preamble state (before trigger),
    /// fill_bitmask() is called every 4 tokens instead of every token, saving
    /// fill_bitmask MUST be called every token to keep the xgrammar NPDA
    /// stacks synchronized with accept_token(). Skipping calls desynchronizes
    /// the FSM and causes fill_next_token_bitmask to hang (~47 tokens in).
    pub fn fill_bitmask(&mut self) -> bool {
        // Guard: calling fill_next_token_bitmask after the matcher has accepted
        // its stop token throws xgrammar::LogFatalError, which std::terminate()s
        // the whole process. Return false so callers skip bitmask application —
        // the grammar is already satisfied and imposes no further constraint.
        if self.matcher.is_terminated() {
            return false;
        }
        // Per-position cache: the mask is invariant until the matcher advances
        // (accept_token / rollback / reset all invalidate). Deduping the
        // redundant same-position refill (sampling-constrain vs stop_legal
        // EOS-legality) keeps exactly ONE real fill_next_token_bitmask per
        // token — preserving the "fill every token" NPDA-sync invariant above —
        // while removing the ~30ms redundant 248K-vocab refill that inflated the
        // MTP verify step and tripped its net-negative gate.
        if self.bitmask_valid {
            return self.bitmask_fill_result;
        }
        let t_fill = std::time::Instant::now();
        reset_token_bitmask(&mut self.bitmask_data);
        let filled = self
            .matcher
            .fill_next_token_bitmask(&mut self.bitmask_data, 0, false);
        self.timing
            .record(crate::scheduler::mtp_timing::Phase::GrammarFill, t_fill);
        self.bitmask_valid = true;
        self.bitmask_fill_result = filled;
        if filled && self.auto_literal_dispatch && self.matcher.is_grammar_completed() {
            self.restore_literal_candidates();
        }
        filled
    }

    /// Raw bitmask data: `ceil(vocab_size / 32)` i32 words.
    ///
    /// Bit `token_id` is at `data[token_id / 32] & (1 << (token_id % 32))`.
    /// A set bit means the token is allowed.
    pub fn bitmask_data(&self) -> &[i32] {
        &self.bitmask_data
    }

    /// Check if a specific token is allowed by the current bitmask.
    pub fn is_token_allowed(&self, token_id: u32) -> bool {
        let word = (token_id / 32) as usize;
        let bit = token_id % 32;
        if word >= self.bitmask_data.len() {
            return false;
        }
        (self.bitmask_data[word] & (1i32 << bit)) != 0
    }

    /// The single grammar-forced next token, if the current state
    /// admits exactly one legal token (xgrammar Tier 3b, Coalescence).
    ///
    /// Returns `Some(token_id)` when the constrained grammar leaves no
    /// choice — the token is fully determined, so the model sampling
    /// step (and the full vocab-wide mask fill) for this position can be
    /// skipped and `token_id` emitted directly. Returns `None` when the
    /// continuation is a genuine choice, the state is dead, or the
    /// matcher has terminated.
    ///
    /// CORRECTNESS: `forced_token` computes the same authoritative
    /// next-token bitmask [`Self::fill_bitmask`] would and reports a
    /// token only when it is the *sole* set bit — so it is, by
    /// construction, the only grammar-legal token. The normal path could
    /// only ever have sampled that exact token (every other token is
    /// masked to `-inf`). The matcher state is left unchanged: the caller
    /// must still feed the returned token back through
    /// [`Self::accept_token`], exactly as for a sampled token.
    ///
    /// Returns `None` once the matcher has terminated (no further
    /// constraint) — symmetric with [`Self::fill_bitmask`]'s guard.
    pub fn forced_token(&mut self) -> Option<i32> {
        if self.matcher.is_terminated() {
            return None;
        }
        // #237: reuse the per-position cached bitmask instead of
        // `matcher.forced_token()`, which recomputes a FULL
        // `fill_next_token_bitmask` of its own on every call — a second
        // authoritative mask fill per token position (the pipeline runs
        // ForcedTokenFastPath before GrammarBitmaskApply, so every decode and
        // every verify position paid the fill twice). EXACTNESS:
        // `matcher.forced_token()` == "compute the authoritative next-token
        // mask, return its sole set bit if exactly one". [`Self::fill_bitmask`]
        // computes that same authoritative mask into `bitmask_data` (the
        // per-position cache is invalidated on every matcher advance, so a
        // cached mask is always the current position's), and
        // `forced_from_bitmask` performs the identical sole-set-bit analysis
        // on it. `fill_bitmask() == false` means the mask is unconstrained
        // (all-ones ⇒ ≥ 2 legal tokens ⇒ not forced) — `None` either way.
        if !self.fill_bitmask() {
            return None;
        }
        let t = std::time::Instant::now();
        let forced = self.matcher.forced_from_bitmask(&mut self.bitmask_data, 0);
        self.timing
            .record(crate::scheduler::mtp_timing::Phase::ForcedTok, t);
        forced
    }

    /// Whether the grammar has been fully matched (all required structure generated).
    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }

    /// Whether a model stop/EOS token is grammar-legal at the current
    /// position — i.e. the response may end *here* with parseable output.
    ///
    /// Used by budget-aware graceful close (#144) to decide whether a close
    /// is needed: a length-truncated structured-output response whose stop
    /// token is *not* legal here ends mid-structure (e.g. inside an open
    /// JSON string) with unparseable output. Returns `true` (no close
    /// needed) once terminated or when the grammar imposes no constraint.
    pub fn stop_legal(&mut self, eos_tokens: &[u32]) -> bool {
        if self.matcher.is_terminated() {
            return true;
        }
        // `fill_bitmask` returning false means the grammar is all-accepting
        // here, so stopping is legal. When it constrains, the stop token is
        // present in the mask iff the root rule can complete now.
        if !self.fill_bitmask() {
            return true;
        }
        eos_tokens.iter().any(|&e| self.is_token_allowed(e))
    }

    /// The shortest grammar-legal close as content token ids (#144), or
    /// `None` if no close is found within `max_bytes`. `Some(empty)` means
    /// the grammar can already stop. Leaves matcher state unchanged.
    pub fn completion_token_ids(&mut self, max_bytes: usize) -> Option<Vec<i32>> {
        self.matcher.find_completion_token_ids(max_bytes)
    }

    /// Number of actual matcher history steps (== tokens `rollback` can undo).
    ///
    /// BUG#3 (2026-06-02): `accept_token` returns `true` for stop/EOS tokens and
    /// in the terminated state WITHOUT advancing the matcher (no history step).
    /// Spec/verify rollback accounting must therefore count actual advances via
    /// the delta of this value across a draft span — NOT the number of
    /// `accept_token`→true calls — or it over-rewinds (corrupt state / rollback
    /// panic) when a stop or terminated token lands inside the span.
    pub fn num_history_steps(&self) -> usize {
        self.matcher.num_history_steps()
    }

    /// Rollback the grammar state by `n` tokens.
    ///
    /// Used for MTP speculative decode: when draft tokens are rejected,
    /// the grammar state must be rewound to match.
    pub fn rollback(&mut self, n: usize) {
        self.bitmask_valid = false;
        self.matcher.rollback(n as i32);
        if self.auto_literal_dispatch {
            let keep = self.literal_xml_history.len().saturating_sub(n);
            self.literal_xml_history.truncate(keep);
            self.literal_xml_state = self.literal_xml_history.last().copied().unwrap_or_default();
        }
    }

    /// Reset the grammar state to the initial position.
    pub fn reset(&mut self) {
        self.bitmask_valid = false;
        self.matcher.reset();
        if self.auto_literal_dispatch {
            self.literal_xml_history.clear();
            self.literal_xml_state = LiteralXmlState::default();
        }
    }

    /// Apply the current bitmask to a slice of f32 logits in-place.
    ///
    /// Masked tokens (disallowed by grammar) are set to `f32::NEG_INFINITY`.
    /// This is the CPU-side application; for GPU-side, a CUDA kernel would
    /// be needed (future optimization).
    pub fn apply_bitmask_to_logits(&self, logits: &mut [f32]) {
        let n = logits.len().min(self.vocab_size);
        for token_id in 0..n {
            let word = token_id / 32;
            let bit = token_id % 32;
            if word < self.bitmask_data.len() && (self.bitmask_data[word] & (1i32 << bit)) == 0 {
                logits[token_id] = f32::NEG_INFINITY;
            }
        }
    }
}
#[path = "state/stop.rs"]
mod stop;
pub use stop::grammar_blocks_stop;

// ── Vocabulary extraction helper ───────────────────────────────────────

// F72 helpers (`decoded_vocab_bytes`, `compute_trigger_breakers`)
// were removed in F73 / fix42. The byte-level partial-trigger anchor
// at the sampler level hung the server in production despite passing
// isolated tests. The xgrammar non-anchored TagDispatch limitation is
// now handled at the streaming-sanitizer + parser layer (envelope_open
// markers in `LeakMarkers`, plus `<minimax:_call>` → `<tool_call>`
// normalisation in `parse_tool_calls`).

#[path = "state/first_token.rs"]
mod first_token;
