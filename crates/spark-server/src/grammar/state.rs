// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request grammar matching state.

use xgrammar::{CompiledGrammar, GrammarMatcher, allocate_token_bitmask, reset_token_bitmask};

use super::engine::GrammarError;
use crate::literal_xml::{LiteralXmlState, mask_literal_lt_bytes_with_state};

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
    /// Native tool token forbidden during a deferred required-tool preamble.
    first_token_suppress: Option<u32>,
    /// GLM's native tool opener, which ends reasoning without a `</think>`
    /// and is itself the first output token. Set for a union grammar
    /// (`GrammarSpec::ToolCallOrJson`), whose matcher must see it; `None`
    /// elsewhere.
    native_opener: Option<u32>,
    /// A tool-call-or-JSON union's answer (JSON) branch first-token mask: the
    /// branch is chosen by mass at the first output token (`grammar::branch`).
    /// (GLM `native_opener` is #199 infra — dropped for main.)
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

    /// Mark this as a `response_format` grammar (see the `strict` field).
    /// `opens_in_thinking`: the prompt leaves the model inside `<think>`.
    #[must_use]
    pub fn strict_output(mut self, opens_in_thinking: bool) -> Self {
        self.strict = true;
        self.opens_in_thinking = opens_in_thinking;
        self
    }

    /// A required tool grammar for a model with an explicit think-end starts
    /// after reasoning, just like its decode-step mask. This does not change
    /// the existing tool grammar's refusal policy.
    pub fn defer_first_token(mut self, tool_opener: Option<u32>) -> Self {
        self.opens_in_thinking = true;
        self.first_token_suppress = tool_opener;
        self
    }

    /// Match the decode-time tool-opener mask during required-tool reasoning.
    pub fn first_token_suppress(&self) -> Option<u32> {
        self.first_token_suppress
    }

    /// Register a union's answer-branch first-token mask (`answer_start`).
    #[must_use]
    pub fn with_answer_start(mut self, answer_start: Box<[i32]>) -> Self {
        self.answer_start = Some(answer_start);
        self
    }

    /// Set this grammar's native opener (the token that ends reasoning and
    /// is itself the first output token). Union grammars only.
    #[must_use]
    pub fn with_native_opener(mut self, native_opener: Option<u32>) -> Self {
        self.native_opener = native_opener;
        self
    }

    /// Enable quote/code-aware matching for an automatic tool-call grammar.
    /// The scheduler opts this in only for `ToolCall(use_triggers=true)`;
    /// required and response-format grammars remain unchanged.
    #[must_use]
    pub fn with_auto_literal_dispatch(mut self, enabled: bool) -> Self {
        self.auto_literal_dispatch = enabled;
        if enabled {
            self.literal_lt_tokens = self
                .tokenizer_info
                .decoded_vocab()
                .iter()
                .enumerate()
                .filter_map(|(token_id, token)| token.contains(&b'<').then_some(token_id as u32))
                .collect::<Vec<_>>()
                .into_boxed_slice();
        } else {
            self.literal_lt_tokens = Box::new([]);
        }
        self
    }

    /// Whether `tok`, emitted inside `<think>`, ends the reasoning and is the
    /// grammar's first output token (the native tool opener).
    pub fn opens_output(&self, tok: u32) -> bool {
        self.native_opener == Some(tok)
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

    /// Whether the first token sampled after prefill is grammar output. False
    /// only for a strict grammar whose response opens inside `<think>`: that
    /// token is reasoning, and the grammar starts after `</think>` as on every
    /// other path. Tool grammars keep masking it (GLM can open a tool call
    /// as its first token, which implicitly ends thinking).
    pub fn masks_first_token(&self) -> bool {
        !self.opens_in_thinking
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

    /// Re-admit tokens whose matcher view escapes a `<` inside a quote/code
    /// span. This is limited to the free dispatch state; markup and quotes in
    /// an actual tool body remain governed by the tool grammar.
    fn restore_literal_candidates(&mut self) {
        let vocab = self.tokenizer_info.decoded_vocab();
        for &token_id in &self.literal_lt_tokens {
            let Some(token) = vocab.get(token_id as usize) else {
                continue;
            };
            let matcher_view = mask_literal_lt_bytes_with_state(token, self.literal_xml_state);
            if matcher_view == *token {
                continue;
            }
            let word = token_id as usize / 32;
            if let Some(mask) = self.bitmask_data.get_mut(word) {
                *mask |= 1i32 << (token_id % 32);
            }
        }
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

    /// Accept a sampled token and advance the grammar state.
    ///
    /// Returns `true` if the token was accepted by the grammar.
    /// Returns `false` if the token violates the grammar (should not happen
    /// if the bitmask was applied correctly).
    ///
    /// Short-circuits with `true` once the matcher has reached its
    /// terminated (accepting) state — feeding tokens past the stop into
    /// xgrammar emits a `grammar_matcher.cc:493` warning ("matcher has
    /// terminated, but is trying to accept new token") for every trailing
    /// token in spec-decode draft runs (Discord 2026-05-08 universe06608).
    /// Returning `true` keeps the spec-decode boundary heuristic in
    /// `truncate_drafts_at_grammar_boundary` consistent — drafts past a
    /// completed grammar are not "rejected by grammar"; they are simply
    /// past the stop, which the EOS handler will terminate independently.
    pub fn accept_token(&mut self, token_id: u32) -> bool {
        // Any advance invalidates the per-position bitmask cache. Done
        // unconditionally (even on the early-return paths that don't move the
        // matcher) — over-invalidation only costs a refill, whereas a missed
        // invalidation would serve a stale mask and desync the NPDA.
        self.bitmask_valid = false;
        if self.matcher.is_terminated() {
            return true;
        }
        // Stop/EOS tokens (e.g. `<|im_end|>`, 248046) are control tokens that
        // terminate the turn — they are never part of the grammar's content
        // language. The model legitimately ends a turn with one (after a
        // text-only reply, or after a complete tool call) while the
        // structural-tag NPDA is still in a non-accepting state; feeding it to
        // `matcher.accept_token` then returns false ("refusal"), which callers
        // mis-read as a grammar desync and abort the response mid-stream
        // (`emit_step`: "Ending response to prevent cascading grammar-mask
        // corruption"). That truncates agentic turns and is the dominant cause
        // of the opencode webserver_ok gap vs vLLM (which parses tools
        // post-hoc and never constrains the stop token). Accept it
        // unconditionally and let the EOS handler terminate. This cannot
        // corrupt tool-call STRUCTURE — only non-stop tokens drive the matcher.
        if self.stop_tokens.contains(&token_id) {
            return true;
        }
        if self.auto_literal_dispatch
            && !self
                .tokenizer_info
                .special_token_ids()
                .contains(&(token_id as i32))
            && let Some(token) = self.tokenizer_info.decoded_vocab().get(token_id as usize)
        {
            let in_dispatch = self.matcher.is_grammar_completed();
            let matcher_view = if in_dispatch {
                mask_literal_lt_bytes_with_state(token, self.literal_xml_state)
            } else {
                token.clone()
            };
            let accepted = self.matcher.accept_bytes(&matcher_view, false);
            if accepted {
                if in_dispatch {
                    let mut actual_state = self.literal_xml_state;
                    actual_state.feed_bytes(token);
                    if self.matcher.is_grammar_completed() {
                        // Ordinary prose remains in dispatch, so preserve the
                        // actual delimiter state for the next token.
                        self.literal_xml_state = actual_state;
                    } else {
                        // The matcher left dispatch and entered a tool body.
                        self.literal_xml_state = LiteralXmlState::default();
                    }
                } else {
                    // The active body is governed solely by its grammar. A
                    // closing tag returns to dispatch with a clean classifier.
                    self.literal_xml_state = LiteralXmlState::default();
                }
                self.literal_xml_history.push(self.literal_xml_state);
            }
            return accepted;
        }
        self.matcher.accept_token(token_id as i32)
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

/// #192: whether the active grammar FORBIDS ending the turn at the current
/// matcher position — the single EOS-suppression predicate for both decode
/// paths (`decode_logits_step` and the MTP/emit path in `emit_step`).
///
/// Replaces the blanket `!is_terminated()` gate: a trigger-based
/// (tool_choice="auto") structural-tag matcher NEVER terminates — its
/// dispatch root loops forever and [`GrammarState::accept_token`] exempts
/// stop/EOS tokens from the matcher — so `!is_terminated()` suppressed EOS
/// for the WHOLE turn whenever no tool call completed (armed-but-unused
/// tools ran to `finish_reason="length"`; live probe #6, 2026-07-02). The
/// correct question is positional: may the response legally end HERE?
///
/// * `None` grammar (plain chat / disengaged)      → `false` (EOS free).
/// * terminated matcher                            → `false`.
/// * dispatch/preamble state, or between completed
///   calls (stop token grammar-legal)              → `false`.
/// * mid-structure (open tag body / JSON string)   → `true` (suppress; the
///   sampler's bitmask already excludes EOS here, so a sampled EOS is a
///   masked-path leak that must not end the turn mid-call).
///
/// Cost note: [`GrammarState::stop_legal`] fills a fresh bitmask — callers
/// must invoke this ONLY when the sampled token is an EOS token (rare),
/// never as a per-token predicate.
pub fn grammar_blocks_stop(gs: Option<&mut GrammarState>, eos_tokens: &[u32]) -> bool {
    match gs {
        None => false,
        Some(gs) => !gs.is_terminated() && !gs.stop_legal(eos_tokens),
    }
}

// ── Vocabulary extraction helper ───────────────────────────────────────

// F72 helpers (`decoded_vocab_bytes`, `compute_trigger_breakers`)
// were removed in F73 / fix42. The byte-level partial-trigger anchor
// at the sampler level hung the server in production despite passing
// isolated tests. The xgrammar non-anchored TagDispatch limitation is
// now handled at the streaming-sanitizer + parser layer (envelope_open
// markers in `LeakMarkers`, plus `<minimax:_call>` → `<tool_call>`
// normalisation in `parse_tool_calls`).
