// SPDX-License-Identifier: AGPL-3.0-only
//
// Optimization pipeline — `RepetitionNormalizer` and `GrammarOptimizer`.
// Split out of `optimizer.rs` to keep each file under the 250-line cap.
// Port of the corresponding functors in `cpp/grammar_functor.cc`.

use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;
use crate::grammar::functor::analyzer::AllowEmptyRuleAnalyzer;
use crate::grammar::functor::fsm_builder::GrammarFsmBuilder;
use crate::grammar::functor::lookahead::LookaheadAssertionAnalyzer;
use crate::grammar::functor::optimizer::{ByteStringFuser, DeadCodeEliminator, RuleInliner};

/// Normalize repetition ranges: if the repeated rule is nullable, the
/// minimum repeat count is forced to 0. Operates in place.
pub struct RepetitionNormalizer;

impl RepetitionNormalizer {
    /// Run the pass on `grammar` in place.
    pub fn apply(grammar: &mut GrammarData) {
        for i in 0..grammar.num_exprs() {
            let expr = grammar.expr(i);
            if expr.kind != GrammarExprType::Repeat {
                continue;
            }
            let repeat_rule_id = expr.data[0];
            grammar.rule_mut(repeat_rule_id).is_exact_lookahead = true;
            if grammar
                .allow_empty_rule_ids
                .binary_search(&repeat_rule_id)
                .is_ok()
            {
                grammar.set_expr_data(i, 1, 0);
            }
        }
    }
}

/// Full optimization pipeline. Port of `GrammarOptimizer`.
pub struct GrammarOptimizer;

/// Whether `grammar` fits the FSM edge encoding: rule ids and repeat-edge
/// aux indices are stored as `i16`, so a grammar with more rules, or with
/// more repeat elements (three aux words each), would wrap silently. The
/// error names the limit; a client-supplied schema that hits it is too
/// large to serve.
pub fn check_fsm_limits(grammar: &GrammarData) -> Result<(), String> {
    let rules = grammar.num_rules() as usize;
    if rules > i16::MAX as usize + 1 {
        return Err(format!(
            "grammar has {rules} rules, over the {} limit",
            i16::MAX as usize + 1
        ));
    }
    let repeats = (0..grammar.num_exprs())
        .filter(|&i| grammar.expr(i).kind == GrammarExprType::Repeat)
        .count();
    if repeats * 3 > i16::MAX as usize {
        return Err(format!(
            "grammar has {repeats} bounded repetitions, over the {} limit",
            i16::MAX as usize / 3
        ));
    }
    Ok(())
}

impl GrammarOptimizer {
    /// Apply byte fusion, inlining, dead-code elimination, lookahead
    /// analysis, allow-empty analysis, repetition normalization and FSM
    /// construction. Always returns a new (optimized) grammar. Panics on a
    /// grammar over [`check_fsm_limits`]; compilers use [`Self::try_apply`].
    pub fn apply(grammar: GrammarData) -> GrammarData {
        Self::try_apply(grammar).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::apply`], refusing a grammar the FSM edges cannot encode
    /// ([`check_fsm_limits`]) before building its FSM.
    pub fn try_apply(grammar: GrammarData) -> Result<GrammarData, String> {
        let mut result = ByteStringFuser::apply(grammar);
        result = RuleInliner::apply(result);
        result = DeadCodeEliminator::apply(result);
        result = LookaheadAssertionAnalyzer::apply(result);
        result.allow_empty_rule_ids = AllowEmptyRuleAnalyzer::apply(&result);
        RepetitionNormalizer::apply(&mut result);
        check_fsm_limits(&result)?;
        GrammarFsmBuilder::apply(&mut result);
        result.optimized = true;
        Ok(result)
    }
}
