// SPDX-License-Identifier: AGPL-3.0-only
//
// Scale regressions: a strict object of N required object-valued
// properties (one rule chain per property) compiled O(N^2) in time and
// memory — every rule's view held a deep copy of the shared FSM and a
// whole-FSM end bitmap, and the pruning table walked the whole FSM per
// rule and was rebuilt for every adaptive mask. Production saw ~2.7 GB
// for N = 64 and an out-of-memory host at N = 96.

use std::sync::Arc;

use super::compiler;
use crate::earley::{EarleyParser, ProductivityTable};

/// `{"m000": {"keep": bool, "reason": str}, ...}`, every key required.
fn keyed_object(n: usize) -> String {
    let props: Vec<String> = (0..n)
        .map(|i| {
            format!(
                "\"m{i:03}\":{{\"type\":\"object\",\"additionalProperties\":false,\
                 \"required\":[\"keep\",\"reason\"],\"properties\":{{\
                 \"keep\":{{\"type\":\"boolean\"}},\"reason\":{{\"type\":\"string\"}}}}}}"
            )
        })
        .collect();
    let keys: Vec<String> = (0..n).map(|i| format!("\"m{i:03}\"")).collect();
    format!(
        "{{\"type\":\"object\",\"additionalProperties\":false,\"required\":[{}],\
         \"properties\":{{{}}}}}",
        keys.join(","),
        props.join(",")
    )
}

fn compile(n: usize) -> crate::compiler::CompiledGrammar {
    compiler(1)
        .compile_json_schema(&keyed_object(n), true, None, None, true, Some(8))
        .unwrap()
}

#[test]
fn every_rule_view_shares_the_one_grammar_fsm() {
    let cg = compile(96);
    let g = cg.grammar();
    let views: Vec<_> = g.per_rule_fsms.iter().flatten().collect();
    assert!(views.len() > 96 * 10, "one rule chain per property");
    for v in views {
        assert!(v.fsm().shares_storage_with(&g.complete_fsm));
        assert_eq!(
            v.ends().len(),
            v.num_states(),
            "ends cover the rule's own nodes"
        );
    }
}

#[test]
fn a_keyed_object_grammar_grows_linearly() {
    let (half, full) = (compile(48), compile(96));
    let (a, b) = (
        half.grammar_memory_size_bytes(),
        full.grammar_memory_size_bytes(),
    );
    assert!(b < a * 5 / 2, "48 keys: {a} B, 96 keys: {b} B");
    assert!(b < 16 << 20, "96 keys: {b} B");
}

#[test]
fn the_pruning_table_is_built_once_per_grammar() {
    let g = Arc::new(compile(8).grammar().clone());
    let (p1, p2) = (
        EarleyParser::from_grammar(g.clone()),
        EarleyParser::from_grammar(g),
    );
    assert!(Arc::ptr_eq(&p1.productivity, &p2.productivity));
}

/// The old whole-FSM fixpoint, per rule: which backing nodes reach one of
/// the rule's end nodes.
fn whole_fsm_reachability(view: &crate::fsm::CompactFsmWithStartEnd) -> Vec<bool> {
    let n = view.backing_num_states();
    let mut productive: Vec<bool> = (0..n).map(|s| view.is_end_state(s)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for s in 0..n {
            if !productive[s]
                && view.fsm().edges(s).iter().any(|e| {
                    e.target >= 0 && (e.target as usize) < n && productive[e.target as usize]
                })
            {
                productive[s] = true;
                changed = true;
            }
        }
    }
    productive
}

#[test]
fn rule_local_pruning_classifies_every_rule_node_as_the_whole_fsm_walk_did() {
    let mut set = vec![("keyed_object_4", compile(4))];
    set.extend(super::golden_tests::golden_grammars());
    for (name, cg) in set {
        let g = cg.grammar();
        let table = ProductivityTable::build(g);
        for (rule, view) in g.per_rule_fsms.iter().enumerate() {
            let Some(view) = view else { continue };
            let reference = whole_fsm_reachability(view);
            for node in view.base()..view.base() + view.num_states() {
                assert_eq!(
                    table.is_node_productive(rule as i32, node as i32),
                    reference[node],
                    "{name}: rule {rule} node {node}"
                );
            }
        }
    }
}

/// ~40 KB of schema whose `[^"]{min,max}` strings expand to more rules
/// than an FSM edge can name (rule ids are `i16`).
pub(super) fn too_many_rules_schema() -> String {
    let props: Vec<String> = (0..760)
        .map(|i| {
            let min = i % 64;
            format!(
                "\"p{i}\":{{\"type\":\"string\",\"minLength\":{min},\"maxLength\":{}}}",
                min + 50 + i % 13
            )
        })
        .collect();
    format!(
        "{{\"type\":\"object\",\"properties\":{{{}}}}}",
        props.join(",")
    )
}

#[test]
fn a_grammar_over_the_fsm_encoding_is_refused_not_wrapped() {
    use crate::compiler::CompileError;
    let schema = too_many_rules_schema();
    assert!(schema.len() < 48 << 10, "{} B", schema.len());
    let err = compiler(1)
        .compile_json_schema(&schema, true, None, None, true, Some(8))
        .unwrap_err();
    assert!(
        matches!(err, CompileError::TooLarge(ref m) if m.contains("rules")),
        "{err}"
    );
    // Bounded repetitions name aux words with `i16` too: 11,000 of them.
    let elems: Vec<String> = (0..11_000).map(|_| "\"a\"{200,300}".to_string()).collect();
    let err = compiler(1)
        .compile_grammar_from_ebnf(&format!("root ::= {}\n", elems.join(" ")), "root")
        .unwrap_err();
    assert!(
        matches!(err, CompileError::TooLarge(ref m) if m.contains("repetitions")),
        "{err}"
    );
}

#[test]
fn an_oversized_schema_grammar_is_refused_and_never_cached() {
    let mut c = compiler(1);
    c.set_max_schema_grammar_bytes(16 << 10);
    let schema = keyed_object(32);
    assert!(
        c.compile_json_schema(&schema, true, None, None, true, Some(8))
            .is_err()
    );
    assert_eq!(c.cache_size_bytes(), 0, "a refused grammar is not cached");
    c.set_max_schema_grammar_bytes(usize::MAX);
    assert!(
        c.compile_json_schema(&schema, true, None, None, true, Some(8))
            .is_ok()
    );
    assert!(c.cache_size_bytes() > 0);
}

#[test]
fn grammar_memory_counts_views_maps_and_pruning() {
    let cg = compile(16);
    let g = cg.grammar();
    let before = cg.grammar_memory_size_bytes();
    assert!(before > g.complete_fsm.memory_size() + g.num_exprs() as usize * 4);
    let _ = g.productivity();
    assert!(
        cg.grammar_memory_size_bytes() > before,
        "the pruning table is counted once built"
    );
}
