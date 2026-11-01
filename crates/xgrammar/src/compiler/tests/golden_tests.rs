// SPDX-License-Identifier: AGPL-3.0-only
//
// Golden mask digests: proof that the linear-compile rework (shared FSM
// views, per-grammar pruning table, in-place reachability) changed no mask.
// The digests were recorded on eb37bf08, before the rework, and must hold
// on every later build: the multiset of all reachable adaptive token masks
// (canonical accepted/uncertain token-id sets, independent of FSM node
// numbering) and the fill_next_token_bitmask output along seeded random
// valid walks, for a varied grammar set. `GOLDEN_PRINT=1` prints them.

use crate::compiler::{AdaptiveTokenMask, CompiledGrammar, GrammarCompiler, StoreType};
use crate::matcher::{GrammarMatcher, bitmask_size};
use crate::tokenizer::{TokenizerInfo, VocabType};

/// ASCII bytes, JSON/tool fragments, multi-byte UTF-8, then `<eos>`.
fn tokenizer() -> TokenizerInfo {
    let mut vocab: Vec<String> = (0u8..128).map(|b| (b as char).to_string()).collect();
    for t in [
        "{\"",
        "\":",
        "\": ",
        ", \"",
        "\"}",
        "\"]",
        "true",
        "false",
        "null",
        "  ",
        "\n  ",
        "ab",
        "abc",
        "name",
        "id",
        "keep",
        "reason",
        "red",
        "green",
        "blue",
        "kids",
        "<tool_call>",
        "</tool_call>",
        "get_weather",
        "{\"name\": \"",
        "\"arguments\": ",
        "12",
        "123",
        "0.",
        "-1",
        "[1",
        "],",
        "\\n",
        "\\\"",
        "\\u00",
        "\u{e9}",
        "\u{65e5}\u{672c}",
        " }",
        "}}",
    ] {
        vocab.push(t.to_string());
    }
    vocab.push("<eos>".to_string());
    let eos = vocab.len() as i32 - 1;
    TokenizerInfo::new(&vocab, VocabType::Raw, None, Some(vec![eos]), false)
}

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

const SCHEMAS: [(&str, &str); 6] = [
    (
        "array_min130_max140",
        r#"{"type":"array","items":{"type":"integer"},"minItems":130,"maxItems":140}"#,
    ),
    (
        "string_len_2_200",
        r#"{"type":"object","additionalProperties":false,"properties":{"s":{"type":"string","minLength":2,"maxLength":200}},"required":["s"]}"#,
    ),
    ("enum", r#"{"enum":["red","green","blue",42,null]}"#),
    (
        "optional_any_of",
        r#"{"type":"object","properties":{"a":{"anyOf":[{"type":"string"},{"type":"number"}]},"b":{"type":"boolean"},"c":{"type":"array","items":{"type":"string"}}},"required":["a"]}"#,
    ),
    (
        "recursive_ref",
        r##"{"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"]}},"$ref":"#/$defs/node"}"##,
    ),
    ("keyed_object_8", ""),
];

/// A TagDispatch tool grammar, shaped as `compile_structural_tag_raw` builds them.
const TOOL_TAG: &str = r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<tool_call>"],"tags":[{"type":"tag","begin":"<tool_call>{\"name\": \"get_weather\", \"arguments\": ","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}},"end":"}</tool_call>"}],"at_least_one":false,"stop_after_first":false}}"#;

/// The golden grammar set (shared with the pruning-equivalence test).
pub(super) fn golden_grammars() -> Vec<(&'static str, CompiledGrammar)> {
    let c = GrammarCompiler::new(tokenizer(), 1, false, -1);
    let mut out = vec![("json_object", c.compile_builtin_json_grammar().unwrap())];
    for (name, schema) in SCHEMAS {
        let schema = if schema.is_empty() {
            keyed_object(8)
        } else {
            schema.to_string()
        };
        let g = c
            .compile_json_schema(&schema, true, None, None, true, Some(8))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        out.push((name, g));
    }
    out.push((
        "tool_tag_dispatch",
        c.compile_structural_tag(TOOL_TAG).unwrap(),
    ));
    out
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// Canonical (accepted ids, uncertain ids) of a mask, hashed.
fn mask_digest(m: &AdaptiveTokenMask, info: &TokenizerInfo) -> u64 {
    let sorted = info.sorted_decoded_vocab();
    let id = |i: &i32| sorted[*i as usize].0;
    let mut uncertain: Vec<i32> = m.uncertain_indices.iter().map(id).collect();
    let mut accepted: Vec<i32> = match m.store_type {
        StoreType::Accepted => m.accepted_indices.iter().map(id).collect(),
        StoreType::Rejected => (0..sorted.len() as i32)
            .filter(|i| !m.rejected_indices.contains(i) && !m.uncertain_indices.contains(i))
            .map(|i| id(&i))
            .collect(),
        StoreType::AcceptedBitset => m.accepted_bitset.iter_ones().map(|i| i as i32).collect(),
    };
    accepted.sort_unstable();
    uncertain.sort_unstable();
    let mut h = 0xcbf29ce484222325u64;
    for v in accepted.iter().chain([&-1]).chain(uncertain.iter()) {
        fnv(&mut h, &v.to_le_bytes());
    }
    h
}

/// `(mask count, mask multiset digest, walk digest)`.
fn digests(cg: &CompiledGrammar) -> (usize, u64, u64) {
    let info = cg.tokenizer_info().clone();
    let mut per_mask: Vec<u64> = cg
        .all_reachable_masks()
        .iter()
        .map(|(_, m)| mask_digest(m, &info))
        .collect();
    per_mask.sort_unstable();
    let mut masks = 0xcbf29ce484222325u64;
    for d in &per_mask {
        fnv(&mut masks, &d.to_le_bytes());
    }
    let vocab = info.vocab_size();
    let mut walks = 0xcbf29ce484222325u64;
    for seed in 1..=6u64 {
        let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut m = GrammarMatcher::new(cg.clone(), None, false, -1);
        for _ in 0..48 {
            let mut bm = vec![0i32; bitmask_size(vocab)];
            if m.fill_next_token_bitmask(&mut bm, 0, false).is_err() {
                break;
            }
            for w in &bm {
                fnv(&mut walks, &w.to_le_bytes());
            }
            let allowed: Vec<i32> = (0..vocab as i32)
                .filter(|&t| bm[t as usize / 32] >> (t % 32) & 1 == 1)
                .collect();
            if allowed.is_empty() {
                break;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let t = allowed[(rng % allowed.len() as u64) as usize];
            assert!(
                m.accept_token(t, false),
                "seed {seed}: allowed token {t} refused"
            );
            if m.is_terminated() {
                break;
            }
        }
    }
    (per_mask.len(), masks, walks)
}

/// Recorded on eb37bf08 (before the linear-compile rework).
const GOLDEN: &[(&str, usize, u64, u64)] = &[
    ("json_object", 44, 0x0ad9b2120531fc49, 0x0c22c68e1cb27f59),
    (
        "array_min130_max140",
        167,
        0xa6bf53bf188cda53,
        0xd24624f6d188e5b1,
    ),
    (
        "string_len_2_200",
        1064,
        0x26dad172a5935f99,
        0xa3de12d828886ae8,
    ),
    ("enum", 16, 0x845367108e2d21b8, 0xe4f246e96da6f667),
    (
        "optional_any_of",
        188,
        0xa5284e64ac0595a5,
        0xab2923007d48422a,
    ),
    ("recursive_ref", 125, 0xed1b55e9ba022b6f, 0x5ffc67be49eb000d),
    (
        "keyed_object_8",
        1010,
        0x3772e51ec5b4e4c5,
        0x10cd187431b91599,
    ),
    (
        "tool_tag_dispatch",
        94,
        0xc1af611d1984414b,
        0xe8c4666241d8c77e,
    ),
];

#[test]
fn masks_and_walk_bitmasks_match_the_pre_rework_build() {
    let mut got = Vec::new();
    for (name, cg) in golden_grammars() {
        let (n, masks, walks) = digests(&cg);
        if std::env::var("GOLDEN_PRINT").is_ok() {
            eprintln!("    (\"{name}\", {n}, {masks:#018x}, {walks:#018x}),");
        }
        got.push((name, n, masks, walks));
    }
    assert_eq!(got, GOLDEN);
}
