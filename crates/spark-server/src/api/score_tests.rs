// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn req(json: &str) -> ScoreRequest {
    serde_json::from_str(json).unwrap()
}

#[test]
fn shared_options_expand_per_branch() {
    let v = validate(
        req(r#"{"prefix":[1,2],"branches":[[3],[4,5]],"options":[7,8]}"#),
        10,
    )
    .unwrap();
    assert!(!v.prefix_bidir);
    assert!(v.pack.is_none());
    assert_eq!(v.prefix, vec![1, 2]);
    assert_eq!(v.branches.len(), 2);
    assert_eq!(v.options, vec![vec![7, 8], vec![7, 8]]);
}

#[test]
fn per_branch_options_must_match_branches() {
    let r = req(r#"{"branches":[[3],[4]],"options":[[7]]}"#);
    assert!(validate(r, 10).is_err());
    let r = req(r#"{"branches":[[3],[4]],"options":[[7],[8,9]]}"#);
    assert!(validate(r, 10).is_ok());
}

#[test]
fn rejects_empty_and_out_of_vocab() {
    assert!(validate(req(r#"{"branches":[],"options":[1]}"#), 10).is_err());
    assert!(validate(req(r#"{"branches":[[]],"options":[1]}"#), 10).is_err());
    assert!(validate(req(r#"{"branches":[[1]],"options":[]}"#), 10).is_err());
    assert!(validate(req(r#"{"branches":[[1]],"options":[10]}"#), 10).is_err());
    assert!(validate(req(r#"{"prefix":[11],"branches":[[1]],"options":[2]}"#), 10).is_err());
}

#[test]
fn attention_mode_is_validated() {
    let ok = req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"attention":"prefix_bidir"}"#);
    assert!(validate(ok, 10).unwrap().prefix_bidir);
    let empty = req(r#"{"branches":[[3]],"options":[7],"attention":"prefix_bidir"}"#);
    assert!(validate(empty, 10).is_err());
    let bad = req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"attention":"full"}"#);
    assert!(validate(bad, 10).is_err());
}

#[test]
fn a_pack_is_tokens_or_an_id_never_both() {
    let t = req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"pack":[4,5]}"#);
    assert_eq!(
        validate(t, 10).unwrap().pack,
        Some(PackRef::Tokens(vec![4, 5]))
    );
    let id = req(r#"{"prefix":[1],"branches":[[3]],"options":[7],"pack_id":"00000000000000ff"}"#);
    assert_eq!(validate(id, 10).unwrap().pack, Some(PackRef::Id(255)));
    let both = req(r#"{"branches":[[3]],"options":[7],"pack":[4],"pack_id":"ff"}"#);
    assert!(validate(both, 10).is_err());
    let state_is_pack =
        req(r#"{"branches":[[3]],"options":[7],"pack":[4],"attention":"prefix_bidir"}"#);
    assert!(validate(state_is_pack, 10).is_ok());
    let empty = req(r#"{"branches":[[3]],"options":[7],"pack":[]}"#);
    assert!(validate(empty, 10).is_err());
    let not_hex = req(r#"{"branches":[[3]],"options":[7],"pack_id":"zz"}"#);
    assert!(validate(not_hex, 10).is_err());
}

#[test]
fn readouts_require_one_to_four_in_branch_offsets_per_branch() {
    let valid = validate(
        req(r#"{"branches":[[3,4],[5]],"options":[7],"readouts":[[1,0],[0]]}"#),
        10,
    )
    .unwrap();
    assert_eq!(valid.readouts, Some(vec![vec![1, 0], vec![0]]));
    for json in [
        r#"{"branches":[[3]],"options":[7],"readouts":[]}"#,
        r#"{"branches":[[3]],"options":[7],"readouts":[[]]}"#,
        r#"{"branches":[[3]],"options":[7],"readouts":[[0,0,0,0,0]]}"#,
        r#"{"branches":[[3]],"options":[7],"readouts":[[1]]}"#,
    ] {
        assert!(validate(req(json), 10).is_err(), "{json}");
    }
    assert!(
        serde_json::from_str::<ScoreRequest>(
            r#"{"branches":[[3]],"options":[7],"readouts":[[-1]]}"#
        )
        .is_err()
    );
}

#[test]
fn response_keeps_legacy_flat_shape_and_nests_requested_readouts() {
    let legacy = ScoreResponse {
        probs: vec![BranchProbabilities::Last(vec![0.75, 0.25])],
        engine_us: 1,
        queue_us: 2,
        batched: 1,
        pack_id: None,
        pack: None,
    };
    assert_eq!(
        serde_json::to_string(&legacy).unwrap(),
        r#"{"probs":[[0.75,0.25]],"engine_us":1,"queue_us":2,"batched":1}"#
    );
    let with_readouts = ScoreResponse {
        probs: vec![BranchProbabilities::Readouts(vec![
            vec![0.1, 0.9],
            vec![0.5, 0.5],
        ])],
        engine_us: 1,
        queue_us: 2,
        batched: 1,
        pack_id: None,
        pack: None,
    };
    assert_eq!(
        serde_json::to_value(with_readouts).unwrap()["probs"],
        serde_json::to_value([vec![vec![0.1_f32, 0.9_f32], vec![0.5_f32, 0.5_f32]]]).unwrap()
    );
}
