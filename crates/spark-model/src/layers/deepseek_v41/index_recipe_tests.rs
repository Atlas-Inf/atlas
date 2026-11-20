// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::deepseek_v41::cache_recipe::bf16_to_f32;

const NONE: f32 = f32::NEG_INFINITY;

#[test]
fn the_best_positions_come_back_in_position_order() {
    let scores = [0.5, 3.0, -1.0, 2.0, 0.0, 2.5];
    assert_eq!(top_positions(&scores, 3), [1, 3, 5]);
    assert_eq!(top_positions(&scores, 1), [1]);
    // Fewer positions than the limit: all of them.
    assert_eq!(top_positions(&scores, 6), [0, 1, 2, 3, 4, 5]);
    assert_eq!(top_positions(&scores, 512), [0, 1, 2, 3, 4, 5]);
    assert!(top_positions(&[], 4).is_empty());
}

#[test]
fn equal_scores_at_the_cut_go_to_the_earliest_positions() {
    // Two places left for four positions that score 1.
    let scores = [1.0, 5.0, 1.0, 1.0, 4.0, 1.0];
    assert_eq!(top_positions(&scores, 4), [0, 1, 2, 4]);
    // Zero and minus zero are one score.
    assert_eq!(top_positions(&[-0.0, 0.0, 0.0], 2), [0, 1]);
}

#[test]
fn candidate_blocks_are_the_best_blocks_and_always_the_newest() {
    // Blocks of 2 scored by their best position: 1, 9, 3, and a newest block
    // of one position that scores worst.
    let scores = [1.0, 0.0, 9.0, -2.0, 3.0, 3.0, -5.0];
    let kept = |blocks| {
        candidate_blocks(&scores, blocks, 2)
            .iter()
            .map(|&k| u8::from(k))
            .collect::<Vec<_>>()
    };
    assert_eq!(kept(2), [0, 0, 1, 1, 0, 0, 1]);
    assert_eq!(kept(3), [0, 0, 1, 1, 1, 1, 1]);
    // No more blocks than there are: every position.
    assert_eq!(kept(4), [1; 7]);
    assert_eq!(kept(2048), [1; 7]);
}

#[test]
fn equal_blocks_at_the_cut_go_to_the_earliest_and_unreachable_blocks_are_dropped() {
    // Three blocks score 2; with the newest pinned, one place is left.
    let scores = [2.0, 0.0, 2.0, 1.0, 2.0, 2.0, 0.0, 0.0];
    assert_eq!(
        candidate_blocks(&scores, 2, 2),
        [true, true, false, false, false, false, true, true]
    );
    // A block no position of which can be reached is not kept, even with
    // places to spare.
    let masked = [NONE, NONE, 1.0, NONE, 0.5, 0.25];
    assert_eq!(
        candidate_blocks(&masked, 3, 2),
        [false, false, true, true, true, true]
    );
}

/// Recorded index scores of layer 2, and what DeepSeek's reference lines keep
/// of them at a top-k of 64 and 16 candidate blocks of 8.
#[test]
fn recorded_scores_give_the_references_choices() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../test_data/deepseek_v41/index_fixture.json"
    ))
    .expect("fixture");
    let count = |key: &str| fixture[key].as_u64().expect("count") as usize;
    let (k, blocks, block) = (count("k"), count("topk_blocks"), count("block_size"));
    let list = |v: &serde_json::Value| -> Vec<i32> {
        v.as_array()
            .expect("list")
            .iter()
            .map(|i| i.as_i64().expect("position") as i32)
            .collect()
    };
    let rows = fixture["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 6);
    let mut tied = 0;
    for row in rows {
        let hex = row["scores"].as_str().expect("scores");
        let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex");
        let mut scores: Vec<f32> = (0..hex.len() / 4)
            .map(|i| bf16_to_f32(u16::from_le_bytes([byte(4 * i), byte(4 * i + 2)])))
            .collect();
        let query = &row["query"];
        tied += usize::from(row["tie_at_the_cut"].as_bool().expect("tie"));
        assert_eq!(top_positions(&scores, k), list(&row["chosen"]), "{query}");
        let kept = candidate_blocks(&scores, blocks, block);
        let want: Vec<bool> = row["candidates"]
            .as_str()
            .expect("candidates")
            .bytes()
            .map(|b| b == b'1')
            .collect();
        assert_eq!(kept, want, "{query}");
        keep_candidates(&mut scores, &kept);
        assert_eq!(
            top_positions(&scores, k),
            list(&row["chosen_among_candidates"]),
            "{query}"
        );
    }
    assert!(tied >= 3, "the fixture holds rows with a tie at the cut");
}
