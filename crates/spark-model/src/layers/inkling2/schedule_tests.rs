// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::parse_config;

const CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../atlas-core/test_data/inkling2_mini_config.json"
));

fn mini() -> Vec<Visit> {
    let config = parse_config(CONFIG).unwrap();
    schedule(config.inkling2.as_ref().unwrap())
}

#[test]
fn mini_runs_48_visits_over_24_layers() {
    let s = mini();
    assert_eq!(s.len(), 48);
    let physical: Vec<usize> = s.iter().map(|v| v.physical).collect();
    let mut want: Vec<usize> = (0..6).collect();
    for _ in 0..3 {
        want.extend(6..18);
    }
    want.extend(18..24);
    assert_eq!(physical, want);
    assert!(s[0].first && s[47].last && s[5].save_prelude);
    assert_eq!(
        s.iter()
            .filter(|v| v.first || v.last || v.save_prelude)
            .count(),
        3
    );
}

#[test]
fn mini_block_stack_follows_the_reference_loop() {
    let s = mini();
    // layer 0 pushes the embedding; the prelude ends with one block
    assert_eq!((s[0].blocks_before, s[0].boundary), (0, true));
    assert_eq!(s[5].blocks_before, 1);
    // first pass: pushes at layers 6 and 12
    assert_eq!(
        (s[6].blocks_before, s[12].blocks_before, s[17].blocks_before),
        (1, 2, 3)
    );
    // later passes restart from the prelude's single block
    for start in [18, 30] {
        assert!(s[start].restart && s[start].physical == 6);
        assert_eq!(s[start].blocks_before, 1);
        // messages read the previous pass's layer-12 push (B[2:] in the reference)
        assert_eq!(s[start].message_blocks, 1);
        assert_eq!(s[start + 11].message_blocks, 1);
    }
    assert_eq!(s.iter().filter(|v| v.restart).count(), 2);
    // coda: push at 18, four blocks before the final residual
    assert_eq!(
        (s[42].physical, s[42].boundary, s[42].blocks_before),
        (18, true, 3)
    );
    assert_eq!(s[47].blocks_before, 4);
    assert_eq!(max_blocks(&s), 4);
}

#[test]
fn mini_passes_and_loop_layers() {
    let s = mini();
    assert!(
        s[..6]
            .iter()
            .chain(&s[42..])
            .all(|v| v.pass == 0 && v.loop_layer.is_none())
    );
    for (i, v) in s[6..42].iter().enumerate() {
        assert_eq!((v.pass, v.loop_layer), (i / 12, Some(i % 12)));
        assert_eq!(v.message_blocks, if v.pass == 0 { 0 } else { 1 });
    }
    let full: Vec<usize> = s
        .iter()
        .enumerate()
        .filter(|(_, v)| v.full)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full, [5, 11, 17, 23, 29, 35, 41, 47]);
}
