// SPDX-License-Identifier: AGPL-3.0-only

//! What a rank reads from its environment for the GLM KV shard, and what
//! the ranks of a pair must agree on.

use super::super::*;

#[test]
fn flag_is_explicit() {
    assert!(!parse("F", None).unwrap());
    assert!(!parse("F", Some("0")).unwrap());
    assert!(parse("F", Some("1")).unwrap());
    for bad in ["", "true", "2", " 1"] {
        assert!(parse("F", Some(bad)).is_err());
    }
}

#[test]
fn merge_tuning_is_explicit_and_the_check_keeps_exchanges_inline() {
    let t = |c, o, check| MergeTuning::parse(c, o, check, true).unwrap();
    assert_eq!(t(None, None, None), MergeTuning::default());
    assert_eq!(t(Some("0"), Some("0"), Some("0")), MergeTuning::default());
    let both = MergeTuning {
        compact: true,
        overlap: true,
        check: false,
    };
    assert_eq!(t(Some("1"), Some("1"), None), both);
    // The check's own exchange would fall inside an overlap window.
    let checked = MergeTuning {
        overlap: false,
        check: true,
        ..both
    };
    assert_eq!(t(Some("1"), Some("1"), Some("1")), checked);
    // A mistyped value is an error, never an arm that silently runs without it.
    for junk in [(Some("yes"), None, None), (None, Some("2"), None)] {
        assert!(MergeTuning::parse(junk.0, junk.1, junk.2, true).is_err());
    }
    let err = MergeTuning::parse(None, None, Some("true"), true).unwrap_err();
    assert!(
        err.to_string().contains("ATLAS_GLM_KV_SHARD_CHECK"),
        "{err}"
    );
}

#[test]
fn a_tuning_without_the_shard_fails_instead_of_being_ignored() {
    let unsharded = |c, o, check| MergeTuning::parse(c, o, check, false);
    let off = unsharded(None, Some("0"), Some("0")).unwrap();
    assert_eq!(off, MergeTuning::default());
    for (c, o, check) in [
        (Some("1"), None, None),
        (None, Some("1"), None),
        (None, None, Some("1")),
        (Some("junk"), None, None),
        // The check does not hide an overlap request.
        (None, Some("1"), Some("1")),
    ] {
        let err = unsharded(c, o, check).unwrap_err().to_string();
        assert!(
            err.contains("ATLAS_GLM_KV_SHARD"),
            "{c:?} {o:?} {check:?}: {err}"
        );
    }
}

#[test]
fn lanes_without_a_sharded_reader_are_named_for_refusal() {
    let env = |set: &'static [(&'static str, &'static str)]| {
        move |name: &str| {
            let hit = set.iter().find(|(n, _)| *n == name);
            hit.map(|(_, v)| v.to_string())
        }
    };
    // The production profile spells its lanes out as 0.
    assert_eq!(unsharded_lane(env(&[])), None);
    let profile = &[
        ("ATLAS_GLM_MTP_REPAIR", "0"),
        ("ATLAS_GLM_C4_DECODE", "0"),
        ("ATLAS_GLM_MLA_MULTI_SEQ", "1"),
        ("ATLAS_GLM_LONG_BATCH_VERIFY", "1"),
        ("ATLAS_GLM_LONG_BATCH_SERIAL", ""),
    ];
    assert_eq!(unsharded_lane(env(profile)), None);
    for lane in UNSHARDED_LANES {
        let on = |name: &str| (name == lane).then(|| "1".to_string());
        assert_eq!(unsharded_lane(on), Some(lane));
    }
    let serial = &[("ATLAS_GLM_LONG_BATCH_SERIAL", "mla")];
    assert_eq!(
        unsharded_lane(env(serial)),
        Some("ATLAS_GLM_LONG_BATCH_SERIAL")
    );
}

#[test]
fn unsharded_ranks_gather_plain_block_counts_and_take_the_minimum() {
    assert_eq!(settings_word(false).unwrap(), 0);
    assert_eq!(blocks_word(177_000, 0).unwrap(), 177_000);
    // A count that would reach the settings bits is refused, not misread.
    assert!(blocks_word(1 << 28, 0).is_err());
    assert_eq!(blocks_word((1 << 28) - 1, 0).unwrap(), (1 << 28) - 1);
    assert_eq!(
        agreed_blocks(0, 177_000, &[177_000, 150_000]).unwrap(),
        150_000
    );
    assert_eq!(
        agreed_blocks(1, 150_000, &[177_000, 150_000]).unwrap(),
        150_000
    );
}

#[test]
fn ranks_must_agree_on_the_shard_and_its_tunings() {
    let plain = MergeTuning::default();
    let compact = MergeTuning {
        compact: true,
        ..plain
    };
    let checked = MergeTuning {
        check: true,
        ..plain
    };
    let word = |blocks, tuning| blocks_word(blocks, settings_bits(tuning)).unwrap();
    let (a, b) = (word(300_000, plain), word(280_000, plain));
    assert_eq!(agreed_blocks(0, a, &[a, b]).unwrap(), 280_000);
    // One rank sharded, one tuning or the check on one rank only: both fail.
    for peer in [
        blocks_word(280_000, 0).unwrap(),
        word(280_000, compact),
        word(280_000, checked),
    ] {
        for (rank, ours) in [(0, a), (1, peer)] {
            let err = agreed_blocks(rank, ours, &[a, peer])
                .unwrap_err()
                .to_string();
            assert!(err.contains("ATLAS_GLM_KV_SHARD settings"), "{err}");
            assert!(err.contains(&format!("rank {rank} has")), "{err}");
        }
    }
}
