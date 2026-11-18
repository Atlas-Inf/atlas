// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
#[test]
fn fp8_lookup_and_four_channel_injection_match_cpu_torch_fixture() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../engram_fixture.json")).unwrap();
    let vector = |name: &str| -> Vec<f32> {
        fixture[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let bytes: Vec<u8> = fixture["row_bytes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    assert_eq!(
        fp8_row(
            &bytes,
            fixture["row_scale"].as_f64().unwrap() as f32,
            &vector("row_gamma"),
            true
        )
        .unwrap(),
        vector("lookup")
    );
    assert_eq!(
        inject(
            &vector("hidden"),
            &vector("keys"),
            &vector("value"),
            &vector("key_norm"),
            &vector("query_norm"),
            &vector("pre_gamma"),
            &vector("scales"),
            true
        )
        .unwrap(),
        vector("output")
    );
}
#[test]
fn numpy_seed_sequence_and_pcg64_raw_draws_are_exact() {
    assert_eq!(
        multipliers(0, 2, 120818, 3).unwrap(),
        [62749575316035, 30426722171175, 18574837598557]
    );
    assert_eq!(
        multipliers(0, 12, 120818, 3).unwrap(),
        [14916611332703, 34290019069965, 64992498366675]
    );
    assert_eq!(
        multipliers(1234, 32, 2000000000, 3).unwrap(),
        [1617354049, 2316058337, 1445165527]
    );
    assert_eq!(
        rng::seed_sequence(42),
        [
            11465652750463011511,
            15382171918060459190,
            9018504550953525431,
            3703499796004394495
        ]
    );
    let mut g = rng::Pcg64::new(42);
    assert_eq!(
        (0..4).map(|_| g.raw()).collect::<Vec<_>>(),
        [
            14276969152011380360,
            8095878257575067585,
            15838336090824644132,
            12864169557245331597
        ]
    );
}
#[test]
fn partition_preserves_heads_and_balances_rows_with_lower_boundary_ties() {
    assert_eq!(head_range(&[2, 8, 2, 2], 0, 2).unwrap(), 0..2);
    assert_eq!(head_range(&[2, 8, 2, 2], 1, 2).unwrap(), 2..4);
    assert_eq!(head_range(&[2, 2, 2], 0, 2).unwrap(), 0..1);
    assert!(head_range(&[], 0, 1).is_err());
    assert!(head_range(&[1], 0, 2).is_err());
}
#[test]
fn hashes_are_request_local_and_chunk_invariant_with_dead_and_padding_history() {
    let l = HashLayout {
        map: vec![0, 2, 1, 3],
        pad: 3,
        heads_per_order: 2,
        multipliers: vec![3, 5, 7],
        sizes: vec![11, 13, 17, 19],
    };
    assert_eq!(l.hash(&[], 1).unwrap(), [9, 20, 35, 50]);
    assert_eq!(l.hash(&[1], 2).unwrap(), [9, 20, 35, 50]);
    assert_eq!(l.hash(&[1, 2], 0).unwrap(), [5, 16, 35, 52]);
    assert_eq!(l.hash(&[999, 1], 2).unwrap(), l.hash(&[-1, 1], 2).unwrap());
    assert_eq!(l.hash(&[], 3).unwrap(), [-1; 4]);
    assert_eq!(l.hash(&[], 99).unwrap(), [-1; 4]);
    let mut bad = l.clone();
    bad.multipliers[0] = u64::MAX;
    assert!(bad.hash(&[], 0).is_err());
}
#[test]
fn row_scale_is_a_multiplier_and_invalid_hashes_do_not_inject() {
    assert_eq!(fp8_row(&[56, 56], 2., &[1., 1.], true).unwrap(), [1., 1.]);
    assert_eq!(
        fp8_row(&[127, 127], 2., &[1., 1.], false).unwrap(),
        [0., 0.]
    );
    assert!(fp8_row(&[127], 1., &[1.], true).is_err());
    let hidden = vec![1.; 8];
    let key = vec![0.; 8];
    let result = inject(
        &hidden,
        &key,
        &[2., 2.],
        &[1.; 8],
        &[1.; 8],
        &[1.; 2],
        &[0.; 8],
        true,
    )
    .unwrap();
    assert_eq!(result, [1.5; 8]);
    assert_eq!(
        inject(
            &hidden,
            &key,
            &[2., 2.],
            &[1.; 8],
            &[1.; 8],
            &[1.; 2],
            &[0.; 8],
            false
        )
        .unwrap(),
        hidden
    );
    assert!(inject(&[], &[], &[], &[], &[], &[], &[], true).is_err());
    assert!(multipliers(0, 2, 0, 3).is_err());
}
