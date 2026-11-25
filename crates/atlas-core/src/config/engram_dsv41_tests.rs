// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The Engram fields of deepseek-ai/DeepSeek-V4.1-Flash-1017's config.json.
fn published() -> DeepseekV41Engram {
    DeepseekV41Engram {
        layer_ids: vec![1, 14],
        num_embeddings: vec![384_006_168, 384_016_682],
        max_ngram_size: 4,
        n_heads: 8,
        head_dim: 256,
        vocab_size: 16_000_000,
        compressed_vocab_size: 99_092,
        // `engram_pad_token_id` 2 maps to compressed id 2 in the 1017 tokenizer.
        pad_id: 2,
    }
}

/// `numpy.random.default_rng(seed).integers(0, bound, size=n, dtype=np.int64)`,
/// recorded with numpy 2.4.3.
#[test]
fn draws_match_numpy_default_rng() {
    let cases: [(u64, u64, &[u64]); 5] = [
        (10007, 46539438283891, &[38316048023122]),
        (140098, 46539438283891, &[33858405369630]),
        (
            0,
            1 << 62,
            &[
                2937467307694268567,
                1244171615822312904,
                188957027462249006,
                76220265684581383,
                3750546991322993742,
                4209342133973288723,
            ],
        ),
        (
            1,
            (1 << 33) + 7,
            &[4396514282, 8164420990, 1238321645, 8148836708, 2678611778],
        ),
        (
            123456789,
            i64::MAX as u64,
            &[
                255604904563013789,
                8362831937566009189,
                8129420665881559387,
                5763660056023947074,
            ],
        ),
    ];
    for (seed, bound, expected) in cases {
        let mut rng = Pcg64::from_seed(seed);
        let got: Vec<u64> = expected.iter().map(|_| rng.below(bound)).collect();
        assert_eq!(got, expected, "seed {seed} bound {bound}");
    }
}

#[test]
fn prime_buckets_tile_the_published_tables() {
    let e = published();
    e.validate(40).expect("the published config is consistent");
    assert_eq!(&e.primes(0)[..3], &[16_000_057, 16_000_079, 16_000_081]);
    assert_eq!(*e.primes(1).last().unwrap(), 16_000_889);
    let all: Vec<u64> = (0..2).flat_map(|i| e.primes(i)).collect();
    assert!(
        all.windows(2).all(|w| w[0] < w[1]),
        "no prime is handed out twice"
    );
}

#[test]
fn published_multipliers() {
    let e = published();
    assert_eq!(
        e.multipliers(0),
        [
            76632096046245,
            4839876093313,
            35959672319349,
            73987337458391
        ]
    );
    assert_eq!(
        e.multipliers(1),
        [
            67716810739261,
            51510806800915,
            30921347202721,
            82619226485591
        ]
    );
}

#[test]
fn a_table_size_that_disagrees_is_refused() {
    let mut e = published();
    e.num_embeddings[1] += 1;
    let err = e.validate(40).unwrap_err().to_string();
    assert!(err.contains("Engram layer 14"), "{err}");
    let mut e = published();
    e.layer_ids = vec![1, 40];
    assert!(e.validate(40).is_err());
}

/// Compressed ids of a 30-token English/French prompt under the 1017
/// tokenizer, and row ids from a transcription of the reference
/// `NgramHashState` over them.
const COMPRESSED: [u32; 30] = [
    237, 4015, 9933, 18307, 14, 237, 335, 994, 9933, 18307, 29, 237, 174, 4015, 36, 4006, 18307,
    16, 39, 54, 39, 35, 9055, 28, 44373, 44373, 5405, 40, 39, 16,
];

#[test]
fn row_ids_match_the_reference() {
    let e = published();
    let tokens: Vec<Option<u32>> = COMPRESSED.iter().copied().map(Some).collect();
    let layer0 = e.row_ids(0, &tokens);
    let layer1 = e.row_ids(1, &tokens);
    // every 8th column: the first head of the 2-, 3- and 4-gram
    let firsts = |row: &[u32]| [row[0], row[8], row[16]];
    assert_eq!(
        firsts(&layer0[0]),
        [5184683, 136968925, 268680887],
        "sequence start"
    );
    assert_eq!(firsts(&layer0[1]), [15897228, 131033500, 269121502]);
    assert_eq!(
        firsts(&layer0[3]),
        [3548953, 142773829, 268216897],
        "first full 4-gram"
    );
    assert_eq!(firsts(&layer0[29]), [2554292, 140551556, 260465386]);
    assert_eq!(
        layer0[12],
        [
            6822039, 17782449, 46417638, 51511744, 75198720, 93773326, 103062917, 115310490,
            143751683, 146324736, 175345306, 176144348, 206387075, 213527776, 232263308, 253966838,
            271908129, 274940174, 299824443, 304840725, 324398032, 346052877, 358755107, 382172179,
        ]
    );
    assert_eq!(
        [
            layer1[0][23],
            layer1[1][23],
            layer1[2][23],
            layer1[3][23],
            layer1[12][23],
            layer1[29][23]
        ],
        [
            370616412, 380642243, 381766319, 376813552, 369342920, 381851844
        ]
    );
    assert!(
        layer1
            .iter()
            .flatten()
            .all(|&id| (id as u64) < e.num_embeddings[1])
    );
}

#[test]
fn an_image_span_breaks_every_ngram_that_reaches_it() {
    let e = published();
    let tokens: Vec<Option<u32>> = COMPRESSED
        .iter()
        .enumerate()
        .map(|(i, &t)| (!(5..10).contains(&i)).then_some(t))
        .collect();
    let ids = e.row_ids(0, &tokens);
    let firsts = |row: &[u32]| [row[0], row[8], row[16]];
    // inside the span every slot is pad
    for (p, row) in ids.iter().enumerate().take(10).skip(5) {
        assert_eq!(firsts(row), [4299726, 129401291, 264341430], "position {p}");
    }
    // just after it, the n-grams start over as at the beginning of a sequence
    assert_eq!(firsts(&ids[10]), [2171627, 133359380, 268063980]);
    assert_eq!(firsts(&ids[11]), [5049114, 143455168, 263603238]);
    assert_eq!(firsts(&ids[12]), [6822039, 143751683, 269042145]);
    assert_eq!(firsts(&ids[13]), [5778927, 136198807, 256335570]);
}
