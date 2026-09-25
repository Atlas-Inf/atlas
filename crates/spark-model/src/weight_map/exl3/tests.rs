// SPDX-License-Identifier: AGPL-3.0-only
// Portions of this file are derived from ExLlamaV3 (MIT, see NOTICE.md).

//! Cross-language checks against `bench/exl3/reference.py`, the NumPy oracle.
//!
//! The hashes are what make this more than a smoke test. The decode is a
//! bitstream, a codebook, a permutation and two Hadamard passes, and each of
//! those can go quietly wrong - a transposed tile, a double rounding, a
//! bit-reversed butterfly - in a way that still yields plausible weights. So
//! each stage is pinned on its own: states bit-exact, W_inner bit-exact, then
//! the assembled f32 weight to the tolerance the f32-vs-f64 Hadamard allows.

use base64::Engine as _;
use half::f16;

use super::Exl3Shape;
use super::cpu_ref::{decode_inner, mul1_decode, reconstruct, tile_states};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[derive(serde::Deserialize)]
struct Fixtures {
    mul1: Mul1,
    blocks: Vec<Block>,
}

#[derive(serde::Deserialize)]
struct Mul1 {
    table_fnv1a64: String,
    spots: Vec<[u64; 2]>,
}

#[derive(serde::Deserialize)]
struct Block {
    name: String,
    bits: u32,
    in_features: usize,
    out_features: usize,
    trellis_i16_b64: String,
    suh_f16_b64: String,
    svh_f16_b64: String,
    states_fnv1a64: String,
    inner_fnv1a64: String,
    inner_spots: Vec<[u64; 3]>,
    w_spots: Vec<[f64; 3]>,
    w_sum: f64,
    w_sumsq: f64,
}

fn fixtures() -> Fixtures {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/exl3/fixtures.json"
    );
    let raw = std::fs::read_to_string(path).expect("regenerate with bench/exl3/make_fixtures.py");
    serde_json::from_str(&raw).expect("fixtures parse")
}

/// Little-endian lanes of a base64 blob (the fixtures store raw tensor bytes).
fn lanes(b64: &str) -> Vec<u16> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .expect("fixture base64");
    raw.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn halves(b64: &str) -> Vec<f16> {
    lanes(b64).iter().copied().map(f16::from_bits).collect()
}

fn shape(b: &Block) -> Exl3Shape {
    Exl3Shape {
        in_features: b.in_features,
        out_features: b.out_features,
        bits: b.bits,
    }
}

/// Every one of the 65536 mul1 codebook entries, bit-exact. The table is shared
/// by every tensor and every later kernel, so one wrong rounding here shows up
/// as a uniform bias in the model rather than as an error.
#[test]
fn mul1_table_is_bit_exact() {
    let f = fixtures();
    let mut table = Vec::with_capacity(2 * 65536);
    for state in 0..=u16::MAX {
        table.extend_from_slice(&mul1_decode(state).to_le_bytes());
    }
    assert_eq!(
        format!("{:016x}", fnv1a64(&table)),
        f.mul1.table_fnv1a64,
        "the 65536-entry mul1 table disagrees with the oracle"
    );
    for &[state, want] in &f.mul1.spots {
        let got = mul1_decode(state as u16).to_bits();
        assert_eq!(got as u64, want, "state {state:#x}");
    }
}

/// Eight real checkpoint blocks, bits 3 to 6, square and non-square.
#[test]
fn checkpoint_blocks_decode_bit_exact() {
    let f = fixtures();
    assert_eq!(f.blocks.len(), 8, "the fixture should carry 8 real blocks");
    for b in &f.blocks {
        let trellis = lanes(&b.trellis_i16_b64);
        let suh = halves(&b.suh_f16_b64);
        let svh = halves(&b.svh_f16_b64);
        let sh = shape(b);
        let words = sh.words_per_tile();
        assert_eq!(
            trellis.len() % words,
            0,
            "{}: trellis is not whole tiles",
            b.name
        );

        // States: tile row i outer, tile column j inner (trellis order), then
        // t = 0..256 within a tile.
        let mut hashed = Vec::with_capacity(trellis.len() / words * 512);
        for chunk in trellis.chunks_exact(words) {
            for state in tile_states(chunk, b.bits) {
                hashed.extend_from_slice(&state.to_le_bytes());
            }
        }
        assert_eq!(
            format!("{:016x}", fnv1a64(&hashed)),
            b.states_fnv1a64,
            "{}: extracted states",
            b.name
        );

        let inner = decode_inner(&trellis, &sh).unwrap();
        let inner_bytes: Vec<u8> = inner.iter().flat_map(|&v| v.to_le_bytes()).collect();
        assert_eq!(
            format!("{:016x}", fnv1a64(&inner_bytes)),
            b.inner_fnv1a64,
            "{}: W_inner",
            b.name
        );
        for &[r, c, want] in &b.inner_spots {
            assert_eq!(
                inner[r as usize * b.out_features + c as usize].to_bits() as u64,
                want,
                "{}: W_inner[{r}][{c}]",
                b.name
            );
        }

        let w = reconstruct(&trellis, &suh, &svh, &sh).unwrap();
        // The oracle's Hadamard runs in f64 and this one in f32, so the weight
        // is pinned to the f32 round-off of an O(1) value, not bit-exact.
        let rms = (b.w_sumsq / (b.in_features * b.out_features) as f64).sqrt() as f32;
        for &[r, c, want] in &b.w_spots {
            let got = w[r as usize * b.out_features + c as usize];
            assert!(
                (got - want as f32).abs() <= 1e-4 * rms + 1e-7,
                "{}: W[{r}][{c}] = {got}, oracle {want}, rms {rms}",
                b.name
            );
        }
        let sum = w.iter().map(|&v| v as f64).sum::<f64>();
        let sumsq = w.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        assert!(
            (sumsq - b.w_sumsq).abs() <= 1e-4 * b.w_sumsq.abs(),
            "{}: w_sumsq = {sumsq}, oracle {}",
            b.name,
            b.w_sumsq
        );
        // A sum can land near zero, so scale the w_sum tolerance by the RMS,
        // not by the (possibly vanishing) sum itself.
        assert!(
            (sum - b.w_sum).abs() <= 1e-4 * b.w_sumsq.sqrt(),
            "{}: w_sum = {sum}, oracle {}",
            b.name,
            b.w_sum
        );
    }
}

#[test]
fn bad_trellis_dims_are_refused() {
    let err = Exl3Shape::from_trellis_dims(&[8, 8]).unwrap_err();
    assert!(format!("{err:#}").contains("rank 3"), "{err:#}");

    for dims in [vec![8usize, 8, 30], vec![8, 8, 0]] {
        let err = Exl3Shape::from_trellis_dims(&dims).unwrap_err();
        assert!(format!("{err:#}").contains("multiple of 16"), "{err:#}");
    }

    let err = Exl3Shape::from_trellis_dims(&[8, 8, 144]).unwrap_err();
    assert!(format!("{err:#}").contains("1..=8"), "{err:#}");

    for dims in [vec![0usize, 8, 64], vec![8, 0, 64]] {
        let err = Exl3Shape::from_trellis_dims(&dims).unwrap_err();
        assert!(format!("{err:#}").contains("zero tile dim"), "{err:#}");
    }

    let ok = Exl3Shape::from_trellis_dims(&[8, 8, 64]).unwrap();
    assert_eq!(
        ok,
        Exl3Shape {
            in_features: 128,
            out_features: 128,
            bits: 4
        }
    );
    assert!(ok.check_scales(128, 128).is_ok());
    assert!(format!("{}", ok.check_scales(64, 128).unwrap_err()).contains("suh"));
    assert!(format!("{}", ok.check_scales(128, 130).unwrap_err()).contains("svh"));
}

/// `validate` is what stops a hand-built shape from silently truncating the
/// decode or tripping `tile_states`; each rejection must name the value.
#[test]
fn invalid_hand_built_shapes_are_refused() {
    for bits in [0u32, 9] {
        let sh = Exl3Shape {
            in_features: 128,
            out_features: 128,
            bits,
        };
        let err = sh.validate().unwrap_err();
        assert!(format!("{err:#}").contains(&bits.to_string()), "{err:#}");
        let err = decode_inner(&vec![0u16; 8 * 8 * 16 * 4], &sh).unwrap_err();
        assert!(
            format!("{err:#}").contains("outside the codebook range"),
            "{err:#}"
        );
    }

    let sh = Exl3Shape {
        in_features: 100,
        out_features: 128,
        bits: 4,
    };
    let err = sh.validate().unwrap_err();
    assert!(format!("{err:#}").contains("in_features 100"), "{err:#}");
    // Without validate this would silently decode 6 of the 6.25 tile rows.
    let err = decode_inner(&vec![0u16; 100 * 128], &sh).unwrap_err();
    assert!(format!("{err:#}").contains("in_features 100"), "{err:#}");

    let sh = Exl3Shape {
        in_features: 128,
        out_features: 0,
        bits: 4,
    };
    let err = sh.validate().unwrap_err();
    assert!(format!("{err:#}").contains("out_features 0"), "{err:#}");

    assert!(
        Exl3Shape {
            in_features: 128,
            out_features: 128,
            bits: 4
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn ragged_features_and_wrong_lengths_are_refused() {
    let f = fixtures();
    let b = &f.blocks[0];
    let trellis = lanes(&b.trellis_i16_b64);
    let suh = halves(&b.suh_f16_b64);
    let svh = halves(&b.svh_f16_b64);
    let sh = shape(b);

    // Feature counts that are not multiples of 128 cannot come from a
    // block-diagonal 128 Hadamard; refuse rather than guess a padding. The
    // scales are made to match so the length check is not what fires.
    let half = Exl3Shape {
        in_features: 64,
        out_features: 128,
        bits: b.bits,
    };
    let err = reconstruct(&trellis[..trellis.len() / 2], &suh[..64], &svh, &half).unwrap_err();
    assert!(
        format!("{err:#}").contains("multiple of the 128"),
        "{err:#}"
    );

    // 144 = 9 tile columns is a multiple of 16 but not of the 128-wide
    // Hadamard block; refuse rather than guess a padding. The scales match, so
    // the length checks are not what fires.
    let ragged = Exl3Shape {
        in_features: 128,
        out_features: 144,
        bits: b.bits,
    };
    let err = reconstruct(&trellis, &suh, &svh, &ragged).unwrap_err();
    assert!(
        format!("{err:#}").contains("multiple of the 128"),
        "{err:#}"
    );

    let err = decode_inner(&trellis[..trellis.len() - 1], &sh).unwrap_err();
    assert!(format!("{err:#}").contains("trellis has"), "{err:#}");
    let err = reconstruct(&trellis[..trellis.len() - 1], &suh, &svh, &sh).unwrap_err();
    assert!(format!("{err:#}").contains("trellis has"), "{err:#}");
    assert!(
        format!(
            "{}",
            reconstruct(&trellis, &suh[..100], &svh, &sh).unwrap_err()
        )
        .contains("suh")
    );
    assert!(
        format!(
            "{}",
            reconstruct(&trellis, &suh, &svh[..100], &sh).unwrap_err()
        )
        .contains("svh")
    );
}

/// The Hadamard pass must scale by exactly the f32 nearest 1/sqrt(128); a
/// one-ulp-off literal would bias every reconstructed weight.
#[test]
fn had_scale_is_the_exact_f32() {
    assert_eq!(super::cpu_ref::HAD_SCALE, 1.0f32 / (128.0f32).sqrt());
}
