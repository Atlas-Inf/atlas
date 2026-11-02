// SPDX-License-Identifier: AGPL-3.0-only
// Portions of this file are derived from ExLlamaV3 (MIT, see NOTICE.md).

//! The `bench/exl3/fixtures.json` loader, shared by this module's CPU tests and
//! by the GPU parity tests in `layers/ops/exl3_gpu_tests.rs` (which decode the
//! same blocks on the device and compare against `decode_inner`).

use base64::Engine as _;
use half::f16;

use super::Exl3Shape;

#[derive(serde::Deserialize)]
pub(crate) struct Fixtures {
    pub(crate) mul1: Mul1,
    pub(crate) blocks: Vec<Block>,
}

#[derive(serde::Deserialize)]
pub(crate) struct Mul1 {
    pub(crate) table_fnv1a64: String,
    pub(crate) spots: Vec<[u64; 2]>,
}

#[derive(serde::Deserialize)]
pub(crate) struct Block {
    pub(crate) name: String,
    pub(crate) bits: u32,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) trellis_i16_b64: String,
    pub(crate) suh_f16_b64: String,
    pub(crate) svh_f16_b64: String,
    pub(crate) states_fnv1a64: String,
    pub(crate) inner_fnv1a64: String,
    pub(crate) inner_spots: Vec<[u64; 3]>,
    pub(crate) w_spots: Vec<[f64; 3]>,
    pub(crate) w_sum: f64,
    pub(crate) w_sumsq: f64,
}

pub(crate) fn fixtures() -> Fixtures {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/exl3/fixtures.json"
    );
    let raw = std::fs::read_to_string(path).expect("regenerate with bench/exl3/make_fixtures.py");
    serde_json::from_str(&raw).expect("fixtures parse")
}

/// Little-endian lanes of a base64 blob (the fixtures store raw tensor bytes).
pub(crate) fn lanes(b64: &str) -> Vec<u16> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .expect("fixture base64");
    raw.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

pub(crate) fn halves(b64: &str) -> Vec<f16> {
    lanes(b64).iter().copied().map(f16::from_bits).collect()
}

pub(crate) fn shape(b: &Block) -> Exl3Shape {
    Exl3Shape {
        in_features: b.in_features,
        out_features: b.out_features,
        bits: b.bits,
    }
}
