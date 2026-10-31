// SPDX-License-Identifier: AGPL-3.0-only

//! `csf_32x2_v1` decoder tests.
//!
//! `golden_scales_decode_byte_identical_to_the_reference` is the one that
//! matters: three real scales range-read from nvidia/GLM-5.3-Flash-FP4-Local
//! @ 76348ddb, decoded by the LIL FlashInfer reference
//! (`scripts/nvfp4_csf32_golden.py`), compared by sha256 of the full matrix,
//! plus one synthetic odd-`C` matrix encoded by the reference ENCODER.

use super::*;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

pub(crate) struct Golden {
    pub tag: String,
    /// bases, codes, page_bases, offsets, exceptions.
    pub parts: [Vec<u8>; 5],
    pub case: serde_json::Value,
}

impl Golden {
    pub fn parts(&self) -> Csf32Parts<'_> {
        view(&self.parts)
    }

    pub fn shape(&self) -> [usize; 2] {
        let s = self.case["decoded_shape"].as_array().unwrap();
        [
            s[0].as_u64().unwrap() as usize,
            s[1].as_u64().unwrap() as usize,
        ]
    }

    pub fn sha(&self) -> &str {
        self.case["decoded_sha256"].as_str().unwrap()
    }
}

pub(crate) fn view(p: &[Vec<u8>; 5]) -> Csf32Parts<'_> {
    Csf32Parts {
        bases: &p[0],
        codes: &p[1],
        page_bases: &p[2],
        offsets: &p[3],
        exceptions: &p[4],
    }
}

pub(crate) fn golden() -> Vec<Golden> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/nvfp4_csf32");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("golden.json")).unwrap()).unwrap();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let tag = c["tag"].as_str().unwrap().to_string();
            let read = |what: &str| std::fs::read(dir.join(format!("{tag}.{what}.bin"))).unwrap();
            Golden {
                parts: ["bases", "codes", "page_bases", "offsets", "exceptions"].map(read),
                tag,
                case: c.clone(),
            }
        })
        .collect()
}

pub(crate) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The golden script's `synthetic()` formula: a per-row window with sparse
/// outliers.
fn synthetic(rows: usize, cols: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out.push(if (r * 131 + c * 7).is_multiple_of(19) {
                ((r + 3 * c) % 0x7F) as u8
            } else {
                (0x28 + (r * 7 + c * 3) % 16 + r % 5) as u8
            });
        }
    }
    out
}

#[test]
fn golden_scales_decode_byte_identical_to_the_reference() {
    let cases = golden();
    assert_eq!(cases.len(), 4, "fixture set changed");
    for g in &cases {
        let (out, shape) =
            decode_nvfp4_csf32(&g.parts()).unwrap_or_else(|e| panic!("{}: {e:#}", g.tag));
        assert_eq!(shape, g.shape(), "{}", g.tag);
        // Localise a failure before the hash says only "different".
        let head: Vec<u8> = g.case["head"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(&out[..head.len()], &head[..], "{}: first bytes", g.tag);
        for s in g.case["exception_samples"].as_array().unwrap() {
            let (pos, val) = (
                s[0].as_u64().unwrap() as usize,
                s[1].as_u64().unwrap() as u8,
            );
            assert_eq!(out[pos], val, "{}: exception at {pos}", g.tag);
        }
        assert_eq!(
            hex(&Sha256::digest(&out)),
            g.sha(),
            "{}: differs from the reference",
            g.tag
        );
        if g.tag.starts_with("syn_") {
            assert_eq!(out, synthetic(shape[0], shape[1]), "{}", g.tag);
        }
        let codes_shape = [g.parts[1].len()];
        assert_eq!(decoded_len_of_codes(&codes_shape), Some(out.len()));
    }
}

pub(crate) fn decoded_len_of_codes(shape: &[usize]) -> Option<usize> {
    super::super::nvfp4_csf::decoded_len(super::super::nvfp4_csf::CsfPart::Codes, shape)
}

/// Decoding a mutated golden must not reproduce the reference: either it is
/// refused or the hash differs.
fn must_not_match(g: &Golden, parts: [Vec<u8>; 5], what: &str) {
    if let Ok((out, _)) = decode_nvfp4_csf32(&view(&parts)) {
        assert_ne!(hex(&Sha256::digest(&out)), g.sha(), "{what} went unnoticed");
    }
}

#[test]
fn mutated_goldens_do_not_match() {
    for g in &golden() {
        // Nibble order swapped in every code byte.
        let mut p = g.parts.clone();
        p[1].iter_mut().for_each(|b| *b = b.rotate_left(4));
        must_not_match(g, p, "a nibble-order swap");
        // One exception record's position off by one.
        let mut p = g.parts.clone();
        p[4][0] ^= 1;
        must_not_match(g, p, "an exception record off by one");
        // The records read one later (page origins shifted a record).
        let mut p = g.parts.clone();
        p[4].rotate_left(2);
        must_not_match(g, p, "a record stream shifted by one");
    }
}

// ---- The reference encoder, re-expressed ------------------------------------

/// `_geometry` from the reference `encode_csf.py`, ported literally (a grid of
/// block ids, tiles taken as transposed 4x2 slices), independently of
/// [`Geometry`]: `(order, starts, pages, first)`.
fn ref_geometry(rows: usize, cols: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let (nr, nc) = (rows / 32, cols.div_ceil(2));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for r in (0..nr).step_by(4) {
        for c in (0..nc).step_by(2) {
            let mut g = Vec::new();
            for j in c..(c + 2).min(nc) {
                for i in r..(r + 4).min(nr) {
                    g.push(i * nc + j);
                }
            }
            groups.push(g);
        }
    }
    let blocks: Vec<usize> = groups.concat();
    let first: Vec<usize> = blocks
        .iter()
        .map(|&b| (b / nc) * 32 * cols + b % nc * 2)
        .collect();
    let width: Vec<usize> = first.iter().map(|&f| 2.min(cols - f % cols)).collect();
    let mut order = Vec::new();
    for (&f, &w) in first.iter().zip(&width) {
        for local in 0..64 {
            if local % 2 < w {
                order.push(f + local / 2 * cols + local % 2);
            }
        }
    }
    let mut starts = vec![0];
    for w in &width {
        starts.push(starts.last().unwrap() + 32 * w);
    }
    let mut group_offsets = vec![0];
    for g in &groups {
        group_offsets.push(group_offsets.last().unwrap() + g.len());
    }
    let mut pages: Vec<usize> = group_offsets[..group_offsets.len() - 1]
        .iter()
        .step_by(32)
        .copied()
        .collect();
    pages.push(*group_offsets.last().unwrap());
    (order, starts, pages, first)
}

/// `encode_scales(source, "nvfp4")`: each row's base is the 16-wide window
/// covering the most bytes (ties to the smallest base); bytes outside it are
/// exceptions.
pub(crate) fn encode(src: &[u8], rows: usize, cols: usize) -> [Vec<u8>; 5] {
    let (order, starts, pages, _) = ref_geometry(rows, cols);
    let mut bases = vec![0u8; rows];
    for (r, base) in bases.iter_mut().enumerate() {
        let mut hist = [0usize; 256];
        src[r * cols..(r + 1) * cols]
            .iter()
            .for_each(|&b| hist[b as usize] += 1);
        let (mut best, mut best_n) = (0, 0);
        for b in 0..=240 {
            let n: usize = hist[b..b + 16].iter().sum();
            if n > best_n {
                (best, best_n) = (b, n);
            }
        }
        *base = best as u8;
    }
    let outside = |pos: usize| {
        let (v, b) = (src[pos], bases[pos / cols]);
        v < b || v - b >= 16
    };
    let fields: Vec<u8> = order
        .iter()
        .map(|&p| {
            if outside(p) {
                0
            } else {
                src[p] - bases[p / cols]
            }
        })
        .collect();
    let codes: Vec<u8> = fields.chunks_exact(2).map(|f| f[0] | (f[1] << 4)).collect();
    let mut records = Vec::new();
    let mut prefix = vec![0usize];
    for blk in starts.windows(2) {
        let mut n = 0;
        for &p in &order[blk[0]..blk[1]] {
            if outside(p) {
                let local = (p / cols % 32 * 2 + p % cols % 2) as u16;
                records.push(local | (u16::from(src[p]) << 8));
                n += 1;
            }
        }
        prefix.push(prefix.last().unwrap() + n);
    }
    let page_bases: Vec<u8> = pages[..pages.len() - 1]
        .iter()
        .flat_map(|&a| (2 * prefix[a] as u32).to_le_bytes())
        .collect();
    let offsets: Vec<u8> = pages
        .windows(2)
        .flat_map(|w| (w[0]..=w[1]).map(move |i| (i, w[0])))
        .flat_map(|(i, a)| ((prefix[i] - prefix[a]) as u16).to_le_bytes())
        .collect();
    let exceptions: Vec<u8> = records.iter().flat_map(|r| r.to_le_bytes()).collect();
    [bases, codes, page_bases, offsets, exceptions]
}

/// Deterministic scales: a per-row window with outliers at `1/density`.
fn random(rows: usize, cols: usize, seed: u64, density: u32) -> Vec<u8> {
    let mut s = seed;
    (0..rows * cols)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (s >> 33) as u32;
            if r.is_multiple_of(density) {
                (r >> 8) as u8 % 0x7F
            } else {
                0x20 + (i / cols % 40) as u8 + (r % 16) as u8
            }
        })
        .collect()
}

#[test]
fn the_ported_encoder_matches_the_reference_encoder() {
    // The synthetic golden was encoded by the reference itself.
    let g = golden()
        .into_iter()
        .find(|g| g.tag.starts_with("syn_"))
        .unwrap();
    let [rows, cols] = g.shape();
    assert_eq!(encode(&synthetic(rows, cols), rows, cols), g.parts);
}

#[test]
fn round_trips_odd_columns_boundary_tiles_and_page_breaks() {
    for (rows, cols) in [
        (32, 1),     // one single-column block
        (32, 2),     // one full block
        (32, 3),     // a full block then a single column in one tile
        (64, 5),     // a tile with one single-column pair
        (96, 7),     // short tile row (3 blocks), odd C
        (160, 133),  // the synthetic golden's shape: 3 pages
        (4128, 9),   // 33 tile rows x 3 tile columns: page break mid-row
        (256, 257),  // 129 column pairs, 2 tile rows, 130 tiles: 5 pages
        (2048, 256), // GLM gate/up
        (4096, 128), // GLM down
    ] {
        for (seed, density) in [(1u64, 50u32), (2, 3), (3, 1_000_000)] {
            let raw = random(rows, cols, seed + rows as u64, density);
            let parts = encode(&raw, rows, cols);
            let (out, shape) = decode_nvfp4_csf32(&view(&parts))
                .unwrap_or_else(|e| panic!("[{rows}, {cols}] seed {seed}: {e:#}"));
            assert_eq!(shape, [rows, cols]);
            assert_eq!(out, raw, "[{rows}, {cols}] seed {seed}");
        }
    }
}

#[test]
fn geometry_counts_blocks_and_pages() {
    for (rows, cols, blocks, pages) in [
        (2048, 256, 64 * 128, 32),
        (4096, 128, 128 * 64, 32),
        (160, 133, 5 * 67, 3),
        (32, 1, 1, 1),
    ] {
        let g = Geometry::new(rows, cols).unwrap();
        assert_eq!(
            (g.blocks.len(), g.page_count()),
            (blocks, pages),
            "[{rows}, {cols}]"
        );
    }
}

/// Throughput at the real workload: GLM-5.3-Flash-FP4-Local's 36,288 csf32
/// matrices (42 layers x 288 experts x gate/up/down), two thirds `[2048, 256]`
/// and one third `[4096, 128]`, at the golden fixtures' exception density.
/// `cargo test --release -p spark-runtime --lib csf32_decode_throughput -- --ignored --nocapture`
#[test]
#[ignore]
fn csf32_decode_throughput() {
    let g = golden();
    let (gate, down) = (&g[0], &g[1]);
    let mut cache = GeometryCache::default();
    let t = std::time::Instant::now();
    let mut bytes = 0usize;
    for i in 0..36_288usize {
        let c = if i % 3 == 2 { down } else { gate };
        let (out, _) = cache.decode(&c.parts()).unwrap();
        bytes += out.len();
    }
    let s = t.elapsed().as_secs_f64();
    let gib = bytes as f64 / (1u64 << 30) as f64;
    eprintln!(
        "decoded 36288 csf32 matrices ({gib:.2} GiB out) in {s:.2} s = {:.2} GiB/s",
        gib / s
    );
}
