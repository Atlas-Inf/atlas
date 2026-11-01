// SPDX-License-Identifier: AGPL-3.0-only

//! CSF decoder tests.
//!
//! `golden_pairs_decode_byte_identical_to_the_vllm_reference` is the one that
//! matters: three real scale pairs range-read from
//! nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6, decoded by vLLM's reference
//! (`scripts/nvfp4_csf_golden.py`), compared by sha256 of the full matrix.

use super::*;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/nvfp4_csf")
}

struct Golden {
    tag: String,
    fixed: Vec<u8>,
    fixed_shape: Vec<usize>,
    exceptions: Vec<u8>,
    case: serde_json::Value,
}

fn golden() -> Vec<Golden> {
    let dir = golden_dir();
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("golden.json")).unwrap()).unwrap();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let tag = c["tag"].as_str().unwrap().to_string();
            Golden {
                fixed: std::fs::read(dir.join(format!("{tag}.fixed.bin"))).unwrap(),
                exceptions: std::fs::read(dir.join(format!("{tag}.exceptions.bin"))).unwrap(),
                fixed_shape: c["fixed_shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
                tag,
                case: c.clone(),
            }
        })
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn golden_pairs_decode_byte_identical_to_the_vllm_reference() {
    let cases = golden();
    assert_eq!(cases.len(), 3, "fixture set changed");
    for g in &cases {
        let (out, shape) = decode_nvfp4_csf(&g.fixed, &g.fixed_shape, &g.exceptions)
            .unwrap_or_else(|e| panic!("{}: {e:#}", g.tag));
        let want_shape: Vec<usize> = g.case["decoded_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(shape.to_vec(), want_shape, "{}", g.tag);
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
            g.case["decoded_sha256"].as_str().unwrap(),
            "{}: decoded matrix differs from the vLLM reference",
            g.tag
        );
        assert_eq!(decoded_len(CsfPart::Fixed, &g.fixed_shape), Some(out.len()));
    }
}

/// vLLM's test encoder (row base = row min capped at 240, out-of-window bytes
/// as exceptions), re-expressed in Rust so round trips can cover geometries
/// and exception densities the three real fixtures do not.
fn encode(raw: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<usize>, Vec<u8>) {
    let width = SLAB * (1 + cols / 2);
    let mut fixed = vec![0u8; rows / SLAB * width];
    let mut exc = Vec::new();
    for r in 0..rows {
        let row = &raw[r * cols..(r + 1) * cols];
        let base = (*row.iter().min().unwrap()).min(MAX_BASE);
        let slab = &mut fixed[r / SLAB * width..(r / SLAB + 1) * width];
        slab[r % SLAB] = base;
        let codes = &mut slab[SLAB + (r % SLAB) * cols / 2..SLAB + (r % SLAB + 1) * cols / 2];
        for c in 0..cols {
            let code = row[c].wrapping_sub(base);
            let code = if row[c] < base || code > 15 {
                exc.extend_from_slice(
                    &(((row[c] as u32) << 24) | (r * cols + c) as u32).to_le_bytes(),
                );
                0
            } else {
                code
            };
            codes[c / 2] |= code << (4 * (c % 2));
        }
    }
    (fixed, vec![rows / SLAB, width], exc)
}

fn synthetic(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    // Deterministic LCG; scales cluster in a window with sparse outliers, the
    // shape the codec is built for.
    let mut s = seed;
    (0..rows * cols)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (s >> 33) as u32;
            if r.is_multiple_of(50) {
                0x7E
            } else {
                0x30 + (r % 16) as u8
            }
        })
        .collect()
}

#[test]
fn round_trips_both_expert_geometries() {
    for (rows, cols) in [(640, 160), (2560, 40), (16, 4)] {
        let raw = synthetic(rows, cols, rows as u64);
        let (fixed, shape, exc) = encode(&raw, rows, cols);
        assert!(!exc.is_empty() || rows == 16);
        let (out, got) = decode_nvfp4_csf(&fixed, &shape, &exc).unwrap();
        assert_eq!(got, [rows, cols]);
        assert_eq!(out, raw);
    }
}

fn expect_err(r: Result<(Vec<u8>, [usize; 2])>, needle: &str) {
    let e = format!("{:#}", r.expect_err("corrupt input must be refused"));
    assert!(
        e.contains(needle),
        "error {e:?} does not mention {needle:?}"
    );
}

#[test]
fn corrupted_inputs_fail_loudly() {
    let g = &golden()[0];
    let words = |e: &[u8]| -> Vec<u32> {
        e.chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    let bytes = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };

    // A row base that would carry into its neighbour.
    let mut f = g.fixed.clone();
    f[3] = 0xF8;
    expect_err(decode_nvfp4_csf(&f, &g.fixed_shape, &g.exceptions), "base");

    // Exception past the end of the matrix.
    let mut w = words(&g.exceptions);
    *w.last_mut().unwrap() = (0x30 << 24) | (640 * 160);
    expect_err(
        decode_nvfp4_csf(&g.fixed, &g.fixed_shape, &bytes(&w)),
        "past the",
    );

    // Two exceptions swapped: out of order.
    let mut w = words(&g.exceptions);
    w.swap(0, 1);
    expect_err(
        decode_nvfp4_csf(&g.fixed, &g.fixed_shape, &bytes(&w)),
        "out of order",
    );

    // Duplicate position.
    let mut w = words(&g.exceptions);
    w[1] = (w[1] & 0xFF00_0000) | (w[0] & 0x00FF_FFFF);
    expect_err(
        decode_nvfp4_csf(&g.fixed, &g.fixed_shape, &bytes(&w)),
        "out of order",
    );

    // An exception value that is not a valid NVFP4 scale (sign bit set).
    let mut w = words(&g.exceptions);
    w[0] = (0xB8 << 24) | (w[0] & 0x00FF_FFFF);
    expect_err(
        decode_nvfp4_csf(&g.fixed, &g.fixed_shape, &bytes(&w)),
        "negative or NaN",
    );

    // Truncated streams and wrong shapes.
    expect_err(
        decode_nvfp4_csf(&g.fixed[..g.fixed.len() - 1], &g.fixed_shape, &g.exceptions),
        "holds",
    );
    expect_err(
        decode_nvfp4_csf(
            &g.fixed,
            &g.fixed_shape,
            &g.exceptions[..g.exceptions.len() - 2],
        ),
        "whole number",
    );
    expect_err(
        decode_nvfp4_csf(&g.fixed, &[40, 1295], &g.exceptions),
        "not [rows/16",
    );
    expect_err(decode_nvfp4_csf(&g.fixed, &[51840], &g.exceptions), "2-D");
}

#[test]
fn assembler_pairs_in_any_order_and_names_the_scale() {
    let g = golden();
    let mut a = CsfAssembler::new();
    let stem = |i: usize| format!("m.layers.0.mlp.experts.{i}.gate_proj.weight_scale");
    // Interleaved, both orders.
    assert!(
        a.offer(
            &format!("{}{CSF_FIXED_SUFFIX}", stem(0)),
            &g[0].fixed_shape,
            &g[0].fixed
        )
        .unwrap()
        .is_none()
    );
    let n = g[2].exceptions.len() / 4;
    assert!(
        a.offer(
            &format!("{}{CSF_EXCEPTIONS_SUFFIX}", stem(1)),
            &[n],
            &g[2].exceptions
        )
        .unwrap()
        .is_none()
    );
    let n0 = g[0].exceptions.len() / 4;
    let d0 = a
        .offer(
            &format!("{}{CSF_EXCEPTIONS_SUFFIX}", stem(0)),
            &[n0],
            &g[0].exceptions,
        )
        .unwrap()
        .expect("pair 0 complete");
    assert_eq!(d0.name, stem(0));
    assert_eq!(d0.shape, vec![640, 160]);
    let d1 = a
        .offer(
            &format!("{}{CSF_FIXED_SUFFIX}", stem(1)),
            &g[2].fixed_shape,
            &g[2].fixed,
        )
        .unwrap()
        .expect("pair 1 complete");
    assert_eq!(d1.name, stem(1));
    assert_eq!(
        hex(&Sha256::digest(&d1.bytes)),
        g[2].case["decoded_sha256"].as_str().unwrap()
    );
    a.finish(|_| Some(vec![640, 1280])).unwrap();
}

#[test]
fn assembler_refuses_lone_duplicate_and_mistyped_halves() {
    let g = &golden()[1];
    let fixed = format!("x.weight_scale{CSF_FIXED_SUFFIX}");
    let exc = format!("x.weight_scale{CSF_EXCEPTIONS_SUFFIX}");
    let n = g.exceptions.len() / 4;

    let mut a = CsfAssembler::new();
    a.offer(&fixed, &g.fixed_shape, &g.fixed).unwrap();
    let e = format!("{:#}", a.finish(|_| None).unwrap_err());
    assert!(e.contains("without a partner"), "{e}");

    let mut a = CsfAssembler::new();
    a.offer(&fixed, &g.fixed_shape, &g.fixed).unwrap();
    assert!(a.offer(&fixed, &g.fixed_shape, &g.fixed).is_err());

    // U8-shaped bytes under the U32 name: the header says n words, the payload
    // holds n bytes.
    let mut a = CsfAssembler::new();
    assert!(a.offer(&exc, &[n], &g.exceptions[..n]).is_err());
    assert!(a.offer("x.weight_scale", &[1], &[0]).is_err());
}

#[test]
fn assembler_refuses_components_after_the_scale_is_complete() {
    let g = golden().remove(0);
    for deferred in [false, true] {
        let mut a = CsfAssembler::new();
        for (name, shape, bytes) in [
            (
                "x.weight_scale.nvfp4_csf_fixed",
                g.fixed_shape.clone(),
                g.fixed.as_slice(),
            ),
            (
                "x.weight_scale.nvfp4_csf_exceptions",
                vec![g.exceptions.len() / 4],
                g.exceptions.as_slice(),
            ),
        ] {
            if deferred {
                a.defer(name, &shape, bytes.len()).unwrap();
            } else {
                a.offer(name, &shape, bytes).unwrap();
            }
        }
        assert!(
            a.offer("x.weight_scale.nvfp4_csf_fixed", &g.fixed_shape, &g.fixed)
                .is_err()
        );
        assert!(a.offer("x.weight_scale.csf32_bases", &[1], &[0]).is_err());
    }
}

/// Shapes come from a file header: every size computed from them must be
/// checked, or a hostile header wraps in release builds (or panics in debug)
/// instead of being refused.
#[test]
fn overflowing_shapes_are_refused_not_wrapped() {
    let big = usize::MAX;
    // geometry: slabs * 16 wraps.
    expect_err(decode_nvfp4_csf(&[], &[big, 32], &[]), "overflow");
    // geometry: rows * cols wraps (2^44 rows x ~2^41 cols).
    let w = 16 * (1 + (1usize << 40));
    expect_err(decode_nvfp4_csf(&[], &[1 << 40, w], &[]), "overflow");
    // No rows at all.
    expect_err(decode_nvfp4_csf(&[], &[0, 32], &[]), "not [rows/16");
    // The pre-flight estimate gets None, not a wrapped size.
    assert_eq!(decoded_len(CsfPart::Fixed, &[big, 32]), None);
    assert_eq!(decoded_len(CsfPart::Fixed, &[1 << 40, w]), None);
    assert_eq!(decoded_len(CsfPart::Fixed, &[40, 1296]), Some(640 * 160));
    // offer: the header's element count wraps, and so does count * 4.
    let mut a = CsfAssembler::new();
    let e = a.offer("x.weight_scale.nvfp4_csf_fixed", &[big, 2], &[0; 2]);
    assert!(format!("{:#}", e.err().unwrap()).contains("overflow"));
    let e = a.offer("x.weight_scale.nvfp4_csf_exceptions", &[big / 2], &[0; 4]);
    assert!(format!("{:#}", e.err().unwrap()).contains("overflow"));
}

/// The decoded scale must be the one its packed weight `[n, k/2]` needs,
/// `[n, k/16]`: a pair from another projection, or a weight from another
/// layout, is refused rather than uploaded and read as scales.
#[test]
fn decoded_scales_are_checked_against_their_weights() {
    let g = &golden()[0]; // [640, 160] E4M3: a [640, 2560] packed-NVFP4 weight
    let name = "e.gate_proj.weight_scale";
    let pair = |a: &mut CsfAssembler| {
        a.offer(
            &format!("{name}{CSF_FIXED_SUFFIX}"),
            &g.fixed_shape,
            &g.fixed,
        )
        .unwrap();
        let n = g.exceptions.len() / 4;
        a.offer(
            &format!("{name}{CSF_EXCEPTIONS_SUFFIX}"),
            &[n],
            &g.exceptions,
        )
        .unwrap()
        .expect("pair complete")
    };
    let finish = |weight: Option<Vec<usize>>| {
        let mut a = CsfAssembler::new();
        pair(&mut a);
        a.finish(|n: &str| {
            (n == "e.gate_proj.weight")
                .then(|| weight.clone())
                .flatten()
        })
    };
    finish(Some(vec![640, 1280])).unwrap();
    for (w, needle) in [
        (Some(vec![640, 1288]), "needs"),       // k/16 = 161
        (Some(vec![1280, 1280]), "needs"),      // rows differ
        (Some(vec![640, 1280, 1]), "needs"),    // not 2-D
        (Some(vec![640, 1281]), "needs"),       // k not a multiple of 16
        (Some(vec![640, usize::MAX]), "needs"), // 2 * k_packed wraps
        (None, "no `e.gate_proj.weight`"),
    ] {
        let e = format!("{:#}", finish(w.clone()).unwrap_err());
        assert!(e.contains(needle), "{w:?}: {e}");
    }
}

/// Plain and CSF copies of one scale are refused in either order (the
/// loaders used to check only CSF-after-plain, and only within one map).
#[test]
fn a_scale_both_plain_and_csf_is_refused_in_either_order() {
    let g = &golden()[0];
    let name = "e.gate_proj.weight_scale";
    let n = g.exceptions.len() / 4;
    let offer_pair = |a: &mut CsfAssembler| -> Result<()> {
        a.offer(
            &format!("{name}{CSF_FIXED_SUFFIX}"),
            &g.fixed_shape,
            &g.fixed,
        )?;
        a.offer(
            &format!("{name}{CSF_EXCEPTIONS_SUFFIX}"),
            &[n],
            &g.exceptions,
        )?;
        Ok(())
    };
    // Plain first, then the pair.
    let mut a = CsfAssembler::new();
    a.note_plain(name).unwrap();
    let e = format!("{:#}", offer_pair(&mut a).unwrap_err());
    assert!(e.contains("both plain and CSF"), "{e}");
    // The pair first, then plain.
    let mut a = CsfAssembler::new();
    offer_pair(&mut a).unwrap();
    let e = format!("{:#}", a.note_plain(name).unwrap_err());
    assert!(e.contains("both plain and CSF"), "{e}");
    // Unrelated plain tensors are fine either side of a pair.
    let mut a = CsfAssembler::new();
    a.note_plain("e.up_proj.weight_scale").unwrap();
    offer_pair(&mut a).unwrap();
    a.note_plain("e.gate_proj.weight").unwrap();
}

#[test]
fn csf_part_names() {
    assert_eq!(
        csf_part("a.weight_scale.nvfp4_csf_fixed"),
        Some(("a.weight_scale", CsfPart::Fixed))
    );
    assert_eq!(
        csf_part("a.weight_scale.nvfp4_csf_exceptions"),
        Some(("a.weight_scale", CsfPart::Exceptions))
    );
    for (suffix, part) in [
        (".csf32_bases", CsfPart::Bases),
        (".csf32_codes", CsfPart::Codes),
        (".csf32_page_bases", CsfPart::PageBases),
        (".csf32_offsets", CsfPart::Offsets),
        (".csf32_exceptions", CsfPart::BlockExceptions),
    ] {
        assert_eq!(
            csf_part(&format!("a.weight_scale{suffix}")),
            Some(("a.weight_scale", part))
        );
        assert_eq!(part.encoding(), CsfEncoding::Blocks32);
        assert_eq!(
            decoded_len(part, &[64]),
            Some(if part == CsfPart::Codes { 128 } else { 0 })
        );
    }
    assert_eq!(decoded_len(CsfPart::Exceptions, &[64]), Some(0));
    assert_eq!(csf_part("a.weight_scale"), None);
    assert_eq!(csf_part("a.weight_scale.csf32_bogus"), None);
}

/// Throughput at the real workload: 73,728 matrices, two thirds `[640, 160]`
/// and one third `[2560, 40]`, at the golden fixtures' exception density.
/// `cargo test --release -p spark-runtime csf_decode_throughput -- --ignored --nocapture`
#[test]
#[ignore]
fn csf_decode_throughput() {
    let g = golden();
    let (gate, down) = (&g[0], &g[1]);
    let t = Instant::now();
    let mut bytes = 0usize;
    for i in 0..73_728usize {
        let c = if i % 3 == 1 { down } else { gate };
        let (out, _) = decode_nvfp4_csf(&c.fixed, &c.fixed_shape, &c.exceptions).unwrap();
        bytes += out.len();
    }
    let s = t.elapsed().as_secs_f64();
    eprintln!(
        "decoded 73728 CSF matrices ({:.2} GiB out) in {s:.2} s = {:.2} GiB/s",
        bytes as f64 / (1u64 << 30) as f64,
        bytes as f64 / (1u64 << 30) as f64 / s
    );
}
