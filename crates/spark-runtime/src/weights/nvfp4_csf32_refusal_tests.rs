// SPDX-License-Identifier: AGPL-3.0-only

//! `csf_32x2_v1` refusals: every corrupt input the decoder and the assembler
//! must refuse loudly rather than decode into wrong scales.

use super::tests::{Golden, decoded_len_of_codes, encode, golden, hex, view};
use super::*;
use crate::weights::nvfp4_csf::CsfAssembler;
use sha2::{Digest, Sha256};

fn expect_err(parts: &[Vec<u8>; 5], needle: &str) {
    let e = format!(
        "{:#}",
        decode_nvfp4_csf32(&view(parts)).expect_err("corrupt input must be refused")
    );
    assert!(
        e.contains(needle),
        "error {e:?} does not mention {needle:?}"
    );
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|w| u16::from_le_bytes([w[0], w[1]]))
        .collect()
}

fn bytes16(w: &[u16]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `(block index, record index)` of the first block holding two records, so
/// order checks have a pair to break.
fn block_with_two(parts: &[Vec<u8>; 5], rows: usize, cols: usize) -> (usize, usize) {
    let g = Geometry::new(rows, cols).unwrap();
    let off = u16s(&parts[3]);
    let base: Vec<usize> = parts[2]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize / 2)
        .collect();
    for p in 0..g.page_count() {
        for b in g.pages[p]..g.pages[p + 1] {
            let (lo, hi) = (off[b + p], off[b + p + 1]);
            if hi - lo >= 2 {
                return (b, base[p] + lo as usize);
            }
        }
    }
    panic!("no block with two records")
}

#[test]
fn structural_corruption_fails_loudly() {
    let g = &golden()[0]; // [2048, 256], 32 pages
    let ok = || g.parts.clone();

    // R not a multiple of 32, and codes that are not R * C / 2.
    let mut p = ok();
    p[0].pop();
    expect_err(&p, "multiple of 32");
    let mut p = ok();
    p[1].truncate(p[1].len() - 2);
    expect_err(&p, "is not R * C / 2");
    // Codes for C = 254: the offsets no longer fit the geometry.
    let mut p = ok();
    p[1].truncate(2048 * 254 / 2);
    expect_err(&p, "offsets:");
    // Wrong page_bases / offsets lengths, a torn record.
    let mut p = ok();
    p[2].truncate(p[2].len() - 4);
    expect_err(&p, "page_bases:");
    let mut p = ok();
    p[3].extend_from_slice(&[0, 0]);
    expect_err(&p, "offsets:");
    let mut p = ok();
    p[4].pop();
    expect_err(&p, "whole number");

    // A row base past 240.
    let mut p = ok();
    p[0][77] = 241;
    expect_err(&p, "row 77 base");

    // Page origins: odd, overlapping the previous page, past the arena.
    let mut p = ok();
    p[2][4] |= 1;
    expect_err(&p, "is odd");
    let mut p = ok();
    p[2][4] -= 2;
    expect_err(&p, "overlap or leave a gap");
    let mut p = ok();
    p[2][4] += 2;
    expect_err(&p, "overlap or leave a gap");
    // The arena loses its last record: the last page runs past it.
    let mut p = ok();
    p[4].truncate(p[4].len() - 2);
    expect_err(&p, "but the arena has");
    // A record no page indexes.
    let mut p = ok();
    p[4].extend_from_slice(&[0, 0x30]);
    expect_err(&p, "but the pages index");

    // Prefixes: page 1 not starting at 0, and decreasing inside a page.
    let per = (8192 + 32) / 32;
    let mut off = u16s(&g.parts[3]);
    off[per] = 1;
    let mut p = ok();
    p[3] = bytes16(&off);
    expect_err(&p, "page 1 prefixes start at 1");
    let mut off = u16s(&g.parts[3]);
    let i = (1..per - 1).find(|&i| off[i] > 0).unwrap();
    off[i + 1] = off[i] - 1;
    let mut p = ok();
    p[3] = bytes16(&off);
    expect_err(&p, "prefixes decrease");

    // A prefix past the page's count, even if later ones come back down:
    // refused before any record is read, not a panic past the arena.
    let mut off = u16s(&g.parts[3]);
    off[1] = u16::MAX;
    let mut p = ok();
    p[3] = bytes16(&off);
    expect_err(&p, "exceeds the page's");

    // Records: out of order, duplicated, outside the block.
    let (_, r) = block_with_two(&g.parts, 2048, 256);
    let rec = u16s(&g.parts[4]);
    let mut w = rec.clone();
    w.swap(r, r + 1);
    let mut p = ok();
    p[4] = bytes16(&w);
    expect_err(&p, "out of order");
    let mut w = rec.clone();
    w[r + 1] = (w[r + 1] & 0xFF00) | (w[r] & 0x00FF);
    let mut p = ok();
    p[4] = bytes16(&w);
    expect_err(&p, "out of order");
    let mut w = rec.clone();
    *w.last_mut().unwrap() = (w.last().unwrap() & 0xFF00) | 64;
    let mut p = ok();
    p[4] = bytes16(&w);
    expect_err(&p, "outside block");

    // A replacement that is not a valid NVFP4 scale (sign bit set).
    let mut w = rec;
    w[0] = (w[0] & 0x00FF) | 0xB800;
    let mut p = ok();
    p[4] = bytes16(&w);
    expect_err(&p, "negative or NaN");
}

/// Arbitrary corruption is refused or decodes; it never panics (a panic is
/// an index past a component, i.e. a check missing ahead of a read).
#[test]
fn random_corruption_never_panics() {
    let cases = golden();
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |n: usize| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as usize % n
    };
    for g in [&cases[3], &cases[0]] {
        let rounds = if g.tag.starts_with("syn_") { 4000 } else { 50 };
        for _ in 0..rounds {
            let mut p = g.parts.clone();
            // Index-bearing components get most of the hits.
            let which = [0, 1, 2, 2, 3, 3, 3, 4, 4, 4][next(10)];
            if !p[which].is_empty() {
                let at = next(p[which].len());
                p[which][at] = next(256) as u8;
            }
            let _ = decode_nvfp4_csf32(&view(&p));
        }
    }
}

/// A single-column block has no column 1: an odd local position in it is
/// outside its extent even though it is below 64.
#[test]
fn a_record_in_the_missing_column_of_a_single_column_block_is_refused() {
    let (rows, cols) = (32, 3);
    let mut raw = vec![0x30u8; rows * cols];
    raw[2] = 0x70; // row 0, col 2: an exception in the single-column block
    let mut p = encode(&raw, rows, cols);
    assert_eq!(u16s(&p[4]), [0x7000]); // local 0 of block 1
    p[4] = bytes16(&[0x7001]); // local row 0, column 1 — which it lacks
    expect_err(&p, "outside block 1 (32 rows x 1 column(s))");
}

#[test]
fn oversized_inputs_are_refused_not_wrapped() {
    // rows * cols wraps.
    let e = Geometry::new(1 << 40, 1 << 40).err().unwrap();
    assert!(format!("{e:#}").contains("overflows"), "{e:#}");
    // No columns, no rows, rows not a multiple of 32.
    assert!(Geometry::new(32, 0).is_err());
    assert!(Geometry::new(0, 2).is_err());
    assert!(Geometry::new(48, 2).is_err());
    // The pre-flight estimate gets None, not a wrapped size.
    assert_eq!(decoded_len_of_codes(&[usize::MAX]), None);
}

/// The arena must be smaller than 4 GiB (page origins are u32 bytes). A
/// 4 GiB slice is too much for a unit test, so the bound is checked alone.
#[test]
fn a_four_gib_arena_is_refused() {
    check_arena(u32::MAX as usize - 1).unwrap();
    let e = check_arena(1 << 32).unwrap_err();
    assert!(format!("{e:#}").contains("< 4 GiB"), "{e:#}");
}

// ---- Assembler --------------------------------------------------------------

const PARTS: [&str; 5] = CSF32_SUFFIXES;

/// Component `i` of `g` as scale `name`: tensor name, header shape, bytes.
fn part(name: &str, g: &Golden, i: usize) -> (String, Vec<usize>, Vec<u8>) {
    let bytes = g.parts[i].clone();
    (
        format!("{name}{}", PARTS[i]),
        vec![bytes.len() / CSF32_ELEM_BYTES[i]],
        bytes,
    )
}

#[test]
fn assembler_gathers_five_parts_in_any_order_across_scales() {
    let g = golden();
    let (gate, down) = (&g[0], &g[1]);
    let (a_name, b_name) = ("m.e0.gate_proj.weight_scale", "m.e0.down_proj.weight_scale");
    let mut a = CsfAssembler::new();
    // Interleaved: gate's parts reversed, down's rotated.
    let order: Vec<(&str, &Golden, usize)> = [4, 3, 2, 1]
        .iter()
        .map(|&i| (a_name, gate, i))
        .zip([2, 3, 4, 0].iter().map(|&i| (b_name, down, i)))
        .flat_map(|(x, y)| [x, y])
        .collect();
    for (name, src, i) in order {
        let (n, shape, bytes) = part(name, src, i);
        assert!(
            a.offer(&n, &shape, &bytes).unwrap().is_none(),
            "{n} completed early"
        );
    }
    for (name, src, i, shape) in [
        (a_name, gate, 0, [2048, 256]),
        (b_name, down, 1, [4096, 128]),
    ] {
        let (n, s, bytes) = part(name, src, i);
        let d = a.offer(&n, &s, &bytes).unwrap().expect("set complete");
        assert_eq!((d.name.as_str(), d.shape.as_slice()), (name, &shape[..]));
        assert_eq!(hex(&Sha256::digest(&d.bytes)), src.sha());
    }
    a.finish(|w| match w {
        "m.e0.gate_proj.weight" => Some(vec![2048, 2048]),
        "m.e0.down_proj.weight" => Some(vec![4096, 1024]),
        _ => None,
    })
    .unwrap();
}

fn all_parts(a: &mut CsfAssembler, name: &str, g: &Golden) -> Result<()> {
    for i in 0..5 {
        let (n, s, b) = part(name, g, i);
        a.offer(&n, &s, &b)?;
    }
    Ok(())
}

#[test]
fn assembler_refuses_incomplete_duplicate_mixed_and_mistyped_sets() {
    let g = &golden()[0];
    let name = "x.weight_scale";
    let err = |r: Result<()>| format!("{:#}", r.unwrap_err());

    // Four of five: the load is incomplete.
    let mut a = CsfAssembler::new();
    for i in 0..4 {
        let (n, s, b) = part(name, g, i);
        a.offer(&n, &s, &b).unwrap();
    }
    let e = err(a.finish(|_| None));
    assert!(e.contains("without a partner"), "{e}");

    // A component twice.
    let mut a = CsfAssembler::new();
    let (n, s, b) = part(name, g, 2);
    a.offer(&n, &s, &b).unwrap();
    let e = err(a.offer(&n, &s, &b).map(drop));
    assert!(e.contains("appears twice"), "{e}");

    // A slab half and a csf32 part for one scale.
    let mut a = CsfAssembler::new();
    a.offer("x.weight_scale.nvfp4_csf_fixed", &[2, 48], &[0; 96])
        .unwrap();
    let (n, s, b) = part(name, g, 0);
    let e = err(a.offer(&n, &s, &b).map(drop));
    assert!(e.contains("both the CSF and the csf_32x2_v1"), "{e}");

    // Components are 1-D.
    let mut a = CsfAssembler::new();
    let e = err(a
        .offer("x.weight_scale.csf32_bases", &[64, 32], &g.parts[0])
        .map(drop));
    assert!(e.contains("are 1-D"), "{e}");

    // U8 bytes under a U16 / U32 name: the element count does not fit.
    for (i, what) in [(3, "U16"), (4, "U16"), (2, "U32")] {
        let mut a = CsfAssembler::new();
        let n = format!("{name}{}", PARTS[i]);
        let e = err(a.offer(&n, &[g.parts[i].len()], &g.parts[i]).map(drop));
        assert!(e.contains(what), "{e}");
    }

    // A corrupt set names the scale it was decoding.
    let mut a = CsfAssembler::new();
    let mut bad = golden().remove(0);
    bad.parts[0][5] = 250;
    let e = err(all_parts(&mut a, name, &bad));
    assert!(
        e.contains("decoding csf_32x2_v1 scale x.weight_scale"),
        "{e}"
    );
}

#[test]
fn assembler_refuses_a_csf32_scale_that_is_also_plain_or_misfits_its_weight() {
    let g = &golden()[0];
    let name = "e.gate_proj.weight_scale";
    // Plain first, then the set; the set first, then plain.
    let mut a = CsfAssembler::new();
    a.note_plain(name).unwrap();
    let e = format!("{:#}", all_parts(&mut a, name, g).unwrap_err());
    assert!(e.contains("both plain and CSF"), "{e}");
    let mut a = CsfAssembler::new();
    all_parts(&mut a, name, g).unwrap();
    let e = format!("{:#}", a.note_plain(name).unwrap_err());
    assert!(e.contains("both plain and CSF"), "{e}");
    // The decoded [2048, 256] needs a [2048, 2048] packed weight.
    for (w, ok) in [
        (vec![2048, 2048], true),
        (vec![2048, 2056], false),
        (vec![1024, 2048], false),
    ] {
        let mut a = CsfAssembler::new();
        all_parts(&mut a, name, g).unwrap();
        let r = a.finish(|n| (n == "e.gate_proj.weight").then(|| w.clone()));
        assert_eq!(r.is_ok(), ok, "{w:?}: {r:?}");
    }
}
