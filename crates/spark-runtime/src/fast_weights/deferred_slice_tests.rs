// SPDX-License-Identifier: AGPL-3.0-only

//! Expert-TP slices of CSF-compressed scales equal slicing the plain scale.
//!
//! Two checkpoints hold the same four real scales (slab gate [640, 160] and
//! down [2560, 40] from Qwen FP4-Local, csf_32x2_v1 gate [2048, 256] and down
//! [4096, 128] from GLM FP4-Local): one as their CSF components, one as the
//! plain E4M3 matrices they decode to. Loaded for expert TP over two ranks,
//! every rank's row slice (gate/up are column-parallel: rows are output
//! features) and column slice (down is row-parallel: columns are input
//! groups) of the compressed scale must be byte-identical to the plain
//! path's, and to slicing the full decode.

use super::*;
use crate::fast_weights::FastSafetensorsLoader;
use crate::weights::WeightLoader;
use crate::weights::csf_load_tests::{Dir, T, write_shard};
use crate::weights::nvfp4_csf::decode_nvfp4_csf;
use crate::weights::nvfp4_csf32::{Csf32Parts, decode_nvfp4_csf32};
use std::path::Path;

fn fixture(dir: &str, name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data")
            .join(dir)
            .join(name),
    )
    .unwrap()
}

/// One scale: its name, its components `(suffix, dtype, shape, bytes)`, and
/// the plain matrix and shape they decode to.
struct Case {
    name: String,
    parts: Vec<(&'static str, &'static str, Vec<usize>, Vec<u8>)>,
    plain: Vec<u8>,
    shape: [usize; 2],
}

fn cases() -> Vec<Case> {
    let scale = |e: usize, proj: &str| {
        format!("model.language_model.layers.3.mlp.experts.{e}.{proj}_proj.weight_scale")
    };
    let mut out = Vec::new();
    for (e, tag, proj, fixed_shape) in [
        (0, "l0_e0_gate", "gate", [40, 1296]),
        (1, "l0_e0_down", "down", [160, 336]),
    ] {
        let fixed = fixture("nvfp4_csf", &format!("{tag}.fixed.bin"));
        let exc = fixture("nvfp4_csf", &format!("{tag}.exceptions.bin"));
        let (plain, shape) = decode_nvfp4_csf(&fixed, &fixed_shape, &exc).unwrap();
        out.push(Case {
            name: scale(e, proj),
            parts: vec![
                (".nvfp4_csf_fixed", "U8", fixed_shape.to_vec(), fixed),
                (".nvfp4_csf_exceptions", "U32", vec![exc.len() / 4], exc),
            ],
            plain,
            shape,
        });
    }
    let comps = [
        ("bases", "U8", 1),
        ("codes", "U8", 1),
        ("page_bases", "U32", 4),
        ("offsets", "U16", 2),
        ("exceptions", "U16", 2),
    ];
    for (e, tag, proj) in [(2, "l3_e0_gate", "gate"), (3, "l3_e0_down", "down")] {
        let p: Vec<Vec<u8>> = comps
            .iter()
            .map(|(c, ..)| fixture("nvfp4_csf32", &format!("{tag}.{c}.bin")))
            .collect();
        let (plain, shape) = decode_nvfp4_csf32(&Csf32Parts {
            bases: &p[0],
            codes: &p[1],
            page_bases: &p[2],
            offsets: &p[3],
            exceptions: &p[4],
        })
        .unwrap();
        let suffixes = crate::weights::nvfp4_csf32::CSF32_SUFFIXES;
        out.push(Case {
            name: scale(e, proj),
            parts: comps
                .iter()
                .zip(p)
                .zip(suffixes)
                .map(|(((_, dtype, elem), bytes), suffix)| {
                    (suffix, *dtype, vec![bytes.len() / elem], bytes)
                })
                .collect(),
            plain,
            shape,
        });
    }
    out
}

/// A checkpoint of every case, compressed or plain, with its packed weight
/// `[n, k/2]` (`k/16` = the scale's columns). `drop` leaves out the named
/// component of the first case; `plain_too` adds a plain twin of it.
fn checkpoint(dir: &Path, cases: &[Case], compressed: bool, drop: Option<&str>, plain_too: bool) {
    let mut owned: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        if compressed {
            for (suffix, dtype, shape, bytes) in &c.parts {
                if i == 0 && drop == Some(*suffix) {
                    continue;
                }
                owned.push((
                    format!("{}{suffix}", c.name),
                    dtype,
                    shape.clone(),
                    bytes.clone(),
                ));
            }
        }
        if !compressed || (i == 0 && plain_too) {
            owned.push((c.name.clone(), "F8_E4M3", c.shape.to_vec(), c.plain.clone()));
        }
        let [n, cols] = c.shape;
        let w = c.name.strip_suffix("_scale").unwrap().to_string();
        owned.push((w, "U8", vec![n, cols * 8], vec![0; n * cols * 8]));
    }
    let t: Vec<T<'_>> = owned
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone()))
        .collect();
    write_shard(&dir.join("tensors/a.safetensors"), &t);
    let map: serde_json::Map<_, _> = owned
        .iter()
        .map(|(n, ..)| (n.clone(), "tensors/a.safetensors".into()))
        .collect();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
}

fn expert_tp_load(dir: &Path) -> Result<WeightStore> {
    let mut fast = FastSafetensorsLoader::with_ep(0, 2, 4);
    fast.expert_tp = true;
    fast.try_direct_io = false;
    fast.load(dir, &crate::gpu::mock::MockGpuBackend::new(), 0)
}

fn slice(store: &WeightStore, c: &Case, axis: SliceAxis, rank: usize) -> Vec<u8> {
    let [rows, cols] = c.shape;
    with_deferred_slice(
        store,
        &c.name,
        rows,
        cols,
        axis,
        rank,
        2,
        |b| Ok(b.to_vec()),
    )
    .unwrap_or_else(|e| panic!("{}: {e:#}", c.name))
}

#[test]
fn csf_scale_slices_equal_plain_scale_slices() {
    let cases = cases();
    let (cdir, pdir) = (Dir::new("tp-csf"), Dir::new("tp-plain"));
    checkpoint(&cdir.0, &cases, true, None, false);
    checkpoint(&pdir.0, &cases, false, None, false);
    let (csf, plain) = (
        expert_tp_load(&cdir.0).unwrap(),
        expert_tp_load(&pdir.0).unwrap(),
    );
    for c in &cases {
        // Nothing of a compressed scale is uploaded: it lives on disk.
        assert!(
            !csf.contains(&c.name) && csf.deferred(&c.name).is_none(),
            "{}",
            c.name
        );
        assert!(decode_deferred_csf_scale(&plain, &c.name).is_none());
        let [rows, cols] = c.shape;
        for rank in 0..2 {
            let want_rows = c.plain[rank * rows / 2 * cols..(rank + 1) * rows / 2 * cols].to_vec();
            let want_cols: Vec<u8> = c
                .plain
                .chunks_exact(cols)
                .flat_map(|r| &r[rank * cols / 2..(rank + 1) * cols / 2])
                .copied()
                .collect();
            for (axis, want) in [(SliceAxis::Rows, &want_rows), (SliceAxis::Cols, &want_cols)] {
                let what = format!("{} {axis:?} rank {rank}", c.name);
                assert_eq!(&slice(&plain, c, axis, rank), want, "plain {what}");
                assert_eq!(&slice(&csf, c, axis, rank), want, "csf {what}");
            }
        }
    }
}

#[test]
fn deferred_csf_sets_are_checked_at_load_and_decoded_loudly() {
    let all = cases();
    // A component missing: the load fails, as for a read set.
    for (case, drop) in [(0, ".nvfp4_csf_exceptions"), (2, ".csf32_page_bases")] {
        let dir = Dir::new("tp-lone");
        let c = &all[case..];
        checkpoint(&dir.0, c, true, Some(drop), false);
        let e = format!("{:#}", expert_tp_load(&dir.0).err().unwrap());
        assert!(e.contains("without a partner"), "{drop}: {e}");
    }
    // Plain and compressed copies of one deferred scale.
    let dir = Dir::new("tp-dup");
    checkpoint(&dir.0, &all[2..], true, None, true);
    let e = format!("{:#}", expert_tp_load(&dir.0).err().unwrap());
    assert!(e.contains("both plain and CSF"), "{e}");
    // A scale that does not fit its weight fails the load by shape alone.
    let dir = Dir::new("tp-misfit");
    let mut c = cases();
    c.truncate(3);
    let gate = c.remove(2);
    let wrong = Case {
        shape: [gate.shape[0], gate.shape[1] + 16],
        ..gate
    };
    checkpoint(&dir.0, std::slice::from_ref(&wrong), true, None, false);
    let e = format!("{:#}", expert_tp_load(&dir.0).err().unwrap());
    assert!(e.contains("needs"), "{e}");
    // Corrupt bytes pass the load's shape check and fail the slice loudly.
    let dir = Dir::new("tp-corrupt");
    let mut c = cases();
    c[2].parts[0].3[5] = 250; // a row base past 240
    checkpoint(&dir.0, &c[2..3], true, None, false);
    let store = expert_tp_load(&dir.0).unwrap();
    let [rows, cols] = c[2].shape;
    let e = with_deferred_slice(
        &store,
        &c[2].name,
        rows,
        cols,
        SliceAxis::Rows,
        0,
        2,
        |_| Ok(()),
    )
    .unwrap_err();
    let e = format!("{e:#}");
    assert!(
        e.contains("decoding deferred csf_32x2_v1 scale") && e.contains("base"),
        "{e}"
    );
}

#[test]
fn column_slice_keeps_each_rows_rank_half() {
    let bytes: Vec<u8> = (0..12).collect(); // 3 rows x 4 bytes
    assert_eq!(column_slice(&bytes, 4, 0, 2), [0, 1, 4, 5, 8, 9]);
    assert_eq!(column_slice(&bytes, 4, 1, 2), [2, 3, 6, 7, 10, 11]);
}
