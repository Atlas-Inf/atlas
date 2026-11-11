// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn spec() -> Nvfp4RowGeom {
    Nvfp4RowGeom {
        dim: 32,
        group: 16,
        global: 0.25,
    }
}

#[test]
fn decodes_rows_with_block_and_global_scales() {
    let s = spec();
    // Row 0: nibble 2 (=1.0) in block 0, nibble 7 (=6.0) in block 1.
    // Row 1: nibble 9 (=-0.5) everywhere.
    let packed = vec![
        [vec![0x22u8; 8], vec![0x77u8; 8]].concat(),
        vec![0x99u8; 16],
    ];
    // Block scales: E4M3 0x38 = 1.0, 0x40 = 2.0.
    let scales = vec![vec![0x38u8, 0x40], vec![0x40u8, 0x40]];
    let out = decode_rows_bf16(&s, &packed, &scales).unwrap();
    let at = |i: usize| {
        f32::from_bits(u32::from(u16::from_le_bytes([out[2 * i], out[2 * i + 1]])) << 16)
    };
    assert_eq!(out.len(), 2 * 32 * 2);
    assert!((0..16).all(|i| at(i) == 0.25)); // 1.0 * 1.0 * 0.25
    assert!((16..32).all(|i| at(i) == 3.0)); // 6.0 * 2.0 * 0.25
    assert!((32..64).all(|i| at(i) == -0.25)); // -0.5 * 2.0 * 0.25
}

#[test]
fn rejects_mismatched_rows_and_strides() {
    let s = spec();
    assert!(decode_rows_bf16(&s, &[vec![0; 16]], &[]).is_err());
    assert!(decode_rows_bf16(&s, &[vec![0; 15]], &[vec![0; 2]]).is_err());
    // The scale arena must be attached:  true is required.
    assert!(check_packed_stride(16, &s, true).is_ok());
    assert!(check_packed_stride(32, &s, true).is_err());
    assert!(check_packed_stride(16, &s, false).is_err());
}

/// The FP4-Local row: 160 values in 80 code bytes and 10 E4M3 scales.
fn fp4_local() -> Nvfp4RowGeom {
    Nvfp4RowGeom {
        dim: 160,
        group: 16,
        global: 1.0,
    }
}

#[test]
fn interleaved_geometry_of_the_fp4_local_table() {
    let g = fp4_local();
    assert_eq!(
        (g.packed_stride(), g.scale_stride(), g.interleaved_stride()),
        (80, 10, 96)
    );
    // Split rows need the scale arena; interleaved rows must not have one.
    assert!(check_packed_stride(80, &g, true).is_ok(), "split");
    assert!(check_packed_stride(96, &g, false).is_ok(), "interleaved");
    assert!(check_packed_stride(90, &g, false).is_err(), "unpadded");
    assert!(
        check_packed_stride(80, &g, false).is_err(),
        "split, no scales"
    );
    assert!(check_packed_stride(96, &g, true).is_err(), "scales twice");
    assert!(check_packed_stride(160, &g, false).is_err(), "an FP8 row");
}

/// The gather reads a split table's scales from their arena at the scale
/// stride, and an interleaved row's right after its codes at the row stride.
#[test]
fn gather_planes_follow_the_row_layout() {
    let g = fp4_local();
    assert_eq!(g.gather_planes(0x1000, Some(0x9000)), (0x9000, 80, 10));
    assert_eq!(g.gather_planes(0x1000, None), (0x1000 + 80, 96, 96));
}

// ── Golden rows of the real checkpoint ──
//
// Fixture: `bench/qwen4_exp/ple_nvfp4_fixtures.json`, from
// `bench/qwen4_exp/ple_nvfp4_golden.py`, which range-reads nine rows (shards
// 0, 1 and 127, three files) of nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6
// off Hugging Face and dequantizes them with flashinfer's oracle math in torch.

struct GoldenRow {
    packed: Vec<u8>,
    scales: Vec<u8>,
    bf16: Vec<u16>,
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// `(geometry with the checkpoint's global scale, rows)` from the fixture.
fn golden() -> (Nvfp4RowGeom, Vec<GoldenRow>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/qwen4_exp/ple_nvfp4_fixtures.json"
    );
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path)
            .expect("run bench/qwen4_exp/ple_nvfp4_golden.py to generate the fixture"),
    )
    .expect("ple_nvfp4_fixtures.json");
    assert_eq!(v["head_dim"].as_u64(), Some(160));
    assert_eq!(v["rows_per_shard"].as_u64(), Some(2_500_012));
    let rows = v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| GoldenRow {
            packed: hex(r["packed"].as_str().unwrap()),
            scales: hex(r["scales"].as_str().unwrap()),
            bf16: r["bf16_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_u64().unwrap() as u16)
                .collect(),
        })
        .collect();
    let global = f32::from_bits(v["weight_scale_2_bits"].as_u64().unwrap() as u32);
    (
        Nvfp4RowGeom {
            global,
            ..fp4_local()
        },
        rows,
    )
}

/// One interleaved row as `ngram_pack` writes it: codes, scales, zero pad.
fn interleave(g: &Nvfp4RowGeom, packed: &[u8], scales: &[u8]) -> Vec<u8> {
    let mut row = [packed, scales].concat();
    row.resize(g.interleaved_stride(), 0);
    row
}

/// Decode one interleaved row, read in place, to BF16 bits.
fn decode_interleaved(g: &Nvfp4RowGeom, row: &[u8]) -> Vec<u16> {
    let (pw, sw) = (g.packed_stride(), g.scale_stride());
    let p = row[..pw].to_vec();
    let s = row[pw..pw + sw].to_vec();
    decode_rows_bf16(g, &[p], &[s])
        .unwrap()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// BF16 bits equal, with -0 == +0: code 8 is -0 in the kernel and in
/// `modelopt_codec`, +0 in the oracle's table.
fn same_bf16(a: u16, b: u16) -> bool {
    a == b || (a & 0x7FFF == 0 && b & 0x7FFF == 0)
}

/// Every fixture row, interleaved and decoded where the gather reads it,
/// equals the oracle's BF16 bit for bit (up to the sign of zero): the number
/// the GPU gather is held to, through either row layout.
#[test]
fn interleaved_rows_decode_to_the_checkpoint_golden() {
    let (g, rows) = golden();
    assert_eq!(rows.len(), 9);
    // weight_scale_2 is a real multiplier here (~3.3e-5): a dequant that
    // dropped or inverted it could not pass.
    assert!(g.global > 0.0 && g.global < 1e-3, "global {}", g.global);
    let mut nonzero = 0;
    for (k, r) in rows.iter().enumerate() {
        assert_eq!(
            (r.packed.len(), r.scales.len(), r.bf16.len()),
            (80, 10, 160)
        );
        let got = decode_interleaved(&g, &interleave(&g, &r.packed, &r.scales));
        for (i, (&a, &w)) in got.iter().zip(&r.bf16).enumerate() {
            assert!(
                same_bf16(a, w),
                "row {k} elem {i}: got {a:#06x} want {w:#06x}"
            );
        }
        nonzero += got.iter().filter(|b| *b & 0x7FFF != 0).count();
    }
    // Eight live rows (~150 nonzero each) and one all-zero padding row.
    assert!(nonzero > 8 * 140, "only {nonzero} nonzero values");
}

/// `weight_scale_2` MULTIPLIES (ModelOpt's amax-derived global, ~3.3e-5
/// here): a divide, or dropping it, misses every nonzero golden value.
#[test]
fn global_scale_multiplies() {
    let (g, rows) = golden();
    let zeros = rows
        .iter()
        .flat_map(|r| &r.bf16)
        .filter(|w| **w & 0x7FFF == 0)
        .count();
    for wrong in [1.0 / g.global, 1.0] {
        let w = Nvfp4RowGeom { global: wrong, ..g };
        let same: usize = rows
            .iter()
            .map(|r| {
                let got = decode_interleaved(&w, &interleave(&w, &r.packed, &r.scales));
                got.iter()
                    .zip(&r.bf16)
                    .filter(|(a, b)| same_bf16(**a, **b))
                    .count()
            })
            .sum();
        assert_eq!(same, zeros, "global {wrong}: only the zeros may agree");
    }
}

/// The whole `ATLAS_PLE_PACKED_DIR` read path on a GPU: the golden rows,
/// scattered among junk rows in a synthetic two-plane table (three shards
/// over two files, scale plane apart from the code plane), packed by
/// `ngram_pack`, faulted through the O_DIRECT row cache (row 42 straddles a
/// 4 KiB block: 42 x 96 = 4032), and gathered by `batched_embed_nvfp4` with
/// the interleaved strides. Every value must equal the oracle's BF16 (golden
/// rows) or the host decode (junk rows), bit for bit.
///
/// GPU + filesystem test:
///   cargo test -p spark-model --release nvfp4_packed_gather -- --ignored
#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU"]
fn nvfp4_packed_gather_matches_the_golden_through_pack_and_cache() {
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_storage::ngram_pack::{PackSpec, Plane, Verify, open_or_pack};

    let (geom, golden_rows) = golden();
    let (cb, sb) = (geom.packed_stride(), geom.scale_stride());
    let rows_per = 50u64;
    // Golden row k lands at these global rows: shard starts, 42 (the block
    // straddle), shard ends, interior rows.
    let at: [u64; 9] = [0, 42, 49, 50, 77, 99, 100, 123, 149];
    let junk = |r: u64, i: usize| (r.wrapping_mul(0x9E37) ^ (i as u64 * 29)) as u8;
    let row_of = |r: u64| -> (Vec<u8>, Vec<u8>) {
        match at.iter().position(|&a| a == r) {
            Some(k) => (golden_rows[k].packed.clone(), golden_rows[k].scales.clone()),
            None => (
                (0..cb).map(|i| junk(r, i)).collect(),
                (0..sb).map(|i| junk(r, 100 + i)).collect(),
            ),
        }
    };
    let dir = std::env::temp_dir().join(format!("nvfp4_packed_gather_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (fa, fb) = (dir.join("a.safetensors"), dir.join("b.safetensors"));
    let (mut a, mut b) = (vec![7u8; 1001], vec![9u8; 333]);
    let (mut codes, mut scales) = (Vec::new(), Vec::new());
    for shard in 0..3u64 {
        // Shard 1's codes live in file B, the rest in A; scales in B except
        // shard 2's. Offsets are odd on purpose.
        let (cbuf, cpath) = if shard == 1 {
            (&mut b, &fb)
        } else {
            (&mut a, &fa)
        };
        codes.push((cpath.clone(), cbuf.len() as u64));
        for r in 0..rows_per {
            cbuf.extend(row_of(shard * rows_per + r).0);
        }
        let (sbuf, spath) = if shard == 2 {
            (&mut a, &fa)
        } else {
            (&mut b, &fb)
        };
        scales.push((spath.clone(), sbuf.len() as u64));
        for r in 0..rows_per {
            sbuf.extend(row_of(shard * rows_per + r).1);
        }
    }
    std::fs::write(&fa, &a).unwrap();
    std::fs::write(&fb, &b).unwrap();
    let spec = PackSpec {
        planes: vec![
            Plane {
                shards: &codes,
                row_bytes: cb,
            },
            Plane {
                shards: &scales,
                row_bytes: sb,
            },
        ],
        rows_per_shard: rows_per,
        stride: geom.interleaved_stride(),
        tag: "nvfp4g16",
    };
    let table = open_or_pack(&spec, &dir.join("packed"), Verify::Full).unwrap();

    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g
        .kernel("embed_from_argmax", "batched_embed_nvfp4")
        .unwrap();
    let mut cache = table.open_row_cache(64).unwrap();
    let scaled = cache.scale_dev_va().unwrap().is_some();
    check_packed_stride(cache.row_stride(), &geom, scaled).unwrap();

    // Every golden row, some junk rows, and repeats (cache hits), shuffled.
    let ids: Vec<u64> = vec![149, 0, 42, 13, 77, 42, 50, 99, 141, 100, 123, 49, 0, 88];
    let mut slots = Vec::new();
    cache.resolve(&ids, &mut slots).unwrap();
    let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
    let slots_dev = g.alloc(slot_bytes.len()).unwrap();
    g.copy_h2d(&slot_bytes, slots_dev).unwrap();
    let out = g.alloc(ids.len() * geom.dim * 2).unwrap();
    let table_va = cache.table_dev_va().unwrap();
    let (scale_va, row_stride, scale_stride) =
        geom.gather_planes(table_va, cache.block_scale_dev_va().unwrap());
    crate::layers::ops::batched_embed_nvfp4(
        g,
        k,
        slots_dev,
        DevicePtr(table_va),
        DevicePtr(scale_va),
        geom.global,
        out,
        ids.len() as u32,
        geom.dim as u32,
        geom.group as u32,
        row_stride as u32,
        scale_stride as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    cache.end_batch();
    let mut raw = vec![0u8; ids.len() * geom.dim * 2];
    g.copy_d2h(out, &mut raw).unwrap();
    let got: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let (mut golden_checked, mut junk_checked) = (0, 0);
    for (n, &id) in ids.iter().enumerate() {
        let want = match at.iter().position(|&a| a == id) {
            Some(k) => {
                golden_checked += 1;
                golden_rows[k].bf16.clone()
            }
            None => {
                junk_checked += 1;
                let (c, s) = row_of(id);
                decode_interleaved(&geom, &interleave(&geom, &c, &s))
            }
        };
        for i in 0..geom.dim {
            let gv = got[n * geom.dim + i];
            assert!(
                same_bf16(gv, want[i]),
                "id {id} elem {i}: got {gv:#06x} want {:#06x}",
                want[i]
            );
        }
    }
    let (hits, misses, _) = cache.stats();
    println!(
        "{} rows gathered ({golden_checked} golden, {junk_checked} junk), \
         cache {hits} hits / {misses} misses: all bit-exact",
        ids.len()
    );
    assert_eq!((golden_checked, junk_checked), (11, 3));
    std::fs::remove_dir_all(&dir).unwrap();
}
