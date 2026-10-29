// SPDX-License-Identifier: AGPL-3.0-only

//! The packer's offset math, on synthetic split-row tables whose shards are
//! scattered the way the FP4-Local checkpoint's are: several files, shards
//! out of order, other bytes in between, and the scale plane in a different
//! region from the data plane.

use super::check::{read_header, sample_row, verify_chunks};
use super::write::header_fits;
use super::*;

mod integrity;

/// A deterministic, row- and plane-unique byte, so a row read from the wrong
/// shard, plane or offset can never compare equal by accident.
fn byte(plane: usize, row: u64, i: usize) -> u8 {
    (row.wrapping_mul(131) ^ (plane as u64 * 0x55) ^ (i as u64 * 7)) as u8
}

struct Fixture {
    dir: PathBuf,
    data: Vec<(PathBuf, u64)>,
    scale: Vec<(PathBuf, u64)>,
    rows: u64,
}

/// Three shards over two files. File A: junk, data shard 2, junk, data
/// shard 0, scale shard 1. File B: scale shard 0, junk, data shard 1, scale
/// shard 2. Data rows 5 B, scale rows 3 B, odd offsets throughout.
fn fixture(tag: &str, rows: u64) -> Fixture {
    let dir = std::env::temp_dir().join(format!("ngram_pack_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (a, b) = (dir.join("a.safetensors"), dir.join("b.safetensors"));
    let (dw, sw) = (5usize, 3usize);
    let mut fa = vec![0xAAu8; 13];
    let mut fb = Vec::new();
    let mut data = vec![(PathBuf::new(), 0u64); 3];
    let mut scale = vec![(PathBuf::new(), 0u64); 3];
    let put = |buf: &mut Vec<u8>, path: &Path, plane: usize, shard: usize, w: usize| {
        let off = buf.len() as u64;
        for r in 0..rows {
            let g = shard as u64 * rows + r;
            buf.extend((0..w).map(|i| byte(plane, g, i)));
        }
        (path.to_path_buf(), off)
    };
    data[2] = put(&mut fa, &a, 0, 2, dw);
    fa.extend([0xEEu8; 9]);
    data[0] = put(&mut fa, &a, 0, 0, dw);
    scale[1] = put(&mut fa, &a, 1, 1, sw);
    scale[0] = put(&mut fb, &b, 1, 0, sw);
    fb.extend([0xBBu8; 4097]);
    data[1] = put(&mut fb, &b, 0, 1, dw);
    scale[2] = put(&mut fb, &b, 1, 2, sw);
    std::fs::write(&a, &fa).unwrap();
    std::fs::write(&b, &fb).unwrap();
    Fixture {
        dir,
        data,
        scale,
        rows,
    }
}

impl Fixture {
    fn spec(&self) -> PackSpec<'_> {
        PackSpec {
            planes: vec![
                Plane {
                    shards: &self.data,
                    row_bytes: 5,
                },
                Plane {
                    shards: &self.scale,
                    row_bytes: 3,
                },
            ],
            rows_per_shard: self.rows,
            stride: 16,
            tag: "test",
        }
    }
}

/// Overwrite `path` in place (same inode) and restore its mtime.
fn rewrite_keeping_mtime(path: &Path, bytes: &[u8]) {
    let mtime = std::fs::metadata(path).unwrap().modified().unwrap();
    std::fs::write(path, bytes).unwrap();
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
}

fn expected_row(g: u64) -> Vec<u8> {
    let mut v: Vec<u8> = (0..5).map(|i| byte(0, g, i)).collect();
    v.extend((0..3).map(|i| byte(1, g, i)));
    v.resize(16, 0);
    v
}

#[test]
fn interleave_concatenates_planes_and_zeroes_the_tail() {
    let a = [1u8, 2, 3, 4];
    let b = [9u8, 8];
    let mut out = [0xFFu8; 8];
    interleave(&[&a, &b], &[2, 1], 4, &mut out);
    assert_eq!(out, [1, 2, 9, 0, 3, 4, 8, 0]);
}

#[test]
fn source_loc_follows_each_shard_and_plane_base() {
    let f = fixture("loc", 10);
    let s = f.spec();
    // Row 23 = shard 2, local 3. Each plane has its own base AND stride.
    assert_eq!(
        s.source_loc(0, 23),
        (f.data[2].0.as_path(), f.data[2].1 + 3 * 5)
    );
    assert_eq!(
        s.source_loc(1, 23),
        (f.scale[2].0.as_path(), f.scale[2].1 + 3 * 3)
    );
    assert_eq!(s.source_loc(0, 10), (f.data[1].0.as_path(), f.data[1].1));
    assert_eq!(packed_offset(0, 96), HEADER);
    assert_eq!(packed_offset(320_001_535, 96), HEADER + 320_001_535 * 96);
    assert_eq!(HEADER % 4096, 0, "row 0 must start O_DIRECT-aligned");
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// Every packed row, read back at `HEADER + row * stride`, is the two source
/// rows it claims to be — across chunk boundaries (chunk 7 does not divide
/// the 10-row shards) and across files.
#[test]
fn pack_reassembles_every_row_from_scattered_shards() {
    let f = fixture("rows", 10);
    let spec = f.spec();
    let out = f.dir.join("packed.bin");
    pack(&spec, &out, 7, 4).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(bytes.len() as u64, packed_offset(30, 16));
    for g in 0..30u64 {
        let at = packed_offset(g, 16) as usize;
        assert_eq!(&bytes[at..at + 16], expected_row(g).as_slice(), "row {g}");
    }
    validate(&spec, &out, Verify::Full).unwrap();
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn open_or_pack_reuses_a_valid_file_and_rebuilds_a_stale_one() {
    let f = fixture("reuse", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    assert_eq!((t.base_offset, t.rows_total, t.stride), (HEADER, 30, 16));
    assert!(t.path.starts_with(&cache));
    assert!(
        t.path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("ngram-test-")
    );

    // A second open must not rewrite it: stamp a byte in the header padding
    // (outside anything validated) and see it survive.
    let mut bytes = std::fs::read(&t.path).unwrap();
    bytes[4000] = 0x5A;
    std::fs::write(&t.path, &bytes).unwrap();
    let again = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    assert_eq!(again.path, t.path);
    assert_eq!(std::fs::read(&t.path).unwrap()[4000], 0x5A);

    // A valid table on a read-only directory (packed at install time) opens
    // without the lock file it could not create.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::remove_file(cache.join(format!("{}.lock", spec.file_name().unwrap())));
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o555)).unwrap();
        let ro = open_or_pack(&spec, &cache, Verify::Full);
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(ro.unwrap().path, t.path);
    }

    // Same layout, new values: change a SAMPLED row (shard 0, row 0) in the
    // source and put its mtime back. Identity is unchanged (same names,
    // lengths, offsets, mtimes, inodes), so only the row spot-check can catch
    // it — and it must, then rebuild.
    let (a, off) = f.data[0].clone();
    let mut src = std::fs::read(&a).unwrap();
    src[off as usize] ^= 0xFF;
    rewrite_keeping_mtime(&a, &src);
    let e = validate(&spec, &t.path, Verify::Full).unwrap_err();
    assert!(
        format!("{e:#}").contains("does not match its sources"),
        "{e:#}"
    );
    let rebuilt = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    let bytes = std::fs::read(&rebuilt.path).unwrap();
    assert_eq!(bytes[HEADER as usize], byte(0, 0, 0) ^ 0xFF);
    assert_eq!(bytes[4000], 0, "rebuilt from scratch");
    let leftovers: Vec<_> = std::fs::read_dir(&cache)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temp files left behind");
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn identity_names_sources_and_geometry() {
    let f = fixture("id", 10);
    let spec = f.spec();
    let id = spec.identity().unwrap();
    let name = spec.file_name().unwrap();
    // A longer source file is a different checkpoint: new identity, new name.
    let (b, _) = &f.data[1];
    let mut src = std::fs::read(b).unwrap();
    src.push(0);
    std::fs::write(b, &src).unwrap();
    assert_ne!(spec.identity().unwrap(), id);
    assert_ne!(spec.file_name().unwrap(), name);
    // Stride is geometry too.
    let wider = PackSpec {
        stride: 32,
        ..f.spec()
    };
    assert_ne!(wider.identity().unwrap(), spec.identity().unwrap());
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn bad_specs_and_bad_files_are_refused() {
    let f = fixture("bad", 10);
    // Planes wider than the stride.
    let narrow = PackSpec {
        stride: 7,
        ..f.spec()
    };
    assert!(narrow.identity().is_err());
    // Mismatched shard counts between planes.
    let short = PackSpec {
        planes: vec![
            Plane {
                shards: &f.data,
                row_bytes: 5,
            },
            Plane {
                shards: &f.scale[..2],
                row_bytes: 3,
            },
        ],
        ..f.spec()
    };
    assert!(short.identity().is_err());
    // Truncated and wrong-magic files.
    let spec = f.spec();
    let out = f.dir.join("p.bin");
    pack(&spec, &out, 64, 4).unwrap();
    let full = std::fs::read(&out).unwrap();
    std::fs::write(&out, &full[..full.len() - 1]).unwrap();
    assert!(validate(&spec, &out, Verify::Full).is_err());
    let mut bad = full.clone();
    bad[0] = b'X';
    std::fs::write(&out, &bad).unwrap();
    assert!(validate(&spec, &out, Verify::Full).is_err());
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn sample_rows_stay_in_range_and_cover_both_ends() {
    assert_eq!(sample_row(0, 2_500_012), 0);
    assert_eq!(sample_row(1, 2_500_012), 2_500_011);
    for s in 0..128 {
        assert!(sample_row(s, 2_500_012) < 2_500_012);
    }
}

/// The FP4-Local geometry: 128 shards per plane over 7 files with 10-digit
/// offsets. Its identity must fit the header, or the first real pack fails.
#[test]
fn identity_of_the_real_table_fits() {
    let dir = std::env::temp_dir().join(format!("ngram_pack_real_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let files: Vec<PathBuf> = (15..22)
        .map(|i| {
            let p = dir.join(format!("model-{i:05}-of-00021.safetensors"));
            std::fs::write(&p, b"x").unwrap();
            p
        })
        .collect();
    let plane = |base: u64| -> Vec<(PathBuf, u64)> {
        (0..128u64)
            .map(|s| (files[(s % 7) as usize].clone(), base + s * 200_000_960))
            .collect()
    };
    let (codes, scales) = (plane(3_488_889_896), plane(2_950_357_208));
    let spec = PackSpec {
        planes: vec![
            Plane {
                shards: &codes,
                row_bytes: 80,
            },
            Plane {
                shards: &scales,
                row_bytes: 10,
            },
        ],
        rows_per_shard: 2_500_012,
        stride: 96,
        tag: "nvfp4g16",
    };
    let id = spec.identity().unwrap();
    let chunks = spec.rows_total().div_ceil(spec.check_rows());
    println!("identity: {} bytes, {chunks} chunk checksums", id.len());
    // The identity format names every packed file: a change to it orphans
    // each table already packed on disk and rebuilds 29 GB on the next boot.
    assert_eq!(
        spec.file_name().unwrap(),
        "ngram-nvfp4g16-7c53f1fcdd8b0ed5.bin"
    );
    assert!(id.len() > 4096, "the case the 64 KiB header exists for");
    assert_eq!(spec.check_rows(), CHUNK_ROWS, "one checksum per pack pass");
    assert_eq!(chunks, 1221);
    header_fits(id.len() as u64, chunks).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}
