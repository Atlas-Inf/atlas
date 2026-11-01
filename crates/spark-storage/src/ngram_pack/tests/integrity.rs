// SPDX-License-Identifier: AGPL-3.0-only

//! Integrity of a packed file once it is on disk: corrupt headers, damaged
//! rows, touched sources, symlinks and swapped files are refused and rebuilt,
//! never served.

use super::*;

/// A corrupt or hostile header length must be refused, never trusted: on the
/// old `16 + len` check a length near `u64::MAX` wrapped past the range test
/// in a release build and `vec![0; len]` aborted the process, so a bad packed
/// file crash-looped the server instead of being rebuilt.
#[test]
fn a_corrupt_header_length_is_refused_and_rebuilt() {
    let f = fixture("hdrlen", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    let good = std::fs::read(&t.path).unwrap();
    for len in [u64::MAX, u64::MAX - 15, u64::MAX / 2, HEADER, HEADER - 15] {
        let mut bad = good.clone();
        bad[8..16].copy_from_slice(&len.to_le_bytes());
        std::fs::write(&t.path, &bad).unwrap();
        let e = validate(&spec, &t.path, Verify::Full).unwrap_err();
        assert!(
            format!("{e:#}").contains("out of range"),
            "len {len}: {e:#}"
        );
        // `open_or_pack` treats it as stale and rebuilds a valid file.
        let again = open_or_pack(&spec, &cache, Verify::Full).unwrap();
        assert_eq!(again.path, t.path);
        validate(&spec, &t.path, Verify::Full).unwrap();
    }
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// A packed file damaged after it was written, in a row the per-shard spot
/// check never reads, is caught by its chunk checksum: always under
/// `Verify::Full`, and under `Verify::Sample` whenever the sample includes the
/// chunk. `open_or_pack` then rebuilds it.
#[test]
fn a_damaged_unsampled_row_fails_its_chunk_checksum() {
    let f = fixture("chunks", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let path = cache.join(spec.file_name().unwrap());
    std::fs::create_dir_all(&cache).unwrap();
    // 30 rows, 4 per chunk: 8 chunks, the last one short.
    pack(&spec, &path, 7, 4).unwrap();
    validate(&spec, &path, Verify::Full).unwrap();
    // Row 13 (chunk 3) is no shard's spot-check row (0, 19, and shard 2's).
    assert!((0..3).all(|s| s * 10 + sample_row(s, 10) != 13));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[packed_offset(13, 16) as usize + 2] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let e = validate(&spec, &path, Verify::Full).unwrap_err();
    assert!(format!("{e:#}").contains("chunk 3"), "{e:#}");
    let f2 = File::open(&path).unwrap();
    let hdr = read_header(&f2).unwrap();
    assert_eq!((hdr.check_rows, hdr.sums.len()), (4, 8));
    // The chunks the sample would have to draw: the damaged one fails, the
    // rest pass, so it is the checksum and not the spot check that caught it.
    assert!(verify_chunks(&f2, &hdr, 30, 16, &[3]).is_err());
    verify_chunks(&f2, &hdr, 30, 16, &[0, 1, 2, 4, 5, 6, 7]).unwrap();
    drop(f2);

    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    assert_eq!(t.path, path);
    let rebuilt = std::fs::read(&path).unwrap();
    let at = packed_offset(13, 16) as usize;
    assert_eq!(&rebuilt[at..at + 16], expected_row(13).as_slice());
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// A source rewritten in place with the same name and length is a new
/// identity through its mtime alone, and is rebuilt under the SAME file name
/// (the name depends on the layout only), leaving no stale copy behind.
#[test]
fn a_touched_source_is_a_new_identity_under_the_same_name() {
    let f = fixture("touch", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let t = open_or_pack(&spec, &cache, Verify::Sample).unwrap();
    let (id, name) = (spec.identity().unwrap(), spec.file_name().unwrap());
    let (b, _) = &f.data[1];
    let later =
        std::fs::metadata(b).unwrap().modified().unwrap() + std::time::Duration::from_secs(5);
    File::options()
        .write(true)
        .open(b)
        .unwrap()
        .set_modified(later)
        .unwrap();
    assert_ne!(spec.identity().unwrap(), id);
    assert_eq!(spec.file_name().unwrap(), name);
    let e = validate(&spec, &t.path, Verify::Sample).unwrap_err();
    assert!(format!("{e:#}").contains("identity differs"), "{e:#}");
    assert_eq!(
        open_or_pack(&spec, &cache, Verify::Sample).unwrap().path,
        t.path
    );
    validate(&spec, &t.path, Verify::Full).unwrap();
    let bins = std::fs::read_dir(&cache)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".bin"))
        .count();
    assert_eq!(bins, 1);
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// Corrupt checksum geometry in the header is refused, never trusted.
#[test]
fn corrupt_checksum_geometry_is_refused() {
    let f = fixture("geom", 10);
    let spec = f.spec();
    let out = f.dir.join("p.bin");
    pack(&spec, &out, 64, 4).unwrap();
    let good = std::fs::read(&out).unwrap();
    for (at, value) in [
        (24, u64::MAX),     // chunk count
        (24, u64::MAX / 4), // chunk count that wraps when multiplied
        (24, HEADER),       // more sums than the header holds
        (24, 7),            // fits, but not ceil(30 / 4)
        (16, 0),            // zero rows per chunk
        (16, 5),            // rows per chunk that does not give 8 chunks
    ] {
        let mut bad = good.clone();
        bad[at..at + 8].copy_from_slice(&value.to_le_bytes());
        std::fs::write(&out, &bad).unwrap();
        assert!(
            validate(&spec, &out, Verify::Full).is_err(),
            "word @{at} = {value} accepted"
        );
    }
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn verify_mode_parses_leniently() {
    assert_eq!(Verify::parse(Some("full")), Verify::Full);
    for v in [None, Some(""), Some("1"), Some("FULL"), Some("bogus")] {
        assert_eq!(Verify::parse(v), Verify::Sample, "{v:?}");
    }
}

/// The cost of the default verify, at the NVFP4 table's real geometry
/// (96-byte rows, 24 MiB chunks), on whatever disk `$TMPDIR` is: packs a
/// 20-chunk (504 MB) single-plane table and times `Sample` (16 chunks) and
/// `Full`. Not a unit test; run by hand:
/// ```text
/// cargo test -p spark-storage --release --lib verify_cost -- --ignored --nocapture
/// ```
#[test]
#[ignore = "benchmark, run by hand"]
fn verify_cost() {
    let dir = std::env::temp_dir().join(format!("ngram_pack_cost_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("src.bin");
    let rows = 20 * (CHECK_BYTES / 96);
    let bytes: Vec<u8> = (0..rows * 90)
        .map(|i| ((i * 2654435761) >> 13) as u8)
        .collect();
    std::fs::write(&src, &bytes).unwrap();
    drop(bytes);
    let shards = vec![(src, 0u64)];
    let spec = PackSpec {
        planes: vec![Plane {
            shards: &shards,
            row_bytes: 90,
        }],
        rows_per_shard: rows,
        stride: 96,
        tag: "cost",
    };
    let t = std::time::Instant::now();
    let table = open_or_pack(&spec, &dir, Verify::Sample).unwrap();
    println!("pack + sample verify: {:.2} s", t.elapsed().as_secs_f64());
    for mode in [Verify::Sample, Verify::Full, Verify::Sample] {
        let t = std::time::Instant::now();
        validate(&spec, &table.path, mode).unwrap();
        println!(
            "{mode:?} verify of 20 chunks: {:.3} s",
            t.elapsed().as_secs_f64()
        );
    }
    let mut h = crc32fast::Hasher::new();
    let buf = vec![7u8; 256 << 20];
    let t = std::time::Instant::now();
    h.update(&buf);
    let _ = h.finalize();
    println!(
        "crc32: {:.2} GB/s",
        buf.len() as f64 / 1e9 / t.elapsed().as_secs_f64()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Nothing the packer opens for writing follows a symlink: a temp file, a
/// lock or a packed file planted as a link to another file must never write
/// through to that file.
#[cfg(unix)]
#[test]
fn symlinks_are_never_followed() {
    use std::os::unix::fs::symlink;
    let f = fixture("links", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    std::fs::create_dir_all(&cache).unwrap();
    let victim = f.dir.join("victim");
    std::fs::write(&victim, b"precious").unwrap();
    let name = spec.file_name().unwrap();

    // The temp file: pack() refuses a path that is a link.
    let tmp = cache.join(format!("{name}.tmp1"));
    symlink(&victim, &tmp).unwrap();
    assert!(pack(&spec, &tmp, 7, 4).is_err());
    std::fs::remove_file(&tmp).unwrap();

    // The lock: refused, so the build is too.
    let lock = cache.join(format!("{name}.lock"));
    symlink(&victim, &lock).unwrap();
    let e = open_or_pack(&spec, &cache, Verify::Full).unwrap_err();
    assert!(format!("{e:#}").contains("lock"), "{e:#}");
    std::fs::remove_file(&lock).unwrap();

    // The packed file itself, linked to a VALID packed table elsewhere: not
    // used through the link, but replaced by a regular file of its own.
    let elsewhere = f.dir.join("elsewhere.bin");
    pack(&spec, &elsewhere, 7, 4).unwrap();
    let before = std::fs::read(&elsewhere).unwrap();
    let path = cache.join(&name);
    symlink(&elsewhere, &path).unwrap();
    assert!(validate(&spec, &path, Verify::Full).is_err());
    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    assert!(!std::fs::symlink_metadata(&t.path).unwrap().is_symlink());
    validate(&spec, &t.path, Verify::Full).unwrap();
    assert_eq!(std::fs::read(&elsewhere).unwrap(), before);
    assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// The table `open_or_pack` returns names the file it VALIDATED, by inode,
/// so a file swapped in under the same name between the check and the row
/// cache's own open is refused rather than served.
#[cfg(unix)]
#[test]
fn a_file_swapped_in_after_validation_is_refused() {
    let f = fixture("swap", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    t.check_same_file(&File::open(&t.path).unwrap()).unwrap();
    // Same bytes, new inode: what a concurrent rebuild's rename leaves.
    let copy = cache.join("copy.bin");
    std::fs::copy(&t.path, &copy).unwrap();
    std::fs::rename(&copy, &t.path).unwrap();
    let e = t
        .check_same_file(&File::open(&t.path).unwrap())
        .unwrap_err();
    assert!(format!("{e:#}").contains("replaced"), "{e:#}");
    std::fs::remove_dir_all(&f.dir).unwrap();
}
