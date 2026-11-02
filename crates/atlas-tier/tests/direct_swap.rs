// SPDX-License-Identifier: AGPL-3.0-only

//! Public-API integration tests for [`DirectSwapFile`] — real O_DIRECT I/O
//! under `target/atlas-tier-tests`, with an EINVAL-skip when the filesystem
//! (tmpfs/overlay) refuses O_DIRECT so containerized CI doesn't break.

use std::path::Path;

use atlas_tier::{DirectSwapFile, Residency, SwapStore, VecSlotArena};

/// A real-filesystem dir for O_DIRECT (tmpfs/overlay EINVALs on O_DIRECT —
/// tolerated as a skip so containerized CI doesn't break).
fn o_direct_file(record_bytes: usize, tag: &str) -> Option<(DirectSwapFile, std::path::PathBuf)> {
    o_direct_file_with(record_bytes, tag, DirectSwapFile::create)
}

fn o_direct_file_with(
    record_bytes: usize,
    tag: &str,
    create: fn(&Path, usize) -> anyhow::Result<DirectSwapFile>,
) -> Option<(DirectSwapFile, std::path::PathBuf)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/atlas-tier-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("dsf-{tag}-{}.swap", std::process::id()));
    let _ = std::fs::remove_file(&path);
    match create(&path, record_bytes) {
        Ok(f) => Some((f, path)),
        Err(e) => {
            // Opt-in enforcement: CI on a real disk sets this so a silent skip
            // can't green-light a run that never touched the O_DIRECT path.
            if std::env::var_os("ATLAS_TIER_REQUIRE_O_DIRECT").is_some() {
                panic!("ATLAS_TIER_REQUIRE_O_DIRECT set but O_DIRECT unavailable: {e:#}");
            }
            eprintln!("skipping O_DIRECT test (filesystem refused O_DIRECT): {e:#}");
            None
        }
    }
}

/// A page-aligned (4 KiB) mutable sub-slice of `storage`, which must be
/// over-allocated by at least 4096 bytes past `len`. Lets a test deterministically
/// hit the O_DIRECT *aligned fast-path* (plain `Vec`s almost never are).
fn page_aligned(storage: &mut [u8], len: usize) -> &mut [u8] {
    let pad = (4096 - (storage.as_ptr() as usize & 0xfff)) & 0xfff;
    &mut storage[pad..pad + len]
}

/// `create_new`: the one open a tier holding private records needs — refuses
/// an existing file or a planted symlink, owner-only, fully usable.
#[test]
fn create_new_is_exclusive_and_owner_only() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file_with(rb, "new", DirectSwapFile::create_new) else {
        return;
    };
    assert!(DirectSwapFile::create_new(&path, rb).is_err(), "exists");
    assert!(DirectSwapFile::create_new(&path, 1000).is_err());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "no group/other access: {mode:o}");
        let link = path.with_extension("link");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(DirectSwapFile::create_new(&link, rb).is_err());
        let _ = std::fs::remove_file(&link);
    }
    let rec = vec![0x5Au8; rb];
    f.write_record(2, &rec).unwrap();
    let mut out = vec![0u8; rb];
    f.read_record(2, &mut out).unwrap();
    assert_eq!(out, rec);
    let _ = std::fs::remove_file(path);
}

#[test]
fn direct_swap_file_rejects_bad_record_bytes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/atlas-tier-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dsf-bad.swap");
    assert!(
        DirectSwapFile::create(&path, 0).is_err(),
        "zero record_bytes rejected"
    );
    assert!(
        DirectSwapFile::create(&path, 1000).is_err(),
        "non-4KiB multiple rejected"
    );
}

/// O_DIRECT write/read round-trips through the page-aligned bounce (a plain
/// `Vec` caller buffer is usually unaligned, exercising both bounce paths).
#[test]
fn direct_swap_file_roundtrips_records() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "rt") else {
        return;
    };
    assert_eq!(f.record_bytes(), rb);
    let mut pat = vec![0u8; rb];
    for (i, b) in pat.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    f.write_record(3, &pat).unwrap(); // sparse: slot 3 before slot 0
    f.write_record(0, &vec![0xEE; rb]).unwrap();
    let mut out = vec![0u8; rb];
    f.read_record(3, &mut out).unwrap();
    assert_eq!(out, pat, "record 3 byte-identical");
    f.read_record(0, &mut out).unwrap();
    assert_eq!(out, vec![0xEE; rb], "record 0 byte-identical");
    // Size validation is a hard error, not a short IO.
    assert!(f.write_record(1, &pat[..100]).is_err());
    let mut short = vec![0u8; 100];
    assert!(f.read_record(0, &mut short).is_err());
    let _ = std::fs::remove_file(path);
}

/// End-to-end: the residency spills to a REAL O_DIRECT file and faults back
/// byte-identical (the exact peer configuration, minus RDMA).
#[test]
fn residency_over_o_direct_swap_byte_identical() {
    let rb = 4096usize;
    let Some((f, path)) = o_direct_file(rb, "resid") else {
        return;
    };
    let mut r = Residency::new(VecSlotArena::new(rb, 2), f).unwrap();
    for k in 0..8u64 {
        r.put_blob(k, &vec![k as u8; rb]).unwrap();
    }
    assert_eq!(r.total_keys(), 8);
    assert!(
        r.stats().spills_to_disk >= 6,
        "cold keys spilled to the O_DIRECT file"
    );
    let mut out = vec![0u8; rb];
    for k in 0..8u64 {
        assert!(r.get_blob(k, &mut out).unwrap(), "key {k}");
        assert_eq!(
            out,
            vec![k as u8; rb],
            "key {k} byte-identical through O_DIRECT"
        );
    }
    let _ = std::fs::remove_file(path);
}

/// Deterministically exercise the O_DIRECT *aligned fast-path* (the direct
/// pwrite/pread, not the bounce staging): pass page-aligned buffers so
/// `write_record`/`read_record` take the `is_aligned()` branch. The existing
/// `Vec`-based tests almost always hit the bounce instead, so this is the only
/// coverage of the zero-copy path the peer's mmap'd arena actually uses.
#[test]
fn direct_swap_aligned_fast_path_roundtrips() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "aligned") else {
        return;
    };
    let mut wstore = vec![0u8; rb + 4096];
    let w = page_aligned(&mut wstore, rb);
    assert_eq!(
        w.as_ptr() as usize & 0xfff,
        0,
        "write buffer is page-aligned → fast-path"
    );
    for (i, b) in w.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    f.write_record(7, w).unwrap(); // aligned src → direct pwrite

    let mut rstore = vec![0u8; rb + 4096];
    let rbuf = page_aligned(&mut rstore, rb);
    assert_eq!(
        rbuf.as_ptr() as usize & 0xfff,
        0,
        "read buffer is page-aligned → fast-path"
    );
    f.read_record(7, rbuf).unwrap(); // aligned dst → direct pread
    for (i, b) in rbuf.iter().enumerate() {
        assert_eq!(*b, (i % 251) as u8, "aligned fast-path byte {i}");
    }
    let _ = std::fs::remove_file(path);
}

/// The other half of "the swap file is 0 bytes": once a record IS written, the
/// file grows immediately. `write_record` is a bare `pwrite`, so it extends
/// `i_size` on the first byte at offset 0 — meaning a 0-byte file proves zero
/// write calls, never a silently-dropped write. Skips where O_DIRECT is refused
/// (tmpfs), like the other tests in this file.
#[test]
fn direct_swap_file_grows_on_first_write() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "grow") else {
        return;
    };
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        0,
        "freshly created (truncated) swap file starts empty"
    );
    f.write_record(0, &vec![0x7C; rb]).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        rb as u64,
        "one record written ⇒ file is exactly one record long"
    );
    // Sparse addressing: a high slot sets i_size to (slot + 1) * record_bytes.
    f.write_record(4, &vec![0x7D; rb]).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 5 * rb as u64);
    let _ = std::fs::remove_file(path);
}

/// `read_records` returns consecutive records in slot order through both the
/// aligned single-`pread` path and the unaligned per-record fallback, and a
/// run reaching past the written end of the file is an error, not zeros.
#[test]
fn direct_swap_file_reads_consecutive_record_runs() {
    let rb = 4096usize;
    let Some((mut f, path)) = o_direct_file(rb, "runs") else {
        return;
    };
    let mut src = vec![0u8; rb + 4096];
    for slot in 0..4usize {
        let rec = page_aligned(&mut src, rb);
        rec.fill(0x10 + slot as u8);
        f.write_record(slot, rec).unwrap();
    }
    let mut storage = vec![0u8; 3 * rb + 4096];
    let aligned = page_aligned(&mut storage, 3 * rb);
    f.read_records(1, aligned).unwrap();
    for (i, rec) in aligned.chunks_exact(rb).enumerate() {
        assert!(
            rec.iter().all(|&b| b == 0x11 + i as u8),
            "aligned record {i}"
        );
    }
    let mut unaligned = vec![0u8; 2 * rb + 1];
    f.read_records(2, &mut unaligned[1..]).unwrap();
    assert!(unaligned[1..=rb].iter().all(|&b| b == 0x12));
    assert!(unaligned[rb + 1..].iter().all(|&b| b == 0x13));
    let mut past = vec![0u8; 2 * rb + 4096];
    assert!(f.read_records(3, page_aligned(&mut past, 2 * rb)).is_err());
    let _ = std::fs::remove_file(path);
}
