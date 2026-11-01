// SPDX-License-Identifier: AGPL-3.0-only

//! [`SharedRecordFile`] and the [`ConcurrentSwapStore`] run contract: a run
//! lands in slot order or back to front, from aligned and unaligned buffers,
//! from several threads at once — and the `Mutex<SwapStore>` adapter agrees
//! with the file byte for byte.

use std::path::Path;
use std::sync::{Arc, Mutex};

use atlas_tier::{ConcurrentSwapStore, MemSwapStore, SharedRecordFile};

const RB: usize = 4096;

/// Scratch space under the cargo target directory (`CARGO_TARGET_DIR` and
/// `build.target-dir` honoured).
fn scratch() -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("atlas-tier-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A real-filesystem file (tmpfs/overlay refuse O_DIRECT — tolerated as a skip
/// unless `ATLAS_TIER_REQUIRE_O_DIRECT` is set, as in `direct_swap.rs`).
fn record_file(tag: &str) -> Option<(SharedRecordFile, std::path::PathBuf)> {
    let dir = scratch();
    let path = dir.join(format!("srf-{tag}-{}.swap", std::process::id()));
    let _ = std::fs::remove_file(&path);
    match SharedRecordFile::create(&path, RB) {
        Ok(f) => Some((f, path)),
        Err(e) => {
            if std::env::var_os("ATLAS_TIER_REQUIRE_O_DIRECT").is_some() {
                panic!("ATLAS_TIER_REQUIRE_O_DIRECT set but O_DIRECT unavailable: {e:#}");
            }
            eprintln!("skipping O_DIRECT test (filesystem refused O_DIRECT): {e:#}");
            None
        }
    }
}

fn record(slot: usize) -> Vec<u8> {
    (0..RB).map(|i| (slot * 37 + i % 251) as u8).collect()
}

/// `n` records for slots `low..low + n`, in slot order or back to front.
fn run(low: usize, n: usize, reversed: bool) -> Vec<u8> {
    let slots: Vec<usize> = if reversed {
        (low..low + n).rev().collect()
    } else {
        (low..low + n).collect()
    };
    slots.into_iter().flat_map(record).collect()
}

/// A 4 KiB-aligned window of `len` bytes inside an over-allocated `Vec`.
fn aligned(storage: &mut Vec<u8>, len: usize) -> &mut [u8] {
    storage.resize(len + 4096, 0);
    let pad = (4096 - (storage.as_ptr() as usize & 0xfff)) & 0xfff;
    &mut storage[pad..pad + len]
}

fn round_trips(store: &dyn ConcurrentSwapStore) {
    assert_eq!(store.record_bytes(), RB);
    // Written back to front, read in slot order — and the other way round.
    store.write_run(4, true, &run(4, 5, true)).unwrap();
    let mut out = vec![0u8; 5 * RB];
    store.read_run(4, false, &mut out).unwrap();
    assert_eq!(out, run(4, 5, false));
    store.write_run(20, false, &run(20, 3, false)).unwrap();
    let mut out = vec![0u8; 3 * RB];
    store.read_run(20, true, &mut out).unwrap();
    assert_eq!(out, run(20, 3, true));
    // A sub-run and a single record of an existing run.
    let mut one = vec![0u8; RB];
    store.read_run(6, true, &mut one).unwrap();
    assert_eq!(one, record(6));
    let mut two = vec![0u8; 2 * RB];
    store.read_run(5, true, &mut two).unwrap();
    assert_eq!(two, run(5, 2, true));
    // Ragged and empty buffers are refused, not truncated.
    assert!(store.read_run(0, false, &mut [0u8; 100]).is_err());
    assert!(store.write_run(0, false, &[]).is_err());
}

#[test]
fn shared_record_file_moves_runs_in_both_directions() {
    let Some((f, path)) = record_file("runs") else {
        return;
    };
    round_trips(&f);
    // Page-aligned buffers take the direct (no bounce) arm.
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let src = aligned(&mut a, 4 * RB);
    src.copy_from_slice(&run(40, 4, true));
    f.write_run(40, true, src).unwrap();
    let dst = aligned(&mut b, 4 * RB);
    f.read_run(40, true, dst).unwrap();
    assert_eq!(dst, &run(40, 4, true)[..]);
    let dst = aligned(&mut b, 4 * RB);
    f.read_run(40, false, dst).unwrap();
    assert_eq!(dst, &run(40, 4, false)[..]);
    // A never-written run past EOF is an error, never stale bytes.
    let mut far = vec![0u8; 2 * RB];
    assert!(f.read_run(10_000, false, &mut far).is_err());
    assert!(f.read_run(10_000, true, &mut far).is_err());
    let _ = std::fs::remove_file(path);
}

#[test]
fn reserved_space_reads_as_never_written_and_takes_records() {
    let Some((f, path)) = record_file("reserve") else {
        return;
    };
    let reserved = f.reserve(64 * RB as u64).unwrap();
    if cfg!(target_os = "linux") {
        // ext4/xfs reserve; a filesystem without fallocate reports `false`.
        if reserved {
            assert_eq!(std::fs::metadata(&path).unwrap().len(), 64 * RB as u64);
            // Reserved, never written: zeros, not an I/O error — the caller's
            // record check (magic + tag) is what rejects it.
            let mut out = vec![0xFFu8; 2 * RB];
            f.read_run(10, false, &mut out).unwrap();
            assert!(out.iter().all(|&b| b == 0));
        }
    } else {
        assert!(!reserved, "no-op off Linux");
    }
    round_trips(&f);
    // An impossible reservation is an error, not a silent sparse file.
    if cfg!(target_os = "linux") && reserved {
        assert!(f.reserve(1 << 60).is_err());
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn mutex_adapter_honours_the_same_contract() {
    round_trips(&Mutex::new(MemSwapStore::new(RB)));
    let boxed: Box<dyn atlas_tier::SwapStore> = Box::new(MemSwapStore::new(RB));
    round_trips(&Mutex::new(boxed));
    assert!(SharedRecordFile::create(Path::new("/nonexistent-dir/x.swap"), RB).is_err());
    assert!(SharedRecordFile::create(Path::new("x.swap"), 1000).is_err());
}

#[test]
fn create_is_exclusive_and_owner_only() {
    let Some((f, path)) = record_file("excl") else {
        return;
    };
    // An existing file — or a symlink planted at the path — is refused.
    assert!(SharedRecordFile::create(&path, RB).is_err());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "no group/other access: {mode:o}");
        let link = path.with_extension("link");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(SharedRecordFile::create(&link, RB).is_err());
        let _ = std::fs::remove_file(&link);
    }
    drop(f);
    let _ = std::fs::remove_file(path);
}

#[test]
fn concurrent_runs_do_not_interfere() {
    let Some((f, path)) = record_file("threads") else {
        return;
    };
    let f = Arc::new(f);
    let workers: Vec<_> = (0..8usize)
        .map(|t| {
            let f = f.clone();
            std::thread::spawn(move || {
                let low = t * 16;
                for round in 0..20 {
                    let reversed = (t + round) % 2 == 0;
                    f.write_run(low, reversed, &run(low, 16, reversed)).unwrap();
                    let mut out = vec![0u8; 16 * RB];
                    f.read_run(low, !reversed, &mut out).unwrap();
                    assert_eq!(out, run(low, 16, !reversed));
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn memory_backed_and_overlay_filesystems_are_named() {
    assert_eq!(atlas_tier::fs_kind(0x0102_1994), Some("tmpfs (host RAM)"));
    assert!(atlas_tier::fs_kind(0x8584_58f6).is_some());
    assert!(atlas_tier::fs_kind(0x794c_7630).is_some_and(|k| k.contains("overlayfs")));
    assert_eq!(atlas_tier::fs_kind(0xEF53), None, "ext4");
    assert_eq!(atlas_tier::fs_kind(0x5846_5342), None, "xfs");
    // Whatever holds the build tree is a real filesystem; a missing directory
    // is left for the open to report.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    if cfg!(not(target_os = "linux")) {
        assert_eq!(atlas_tier::unsuitable_swap_fs(dir), None);
    }
    assert_eq!(
        atlas_tier::unsuitable_swap_fs(Path::new("/no/such/dir")),
        None
    );
}

#[test]
fn stale_sweep_takes_only_matching_swap_files() {
    let dir = scratch().join(format!("sweep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["kv.1.r0.swap", "kv.77.r1.swap", "kv.notes", "ssm.1.swap"] {
        std::fs::write(dir.join(name), b"x").unwrap();
    }
    assert_eq!(atlas_tier::remove_stale_swap_files(&dir, "kv."), 2);
    let mut left: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left, ["kv.notes", "ssm.1.swap"]);
    assert_eq!(
        atlas_tier::remove_stale_swap_files(&dir.join("absent"), "kv."),
        0
    );
    let _ = std::fs::remove_dir_all(&dir);
}
