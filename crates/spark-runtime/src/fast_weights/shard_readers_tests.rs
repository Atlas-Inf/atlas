// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::io::Write;
use std::os::unix::io::AsRawFd;

/// A scratch file of `blocks` 4 KiB blocks, each byte its block's number.
struct Scratch {
    path: std::path::PathBuf,
    file: std::fs::File,
}

impl Scratch {
    fn new(tag: &str, blocks: usize) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("atlas-readers-{tag}-{}-{n}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        for block in 0..blocks {
            f.write_all(&[block as u8; 4096]).unwrap();
        }
        let file = std::fs::File::open(&path).unwrap();
        Self { path, file }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn blocks(n: usize) -> Vec<(u64, usize)> {
    (0..n).map(|b| (b as u64 * 4096, 4096)).collect()
}

/// Panics rather than hangs the suite when `f` does not return.
fn within_seconds(limit: u64, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    rx.recv_timeout(std::time::Duration::from_secs(limit))
        .expect("readers did not stop");
}

#[test]
fn reads_arrive_in_tensor_order() {
    let s = Scratch::new("order", 256);
    let mut readers = ShardReaders::spawn(s.file.as_raw_fd(), false, blocks(256));
    let mut seen = 0;
    for read in &mut readers {
        let (idx, buf, start) = read.unwrap();
        assert_eq!(idx, seen);
        assert_eq!(buf.as_slice()[start], idx as u8);
        seen += 1;
    }
    assert_eq!(seen, 256);
    readers.join().unwrap();
}

#[test]
fn no_tensors_is_an_empty_stream() {
    let s = Scratch::new("empty", 1);
    let mut readers = ShardReaders::spawn(s.file.as_raw_fd(), false, Vec::new());
    assert!(readers.next().is_none());
    readers.join().unwrap();
}

#[test]
fn a_failed_read_reaches_the_copier() {
    let s = Scratch::new("short", 2);
    let tensors = vec![(0, 4096), (1 << 30, 4096)];
    let mut readers = ShardReaders::spawn(s.file.as_raw_fd(), false, tensors);
    assert!(readers.any(|read| read.is_err()));
}

/// An upload failing mid-shard leaves readers waiting on the read-ahead bound
/// and blocked on a full channel; dropping must free both and wait for them,
/// or they would go on reading a closed descriptor.
#[test]
fn dropping_midway_stops_every_reader() {
    let s = Scratch::new("drop", 256);
    let fd = s.file.as_raw_fd();
    within_seconds(10, move || {
        let mut readers = ShardReaders::spawn(fd, false, blocks(256));
        assert_eq!(readers.next().unwrap().unwrap().0, 0);
        let shared = readers.shared.clone();
        drop(readers);
        // Each reader held a clone of `shared`, gone once it is joined.
        assert_eq!(Arc::strong_count(&shared), 1);
    });
}
