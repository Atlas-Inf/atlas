// SPDX-License-Identifier: AGPL-3.0-only
//! The uploader leaves scratch byte-for-byte what the full host image left.
use super::super::glm_long_verify::META_BLOCK_TABLE;
use super::*;
use anyhow::bail;
use spark_runtime::gpu::{KernelHandle, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};

const STREAM: u64 = 19;
/// verify_d's metadata base in scratch; the owner lane's joint block too.
const META_BASE: usize = 32768;
/// Scratch offset of verify_d's block table.
const KGAMMA: usize = META_BASE + META_BLOCK_TABLE;
/// Full-capacity table of the 512K-context profile.
const MAX_BLOCKS_512K: usize = 32769;

/// Simulated scratch arena recording what reached it, and on which stream.
struct Device {
    inner: MockGpuBackend,
    scratch: DevicePtr,
    /// Host bytes of each H2D copy, in order.
    uploads: Mutex<Vec<usize>>,
    /// Device bytes of each memset, in order.
    fills: Mutex<Vec<usize>>,
}

impl GpuBackend for Device {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, _: &[u8], _: DevicePtr) -> Result<()> {
        bail!("verify metadata uploads are stream-ordered")
    }
    fn copy_h2d_async(&self, s: &[u8], d: DevicePtr, stream: u64) -> Result<()> {
        assert_eq!(stream, STREAM);
        self.uploads.lock().unwrap().push(s.len());
        self.inner.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        self.inner.kernel(module, symbol)
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut c_void],
    ) -> Result<()> {
        bail!("a metadata upload launches no kernel")
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        bail!("verify metadata fills are stream-ordered")
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, stream: u64) -> Result<()> {
        assert_eq!(stream, STREAM);
        self.fills.lock().unwrap().push(n);
        self.inner.memset(p, v, n)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
}

/// The image the verify lanes used to build on the host and upload whole:
/// `tables.len()` rows of `max_blocks` i32, each table zero-padded.
fn full_image(max_blocks: usize, tables: &[&[u32]]) -> Vec<u8> {
    let mut image = vec![0i32; tables.len() * max_blocks];
    for (row, table) in tables.iter().enumerate() {
        for (j, &block) in table.iter().enumerate().take(max_blocks) {
            image[row * max_blocks + j] = block as i32;
        }
    }
    image.iter().flat_map(|b| b.to_ne_bytes()).collect()
}

/// Host and device bytes one step moved.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    /// Host bytes uploaded, per H2D copy.
    uploads: Vec<usize>,
    /// Device bytes zeroed, per memset.
    fills: Vec<usize>,
}

/// A scratch arena and what full-image uploads would have left in it.
struct Sim {
    gpu: Device,
    expect: Vec<u8>,
}

impl Sim {
    /// Scratch of `bytes`, holding another user's leftovers rather than zeros.
    fn new(bytes: usize) -> Self {
        let inner = MockGpuBackend::new();
        let scratch = inner.alloc(bytes).unwrap();
        inner.memset(scratch, 0xA5, bytes).unwrap();
        Self {
            gpu: Device {
                inner,
                scratch,
                uploads: Mutex::default(),
                fills: Mutex::default(),
            },
            expect: vec![0xA5; bytes],
        }
    }

    /// One upload at scratch offset `at`, checked against the full image over
    /// the whole arena (so nothing outside the rows may change either).
    fn step(&mut self, at: usize, max_blocks: usize, tables: &[&[u32]]) -> Moved {
        upload_block_table_rows(
            &self.gpu,
            self.gpu.scratch.offset(at),
            max_blocks,
            tables.iter().copied(),
            STREAM,
        )
        .unwrap();
        let image = full_image(max_blocks, tables);
        self.expect[at..at + image.len()].copy_from_slice(&image);
        let device = self.gpu.inner.read_alloc(self.gpu.scratch).unwrap();
        if let Some(byte) = (0..device.len()).find(|&i| device[i] != self.expect[i]) {
            panic!(
                "scratch byte {byte} is {:#04x}, the full image leaves {:#04x} \
                 (upload at {at}, {} rows x {max_blocks})",
                device[byte],
                self.expect[byte],
                tables.len()
            );
        }
        Moved {
            uploads: std::mem::take(&mut *self.gpu.uploads.lock().unwrap()),
            fills: std::mem::take(&mut *self.gpu.fills.lock().unwrap()),
        }
    }

    /// Another scratch user writes `len` bytes at `at` between two uploads.
    fn scribble(&mut self, at: usize, len: usize) {
        let junk: Vec<u8> = (0..len).map(|i| (i % 251) as u8 | 1).collect();
        self.gpu
            .inner
            .copy_h2d(&junk, self.gpu.scratch.offset(at))
            .unwrap();
        self.expect[at..at + len].copy_from_slice(&junk);
    }
}

/// A sequence's block table: `len` distinct non-zero physical blocks.
fn table(first: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|j| first + j).collect()
}

fn rows(table: &[u32], k: usize) -> Vec<&[u32]> {
    vec![table; k]
}

#[test]
fn growing_table_matches_the_full_image_every_step() {
    let mb = 64;
    let mut sim = Sim::new(KGAMMA + 8 * mb * 4);
    // Past `mb` the row holds the table's first `mb` entries, as before.
    for len in 1..=mb + 6 {
        sim.step(KGAMMA, mb, &rows(&table(100, len), 8));
    }
}

#[test]
fn shrinking_table_zeroes_the_stale_tail() {
    let mb = 64;
    let mut sim = Sim::new(KGAMMA + 8 * mb * 4);
    sim.step(KGAMMA, mb, &rows(&table(100, 40), 8));
    // Rollback, then a shorter sequence, then one with no blocks at all.
    sim.step(KGAMMA, mb, &rows(&table(100, 12), 8));
    sim.step(KGAMMA, mb, &rows(&table(7000, 3), 8));
    sim.step(KGAMMA, mb, &rows(&[], 8));
    sim.step(KGAMMA, mb, &rows(&table(100, 40), 8));
}

#[test]
fn row_count_changes_touch_only_the_rows_of_the_call() {
    let mb = 64;
    let mut sim = Sim::new(KGAMMA + 8 * mb * 4);
    sim.step(KGAMMA, mb, &rows(&table(100, 30), 8));
    // Rows 5..8 keep the 30-entry table of the previous call.
    sim.step(KGAMMA, mb, &rows(&table(900, 10), 5));
    sim.step(KGAMMA, mb, &rows(&table(500, 20), 8));
    sim.step(KGAMMA, mb, &rows(&table(500, 21), 2));
}

#[test]
fn table_longer_than_the_row_is_truncated() {
    let mb = 16;
    let mut sim = Sim::new(KGAMMA + 8 * mb * 4);
    let moved = sim.step(KGAMMA, mb, &rows(&table(1, mb + 9), 8));
    assert_eq!(moved.uploads, vec![mb * 4; 8]);
}

#[test]
fn empty_table_zeroes_its_rows_without_an_upload() {
    let mb = 64;
    let mut sim = Sim::new(KGAMMA + 8 * mb * 4);
    let moved = sim.step(KGAMMA, mb, &rows(&[], 8));
    assert_eq!(moved.uploads, Vec::<usize>::new());
}

#[test]
fn interleaved_destinations_do_not_share_state() {
    let mb = 64;
    let other = KGAMMA + 8 * mb * 4 + 256 + META_BLOCK_TABLE;
    let mut sim = Sim::new(other + 8 * mb * 4 + 1024);
    sim.step(KGAMMA, mb, &rows(&table(100, 40), 8));
    sim.step(other, mb, &rows(&table(300, 9), 8));
    sim.step(KGAMMA, mb, &rows(&table(100, 8), 8));
    sim.step(other, mb, &rows(&table(300, 33), 8));
    sim.step(KGAMMA, mb, &rows(&table(100, 41), 8));
    // A destination that moved, as the fused chunk's owner blocks do: it
    // overlaps the rows both earlier destinations left behind.
    sim.step(KGAMMA + 8 * mb * 4 - 300, mb, &rows(&table(5, 2), 3));
    sim.step(other, mb, &rows(&table(300, 1), 8));
}

#[test]
fn max_blocks_change_relays_the_rows() {
    let mut sim = Sim::new(KGAMMA + 8 * 64 * 4);
    sim.step(KGAMMA, 64, &rows(&table(100, 50), 8));
    sim.step(KGAMMA, 16, &rows(&table(100, 50), 8));
    sim.step(KGAMMA, 16, &rows(&table(100, 4), 8));
    sim.step(KGAMMA, 64, &rows(&table(100, 4), 8));
}

/// The owner lane: one block per owner, then the joint block of every row,
/// owner-major, each owner's rows carrying that owner's table.
#[test]
fn owner_batched_blocks_carry_each_owners_table() {
    let (mb, width) = (64, 8);
    let align = |bytes: usize| bytes.div_ceil(256) * 256;
    let owner_at = |owners: usize, o: usize| {
        META_BASE
            + align(META_BLOCK_TABLE + owners * width * mb * 4)
            + o * align(META_BLOCK_TABLE + width * mb * 4)
            + META_BLOCK_TABLE
    };
    let mut sim = Sim::new(owner_at(4, 4));
    let step = |sim: &mut Sim, tables: &[Vec<u32>]| {
        let mut joint = Vec::new();
        for (o, table) in tables.iter().enumerate() {
            let own = rows(table, width);
            sim.step(owner_at(tables.len(), o), mb, &own);
            joint.extend(own);
        }
        sim.step(KGAMMA, mb, &joint);
    };
    let first = [table(10, 40), table(200, 3), table(3000, 64), table(1, 17)];
    step(&mut sim, &first);
    // Owner 0 rolled back, owner 1 is a new shorter sequence, owner 3 grew.
    let second = [table(10, 39), table(900, 1), table(3000, 64), table(1, 18)];
    step(&mut sim, &second);
    // Fewer owners: the blocks move, over what the wider call left.
    step(&mut sim, &[table(44, 5), table(10, 39)]);
    step(&mut sim, &first);
}

/// Scratch is shared: decode and prefill metadata, the other verify lanes and
/// MoE routing all write inside these rows between two verify steps.
#[test]
fn another_scratch_writer_between_steps_is_overwritten() {
    let mb = 64;
    let bytes = KGAMMA + 8 * mb * 4;
    let mut sim = Sim::new(bytes);
    let table = table(100, 20);
    sim.step(KGAMMA, mb, &rows(&table, 8));
    // Single-sequence decode uploads its table at +256, straddling row 0.
    sim.scribble(META_BASE + 256, 300 * 4);
    sim.step(KGAMMA, mb, &rows(&table, 8));
    sim.scribble(0, bytes);
    sim.step(KGAMMA, mb, &rows(&table, 8));
}

#[test]
fn uploads_only_the_live_prefixes() {
    let mb = MAX_BLOCKS_512K;
    let live = 300;
    let mut sim = Sim::new(KGAMMA + 8 * mb * 4);
    for k in [8, 5, 8] {
        let table = table(100, live);
        // The full image was k * 131076 bytes: 1 MiB at k = 8.
        let moved = sim.step(KGAMMA, mb, &rows(&table, k));
        assert_eq!(moved.uploads, vec![live * 4; k]);
        assert!(moved.uploads.iter().sum::<usize>() < 64 * 1024);
        // The repeat step sends the live prefixes again and nothing more.
        assert_eq!(sim.step(KGAMMA, mb, &rows(&table, k)), moved);
        // The rows are zeroed on the device, in one fill.
        assert_eq!(moved.fills, vec![k * mb * 4]);
    }
}
