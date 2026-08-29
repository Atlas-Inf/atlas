// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe-backed row cache for the n-gram embedding tables.
//!
//! The n-gram tables of the LongCat / Qwen3.8-Flash-Next family are the
//! model's largest tensors by far (31.4 B params on LongCat-Flash-Lite,
//! ~51 B announced for Flash-Next) and simultaneously its *least*
//! bandwidth-hungry: a token touches exactly one row per table — 12 rows,
//! ~3 KB — regardless of sequence length. Pure capacity, near-zero
//! bandwidth, which makes them the best demotion candidate in the model.
//!
//! Design, and why it needs no CUDA kernel change:
//!
//! * The cache is a flat PINNED arena of `slots × row_stride` bytes. On
//!   GB10 pinned host memory is GPU-addressable at the SAME virtual address
//!   ([`ExpertArena`] asserts this), so the arena *is* a
//!   `[slots, dim]` device-side table.
//! * The n-gram row ids are computed HOST-side (they are a pure function of
//!   token ids), so a lookup resolves `row_id -> slot` on the host and hands
//!   the gather kernel the SLOT INDEX in place of the row id. `batched_embed`
//!   / `batched_embed_fp8` then run verbatim against the arena base.
//! * A miss reads the row straight off NVMe into its pinned slot — no
//!   `cuMemcpyHtoD` anywhere on the path.
//!
//! Eviction is CLOCK (second-chance): O(1), no per-hit bookkeeping, and it
//! approximates LRU well for the power-law access pattern these tables have.
//! Rows touched by the CURRENT batch are pinned so a large prefill can never
//! evict a row it is still about to read.
//!
//! O_DIRECT requires 4 KiB-aligned reads, while a row is typically 256 B
//! (FP8, dim 256). Reads are therefore issued as the containing 4 KiB block
//! into a bounce buffer and the row copied out — the block is the disk's
//! minimum transfer anyway, so this costs no extra I/O, only a 256 B host
//! memcpy. Cache capacity stays row-granular, which matters because the
//! hash scatters ids: neighbouring rows in a table are unrelated.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::expert_arena::ExpertArena;

/// O_DIRECT transfer granularity (also `ExpertArena`'s stride requirement).
const BLOCK: usize = 4096;

/// Queue depth for the fault pass. The misses of one prefill are independent
/// 4 KiB O_DIRECT reads, and O_DIRECT means no page cache and no kernel
/// readahead -- the device only overlaps what we hand it at once. Measured on
/// one prefill of 4656 tokens: 22,462 misses x ~74us ISSUED SERIALLY = 1657 ms,
/// 17% of that request's TTFT. Raising the cache does not help; the misses are
/// compulsory first-touch (65536 -> 1048576 slots changed the miss count by
/// zero), so the depth is the whole lever.
///
/// Measured on that prefill, resolve time by depth -- monotone, so the default
/// is the deepest measured rather than the knee:
///
///     QD    resolve    vs serial
///      1    1631 ms      1.00x
///      8     424 ms      3.85x
///     16     257 ms      6.35x
///     32     171 ms      9.55x
///
/// `1` restores the old strictly-serial behaviour for a bisect.
fn fault_threads() -> usize {
    std::env::var("ATLAS_PLE_FAULT_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32)
}

/// One scheduled miss: read row `id` into `slot`. Produced by the decision
/// pass, consumed by the fault pass.
struct Fault {
    id: u64,
    slot: u32,
}

/// Where a worker writes. Raw because the slots a batch owns are disjoint, so
/// the workers do not alias and cannot be expressed as split `&mut` slices
/// without carving the arena up first.
///
/// SAFETY (the `Send`/`Sync`): both pointers are into pinned allocations that
/// outlive the `thread::scope` below -- the cache owns them and is borrowed for
/// the whole scope. Every worker writes ONLY `[slot*stride, +stride)` for slots
/// this batch pinned, and `victim` never hands the same slot out twice while
/// pinned, so no two workers touch the same bytes.
struct ArenaPtrs {
    rows: *mut u8,
    scales: Option<*mut u8>,
}
unsafe impl Send for ArenaPtrs {}
unsafe impl Sync for ArenaPtrs {}

/// The immutable half of the cache a worker needs: which file holds a row, and
/// where. Split out so the fault pass can borrow it shared across threads while
/// each worker keeps its OWN bounce buffer (the old single `self.bounce` was a
/// second serialisation point, independent of the I/O).
struct RowSource<'a> {
    file: &'a File,
    base_offset: u64,
    segments: Option<&'a Segments>,
    row_stride: usize,
    scale_file: Option<&'a File>,
}

impl RowSource<'_> {
    /// The file holding row `id`, and the row's byte offset within it. Same
    /// divide as the method it replaces -- a shard carries its own base offset
    /// AND its own backing file, and resolving one without the other was the
    /// bug this signature exists to prevent.
    fn row_loc(&self, id: u64) -> (&File, u64) {
        match self.segments {
            None => (self.file, self.base_offset + id * self.row_stride as u64),
            Some(seg) => {
                let shard = (id / seg.rows_per) as usize;
                let local = id % seg.rows_per;
                let file = &seg.files[seg.shard_file[shard] as usize];
                (file, seg.bases[shard] + local * self.row_stride as u64)
            }
        }
    }
}

/// Read one row (and its FP8 scale, when the table has a per-row scale file)
/// into `slot`. Free function, not a method: it must be callable from several
/// worker threads at once, each with its own `bounce`.
fn fetch_row(
    src: &RowSource<'_>,
    ptrs: &ArenaPtrs,
    bounce: &mut AlignedBlock,
    id: u64,
    slot: u32,
) -> Result<()> {
    let (file, byte) = src.row_loc(id);
    let block_off = byte - (byte % BLOCK as u64);
    let within = (byte - block_off) as usize;
    // One block unless the row crosses the boundary (possible whenever the
    // table's base offset is not 4 KiB-aligned, i.e. reading in place from a
    // safetensors shard).
    let nblocks = if within + src.row_stride > BLOCK { 2 } else { 1 };
    atlas_tier::pio::read_exact_at(file, bounce.blocks(nblocks), block_off)
        .with_context(|| format!("NgramRowCache: read row {id}"))?;
    // SAFETY: slot is one this batch pinned, so it is < slots and no other
    // worker writes it; the arena holds slots*row_stride bytes.
    let dst = unsafe {
        std::slice::from_raw_parts_mut(
            ptrs.rows.add(slot as usize * src.row_stride),
            src.row_stride,
        )
    };
    dst.copy_from_slice(&bounce.blocks(nblocks)[within..within + src.row_stride]);

    // A constant per-tensor scale needs no per-row refresh: every slot already
    // holds it (see `set_constant_scale`), and `scale_file` is None then.
    if let (Some(sfile), Some(sbase)) = (src.scale_file, ptrs.scales) {
        let sbyte = id * 4;
        let sblock = sbyte - (sbyte % BLOCK as u64);
        let swithin = (sbyte - sblock) as usize;
        atlas_tier::pio::read_exact_at(sfile, bounce.blocks(1), sblock)
            .with_context(|| format!("NgramRowCache: read scale {id}"))?;
        // SAFETY: as above; the scale arena holds slots*4 bytes.
        let sdst = unsafe { std::slice::from_raw_parts_mut(sbase.add(slot as usize * 4), 4) };
        sdst.copy_from_slice(&bounce.blocks(1)[swithin..swithin + 4]);
    }
    Ok(())
}

/// One table's on-NVMe backing file plus its resident row cache.
pub struct NgramRowCache {
    /// Flat pinned, GPU-addressable `[slots, row_stride]` region.
    arena: ExpertArena,
    /// Backing file: row `i` at byte offset `base_offset + i * row_stride`.
    /// `base_offset` lets the cache read STRAIGHT OUT OF A SAFETENSORS SHARD
    /// — a table is already a contiguous row-major blob there, so no repack
    /// or re-save is needed. Because that offset is only 8-byte aligned, a
    /// row may straddle a 4 KiB O_DIRECT block; `fetch_into` handles the seam.
    file: File,
    base_offset: u64,
    /// SEGMENTED tables: one base offset per equal-sized shard.
    ///
    /// LongCat ships each n-gram table as ONE contiguous safetensors tensor,
    /// so `base_offset` alone locates every row. Qwen3.8-Flash-Next splits its
    /// single 320M-row table across 128 shard tensors which are NOT laid out
    /// consecutively in the file — the shards interleave with other weights,
    /// so a global row id needs its shard's own base. `None` keeps the
    /// original single-offset behaviour byte for byte.
    segments: Option<Segments>,
    /// Per-row scale file mirror (FP8 tables), `None` for BF16 tables.
    scales: Option<ScaleCache>,
    row_stride: usize,
    slots: usize,
    rows_total: u64,
    /// row_id -> slot.
    map: HashMap<u64, u32>,
    /// slot -> resident row id (`u64::MAX` = empty).
    slot_row: Vec<u64>,
    /// CLOCK reference bits.
    refbit: Vec<bool>,
    /// Slots pinned for the batch in flight (never evicted).
    pinned: Vec<bool>,
    hand: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// A table split across equal-sized shards at scattered file offsets, which
/// may live in DIFFERENT files.
struct Segments {
    /// Byte offset of each shard's first row, indexed by shard.
    bases: Vec<u64>,
    /// Rows per shard. Every shard but conceivably the last holds exactly
    /// this many; `open_segmented` requires them all equal so the mapping is
    /// a divide rather than a search.
    rows_per: u64,
    /// The distinct backing files, in first-use order. A sharded table is NOT
    /// necessarily confined to one file: the RadixArk NVFP4 conversion of
    /// Qwen3.8-Flash-Next spreads its 128 PLE shards over 10
    /// `model-plefp8-*.safetensors` files, and interleaved rather than in
    /// order (shards 0 and 1 in the first file, shard 2 in the fourth).
    files: Vec<File>,
    /// `shard_file[i]` indexes `files` for shard `i`.
    shard_file: Vec<u32>,
}

/// Dequant scales for an FP8 table, mirrored into a device-visible `[slots]`
/// f32 array indexed by SLOT (parallel to the arena), which is what
/// `batched_embed_fp8` reads.
///
/// `file` is `Some` for a PER-ROW scale file, whose entry for a row is
/// refreshed into the row's slot on every fault. It is `None` for a single
/// PER-TENSOR scale: RadixArk's NVFP4 conversion of Qwen3.8-Flash-Next stores
/// its FP8 PLE table's scale as one BF16 scalar
/// (`ngram_embedding.weight_scale`, shape `[1]`), so every slot holds the same
/// value, written once at open and never touched again.
struct ScaleCache {
    arena: ExpertArena,
    file: Option<File>,
}

/// A 4 KiB-aligned host buffer for O_DIRECT reads.
struct AlignedBlock {
    buf: Vec<u8>,
    off: usize,
}

impl AlignedBlock {
    /// Two blocks: a row whose base offset is not 4 KiB-aligned (every row of
    /// a table read in place from a safetensors shard) can straddle one
    /// boundary, and two blocks always cover it since `row_stride <= BLOCK`.
    fn new() -> Self {
        // Over-allocate and take an aligned window (portable, no libc::memalign).
        let buf = vec![0u8; BLOCK * 3];
        let addr = buf.as_ptr() as usize;
        let off = (BLOCK - (addr % BLOCK)) % BLOCK;
        Self { buf, off }
    }
    /// `n` whole blocks of aligned scratch (`n <= 2`).
    fn blocks(&mut self, n: usize) -> &mut [u8] {
        &mut self.buf[self.off..self.off + n * BLOCK]
    }
}

impl NgramRowCache {
    /// Open `path` as the backing store for a table of `rows_total` rows of
    /// `row_stride` bytes, caching `slots` of them in pinned GPU-addressable
    /// memory. `scale_path` supplies the per-row f32 scales of an FP8 table.
    pub fn open(
        path: &Path,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        Self::open_at(path, 0, scale_path, rows_total, row_stride, slots)
    }

    /// As [`Self::open`], but the table starts at `base_offset` inside the
    /// file — the safetensors-shard case (`data_offsets[0]` + the header
    /// length), which needs no re-save of the checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn open_at(
        path: &Path,
        base_offset: u64,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if row_stride == 0 || slots == 0 {
            bail!("NgramRowCache: zero geometry (row_stride={row_stride}, slots={slots})");
        }
        if row_stride > BLOCK {
            bail!(
                "NgramRowCache: row_stride {row_stride} exceeds the {BLOCK}-byte \
                 O_DIRECT block; a row would span more than the two blocks the \
                 seam-handling fetch reads"
            );
        }
        // One flat pinned region: `slots * row_stride` bytes, rounded up to the
        // arena's 4 KiB stride requirement.
        let bytes = slots * row_stride;
        let blocks = bytes.div_ceil(BLOCK);
        let arena =
            ExpertArena::new(1, blocks as u32, BLOCK).context("NgramRowCache: pinned arena")?;
        let file = open_direct(path)?;
        let scales = match scale_path {
            Some(sp) => {
                let sbytes = slots * 4;
                let sblocks = sbytes.div_ceil(BLOCK);
                Some(ScaleCache {
                    arena: ExpertArena::new(1, sblocks as u32, BLOCK)
                        .context("NgramRowCache: scale arena")?,
                    file: Some(open_direct(sp)?),
                })
            }
            None => None,
        };
        Ok(Self {
            arena,
            file,
            base_offset,
            segments: None,
            scales,
            row_stride,
            slots,
            rows_total,
            map: HashMap::with_capacity(slots * 2),
            slot_row: vec![u64::MAX; slots],
            refbit: vec![false; slots],
            pinned: vec![false; slots],
            hand: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    /// Device VA of the cache's row table — the `embed_table` argument of the
    /// gather kernels, which then index it by SLOT.
    /// Bytes per row, i.e. `head_dim * element_size`.
    ///
    /// Exposed so a caller can CHECK that the gather kernel it is about to
    /// pick matches the element type the cache was opened for. Those two
    /// facts living apart is what let an F8_E4M3 table be gathered by the
    /// BF16 kernel: silently wrong rows, no error anywhere.
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// A resident slot's raw bytes, for tests that verify the gather returned
    /// the row it claimed. The arena is pinned host memory that is ALSO
    /// GPU-addressable, so a host read here sees exactly what the kernel does.
    #[cfg(test)]
    pub(crate) fn slot_bytes(&self, slot: u32) -> Result<&[u8]> {
        // SAFETY: slot < self.slots (checked) and the arena holds
        // slots * row_stride bytes.
        unsafe {
            let base = self.arena.slot_host_ptr(0, 0)?;
            anyhow::ensure!((slot as usize) < self.slots, "slot {slot} out of range");
            Ok(std::slice::from_raw_parts(
                base.add(slot as usize * self.row_stride),
                self.row_stride,
            ))
        }
    }

    pub fn table_dev_va(&self) -> Result<u64> {
        self.arena.slot_dev_va(0, 0)
    }

    /// Device VA of the `[slots]` f32 scale array (FP8 tables only).
    pub fn scale_dev_va(&self) -> Result<Option<u64>> {
        match &self.scales {
            Some(s) => Ok(Some(s.arena.slot_dev_va(0, 0)?)),
            None => Ok(None),
        }
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    /// Resolve `row_ids` to slot indices, faulting misses in from NVMe.
    ///
    /// Every returned slot is PINNED for the caller's batch: the gather runs
    /// after this returns, so a later resolve in the same batch must not
    /// evict a row the kernel is about to read. Call [`Self::end_batch`] once
    /// the gather has been issued.
    pub fn resolve(&mut self, row_ids: &[u64], out_slots: &mut Vec<u32>) -> Result<()> {
        out_slots.clear();
        out_slots.reserve(row_ids.len());

        // PASS 1 -- DECIDE. Serial, and deliberately so: the CLOCK hand, the
        // eviction order and the slot handed to each id are exactly what the
        // old single-pass loop produced, in the same order. Only the I/O moves.
        //
        // The residency bookkeeping that `fetch_into` used to do at its END now
        // happens HERE, at decision time. That is what makes a repeated id
        // inside one batch resolve as a hit to the slot already scheduled for
        // it, instead of being faulted twice into two slots.
        let mut faults: Vec<Fault> = Vec::new();
        for &id in row_ids {
            if id >= self.rows_total {
                bail!(
                    "NgramRowCache: row id {id} >= table rows {} (hash/table mismatch)",
                    self.rows_total
                );
            }
            let slot = match self.map.get(&id) {
                Some(&s) => {
                    self.hits += 1;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    s
                }
                None => {
                    self.misses += 1;
                    // Oversubscription still REFUSES here, before a single byte
                    // of I/O is issued -- `victim` bails when every slot is
                    // pinned by the batch in flight.
                    let s = self.victim()?;
                    self.map.insert(id, s);
                    self.slot_row[s as usize] = id;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    faults.push(Fault { id, slot: s });
                    s
                }
            };
            out_slots.push(slot);
        }

        // PASS 2 -- FAULT. The misses are independent: distinct slots (every
        // slot handed out above is pinned, and `victim` skips pinned slots), so
        // the arena writes are disjoint, and `read_exact_at` is positional so a
        // shared `&File` has no cursor to race on.
        if let Err(e) = self.fetch_many(&faults) {
            // A partially-written slot must not stay claimed as resident: a
            // wrong n-gram row reads as fluent output with wrong logits, which
            // is the failure mode this cache exists to avoid. Drop the whole
            // batch's claims -- conservative, and only on the error path.
            for f in &faults {
                self.map.remove(&f.id);
                self.slot_row[f.slot as usize] = u64::MAX;
            }
            return Err(e);
        }
        Ok(())
    }

    /// Issue `faults` concurrently. `&self`: the residency bookkeeping is
    /// already done (pass 1) and every write below goes through a raw pointer
    /// into a slot this batch owns exclusively.
    fn fetch_many(&self, faults: &[Fault]) -> Result<()> {
        if faults.is_empty() {
            return Ok(());
        }
        let src = RowSource {
            file: &self.file,
            base_offset: self.base_offset,
            segments: self.segments.as_ref(),
            row_stride: self.row_stride,
            scale_file: self.scales.as_ref().and_then(|sc| sc.file.as_ref()),
        };
        let ptrs = ArenaPtrs {
            rows: self.arena.slot_host_ptr(0, 0)?,
            scales: match self.scales.as_ref().filter(|sc| sc.file.is_some()) {
                Some(sc) => Some(sc.arena.slot_host_ptr(0, 0)?),
                None => None,
            },
        };

        let want = fault_threads();
        if want <= 1 || faults.len() == 1 {
            let mut bounce = AlignedBlock::new();
            for f in faults {
                fetch_row(&src, &ptrs, &mut bounce, f.id, f.slot)?;
            }
            return Ok(());
        }

        // One chunk per worker rather than a shared queue: the faults cost the
        // same each (one or two 4 KiB O_DIRECT reads), so static partitioning
        // needs no synchronisation and leaves no tail worth stealing.
        let nthreads = want.min(faults.len());
        let chunk = faults.len().div_ceil(nthreads);
        let first_err: std::sync::Mutex<Option<anyhow::Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for part in faults.chunks(chunk) {
                let (src, ptrs, first_err) = (&src, &ptrs, &first_err);
                scope.spawn(move || {
                    let mut bounce = AlignedBlock::new();
                    for f in part {
                        if let Err(e) = fetch_row(src, ptrs, &mut bounce, f.id, f.slot) {
                            let mut g = first_err.lock().expect("fault error mutex");
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                    }
                });
            }
        });
        match first_err.into_inner().expect("fault error mutex") {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Release the batch's pins (call after the gather kernels are issued).
    pub fn end_batch(&mut self) {
        for p in &mut self.pinned {
            *p = false;
        }
    }

    /// CLOCK second-chance victim among the unpinned slots.
    fn victim(&mut self) -> Result<u32> {
        for _ in 0..(self.slots * 2) {
            let s = self.hand;
            self.hand = (self.hand + 1) % self.slots;
            if self.pinned[s] {
                continue;
            }
            if self.refbit[s] {
                self.refbit[s] = false;
                continue;
            }
            if self.slot_row[s] != u64::MAX {
                let old = self.slot_row[s];
                self.map.remove(&old);
                self.evictions += 1;
            }
            return Ok(s as u32);
        }
        bail!(
            "NgramRowCache: every one of {} slots is pinned by the batch in flight — \
             raise the cache size or lower max-prefill-tokens",
            self.slots
        )
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_direct(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("NgramRowCache: open {} (O_DIRECT)", path.display()))
}

/// macOS has no `O_DIRECT`; `F_NOCACHE` is the nearest equivalent and is set
/// AFTER the open, so this arm opens normally and then asks the kernel not to
/// keep the pages. Best-effort by design: if the fcntl fails the reads are
/// still correct, just cached — and this tier is Linux-only in production, so
/// the arm exists to let the workspace build on an Apple-silicon dev box.
#[cfg(target_os = "macos")]
fn open_direct(path: &Path) -> Result<File> {
    use std::os::unix::io::AsRawFd;
    let file = File::open(path)
        .with_context(|| format!("NgramRowCache: open {} (F_NOCACHE)", path.display()))?;
    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    Ok(file)
}

#[cfg(not(unix))]
fn open_direct(path: &Path) -> Result<File> {
    File::open(path).with_context(|| format!("NgramRowCache: open {}", path.display()))
}

/// Segmented (multi-shard, multi-file) tables and per-tensor FP8 scales.
mod segmented;

#[cfg(test)]
mod tests;
