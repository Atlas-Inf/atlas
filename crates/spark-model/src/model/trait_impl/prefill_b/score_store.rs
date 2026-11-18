// SPDX-License-Identifier: AGPL-3.0-only

//! The score pack store: a compressed copy of every built pack, so a pack
//! evicted from the held set (which holds one SSM slot and its K/V blocks a
//! pack) comes back without its pass. A photo asked many questions is the
//! case it is for: the image prefix, encoded once, held compressed.
//!
//! A stored pack is one device allocation: per GDN layer its FP32 h state as
//! E4M3 rows of `kd` with an FP32 scale a row and its conv window as is, then
//! per attention layer and pack block its BF16 K and V as E4M3 rows of
//! `head_dim` (a token and head) with a scale a row (`rows_to_fp8`). About
//! 60 MB for a 1024x576 photo's 685 positions against 200 MB held.
//!
//! The FP8 rounding is measured at the noise level of this NVFP4 model (any
//! change of its numerics, FP16 state included, moves its answers as much),
//! and a pack's answers never depend on the store's history: a built pack is
//! put back from its stored copy before it is used, so a fresh pack and a
//! restored one hold the same bytes.
//!
//! Budget `ATLAS_SCORE_STORE_GB` (unset or 0: no store); least recently used
//! out first, to the NVMe tier when one is configured (`score_store_disk.rs`),
//! which hands a pack back to device memory when it is asked for again; the
//! scheduler names the packs its queued jobs need ahead of their passes
//! (`store_prefetch`) so their disk reads overlap other passes
//! (`score_store_ahead.rs`). The store lock is taken under the held-pack
//! lock, never the other way.

use std::collections::HashMap;

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::super::types::TransformerModel;
use super::score_pack::ssm_ptrs_of;
use super::score_store_ahead::{Ahead, MAX_SLOT_BYTES};
use super::score_store_disk::Disk;
use crate::traits::SequenceState;

/// A pack's identity and geometry, which a restore needs besides its bytes.
#[derive(Clone)]
pub(in crate::model) struct PackMeta {
    pub tokens: Vec<u32>,
    pub bidir: bool,
    pub grids: Vec<(usize, usize, usize)>,
    pub hashes: Vec<u128>,
}

struct Stored {
    meta: PackMeta,
    bytes: usize,
    /// The device copy, if the pack has one.
    buf: Option<DevicePtr>,
    /// Its NVMe records (empty: none) and the write's generation.
    disk: Vec<usize>,
    generation: u64,
    last_use: u64,
}

#[derive(Default)]
struct Inner {
    map: HashMap<u64, Stored>,
    tick: u64,
    /// Device bytes held.
    used: usize,
    disk: Option<Disk>,
    disk_opened: bool,
}

impl Inner {
    /// Open the NVMe tier the first time the store is used. A tier that
    /// cannot open is said once, loudly, and the store runs without it.
    fn open_disk(&mut self) {
        if !std::mem::replace(&mut self.disk_opened, true) {
            match Disk::open() {
                Ok(disk) => self.disk = disk,
                Err(e) => tracing::error!("score pack store: NVMe tier OFF: {e:#}"),
            }
        }
    }

    /// The least recently used pack `has` matches, other than `keep`.
    fn coldest(&self, keep: u64, has: impl Fn(u64, &Stored) -> bool) -> Option<u64> {
        self.map
            .iter()
            .filter(|&(&id, s)| id != keep && has(id, s))
            .min_by_key(|(_, s)| s.last_use)
            .map(|(&id, _)| id)
    }

    /// Whether pack `id`'s records are being read ahead (they stay put).
    fn reading(&self, id: u64) -> bool {
        let ahead = self.disk.as_ref().and_then(|d| d.ahead.as_ref());
        ahead.is_some_and(|a| a.state(id) == Some(Ahead::Reading))
    }

    /// Room on disk for `n` records, dropping the coldest packs' (not
    /// `keep`'s, nor any being read ahead); false when the tier cannot hold
    /// them.
    fn disk_room(&mut self, n: usize, keep: u64) -> bool {
        loop {
            let Some(disk) = self.disk.as_ref() else {
                return false;
            };
            if disk.free_records() >= n {
                return true;
            }
            let Some(old) = self.coldest(keep, |id, s| !s.disk.is_empty() && !self.reading(id))
            else {
                return false;
            };
            let s = self.map.get_mut(&old).expect("listed");
            let records = std::mem::take(&mut s.disk);
            if s.buf.is_none() {
                self.map.remove(&old);
            }
            self.disk.as_mut().expect("checked").give_back(records);
        }
    }
}

/// The model's stored packs.
#[derive(Default)]
pub(in crate::model) struct ScoreStore(parking_lot::Mutex<Inner>);

/// `ATLAS_SCORE_STORE_GB` in bytes (0: no store).
fn budget() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let gb: f64 = std::env::var("ATLAS_SCORE_STORE_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        (gb.max(0.0) * (1u64 << 30) as f64) as usize
    })
}

/// `ATLAS_SCORE_TRACE=1`: say each disk move and its time.
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_SCORE_TRACE").is_ok_and(|v| v == "1"))
}

/// One piece of a pack: rows of `len` elements (`dtype` 0 = FP32, 1 = BF16)
/// kept as E4M3 rows, or bytes kept as they are.
enum Part {
    Fp8 {
        at: DevicePtr,
        rows: usize,
        len: usize,
        dtype: u32,
    },
    Raw {
        at: DevicePtr,
        bytes: usize,
    },
}

impl Part {
    /// Its store bytes (16-byte aligned pieces): codes, then scales.
    fn bytes(&self) -> usize {
        match *self {
            Part::Fp8 { rows, len, .. } => {
                (rows * len).next_multiple_of(16) + (rows * 4).next_multiple_of(16)
            }
            Part::Raw { bytes, .. } => bytes.next_multiple_of(16),
        }
    }
}

impl TransformerModel {
    /// The pieces of the pack sequence `seq` the store keeps, in store order.
    fn store_parts(&self, seq: &SequenceState, kv: &PagedKvCache) -> Result<Vec<Part>> {
        ensure!(
            self.ssm_pool.h_stored_bytes == self.config.ssm_h_state_bytes(),
            "score store: the SSM pool must hold FP32 state"
        );
        let kd = self.config.linear_key_head_dim;
        let h_rows = self.config.ssm_h_state_bytes() / 4 / kd;
        let mut parts = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.is_ssm_layer() {
                let (h, conv) = ssm_ptrs_of(seq, i)?;
                parts.push(Part::Fp8 {
                    at: h,
                    rows: h_rows,
                    len: kd,
                    dtype: 0,
                });
                parts.push(Part::Raw {
                    at: conv,
                    bytes: self.ssm_pool.conv_bytes,
                });
            }
        }
        let d = self.config.head_dim;
        for l in 0..kv.num_layers() {
            let (k, v) = (
                kv.k_block_stride_bytes_for_layer(l),
                kv.v_block_stride_bytes_for_layer(l),
            );
            if k == 0 && v == 0 {
                continue;
            }
            ensure!(
                kv.dtype_for_layer(l) == KvCacheDtype::Bf16 && k % (2 * d) == 0 && v % (2 * d) == 0,
                "score store: needs a BF16 KV cache"
            );
            for &b in &seq.block_table {
                parts.push(Part::Fp8 {
                    at: kv.k_cache_ptr(l, b),
                    rows: k / 2 / d,
                    len: d,
                    dtype: 1,
                });
                parts.push(Part::Fp8 {
                    at: kv.v_cache_ptr(l, b),
                    rows: v / 2 / d,
                    len: d,
                    dtype: 1,
                });
            }
        }
        Ok(parts)
    }

    /// Move `parts` into the store buffer `buf` (`into`) or back out of it.
    fn store_move(&self, parts: &[Part], buf: DevicePtr, into: bool, stream: u64) -> Result<()> {
        let gpu = self.gpu.as_ref();
        let name = if into { "rows_to_fp8" } else { "rows_from_fp8" };
        // Asked first: looked up here (after the boot audit sealed), a kernel this
        // build lacks would abort the server instead of refusing the store.
        ensure!(
            gpu.ships_kernel("residual_add", name),
            "score store: {name} is not in this build"
        );
        let kernel = crate::layers::try_kernel(gpu, "residual_add", name);
        ensure!(kernel.0 != 0, "score store: {name} is not in this build");
        let mut off = 0;
        for part in parts {
            match *part {
                Part::Fp8 {
                    at,
                    rows,
                    len,
                    dtype,
                } => {
                    let (codes, scales) = (
                        buf.offset(off),
                        buf.offset(off + (rows * len).next_multiple_of(16)),
                    );
                    let launch = KernelLaunch::new(gpu, kernel)
                        .grid([rows.div_ceil(8) as u32, 1, 1])
                        .block([256, 1, 1]);
                    let launch = match into {
                        true => launch.arg_ptr(at).arg_ptr(codes).arg_ptr(scales),
                        false => launch.arg_ptr(codes).arg_ptr(scales).arg_ptr(at),
                    };
                    launch
                        .arg_u32(rows as u32)
                        .arg_u32(len as u32)
                        .arg_u32(dtype)
                        .launch(stream)?;
                }
                Part::Raw { at, bytes } => match into {
                    true => gpu.copy_d2d_async(at, buf.offset(off), bytes, stream)?,
                    false => gpu.copy_d2d_async(buf.offset(off), at, bytes, stream)?,
                },
            }
            off += part.bytes();
        }
        Ok(())
    }

    /// Store the just-built pack `seq` under `id` and put it back from the
    /// stored copy, so it holds what a later restore would. No store, a pack
    /// over the whole budget, a model the store cannot keep (said once) or no
    /// device memory for the copy (said): nothing happens, and the pack is
    /// used as built (never stored, it is never restored either). An error
    /// means the copy back failed, and `seq` must not be used.
    pub(super) fn store_insert(
        &self,
        id: u64,
        meta: &PackMeta,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        let budget = budget();
        if budget == 0 {
            return Ok(());
        }
        let parts = match self.store_parts(seq, &self.kv_cache.lock()) {
            Ok(parts) => parts,
            Err(e) => {
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| tracing::warn!("score pack store OFF for this model: {e:#}"));
                return Ok(());
            }
        };
        let bytes: usize = parts.iter().map(Part::bytes).sum();
        if bytes > budget {
            return Ok(());
        }
        let mut st = self.score_store.0.lock();
        st.open_disk();
        if st.map.contains_key(&id) {
            return Ok(());
        }
        self.store_room(&mut st, bytes, id, stream)?;
        let buf = match self.gpu.alloc(bytes) {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!("score pack store: no device memory to keep pack {id:016x}: {e:#}");
                return Ok(());
            }
        };
        let moved = self
            .store_move(&parts, buf, true, stream)
            .and_then(|()| self.store_move(&parts, buf, false, stream))
            .and_then(|()| self.gpu.synchronize(stream));
        if let Err(e) = moved {
            self.gpu.free(buf)?;
            return Err(e);
        }
        st.tick += 1;
        let last_use = st.tick;
        st.used += bytes;
        let stored = Stored {
            meta: meta.clone(),
            bytes,
            buf: Some(buf),
            disk: Vec::new(),
            generation: 0,
            last_use,
        };
        st.map.insert(id, stored);
        Ok(())
    }

    /// Device room for `bytes` more (`keep` stays): the coldest device
    /// copies go, written to the NVMe tier first when it does not hold them
    /// (a failed write drops the pack, said; it is rebuilt when asked for).
    fn store_room(&self, st: &mut Inner, bytes: usize, keep: u64, stream: u64) -> Result<()> {
        let budget = budget();
        while st.used + bytes > budget {
            let Some(old) = st.coldest(keep, |_, s| s.buf.is_some()) else {
                break;
            };
            let mut s = st.map.remove(&old).expect("listed");
            let buf = s.buf.take().expect("filtered");
            // `old` is out of the map already; `keep` (a pack being promoted) must keep its records.
            if s.disk.is_empty() && st.disk_room(Disk::records(s.bytes), keep) {
                let t0 = std::time::Instant::now();
                let disk = st.disk.as_mut().expect("room");
                match disk.write(self.gpu.as_ref(), buf, s.bytes, stream) {
                    Ok((records, generation)) => {
                        (s.disk, s.generation) = (records, generation);
                        if trace() {
                            tracing::info!(
                                "score store: pack {old:016x} to disk ({} us)",
                                t0.elapsed().as_micros()
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("score pack store: writing pack {old:016x} to disk: {e:#}")
                    }
                }
            }
            st.used -= s.bytes;
            self.gpu.free(buf)?;
            if !s.disk.is_empty() {
                st.map.insert(old, s);
            }
        }
        Ok(())
    }

    /// Put the disk-only pack `id` back in device memory; false when its
    /// read fails (said; the pack is dropped and rebuilt when asked for).
    fn store_promote(&self, st: &mut Inner, id: u64, stream: u64) -> Result<bool> {
        let t0 = std::time::Instant::now();
        let (bytes, records) = (st.map[&id].bytes, st.map[&id].disk.clone());
        self.store_room(st, bytes, id, stream)?;
        if st.map.get(&id).is_none_or(|s| s.disk != records) {
            return Ok(false); // never expected (room is made around `id`); rebuilt when asked for
        }
        let buf = self.gpu.alloc(bytes)?;
        let disk = st.disk.as_mut().expect("a pack with records");
        if let Err(e) = disk.read(self.gpu.as_ref(), &records, buf, bytes, stream) {
            tracing::warn!("score pack store: reading pack {id:016x} from disk: {e:#}");
            self.gpu.free(buf)?;
            let gone = st.map.remove(&id).expect("listed");
            st.disk.as_mut().expect("checked").give_back(gone.disk);
            return Ok(false);
        }
        st.used += bytes;
        st.map.get_mut(&id).expect("listed").buf = Some(buf);
        if trace() {
            tracing::info!(
                "score store: pack {id:016x} from disk ({} us)",
                t0.elapsed().as_micros()
            );
        }
        Ok(true)
    }

    /// Which of the packs `ids` (`held[i]`: held already) are ready for a
    /// pass and whether reads are running; a read ahead is started for each
    /// one only on disk. See [`crate::traits::Model::score_packs_prefetch`].
    pub(super) fn store_prefetch(&self, ids: &[u64], held: &[bool]) -> (Vec<bool>, bool) {
        let mut st = self.score_store.0.lock();
        let st = &mut *st;
        let Some(disk) = st.disk.as_mut().filter(|d| d.ahead.is_some()) else {
            return (vec![true; ids.len()], false);
        };
        let ahead = disk.ahead.as_mut().expect("filtered");
        ahead.collect();
        // Slots of packs that were promoted, dropped or rewritten since.
        ahead.retain(|id, generation| {
            st.map
                .get(&id)
                .is_some_and(|s| s.buf.is_none() && s.generation == generation)
        });
        let wanted: Vec<u64> = ids
            .iter()
            .zip(held)
            .filter(|&(_, &h)| !h)
            .map(|(&id, _)| id)
            .collect();
        let mut ready = Vec::with_capacity(ids.len());
        for (&id, &h) in ids.iter().zip(held) {
            let on_disk_only = st.map.get(&id).filter(|s| !h && s.buf.is_none());
            let Some(s) = on_disk_only else {
                ready.push(true);
                continue;
            };
            // Too big for a slot, or its read ahead failed: the pass reads it
            // itself (and drops it if the disk fails again).
            let by_pass = disk.ahead.as_ref().is_none_or(|a| a.failed(id));
            if s.bytes > MAX_SLOT_BYTES || by_pass {
                ready.push(true);
                continue;
            }
            let state = disk.ahead.as_ref().and_then(|a| a.state(id));
            ready.push(match state {
                Some(Ahead::Ready) => true,
                Some(Ahead::Reading) => false,
                None => {
                    match disk.prefetch(self.gpu.as_ref(), id, s.generation, &s.disk, &wanted) {
                        Ok(started) => !started && disk.ahead.is_none(),
                        // Its host slots cannot be had: said once, and the passes
                        // read their packs themselves from here on.
                        Err(e) => {
                            tracing::error!("score pack store: read-ahead OFF: {e:#}");
                            if let Some(mut ahead) = disk.ahead.take()
                                && let Err(f) = ahead.release(self.gpu.as_ref())
                            {
                                tracing::error!("score pack store: releasing read-ahead: {f:#}");
                            }
                            true
                        }
                    }
                }
            });
        }
        let reading = disk.ahead.as_ref().is_some_and(|a| a.reading(&wanted));
        (ready, reading)
    }

    /// Wait up to `timeout` for a read ahead to finish.
    pub(in crate::model) fn store_wait(&self, timeout: std::time::Duration) {
        let mut st = self.score_store.0.lock();
        if let Some(ahead) = st.disk.as_mut().and_then(|d| d.ahead.as_mut()) {
            ahead.wait(timeout);
        }
    }

    /// The stored pack `id` restored into a new sequence (its SSM slot and
    /// K/V blocks), with its identity; `None` when the store does not hold it.
    pub(super) fn store_restore(
        &self,
        id: u64,
        stream: u64,
    ) -> Result<Option<(PackMeta, SequenceState)>> {
        let mut st = self.score_store.0.lock();
        st.tick += 1;
        let tick = st.tick;
        let Some(stored) = st.map.get_mut(&id) else {
            return Ok(None);
        };
        stored.last_use = tick;
        let (meta, bytes, generation) = (stored.meta.clone(), stored.bytes, stored.generation);
        let buf = match stored.buf {
            Some(buf) => buf,
            None => {
                // Read ahead: restored straight from its host slot, which the
                // GPU reads in place (unified addressing: a page-locked host
                // buffer's address is its device address), so no device copy
                // is made; a library that outgrows the device budget is swept
                // through it anyway.
                let t0 = std::time::Instant::now();
                let ahead = st.disk.as_mut().and_then(|d| d.ahead.as_mut());
                let from_slot = match ahead {
                    Some(a) => a.take(id, generation, bytes, |src| {
                        self.store_unpack(&meta, DevicePtr(src.as_ptr() as u64), stream)
                    })?,
                    None => None,
                };
                if let Some(seq) = from_slot {
                    if trace() {
                        tracing::info!(
                            "score store: pack {id:016x} from read-ahead ({} us)",
                            t0.elapsed().as_micros()
                        );
                    }
                    return Ok(Some((meta, seq)));
                }
                if !self.store_promote(&mut st, id, stream)? {
                    return Ok(None);
                }
                st.map[&id].buf.expect("promoted")
            }
        };
        self.store_unpack(&meta, buf, stream)
            .map(|seq| Some((meta, seq)))
    }

    /// A new sequence (its SSM slot and K/V blocks) holding the stored pack
    /// at `src` (device-readable, `meta`'s geometry), synchronized.
    fn store_unpack(&self, meta: &PackMeta, src: DevicePtr, stream: u64) -> Result<SequenceState> {
        let mut seq = self.alloc_sequence_reset(false)?;
        let restored = (|| -> Result<()> {
            let mut kv = self.kv_cache.lock();
            let bs = kv.block_size();
            let blocks = meta.tokens.len().div_ceil(bs);
            while kv.num_free_blocks() < blocks {
                let evicted = self.prefix_cache.evict(blocks - kv.num_free_blocks());
                if evicted.is_empty() {
                    break;
                }
                self.apply_evicted(evicted, &mut kv);
            }
            ensure!(
                kv.num_free_blocks() >= blocks,
                "score store: {blocks} KV blocks for a restore, {} free",
                kv.num_free_blocks()
            );
            super::super::super::block_mgmt::ensure_blocks_through_prefill(
                &mut seq,
                blocks - 1,
                &mut kv,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;
            let parts = self.store_parts(&seq, &kv)?;
            drop(kv);
            self.store_move(&parts, src, false, stream)?;
            self.gpu.synchronize(stream)
        })();
        if let Err(e) = restored {
            self.free_sequence_dispatch(&mut seq, false)?;
            return Err(e);
        }
        Ok(seq)
    }

    /// Free every stored pack (model teardown).
    pub(in crate::model) fn score_store_release(&self) -> Result<()> {
        let mut st = self.score_store.0.lock();
        let mut first = Ok(());
        let bufs: Vec<DevicePtr> = st.map.drain().filter_map(|(_, s)| s.buf).collect();
        for buf in bufs {
            let r = self.gpu.free(buf);
            if first.is_ok() {
                first = r;
            }
        }
        if let Some(mut disk) = st.disk.take() {
            let r = disk.release(self.gpu.as_ref());
            if first.is_ok() {
                first = r;
            }
        }
        st.used = 0;
        first
    }
}

#[cfg(test)]
mod tests {
    use super::Part;
    use spark_runtime::gpu::DevicePtr;

    #[test]
    fn parts_take_aligned_codes_and_scales() {
        let fp8 = Part::Fp8 {
            at: DevicePtr::NULL,
            rows: 6144,
            len: 128,
            dtype: 0,
        };
        assert_eq!(fp8.bytes(), 6144 * 128 + 6144 * 4);
        let odd = Part::Fp8 {
            at: DevicePtr::NULL,
            rows: 3,
            len: 5,
            dtype: 1,
        };
        assert_eq!(odd.bytes(), 16 + 16);
        assert_eq!(
            Part::Raw {
                at: DevicePtr::NULL,
                bytes: 17
            }
            .bytes(),
            32
        );
    }
}
