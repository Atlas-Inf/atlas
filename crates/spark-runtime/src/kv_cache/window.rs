// SPDX-License-Identifier: AGPL-3.0-only

//! Window-bounded KV for sliding-window layers.
//!
//! A sliding-window layer reads only the last `window` keys of a sequence,
//! so its rows need not live as long as the full-attention layers' rows.
//! The window layers keep their K/V in pools of their own, a few blocks per
//! live sequence, at a *local* block the primary block maps to:
//! `local = map[primary]`, [`NO_WINDOW`] when it holds no window rows. A
//! local block follows its primary block, so sequences that share a primary
//! block (a cached prefix) share its window rows too.
//!
//! Holders keep a mapping alive. A live sequence holds the blocks its next
//! forward reads or writes ([`PagedKvCache::window_hold`]); a snapshot pins
//! the window it resumes from ([`PagedKvCache::window_pin`]), with a primary
//! ref on each block so the mapping cannot outlive the rows it names. Pins
//! go first when the pool runs short. Local block 0 is the null block: map
//! entries the kernels never read, and graph padding rows, point there.
//!
//! Every reader reaches the rows through the map at the same logical
//! positions as before, so attention sums the same values in the same order.

use std::collections::VecDeque;

use anyhow::{Result, bail, ensure};

use super::{KvCacheConfig, KvPlacement, PagedKvCache};

#[path = "window_plan.rs"]
mod plan;
use crate::gpu::{DevicePtr, GpuBackend};
pub use plan::WindowPlan;

/// Map entry of a primary block that holds no window rows.
pub const NO_WINDOW: u32 = u32::MAX;

/// One window layer's local pools.
struct WindowLayer {
    layer: usize,
    k: DevicePtr,
    v: DevicePtr,
    k_stride: usize,
    v_stride: usize,
}

pub(super) struct WindowPool {
    plan: WindowPlan,
    layers: Vec<WindowLayer>,
    /// Device `u32[primary blocks]` primary -> local map.
    map: DevicePtr,
    local_of: Vec<u32>,
    refs: Vec<u32>,
    free: Vec<u32>,
    dirty: Vec<u32>,
    pins: VecDeque<Vec<u32>>,
    /// Per stream, an event recorded after its last release: a local block
    /// lent on another stream waits for it, since the releasing stream's
    /// queued forwards may still read the block.
    released: Vec<(u64, u64)>,
}

impl WindowPool {
    fn new(
        config: &KvCacheConfig,
        primary_blocks: usize,
        plan: WindowPlan,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            plan.blocks >= 2 && u32::try_from(plan.blocks).is_ok() && plan.window > 0,
            "window pool of {} blocks over {} keys",
            plan.blocks,
            plan.window
        );
        let mut pool = Self {
            layers: Vec::with_capacity(plan.layers.len()),
            map: DevicePtr::NULL,
            local_of: vec![NO_WINDOW; primary_blocks],
            refs: vec![0; plan.blocks],
            free: (1..plan.blocks as u32).rev().collect(),
            dirty: Vec::new(),
            pins: VecDeque::new(),
            released: Vec::new(),
            plan,
        };
        let built = pool.allocate(config, primary_blocks, gpu);
        if built.is_err() {
            pool.free_device(gpu);
        }
        built.map(|()| pool)
    }

    fn allocate(
        &mut self,
        config: &KvCacheConfig,
        primary_blocks: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.map = gpu.alloc(primary_blocks * 4)?;
        gpu.memset(self.map, 0xFF, primary_blocks * 4)?;
        for &layer in &self.plan.layers {
            let (k_stride, v_stride) = (
                config.k_block_bytes_for_layer(layer),
                config.v_block_bytes_for_layer(layer),
            );
            let k = gpu.alloc(self.plan.blocks * k_stride)?;
            let v = gpu.alloc(self.plan.blocks * v_stride).inspect_err(|_| {
                let _ = gpu.free(k);
            })?;
            self.layers.push(WindowLayer {
                layer,
                k,
                v,
                k_stride,
                v_stride,
            });
        }
        Ok(())
    }

    pub(super) fn free_device(&mut self, gpu: &dyn GpuBackend) -> Option<anyhow::Error> {
        let mut first = None;
        let ptrs = self.layers.drain(..).flat_map(|l| [l.k, l.v]);
        for ptr in ptrs.chain([std::mem::replace(&mut self.map, DevicePtr::NULL)]) {
            if !ptr.is_null()
                && let Err(e) = gpu.free(ptr)
            {
                first.get_or_insert(e);
            }
        }
        for (_, event) in self.released.drain(..) {
            let _ = gpu.destroy_event(event);
        }
        first
    }

    fn layer(&self, layer: usize) -> Option<&WindowLayer> {
        self.layers.iter().find(|l| l.layer == layer)
    }

    fn mapped(&self, block: u32) -> Option<u32> {
        let local = self.local_of[block as usize];
        (local != NO_WINDOW).then_some(local)
    }

    /// Drop one ref of `block`'s rows; true when its local block went free.
    fn release(&mut self, block: u32) -> bool {
        let Some(local) = self.mapped(block) else {
            return false;
        };
        let refs = &mut self.refs[local as usize];
        *refs = refs.saturating_sub(1);
        if *refs > 0 {
            return false;
        }
        self.local_of[block as usize] = NO_WINDOW;
        self.free.push(local);
        self.dirty.push(block);
        true
    }

    /// Drop the oldest pin's local refs; returns its primary blocks.
    fn evict_pin(&mut self) -> Option<Vec<u32>> {
        let pin = self.pins.pop_front()?;
        for &block in &pin {
            self.release(block);
        }
        Some(pin)
    }

    /// Lend local blocks to every unmapped block of `blocks` and add a ref
    /// to each, evicting pins for room; all or nothing. Returns the fresh
    /// local blocks and the evicted pins' primary blocks.
    fn hold(&mut self, blocks: &[u32]) -> (Result<Vec<u32>>, Vec<u32>) {
        let (mut fresh, mut evicted, mut held) = (Vec::new(), Vec::new(), Vec::new());
        for &block in blocks {
            if let Some(local) = self.mapped(block) {
                self.refs[local as usize] += 1;
                held.push(block);
                continue;
            }
            let local = loop {
                if let Some(local) = self.free.pop() {
                    break Some(local);
                }
                match self.evict_pin() {
                    Some(pin) => evicted.extend(pin),
                    None => break None,
                }
            };
            let Some(local) = local else {
                for &b in &held {
                    self.release(b);
                }
                let (lent, total) = (self.plan.blocks - 1 - self.free.len(), self.plan.blocks - 1);
                return (
                    Err(anyhow::anyhow!(
                        "KV cache exhausted: window pool has no free block for primary block \
                         {block} ({lent} of {total} lent, no pins left)"
                    )),
                    evicted,
                );
            };
            self.local_of[block as usize] = local;
            self.refs[local as usize] = 1;
            self.dirty.push(block);
            fresh.push(local);
            held.push(block);
        }
        (Ok(fresh), evicted)
    }

    fn flush(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        self.dirty.sort_unstable();
        self.dirty.dedup();
        for run in self.dirty.chunk_by(|a, b| *b == *a + 1) {
            let first = run[0] as usize;
            let entries = &self.local_of[first..first + run.len()];
            let bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
            gpu.copy_h2d_async(&bytes, self.map.offset(first * 4), stream)?;
        }
        self.dirty.clear();
        Ok(())
    }

    /// Zero `fresh` local blocks on `stream`, after every other stream's
    /// last release.
    fn fill(&mut self, fresh: &[u32], gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if fresh.is_empty() {
            return Ok(());
        }
        for &(s, event) in &self.released {
            if s != stream {
                gpu.stream_wait_event(stream, event)?;
            }
        }
        let mut sorted = fresh.to_vec();
        sorted.sort_unstable();
        for run in sorted.chunk_by(|a, b| *b == *a + 1) {
            let (first, count) = (run[0] as usize, run.len());
            for l in &self.layers {
                for (base, stride) in [(l.k, l.k_stride), (l.v, l.v_stride)] {
                    gpu.memset_async(base.offset(first * stride), 0, count * stride, stream)?;
                }
            }
        }
        Ok(())
    }

    fn mark_released(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let event = match self.released.iter().find(|(s, _)| *s == stream) {
            Some(&(_, event)) => event,
            None => {
                let event = gpu.create_event()?;
                self.released.push((stream, event));
                event
            }
        };
        gpu.record_event(event, stream)
    }
}

impl PagedKvCache {
    /// A cache whose `plan.layers` keep window-bounded KV in a pool of their
    /// own; the other layers keep one row per primary block, as
    /// [`Self::new`] lays them out.
    pub fn new_windowed(
        config: KvCacheConfig,
        num_blocks: usize,
        plan: WindowPlan,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            config.cache_blocks_per_seq.is_none()
                && plan.layers.iter().all(|&l| l < config.num_layers),
            "window layers {:?} need in-range layers and no --high-speed-swap",
            plan.layers
        );
        let skip = plan.layers.clone();
        let mut pool = WindowPool::new(&config, num_blocks, plan, gpu)?;
        let placement = KvPlacement::default();
        match Self::new_with_k_slots(config, num_blocks, num_blocks, gpu, false, placement, &skip) {
            Ok(mut cache) => {
                cache.window = Some(pool);
                Ok(cache)
            }
            Err(e) => {
                pool.free_device(gpu);
                Err(e)
            }
        }
    }

    pub fn window_plan(&self) -> Option<&WindowPlan> {
        self.window.as_ref().map(|w| &w.plan)
    }

    pub fn is_window_layer(&self, layer: usize) -> bool {
        self.window
            .as_ref()
            .is_some_and(|w| w.layer(layer).is_some())
    }

    /// `(K, V)` local pool of a window layer.
    pub(super) fn window_pools(&self, layer: usize) -> Option<(DevicePtr, DevicePtr)> {
        let l = self.window.as_ref()?.layer(layer)?;
        Some((l.k, l.v))
    }

    /// Device `u32[num_blocks]` primary -> local map; NULL without a window.
    pub fn window_map_ptr(&self) -> DevicePtr {
        self.window.as_ref().map_or(DevicePtr::NULL, |w| w.map)
    }

    pub fn window_mapped(&self, block: u32) -> bool {
        self.window
            .as_ref()
            .is_some_and(|w| w.mapped(block).is_some())
    }

    pub fn window_free_blocks(&self) -> usize {
        self.window.as_ref().map_or(0, |w| w.free.len())
    }

    pub fn window_pins(&self) -> usize {
        self.window.as_ref().map_or(0, |w| w.pins.len())
    }

    /// Add a holder to each of `blocks`' window rows, lending zeroed local
    /// blocks to unmapped ones, and publish the map before any kernel on
    /// `stream` reads it. All or nothing; an exhausted pool is a
    /// `KV cache exhausted` refusal.
    pub fn window_hold(&mut self, blocks: &[u32], gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let Some(w) = self.window.as_mut() else {
            return Ok(());
        };
        let (fresh, evicted) = w.hold(blocks);
        for block in evicted {
            self.dec_ref(block);
        }
        let fresh = fresh?;
        let Some(w) = self.window.as_mut() else {
            bail!("window pool vanished");
        };
        w.fill(&fresh, gpu, stream)?;
        w.flush(gpu, stream)
    }

    /// Drop a holder from each of `blocks`' window rows. A local block that
    /// goes free is lent again only after `stream`'s work queued so far.
    pub fn window_release(
        &mut self,
        blocks: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(w) = self.window.as_mut() else {
            return Ok(());
        };
        let mut freed = false;
        for &block in blocks {
            freed |= w.release(block);
        }
        if freed {
            w.mark_released(gpu, stream)?;
        }
        Ok(())
    }

    /// Pin a snapshot's window, `blocks` its logical blocks from
    /// [`WindowPlan::first_block`] through its last row. False (nothing
    /// pinned) when a row of it is already gone.
    pub fn window_pin(&mut self, blocks: &[u32]) -> bool {
        let Some(w) = self.window.as_mut() else {
            return false;
        };
        if blocks.is_empty() || blocks.iter().any(|&b| w.mapped(b).is_none()) {
            return false;
        }
        for &block in blocks {
            let local = w.local_of[block as usize];
            w.refs[local as usize] += 1;
        }
        w.pins.push_back(blocks.to_vec());
        let mut evicted = Vec::new();
        while w.pins.len() > w.plan.pins {
            evicted.extend(w.evict_pin().unwrap_or_default());
        }
        for &block in blocks {
            self.inc_ref(block);
        }
        for block in evicted {
            self.dec_ref(block);
        }
        true
    }

    /// A resume read the window pinned as `blocks`: evict it last.
    pub fn window_touch(&mut self, blocks: &[u32]) {
        let Some(w) = self.window.as_mut() else {
            return;
        };
        if let Some(i) = w.pins.iter().position(|p| p.as_slice() == blocks)
            && let Some(pin) = w.pins.remove(i)
        {
            w.pins.push_back(pin);
        }
    }

    /// A primary block's refcount reached zero: what rode on it goes too.
    pub(super) fn block_freed(&mut self, block: u32) {
        self.release_tail_slot_if_freed(block);
        self.window_drop_primary(block);
    }

    /// A primary block went free: its window rows go with it. Every holder
    /// holds a primary ref, so a mapped block here is a refcount bug.
    fn window_drop_primary(&mut self, block: u32) {
        let Some(w) = self.window.as_mut() else {
            return;
        };
        let Some(local) = w.mapped(block) else {
            return;
        };
        tracing::error!(
            "KV window: primary block {block} freed with {} holder(s) of its window rows",
            w.refs[local as usize]
        );
        w.refs[local as usize] = 0;
        w.local_of[block as usize] = NO_WINDOW;
        w.free.push(local);
        w.dirty.push(block);
    }
}
