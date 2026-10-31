// SPDX-License-Identifier: AGPL-3.0-only
//! Mock GPU backend for unit tests (no GPU required).

use super::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

mod backend_impl;
mod carveout;

#[derive(Debug)]
pub struct MockAlloc {
    pub bytes: usize,
    pub data: Vec<u8>,
}

/// Records kernel launches and memory operations for test assertions.
pub struct MockGpuBackend {
    op_cache: crate::op_cache::OpCache,
    allocs: Mutex<HashMap<u64, MockAlloc>>,
    next_ptr: Mutex<u64>,
    launches: Mutex<Vec<MockLaunch>>,
    /// Copy/sync shape counters. These exist so tests can assert the SHAPE of a
    /// bulk transfer, not just its bytes: the SSM snapshot spill regressed to
    /// 60 blocking `copy_d2h` calls (one full stream drain each, ~400 ms for
    /// 66 MB) and nothing caught it, because the bytes were correct.
    syncs: AtomicUsize,
    d2h_blocking: AtomicUsize,
    d2h_async: AtomicUsize,
    /// `copy_d2d`/`copy_d2d_async` calls — one eager launch each on the real
    /// backend. The SSM verify rollback issued 2 per SSM layer per sequence
    /// (96 on the 27B), so this counter is what proves a batched form
    /// actually batched rather than merely looking different.
    d2d: AtomicUsize,
    /// `copy_d2d_2d_async` calls — ONE `cudaMemcpy2DAsync` each on the real
    /// backend regardless of `height`. Counted apart from `d2d` so a test can
    /// assert the SHAPE of the transfer, not just the bytes.
    d2d_2d: AtomicUsize,
    /// Pitched host↔device copies — ONE enqueue each on the real backend,
    /// whatever the row count.
    host_pitched: AtomicUsize,
    /// `memset_async` calls — one eager launch each on the real backend.
    memsets: AtomicUsize,
    host_pinned_allocs: AtomicUsize,
    /// Pending `free` failure injections (see [`Self::fail_next_free`]).
    pending_free_failures: AtomicUsize,
    /// `free` calls on a pointer that is not the base of a live allocation
    /// (see [`Self::invalid_free_count`]).
    invalid_frees: AtomicUsize,
    /// `destroy_graph` call counter (see [`Self::destroy_graph_count`]).
    destroyed_graphs: AtomicUsize,
    /// Streams and events created and not yet destroyed (see
    /// [`Self::live_streams_and_events`]).
    live_streams: AtomicUsize,
    live_events: AtomicUsize,
    /// Lendable carveout ([`Self::with_carveout`]); `None` = no carveout.
    carveout: Mutex<Option<super::carveout::CarveoutArena>>,
}

#[derive(Debug, Clone)]
pub struct MockLaunch {
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Kernel parameter count (`kernelParams` entries).
    pub args: usize,
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            op_cache: crate::op_cache::OpCache::new(),
            allocs: Mutex::new(HashMap::new()),
            next_ptr: Mutex::new(0x1000_0000),
            launches: Mutex::new(Vec::new()),
            syncs: AtomicUsize::new(0),
            d2h_blocking: AtomicUsize::new(0),
            d2h_async: AtomicUsize::new(0),
            d2d: AtomicUsize::new(0),
            d2d_2d: AtomicUsize::new(0),
            host_pitched: AtomicUsize::new(0),
            memsets: AtomicUsize::new(0),
            host_pinned_allocs: AtomicUsize::new(0),
            pending_free_failures: AtomicUsize::new(0),
            invalid_frees: AtomicUsize::new(0),
            destroyed_graphs: AtomicUsize::new(0),
            live_streams: AtomicUsize::new(0),
            live_events: AtomicUsize::new(0),
            carveout: Mutex::new(None),
        }
    }

    /// `(streams, events)` created and not yet destroyed: the leak check of
    /// whatever owns them.
    pub fn live_streams_and_events(&self) -> (usize, usize) {
        (
            self.live_streams.load(Ordering::Relaxed),
            self.live_events.load(Ordering::Relaxed),
        )
    }

    /// Queue one `free` failure: the next `free` call returns an error and
    /// consumes one queued failure. Lets cleanup-path tests prove a failed
    /// free keeps its resource observable/retryable instead of silently
    /// leaking (queue N to fail N consecutive frees).
    pub fn fail_next_free(&self) {
        self.pending_free_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// `destroy_graph` calls so far — the tripwire proving a cleanup error
    /// path does NOT destroy pooled graphs.
    pub fn destroy_graph_count(&self) -> usize {
        self.destroyed_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `free` calls whose pointer was not the base of a live allocation: a
    /// double free, or an offset view into a block. The real backend rejects
    /// both with `CUDA_ERROR_INVALID_VALUE` and puts the pointer back on its
    /// cleanup ledger; the mock accepts them silently, so this counter is how
    /// a teardown test sees them.
    pub fn invalid_free_count(&self) -> usize {
        self.invalid_frees.load(Ordering::Relaxed)
    }

    pub fn alloc_count(&self) -> usize {
        self.allocs.lock().len()
    }

    pub fn launch_count(&self) -> usize {
        self.launches.lock().len()
    }

    /// `synchronize` calls so far — a proxy for "full stream drains", the cost
    /// a batched gather exists to amortize.
    pub fn sync_count(&self) -> usize {
        self.syncs.load(Ordering::Relaxed)
    }

    /// BLOCKING `copy_d2h` calls (each one drains the stream on the real
    /// backend). A bulk gather must have zero of these.
    pub fn d2h_blocking_count(&self) -> usize {
        self.d2h_blocking.load(Ordering::Relaxed)
    }

    /// `copy_d2h_async` calls (enqueue-only).
    pub fn d2h_async_count(&self) -> usize {
        self.d2h_async.load(Ordering::Relaxed)
    }

    /// `copy_d2d` + `copy_d2d_async` calls so far — one eager launch each on
    /// the real backend.
    pub fn d2d_count(&self) -> usize {
        self.d2d.load(Ordering::Relaxed)
    }

    /// `copy_d2d_2d_async` calls so far — one `cudaMemcpy2DAsync` each,
    /// whatever the row count.
    pub fn d2d_2d_count(&self) -> usize {
        self.d2d_2d.load(Ordering::Relaxed)
    }

    /// Pitched host↔device copies so far (either direction).
    pub fn host_pitched_count(&self) -> usize {
        self.host_pitched.load(Ordering::Relaxed)
    }

    /// `memset_async` calls so far.
    pub fn memset_count(&self) -> usize {
        self.memsets.load(Ordering::Relaxed)
    }

    /// `alloc_host_pinned` calls — the tripwire for a staging buffer that is
    /// re-allocated per event instead of reused.
    pub fn host_pinned_alloc_count(&self) -> usize {
        self.host_pinned_allocs.load(Ordering::Relaxed)
    }

    pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.allocs.lock().get(&ptr.0).map(|a| a.data.clone())
    }

    /// `bytes` from `src` to `dst` inside the simulated device memory.
    ///
    /// Real byte movement, not a no-op: a D2D that silently succeeds without
    /// moving anything lets a test "pass" while asserting the destination is
    /// still zero — the exact shape of a rollback bug this backend exists to
    /// catch. Source is staged through a temporary so `src` and `dst` may sit
    /// in the same allocation (the borrow checker would otherwise reject it,
    /// and the real `cudaMemcpyAsync` accepts it for non-overlapping ranges).
    fn blit(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut allocs = self.allocs.lock();
        let staged = {
            let (offset, alloc) = find_alloc(&allocs, src)
                .ok_or_else(|| anyhow::anyhow!("copy_d2d: src {src} not allocated"))?;
            if offset + bytes > alloc.bytes {
                anyhow::bail!("copy_d2d: src {src} + {bytes} overruns its allocation");
            }
            alloc.data[offset..offset + bytes].to_vec()
        };
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_d2d: dst {dst} not allocated"))?;
        if offset + bytes > alloc.bytes {
            anyhow::bail!("copy_d2d: dst {dst} + {bytes} overruns its allocation");
        }
        alloc.data[offset..offset + bytes].copy_from_slice(&staged);
        Ok(())
    }

    /// Every launch recorded so far, in dispatch order. Lets a test assert
    /// WHICH kernel shape ran (grid/block signature), not just how many —
    /// the mock's `kernel()` hands out one shared handle, so geometry is
    /// the only per-launch identity available.
    pub fn launches_snapshot(&self) -> Vec<MockLaunch> {
        self.launches.lock().clone()
    }
}

/// Find the allocation containing `ptr` (supports offset pointers).
fn find_alloc(allocs: &HashMap<u64, MockAlloc>, ptr: DevicePtr) -> Option<(usize, &MockAlloc)> {
    for (&base, alloc) in allocs.iter() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

/// Mutable version of find_alloc.
fn find_alloc_mut(
    allocs: &mut HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &mut MockAlloc)> {
    for (&base, alloc) in allocs.iter_mut() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

impl HostPitched for MockGpuBackend {
    fn h2d_retained(&self, src: &[u8], dst: DevicePtr, shape: Pitched, _stream: u64) -> Result<()> {
        // One tick per call, as `copy_d2d_2d_async`: the rows are emulation.
        self.host_pitched.fetch_add(1, Ordering::Relaxed);
        for r in 0..shape.height {
            let row = &src[r * shape.host_pitch..][..shape.width];
            self.copy_h2d(row, dst.offset(r * shape.dev_pitch))?;
        }
        Ok(())
    }

    fn d2h(&self, src: DevicePtr, dst: &mut [u8], shape: Pitched, _stream: u64) -> Result<()> {
        self.host_pitched.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        for r in 0..shape.height {
            let from = src.offset(r * shape.dev_pitch);
            let (offset, alloc) = find_alloc(&allocs, from)
                .ok_or_else(|| anyhow::anyhow!("pitched D2H: ptr {from} not allocated"))?;
            dst[r * shape.host_pitch..][..shape.width]
                .copy_from_slice(&alloc.data[offset..offset + shape.width]);
        }
        Ok(())
    }
}
