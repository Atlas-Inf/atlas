// SPDX-License-Identifier: AGPL-3.0-only
//! `GpuBackend` impl for [`MockGpuBackend`].

use super::*;

impl GpuBackend for MockGpuBackend {
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut next = self.next_ptr.lock();
        let ptr = *next;
        *next += bytes as u64;
        // Align to 256 bytes
        *next = (*next + 255) & !255;
        self.allocs.lock().insert(
            ptr,
            MockAlloc {
                bytes,
                data: vec![0u8; bytes],
            },
        );
        Ok(DevicePtr(ptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes) // Mock: same as regular alloc
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if self
            .pending_free_failures
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("mock free failed (injected): ptr {ptr}");
        }
        self.carveout_release(ptr);
        if self.allocs.lock().remove(&ptr.0).is_none() && !ptr.is_null() {
            self.invalid_frees.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn carveout_capacity(&self) -> usize {
        self.carveout_bytes()
    }

    fn alloc_carveout(&self, bytes: usize) -> Result<DevicePtr> {
        self.carveout_alloc(bytes)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing dst
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_h2d: ptr {dst} not allocated"))?;
        alloc.data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.d2h_blocking.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing src
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], _stream: u64) -> Result<()> {
        // Counted separately from `copy_d2h` and NOT delegating to it, so a
        // test can distinguish the batched shape from the blocking one (the
        // trait's default impl forwards, which would make them indistinguishable).
        self.d2h_async.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h_async: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        // NOT delegating to `copy_d2d`: the trait default forwards, which
        // would make the two indistinguishable to `d2d_count` consumers only
        // by accident. Counted here so both forms land in one counter on
        // purpose.
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_2d_async(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        _stream: u64,
    ) -> Result<()> {
        // ONE launch on the real backend (`cudaMemcpy2DAsync`), so ONE tick —
        // the row loop below is emulation, not dispatch, and must not inflate
        // `d2d_count` (which exists to prove a batched form batched).
        self.d2d_2d.fetch_add(1, Ordering::Relaxed);
        if width_bytes > src_pitch || width_bytes > dst_pitch {
            anyhow::bail!(
                "copy_d2d_2d_async: width {width_bytes} exceeds pitch \
                 (src {src_pitch}, dst {dst_pitch}) — rows would overlap"
            );
        }
        for r in 0..height {
            self.blit(
                src.offset(r * src_pitch),
                dst.offset(r * dst_pitch),
                width_bytes,
            )?;
        }
        Ok(())
    }

    fn host_pitched(&self) -> Option<&dyn HostPitched> {
        Some(self)
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
            args: params.len(),
        });
        Ok(())
    }

    fn synchronize(&self, _stream: u64) -> Result<()> {
        self.syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    #[track_caller]
    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(0xDEAD))
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let mut allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc_mut(&mut allocs, ptr)
            .ok_or_else(|| anyhow::anyhow!("memset: ptr {ptr} not allocated"))?;
        alloc.data[offset..offset + bytes].fill(value);
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.memsets.fetch_add(1, Ordering::Relaxed);
        self.memset(ptr, value, bytes)
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // Same heap allocation as the trait default (the mock cannot page-lock);
        // overridden solely to COUNT, so a test can prove a staging buffer is
        // allocated once and reused rather than per event.
        self.host_pinned_allocs.fetch_add(1, Ordering::Relaxed);
        let layout = std::alloc::Layout::from_size_align(bytes, HOST_PINNED_ALIGN)
            .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            anyhow::bail!("host alloc failed: {bytes} bytes");
        }
        Ok(ptr)
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(128 * 1024 * 1024 * 1024) // 128 GB
    }

    fn sm_count(&self) -> Result<u32> {
        // Rationale (PCND): tests that exercise occupancy-gated dispatch need
        // SOME machine width; 48 is the GB10 value the model targets, chosen
        // so mock runs take the same branch production does.
        Ok(48)
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(120 * 1024 * 1024 * 1024) // 120 GB
    }

    fn destroy_graph(&self, _graph: GraphHandle) -> Result<()> {
        self.destroyed_graphs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn create_stream(&self) -> Result<u64> {
        self.live_streams.fetch_add(1, Ordering::Relaxed);
        Ok(0)
    }

    fn destroy_stream(&self, _stream: u64) -> Result<()> {
        self.live_streams.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        self.live_events.fetch_add(1, Ordering::Relaxed);
        Ok(0)
    }

    fn destroy_event(&self, _event: u64) -> Result<()> {
        self.live_events.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }
}
