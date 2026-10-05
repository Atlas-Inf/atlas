// SPDX-License-Identifier: AGPL-3.0-only

//! The GB10 display carveout as device memory (see `crate::gpu::carveout`).
//!
//! `spark display-carveout` passes an RM object fd in
//! [`carveout::FD_ENV`]. The first [`alloc`] imports it with the CUDA VMM
//! API, maps it once for the life of the process, and zeroes it (it still
//! holds whatever the previous borrower wrote). That happens at KV
//! allocation, after the KV budget is measured, so the mapping never counts
//! as this process's footprint in that measurement.
//!
//! The GPU maps it L2-uncached: RM creates every memory-list descriptor with
//! `NV_MEMORY_UNCACHED` and the sysmem list path never changes it
//! (open-gpu-kernel-modules 580.173.02 `mem_desc.c`, `mem_list.c`). One pass
//! over it streams at `cuMemAlloc` speed (~250 GB/s on spark03), but data
//! read again comes from DRAM each time: a 4 MiB region re-read runs at
//! 266 GB/s against 1,679 GB/s, and a dependent load takes 408 ns against
//! 161 ns. Copy-engine transfers (`cuMemcpy*`) run at about half speed, which
//! only the NVMe prefix tier's block spills use. So only buffers that are
//! seldom re-read belong here (`kv_cache::placement`).

use std::sync::OnceLock;

use anyhow::Result;
use parking_lot::Mutex;

use crate::gpu::DevicePtr;
use crate::gpu::carveout::{self, CarveoutArena};

static ARENA: Mutex<Option<CarveoutArena>> = Mutex::new(None);

/// The fd and size from the environment, checked once: `None` without a
/// carveout, or when the fd is not open in this process (a child that only
/// inherited the environment).
fn exported() -> Option<(i32, usize)> {
    static EXPORTED: OnceLock<Option<(i32, usize)>> = OnceLock::new();
    *EXPORTED.get_or_init(|| match carveout::from_env() {
        Ok(Some((fd, size))) if unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0 => Some((fd, size)),
        Ok(Some((fd, _))) => {
            tracing::warn!(
                "{} names fd {fd}, which is not open; no carveout",
                carveout::FD_ENV
            );
            None
        }
        Ok(None) => None,
        Err(e) => {
            tracing::error!("display carveout environment: {e:#}; no carveout");
            None
        }
    })
}

pub(super) fn capacity() -> usize {
    if cfg!(atlas_scale) {
        return 0;
    }
    exported().map_or(0, |(_, size)| size)
}

pub(super) fn alloc(bytes: usize) -> Result<DevicePtr> {
    let mut arena = ARENA.lock();
    if arena.is_none() {
        *arena = Some(map()?);
    }
    let arena = arena.as_mut().expect("mapped above");
    Ok(DevicePtr(arena.alloc(bytes)?))
}

/// Releases `ptr` if it is a carveout piece.
pub(super) fn free(ptr: DevicePtr) -> bool {
    ARENA.lock().as_mut().is_some_and(|a| a.free(ptr.0))
}

#[cfg(atlas_scale)]
fn map() -> Result<CarveoutArena> {
    anyhow::bail!("the display carveout is GB10-only")
}

#[cfg(not(atlas_scale))]
fn map() -> Result<CarveoutArena> {
    use anyhow::{Context, bail};
    use std::ffi::c_void;

    const HANDLE_TYPE_POSIX_FD: u32 = 1;
    const LOCATION_DEVICE: u32 = 1;
    const ACCESS_READ_WRITE: u32 = 3;
    /// `CUmemAccessDesc`.
    #[repr(C)]
    struct AccessDesc {
        location_type: u32,
        location_id: i32,
        flags: u32,
    }
    unsafe extern "C" {
        fn cuMemImportFromShareableHandle(handle: *mut u64, os: *mut c_void, kind: u32) -> i32;
        fn cuMemAddressReserve(ptr: *mut u64, size: usize, align: usize, addr: u64, f: u64) -> i32;
        fn cuMemMap(ptr: u64, size: usize, offset: usize, handle: u64, flags: u64) -> i32;
        fn cuMemRelease(handle: u64) -> i32;
        fn cuMemSetAccess(ptr: u64, size: usize, desc: *const AccessDesc, count: usize) -> i32;
    }
    let check = |status: i32, what: &str| -> Result<()> {
        if status != 0 {
            bail!(
                "display carveout {what}: {}",
                atlas_core::registry::cuda_error_text(status)
            );
        }
        Ok(())
    };

    let (fd, size) = exported().context("no display carveout was exported to this process")?;
    let mut device = 0i32;
    check(unsafe { super::cuCtxGetDevice(&mut device) }, "device")?;
    let mut handle = 0u64;
    // The fd is a plain integer handle for this call, not a pointer.
    let os_handle = fd as isize as *mut c_void;
    check(
        unsafe { cuMemImportFromShareableHandle(&mut handle, os_handle, HANDLE_TYPE_POSIX_FD) },
        "import",
    )?;
    // The import holds its own reference; the fd is no longer needed.
    unsafe { libc::close(fd) };
    let mut va = 0u64;
    check(
        unsafe { cuMemAddressReserve(&mut va, size, 2 << 20, 0, 0) },
        "address reserve",
    )?;
    check(unsafe { cuMemMap(va, size, 0, handle, 0) }, "map")?;
    // The mapping keeps the memory alive; drop the handle's reference.
    check(unsafe { cuMemRelease(handle) }, "release handle")?;
    let access = AccessDesc {
        location_type: LOCATION_DEVICE,
        location_id: device,
        flags: ACCESS_READ_WRITE,
    };
    check(
        unsafe { cuMemSetAccess(va, size, &access, 1) },
        "set access",
    )?;
    check(unsafe { super::cuMemsetD8Async(va, 0, size, 0) }, "zero")?;
    check(unsafe { super::cuStreamSynchronize(0) }, "zero sync")?;
    tracing::info!(
        "Display carveout: {} MiB mapped at 0x{va:x} for KV pools",
        size >> 20
    );
    Ok(CarveoutArena::new(va, size))
}
