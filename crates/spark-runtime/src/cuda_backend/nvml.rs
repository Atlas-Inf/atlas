// SPDX-License-Identifier: AGPL-3.0-only

//! The driver's per-process device-memory accounting, read through NVML.
//!
//! `libnvidia-ml` is opened at run time rather than linked, so a host or
//! container without it loses this one reading and nothing else: every
//! failure below is `None`, and the caller falls back to the backend's
//! allocation ledger (see [`crate::own_footprint`]).
//!
//! Checked on GB10 (driver 580.178, `docs/campaigns/kv-sizing-2026-09`):
//! `usedGpuMemory` moves by exactly what `cuMemAlloc` takes, driver rounding
//! included, and inside a container the library lists that container's
//! processes under their in-namespace PIDs (the probe ran as PID 1), so
//! `std::process::id()` is the right key in both places. `docker run --gpus
//! all` mounted the library there, including with
//! `NVIDIA_DRIVER_CAPABILITIES=compute`.
//!
//! Unix only (`dlopen`); the parent module does not compile this elsewhere.

use std::ffi::{CStr, c_void};

/// `nvmlProcessInfo_t` as `nvmlDeviceGetComputeRunningProcesses_v2` lays it
/// out (24 bytes; the unversioned symbol's 16-byte struct is not used).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: u32,
    used_gpu_memory: u64,
    gpu_instance_id: u32,
    compute_instance_id: u32,
}

const NVML_SUCCESS: i32 = 0;
const NVML_ERROR_INSUFFICIENT_SIZE: i32 = 7;
/// `NVML_VALUE_NOT_AVAILABLE`: the driver lists the process without a figure.
const NOT_AVAILABLE: u64 = u64::MAX;

type InitFn = unsafe extern "C" fn() -> i32;
type CountFn = unsafe extern "C" fn(*mut u32) -> i32;
type HandleFn = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
type ProcessesFn = unsafe extern "C" fn(*mut c_void, *mut u32, *mut ProcessInfo) -> i32;

/// Device bytes the driver attributes to this process, summed over devices.
/// `None` when NVML is absent, fails, or does not list this process.
pub(super) fn process_device_bytes() -> Option<usize> {
    // Never `dlclose`d: the library may keep state behind `nvmlShutdown`, and
    // a second `dlopen` of a loaded library only bumps a reference count.
    let lib = unsafe { libc::dlopen(c"libnvidia-ml.so.1".as_ptr(), libc::RTLD_NOW) };
    if lib.is_null() {
        return None;
    }
    // SAFETY: each symbol is looked up by its documented name and cast to its
    // documented C signature; a null lookup is `None`, never a call.
    let sym = |name: &CStr| {
        let p = unsafe { libc::dlsym(lib, name.as_ptr()) };
        (!p.is_null()).then_some(p)
    };
    let init: InitFn = unsafe { std::mem::transmute(sym(c"nvmlInit_v2")?) };
    let shutdown: InitFn = unsafe { std::mem::transmute(sym(c"nvmlShutdown")?) };
    let count: CountFn = unsafe { std::mem::transmute(sym(c"nvmlDeviceGetCount_v2")?) };
    let handle: HandleFn = unsafe { std::mem::transmute(sym(c"nvmlDeviceGetHandleByIndex_v2")?) };
    // `_v2` first: its 24-byte entry is the same on every driver that exports
    // it, while `_v3` is reported to have carried a longer one on some 530 to
    // 535 drivers. On 580 the two return the same figure.
    let processes: ProcessesFn = unsafe {
        std::mem::transmute(
            sym(c"nvmlDeviceGetComputeRunningProcesses_v2")
                .or_else(|| sym(c"nvmlDeviceGetComputeRunningProcesses_v3"))?,
        )
    };
    if unsafe { init() } != NVML_SUCCESS {
        return None;
    }
    let bytes = unsafe { sum_for_pid(count, handle, processes, std::process::id()) };
    unsafe { shutdown() };
    bytes
}

/// # Safety
/// The three pointers must be the NVML functions of their types, called
/// between `nvmlInit_v2` and `nvmlShutdown`.
unsafe fn sum_for_pid(
    count: CountFn,
    handle: HandleFn,
    processes: ProcessesFn,
    pid: u32,
) -> Option<usize> {
    let mut devices = 0u32;
    if unsafe { count(&mut devices) } != NVML_SUCCESS {
        return None;
    }
    let mut total: Option<u64> = None;
    for index in 0..devices {
        let mut device: *mut c_void = std::ptr::null_mut();
        if unsafe { handle(index, &mut device) } != NVML_SUCCESS {
            return None;
        }
        let mut infos = vec![ProcessInfo::default(); 64];
        let mut len = infos.len() as u32;
        let mut status = unsafe { processes(device, &mut len, infos.as_mut_ptr()) };
        if status == NVML_ERROR_INSUFFICIENT_SIZE {
            // `len` now holds the count needed; leave room for late arrivals.
            infos.resize(len as usize + 16, ProcessInfo::default());
            len = infos.len() as u32;
            status = unsafe { processes(device, &mut len, infos.as_mut_ptr()) };
        }
        if status != NVML_SUCCESS {
            return None;
        }
        for info in infos.iter().take(len as usize).filter(|i| i.pid == pid) {
            if info.used_gpu_memory == NOT_AVAILABLE {
                return None;
            }
            total = Some(total.unwrap_or(0).saturating_add(info.used_gpu_memory));
        }
    }
    total.and_then(|bytes| usize::try_from(bytes).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_info_matches_the_nvml_v2_abi() {
        assert_eq!(std::mem::size_of::<ProcessInfo>(), 24);
        assert_eq!(std::mem::offset_of!(ProcessInfo, used_gpu_memory), 8);
        assert_eq!(std::mem::offset_of!(ProcessInfo, gpu_instance_id), 16);
        assert_eq!(std::mem::offset_of!(ProcessInfo, compute_instance_id), 20);
    }

    mod fake {
        use super::super::*;

        pub(super) unsafe extern "C" fn two_devices(n: *mut u32) -> i32 {
            unsafe { *n = 2 };
            NVML_SUCCESS
        }
        pub(super) unsafe extern "C" fn handle(index: u32, out: *mut *mut c_void) -> i32 {
            unsafe { *out = (index as usize + 1) as *mut c_void };
            NVML_SUCCESS
        }
        /// Device 1 lists 70 processes (more than the first buffer holds), with
        /// pid 7 holding 5 bytes; device 2 lists pid 7 with 11 and pid 9
        /// without a figure.
        pub(super) unsafe extern "C" fn processes(
            device: *mut c_void,
            len: *mut u32,
            out: *mut ProcessInfo,
        ) -> i32 {
            let listed: Vec<(u32, u64)> = if device as usize == 1 {
                (100..169).map(|p| (p, 1)).chain([(7, 5)]).collect()
            } else {
                vec![(7, 11), (9, NOT_AVAILABLE)]
            };
            let room = unsafe { *len } as usize;
            unsafe { *len = listed.len() as u32 };
            if room < listed.len() {
                return NVML_ERROR_INSUFFICIENT_SIZE;
            }
            for (i, (pid, used)) in listed.into_iter().enumerate() {
                let info = ProcessInfo {
                    pid,
                    used_gpu_memory: used,
                    ..ProcessInfo::default()
                };
                unsafe { out.add(i).write(info) };
            }
            NVML_SUCCESS
        }
    }

    #[test]
    fn sums_this_pid_across_devices_and_regrows_a_short_buffer() {
        let read =
            |pid| unsafe { sum_for_pid(fake::two_devices, fake::handle, fake::processes, pid) };
        assert_eq!(read(7), Some(16));
        assert_eq!(read(100), Some(1));
        assert_eq!(read(8), None, "a process the driver does not list");
        assert_eq!(
            read(9),
            None,
            "listed without a figure is unknown, not zero"
        );
    }

    #[test]
    fn a_host_without_the_library_or_a_context_does_not_fail() {
        // `None` on a host with no NVML and for a process the driver does not
        // list; either way the read returns instead of taking the boot down.
        let _ = process_device_bytes();
    }
}
