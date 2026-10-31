// SPDX-License-Identifier: AGPL-3.0-only

//! The NVIDIA resource-manager (RM) calls that allocate the GB10 display
//! carveout from the driver and export it as an fd CUDA can import.
//!
//! The driver owns the carveout. RM knows GB10 as chip GB20B, for which it
//! manages the firmware's `DISPLAY_FRM` region as its scanout heap: display
//! surfaces come from it (a few MiB with no monitor attached), and any
//! client may allocate from it by asking for
//! `NVOS32_ATTR2_USE_SCANOUT_CARVEOUT` (open-gpu-kernel-modules
//! `mem_mgr_gb10b_phys.c`, `mem_utils.c`, `system_mem.c`). Allocating through
//! the heap needs no privilege and never overlaps what the display driver
//! holds. The object asks for `GPU_CACHEABLE`, so the GPU caches it in L2
//! like `cuMemAlloc` memory (RM's default for system memory is uncached).
//!
//! Driver compatibility: every number and layout here, and the scanout heap
//! behind them, is the same in all 52 open-gpu-kernel-modules releases from
//! 580.76.05 through 615.78.08 (a probe compiled against each tag's headers,
//! plus the heap code; 580.65.06 has the numbers but no heap). No driver
//! version is required or trusted:
//! - The kernel checks each ioctl's argument size and RM each control's
//!   parameter size, so a changed layout fails cleanly.
//! - RM copies its own size of an allocation's parameters in and out, so
//!   those buffers carry a zeroed tail ([`Padded`]).
//! - The object must prove it came from the scanout heap ([`largest_fit`]).
//!
//! An exported object outlives its exporting client (`rmobjexportimport.c`),
//! so the exporter can exec away.

use std::ffi::CString;
use std::io::Error;

use anyhow::{Result, bail};

const IOCTL_RM_ALLOC: libc::c_ulong = 0xc020_462b;
const IOCTL_RM_CONTROL: libc::c_ulong = 0xc020_462a;
const IOCTL_CHECK_VERSION_STR: libc::c_ulong = 0xc048_46d2;
const IOCTL_REGISTER_FD: libc::c_ulong = 0xc004_46c9;
const NV01_ROOT_CLIENT: u32 = 0x41;
const NV01_DEVICE_0: u32 = 0x80;
const NV20_SUBDEVICE_0: u32 = 0x2080;
const NV01_MEMORY_SYSTEM: u32 = 0x3e;
const CMD_FB_GET_CARVEOUT_REGION_INFO: u32 = 0x2080_1360;
const CMD_OS_UNIX_EXPORT_OBJECT_TO_FD: u32 = 0x3d05;
const EXPORT_OBJECT_TYPE_RM: u32 = 1;
const CARVEOUT_DISPLAY_FRM: u32 = 0;
/// `NVOS32_ATTR_LOCATION_PCI | _PHYSICALITY_CONTIGUOUS | _COHERENCY_CACHED`.
const ATTR: u32 = (1 << 25) | (2 << 27) | (1 << 29);
/// `NVOS32_ATTR2_USE_SCANOUT_CARVEOUT_TRUE | _GPU_CACHEABLE_YES`.
const ATTR2: u32 = (1 << 10) | (1 << 2);
/// Any non-zero owner tag: RM refuses owner 0.
const OWNER: u32 = u32::from_le_bytes(*b"SPRK");
/// `NV_RM_API_VERSION_CMD_QUERY`: report the version, check nothing.
const VERSION_CMD_QUERY: u32 = b'2' as u32;
/// `NV_ERR_NO_MEMORY`: the heap has no free range that large.
const STATUS_NO_MEMORY: u32 = 0x51;
/// Size steps of the search: the GPU maps the object in 2 MiB pages.
const STEP: u64 = 2 << 20;
const H_DEVICE: u32 = 0x5c00_0001;
const H_SUBDEVICE: u32 = 0x5c00_0002;
const H_MEMORY: u32 = 0x5c00_0010;

/// `nv_ioctl_rm_api_version_t`.
#[repr(C)]
struct ApiVersion {
    cmd: u32,
    reply: u32,
    version: [u8; 64],
}
/// `NVOS21_PARAMETERS` (RM_ALLOC).
#[repr(C)]
struct Alloc {
    h_root: u32,
    h_parent: u32,
    h_new: u32,
    h_class: u32,
    params: u64,
    params_size: u32,
    status: u32,
}
/// `NVOS54_PARAMETERS` (RM_CONTROL).
#[repr(C)]
struct Control {
    h_client: u32,
    h_object: u32,
    cmd: u32,
    flags: u32,
    params: u64,
    params_size: u32,
    status: u32,
}
/// `NV2080_CTRL_FB_GET_CARVEOUT_REGION_INFO`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Region {
    base: u64,
    size: u64,
    kind: u32,
    _pad: u32,
}
/// `NV2080_CTRL_FB_GET_CARVEOUT_REGION_INFO_PARAMS`.
#[repr(C)]
#[derive(Default)]
struct RegionInfo {
    count: u32,
    _pad: u32,
    regions: [Region; 8],
}
/// The fields used of `NV_MEMORY_ALLOCATION_PARAMS` (128 bytes).
#[repr(C)]
#[derive(Default)]
struct MemoryAlloc {
    owner: u32,
    kind: u32,
    flags: u32,
    _surface: [u32; 3],
    attr: u32,
    attr2: u32,
    _format: [u32; 4],
    _range: [u64; 2],
    size: u64,
    _rest: [u64; 7],
}
/// `NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS`.
#[repr(C)]
struct ExportToFd {
    kind: u32,
    h_device: u32,
    h_parent: u32,
    h_object: u32,
    fd: i32,
    flags: u32,
}

/// Allocation parameters with a zeroed tail: RM copies in and out its own
/// size of a class's parameters, whatever size the caller passes, so a
/// release whose parameters grow writes into the tail and reads zeros
/// (each new field's default) from it.
#[repr(C)]
#[derive(Default)]
struct Padded<T> {
    params: T,
    _tail: Tail,
}
#[repr(C)]
struct Tail([u8; 1024]);
impl Default for Tail {
    fn default() -> Self {
        Self([0; 1024])
    }
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<ApiVersion>() == 72);
    assert!(size_of::<Alloc>() == 32 && size_of::<Control>() == 32);
    assert!(offset_of!(Alloc, status) == 28 && offset_of!(Control, status) == 28);
    assert!(size_of::<RegionInfo>() == 200 && offset_of!(RegionInfo, regions) == 8);
    assert!(size_of::<Region>() == 24 && offset_of!(Region, kind) == 16);
    assert!(size_of::<MemoryAlloc>() == 128);
    assert!(offset_of!(MemoryAlloc, attr) == 24 && offset_of!(MemoryAlloc, attr2) == 28);
    assert!(offset_of!(MemoryAlloc, size) == 64);
    assert!(size_of::<ExportToFd>() == 24 && offset_of!(ExportToFd, fd) == 16);
};

/// The exported carveout: an inheritable fd holding the RM object.
pub(super) struct Exported {
    pub(super) fd: i32,
    pub(super) size: u64,
    /// `DISPLAY_FRM`'s size; the display driver holds the rest.
    pub(super) region: u64,
    pub(super) driver: String,
}

fn ioctl<T>(fd: i32, request: libc::c_ulong, arg: &mut T, what: &str) -> Result<()> {
    if unsafe { libc::ioctl(fd, request as _, arg as *mut T) } < 0 {
        bail!("{what}: {}", Error::last_os_error());
    }
    Ok(())
}

fn open(path: &str, cloexec: bool) -> Result<i32> {
    let c = CString::new(path)?;
    let flags = libc::O_RDWR | if cloexec { libc::O_CLOEXEC } else { 0 };
    let fd = unsafe { libc::open(c.as_ptr(), flags) };
    if fd < 0 {
        bail!("open {path}: {}", Error::last_os_error());
    }
    Ok(fd)
}

/// An RM client with device 0 and its subdevice. Its fds close on exec,
/// which frees the client; exported objects outlive it.
struct Client {
    ctl: i32,
    handle: u32,
    driver: String,
}

impl Client {
    fn open() -> Result<Self> {
        let ctl = open("/dev/nvidiactl", true)?;
        let gpu = open("/dev/nvidia0", true)?;
        let mut version = ApiVersion {
            cmd: VERSION_CMD_QUERY,
            reply: 0,
            version: [0; 64],
        };
        ioctl(ctl, IOCTL_CHECK_VERSION_STR, &mut version, "RM version")?;
        let end = version.version.iter().position(|&b| b == 0).unwrap_or(64);
        let driver = String::from_utf8_lossy(&version.version[..end]).into_owned();
        let mut register = ctl;
        ioctl(gpu, IOCTL_REGISTER_FD, &mut register, "register fd")?;
        let mut client = Self {
            ctl,
            handle: 0,
            driver,
        };
        client.handle = client.alloc::<()>(0, 0, NV01_ROOT_CLIENT, None, "client")?;
        // NV0080_ALLOC_PARAMETERS and NV2080_ALLOC_PARAMETERS, all zero:
        // device 0, subdevice 0.
        client.alloc(
            client.handle,
            H_DEVICE,
            NV01_DEVICE_0,
            Some(&mut Padded::<[u64; 7]>::default()),
            "device",
        )?;
        client.alloc(
            H_DEVICE,
            H_SUBDEVICE,
            NV20_SUBDEVICE_0,
            Some(&mut Padded::<u32>::default()),
            "subdevice",
        )?;
        Ok(client)
    }

    /// Allocates `class` as `handle` (0 lets RM choose) and returns the
    /// object's handle.
    fn alloc<P>(
        &self,
        parent: u32,
        handle: u32,
        class: u32,
        params: Option<&mut P>,
        what: &str,
    ) -> Result<u32> {
        match self.try_alloc(parent, handle, class, params, what)? {
            (0, h_new) => Ok(h_new),
            (status, _) => bail!("RM alloc {what}: status 0x{status:x}"),
        }
    }

    /// The RM status of the allocation (0 on success) and the new handle.
    fn try_alloc<P>(
        &self,
        parent: u32,
        handle: u32,
        class: u32,
        params: Option<&mut P>,
        what: &str,
    ) -> Result<(u32, u32)> {
        let (ptr, size) = params.map_or((0, 0), |p| (p as *mut P as u64, size_of::<P>() as u32));
        let mut a = Alloc {
            h_root: self.handle,
            h_parent: parent,
            h_new: handle,
            h_class: class,
            params: ptr,
            params_size: size,
            status: 0,
        };
        ioctl(self.ctl, IOCTL_RM_ALLOC, &mut a, what)?;
        Ok((a.status, a.h_new))
    }

    fn control<P>(&self, object: u32, cmd: u32, params: &mut P, what: &str) -> Result<()> {
        let mut c = Control {
            h_client: self.handle,
            h_object: object,
            cmd,
            flags: 0,
            params: params as *mut P as u64,
            params_size: size_of::<P>() as u32,
            status: 0,
        };
        ioctl(self.ctl, IOCTL_RM_CONTROL, &mut c, what)?;
        if c.status != 0 {
            bail!("RM control {what}: status 0x{:x}", c.status);
        }
        Ok(())
    }

    /// Asks the scanout heap for `size` bytes as [`H_MEMORY`]: `false` if no
    /// free range is that large.
    fn alloc_scanout(&self, size: u64) -> Result<bool> {
        let mut params = Padded {
            params: MemoryAlloc {
                owner: OWNER,
                attr: ATTR,
                attr2: ATTR2,
                size,
                ..Default::default()
            },
            ..Default::default()
        };
        let (status, _) = self.try_alloc(
            H_DEVICE,
            H_MEMORY,
            NV01_MEMORY_SYSTEM,
            Some(&mut params),
            "scanout memory",
        )?;
        match status {
            0 => Ok(true),
            STATUS_NO_MEMORY => Ok(false),
            _ => bail!("RM alloc scanout memory of {size} bytes: status 0x{status:x}"),
        }
    }
}

/// The largest whole number of [`STEP`]s up to `region` that `fits`
/// allocates, searching down and stopping at the first that does (that
/// allocation is kept). It must also prove the allocator is the scanout
/// heap over `region`: one step more than `region` must not fit, and the
/// result must be at least half of `region`, which the kernel cannot give
/// as contiguous system memory.
fn largest_fit(region: u64, mut fits: impl FnMut(u64) -> Result<bool>) -> Result<u64> {
    let top = region / STEP * STEP;
    if top == 0 {
        bail!("the {region}-byte DISPLAY_FRM carveout is too small");
    }
    if fits(top + STEP)? {
        bail!("the driver allocated more than DISPLAY_FRM: not its scanout heap");
    }
    let floor = region / 2;
    let mut size = top;
    while size >= floor {
        if fits(size)? {
            return Ok(size);
        }
        size -= STEP;
    }
    bail!("the scanout heap has no free range of half of DISPLAY_FRM ({floor} bytes)")
}

/// Allocates the largest free range of the `DISPLAY_FRM` scanout heap and
/// exports it to a new, inheritable `/dev/nvidiactl` fd.
pub(super) fn export_display_frm() -> Result<Exported> {
    let client = Client::open()?;
    let mut info = RegionInfo::default();
    client.control(
        H_SUBDEVICE,
        CMD_FB_GET_CARVEOUT_REGION_INFO,
        &mut info,
        "carveout info",
    )?;
    let regions = &info.regions[..(info.count as usize).min(info.regions.len())];
    let Some(frm) = regions
        .iter()
        .find(|r| r.kind == CARVEOUT_DISPLAY_FRM && r.size > 0)
    else {
        bail!("the driver reports no DISPLAY_FRM carveout");
    };
    let size = largest_fit(frm.size, |size| client.alloc_scanout(size))?;
    let fd = open("/dev/nvidiactl", false)?;
    let mut export = ExportToFd {
        kind: EXPORT_OBJECT_TYPE_RM,
        h_device: H_DEVICE,
        h_parent: H_DEVICE,
        h_object: H_MEMORY,
        fd,
        flags: 0,
    };
    if let Err(e) = client.control(
        client.handle,
        CMD_OS_UNIX_EXPORT_OBJECT_TO_FD,
        &mut export,
        "export",
    ) {
        unsafe { libc::close(fd) };
        return Err(e);
    }
    Ok(Exported {
        fd,
        size,
        region: frm.size,
        driver: client.driver,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const FRM: u64 = 2046 * MIB;

    /// A heap with `free` bytes in one range: records each request.
    fn heap(free: u64, asked: &mut Vec<u64>) -> impl FnMut(u64) -> Result<bool> + '_ {
        move |size| {
            asked.push(size);
            Ok(size <= free)
        }
    }

    #[test]
    fn the_largest_free_range_is_kept() {
        // The display driver holds a few MiB of the region (seen on spark03).
        let mut asked = vec![];
        assert_eq!(
            largest_fit(FRM, heap(2043 * MIB, &mut asked)).unwrap(),
            2042 * MIB
        );
        assert_eq!(asked, [2048 * MIB, 2046 * MIB, 2044 * MIB, 2042 * MIB]);
        let mut asked = vec![];
        assert_eq!(largest_fit(FRM, heap(FRM, &mut asked)).unwrap(), FRM);
    }

    #[test]
    fn an_allocator_larger_than_the_region_is_refused() {
        // Plain system memory, not the scanout heap.
        let mut asked = vec![];
        let err = largest_fit(FRM, heap(u64::MAX, &mut asked)).unwrap_err();
        assert!(err.to_string().contains("not its scanout heap"), "{err}");
        assert_eq!(asked, [2048 * MIB], "nothing else is asked for");
    }

    #[test]
    fn less_than_half_the_region_is_refused() {
        // What the kernel could still give as contiguous memory if the
        // scanout bit were ignored.
        let mut asked = vec![];
        let err = largest_fit(FRM, heap(512 * MIB, &mut asked)).unwrap_err();
        assert!(err.to_string().contains("half of DISPLAY_FRM"), "{err}");
        assert!(asked.iter().all(|&s| s >= FRM / 2 || s == 2048 * MIB));
    }

    #[test]
    fn an_rm_error_stops_the_search() {
        let mut calls = 0;
        let err = largest_fit(FRM, |_| {
            calls += 1;
            bail!("status 0x1f")
        })
        .unwrap_err();
        assert!(err.to_string().contains("0x1f"));
        assert_eq!(calls, 1);
    }
}
