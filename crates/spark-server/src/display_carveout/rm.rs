// SPDX-License-Identifier: AGPL-3.0-only

//! The NVIDIA resource-manager (RM) calls that export the GB10 display
//! carveout as an fd CUDA can import.
//!
//! Layouts and numbers are from open-gpu-kernel-modules (MIT), identical at
//! the 580.173.02 and 580.178.04 tags; the size and offset assertions below
//! were taken from a probe compiled against those headers. RM structures
//! change between driver releases, which is why the launcher only runs this
//! on [`super::VALIDATED_DRIVERS`].
//!
//! Why the memory is free on GB10 (580.173.02 source): every allocation path
//! into the scanout carveout heap requires `PDB_PROP_GPU_IS_SOC_SDM`, which
//! only GB20B and GB20C set (`g_gpu_nvoc.c`), so nothing on GB10 allocates
//! from it. An exported object survives its exporting client
//! (`rmobjexportimport.c`), so the exporter can exec away.

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
const NV01_MEMORY_LIST_SYSTEM: u32 = 0x81;
const CMD_FB_GET_CARVEOUT_REGION_INFO: u32 = 0x2080_1360;
const CMD_OS_UNIX_EXPORT_OBJECT_TO_FD: u32 = 0x3d05;
const EXPORT_OBJECT_TYPE_RM: u32 = 1;
const CARVEOUT_DISPLAY_FRM: u32 = 0;
/// `NVOS02_FLAGS_PHYSICALITY_CONTIGUOUS | NVOS02_FLAGS_COHERENCY_CACHED`.
const OS02_CONTIGUOUS_CACHED: u32 = 0x1000;
/// `NV_RM_API_VERSION_CMD_QUERY`: report the version, check nothing.
const VERSION_CMD_QUERY: u32 = b'2' as u32;
/// RM's page numbers are 4 KiB whatever the kernel's page size.
const RM_PAGE_SHIFT: u32 = 12;
const H_DEVICE: u32 = 0x5c00_0001;
const H_SUBDEVICE: u32 = 0x5c00_0002;
const H_LIST: u32 = 0x5c00_0010;

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
/// `NV_MEMORY_LIST_ALLOCATION_PARAMS`.
#[repr(C)]
#[derive(Default)]
struct MemoryList {
    h_client: u32,
    h_parent: u32,
    h_object: u32,
    h_hw_res_client: u32,
    h_hw_res_device: u32,
    h_hw_res_handle: u32,
    pte_adjust: u32,
    reserved_0: u32,
    kind: u32,
    flags: u32,
    attr: u32,
    attr2: u32,
    height: u32,
    width: u32,
    format: u32,
    comprcovg: u32,
    zcullcovg: u32,
    page_count: u32,
    heap_owner: u32,
    _pad0: u32,
    guest_id: u64,
    range_begin: u64,
    range_end: u64,
    pitch: u32,
    ctag_offset: u32,
    size: u64,
    align: u64,
    page_number_list: u64,
    limit: u64,
    flags_os02: u32,
    _pad1: u32,
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

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<ApiVersion>() == 72);
    assert!(size_of::<Alloc>() == 32 && size_of::<Control>() == 32);
    assert!(size_of::<RegionInfo>() == 200 && offset_of!(RegionInfo, regions) == 8);
    assert!(size_of::<MemoryList>() == 152);
    assert!(offset_of!(MemoryList, page_count) == 68);
    assert!(offset_of!(MemoryList, size) == 112);
    assert!(offset_of!(MemoryList, page_number_list) == 128);
    assert!(offset_of!(MemoryList, limit) == 136);
    assert!(offset_of!(MemoryList, flags_os02) == 144);
    assert!(size_of::<ExportToFd>() == 24 && offset_of!(ExportToFd, fd) == 16);
};

/// The exported carveout: an inheritable fd holding the RM object.
pub(super) struct Exported {
    pub(super) fd: i32,
    pub(super) base: u64,
    pub(super) size: u64,
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
        let mut register = ctl;
        ioctl(gpu, IOCTL_REGISTER_FD, &mut register, "register fd")?;
        let mut client = Self { ctl, handle: 0 };
        client.handle = client.alloc::<()>(0, 0, NV01_ROOT_CLIENT, None, "client")?;
        let mut device = [0u8; 56]; // NV0080_ALLOC_PARAMETERS, deviceId 0
        client.alloc(
            client.handle,
            H_DEVICE,
            NV01_DEVICE_0,
            Some(&mut device),
            "device",
        )?;
        let mut subdevice = 0u32; // NV2080_ALLOC_PARAMETERS, subDeviceId 0
        client.alloc(
            H_DEVICE,
            H_SUBDEVICE,
            NV20_SUBDEVICE_0,
            Some(&mut subdevice),
            "subdevice",
        )?;
        Ok(client)
    }

    fn alloc<P>(
        &self,
        parent: u32,
        handle: u32,
        class: u32,
        params: Option<&mut P>,
        what: &str,
    ) -> Result<u32> {
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
        if a.status != 0 {
            bail!("RM alloc {what}: status 0x{:x}", a.status);
        }
        Ok(a.h_new)
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
}

/// Wraps the whole `DISPLAY_FRM` carveout in a contiguous memory-list object
/// and exports it to a new, inheritable `/dev/nvidiactl` fd. Creating the
/// object needs CAP_SYS_ADMIN (RM status 0x1b without it).
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
    let mut pfn = frm.base >> RM_PAGE_SHIFT;
    let mut list = MemoryList {
        page_count: 1,
        size: frm.size,
        limit: frm.size - 1,
        page_number_list: &mut pfn as *mut u64 as u64,
        flags_os02: OS02_CONTIGUOUS_CACHED,
        ..Default::default()
    };
    client.alloc(
        H_DEVICE,
        H_LIST,
        NV01_MEMORY_LIST_SYSTEM,
        Some(&mut list),
        "memory list",
    )?;
    let fd = open("/dev/nvidiactl", false)?;
    let mut export = ExportToFd {
        kind: EXPORT_OBJECT_TYPE_RM,
        h_device: H_DEVICE,
        h_parent: H_DEVICE,
        h_object: H_LIST,
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
        base: frm.base,
        size: frm.size,
    })
}
