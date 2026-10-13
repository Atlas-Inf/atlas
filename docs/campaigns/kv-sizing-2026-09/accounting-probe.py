#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""GB10 unified-memory accounting probe for Atlas KV sizing (own_footprint.rs).

For ONE process, which counters move when it takes memory each way Atlas does:
  MemAvailable (what free-memory sizing reads), the driver's per-process figure
  (NVML usedGpuMemory, read through the library ABI under this process's PID),
  and RssAnon / RssShmem from /proc/self/status.

SETTLE=<seconds> (default 0.4) is the wait before every sample.

Bounded: at most 4.3 GiB held at once, refuses the 4 GiB step unless 16 GiB
would stay available, frees everything and destroys the context on exit.
"""
import ctypes, os, time

MIB = 1 << 20
cuda = ctypes.CDLL("libcuda.so.1")
nv = ctypes.CDLL("libnvidia-ml.so.1")


class Info(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong),
                ("gi", ctypes.c_uint), ("ci", ctypes.c_uint)]


def chk(status, what):
    if status != 0:
        raise SystemExit(f"{what} failed: status {status}")


def nvml_self(sym="nvmlDeviceGetComputeRunningProcesses_v3"):
    chk(nv.nvmlInit_v2(), "nvmlInit")
    try:
        n = ctypes.c_uint()
        chk(nv.nvmlDeviceGetCount_v2(ctypes.byref(n)), "count")
        total, seen = 0, False
        for i in range(n.value):
            h = ctypes.c_void_p()
            chk(nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)), "handle")
            cnt = ctypes.c_uint(256)
            buf = (Info * 256)()
            chk(getattr(nv, sym)(h, ctypes.byref(cnt), buf), sym)
            for j in range(cnt.value):
                if buf[j].pid == os.getpid():
                    total += buf[j].used
                    seen = True
        return total if seen else None
    finally:
        nv.nvmlShutdown()


def meminfo(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise SystemExit(key)


def status():
    out = {}
    for line in open("/proc/self/status"):
        k, v = line.split(":", 1)
        if k in ("RssAnon", "RssFile", "RssShmem"):
            out[k] = int(v.split()[0]) * 1024
    return out


SETTLE = float(os.environ.get("SETTLE", "0.4"))


def snap():
    time.sleep(SETTLE)  # freed pinned memory returns to MemAvailable about a second late
    s = status()
    return dict(avail=meminfo("MemAvailable"), nvml=nvml_self(), anon=s["RssAnon"], shmem=s["RssShmem"])


def show(label, base, now):
    d = lambda k: (now[k] - base[k]) / MIB
    print(f"{label:46s} MemAvailable {-d('avail'):+9.1f}  driver {d('nvml'):+9.1f}  RssAnon {d('anon'):+7.1f}"
          f"  RssShmem {d('shmem'):+7.1f}  tracked(driver+anon+shmem) {d('nvml') + d('anon') + d('shmem'):+9.1f}  (MiB)",
          flush=True)


def device_allocs(label, count, size):
    base = snap()
    ptrs = []
    for _ in range(count):
        q = ctypes.c_uint64()
        chk(cuda.cuMemAlloc_v2(ctypes.byref(q), ctypes.c_size_t(size)), "cuMemAlloc")
        ptrs.append(q)
    show(f"{label} ({count * size / MIB:.1f} MiB requested)", base, snap())
    for q in ptrs:
        chk(cuda.cuMemFree_v2(q), "cuMemFree")
    show(f"{label} freed", base, snap())


print(f"pid {os.getpid()}  in-container: {os.path.exists('/.dockerenv')}  NVML self before context: {nvml_self()}")
chk(cuda.cuInit(0), "cuInit")
dev = ctypes.c_int()
chk(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
chk(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
try:
    v3 = nvml_self("nvmlDeviceGetComputeRunningProcesses_v3")
    v2 = nvml_self("nvmlDeviceGetComputeRunningProcesses_v2") if hasattr(nv, "nvmlDeviceGetComputeRunningProcesses_v2") else "symbol missing"
    print(f"context only: NVML self via _v3 {v3}  via _v2 {v2}  (bytes; 24-byte struct both)")
    base = snap()
    for i in range(3):
        show(f"noise, idle sample {i}", base, snap())

    # A: one 1 GiB device allocation, untouched then touched.
    base = snap()
    p = ctypes.c_uint64()
    chk(cuda.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(1024 * MIB)), "cuMemAlloc 1 GiB")
    show("A cuMemAlloc 1024 MiB, untouched", base, snap())
    chk(cuda.cuMemsetD8_v2(p, ctypes.c_ubyte(1), ctypes.c_size_t(1024 * MIB)), "memset")
    chk(cuda.cuCtxSynchronize(), "sync")
    show("A touched (memset)", base, snap())
    chk(cuda.cuMemFree_v2(p), "free")
    show("A freed", base, snap())

    # B: driver rounding, three allocation sizes.
    device_allocs("B1 2000 x 100 KiB", 2000, 100 * 1024)
    device_allocs("B2 300 x (1 MiB + 1 B)", 300, MIB + 1)
    device_allocs("B3 200 x (3 MiB + 1 B)", 200, 3 * MIB + 1)
    device_allocs("B4 100 x (2 MiB exactly)", 100, 2 * MIB)

    # C: page-locked host memory, plain (cuMemAllocHost).
    base = snap()
    hp = ctypes.c_void_p()
    chk(cuda.cuMemAllocHost_v2(ctypes.byref(hp), ctypes.c_size_t(256 * MIB)), "cuMemAllocHost")
    ctypes.memset(hp, 1, 256 * MIB)
    show("C cuMemAllocHost 256 MiB, touched", base, snap())
    chk(cuda.cuMemFreeHost(hp), "cuMemFreeHost")
    show("C freed", base, snap())

    # D: page-locked host memory mapped into the device address space, as the
    # RDMA pair region is (cuMemHostAlloc PORTABLE|DEVICEMAP = 0x3).
    base = snap()
    hp = ctypes.c_void_p()
    chk(cuda.cuMemHostAlloc(ctypes.byref(hp), ctypes.c_size_t(256 * MIB), ctypes.c_uint(0x3)), "cuMemHostAlloc")
    ctypes.memset(hp, 1, 256 * MIB)
    show("D cuMemHostAlloc PORTABLE|DEVICEMAP 256 MiB", base, snap())
    chk(cuda.cuMemFreeHost(hp), "cuMemFreeHost")
    show("D freed", base, snap())

    # E: managed memory (cuMemAllocManaged, ATTACH_GLOBAL), touched on device.
    base = snap()
    mp = ctypes.c_uint64()
    chk(cuda.cuMemAllocManaged(ctypes.byref(mp), ctypes.c_size_t(256 * MIB), ctypes.c_uint(1)), "cuMemAllocManaged")
    show("E cuMemAllocManaged 256 MiB, untouched", base, snap())
    chk(cuda.cuMemsetD8_v2(mp, ctypes.c_ubyte(1), ctypes.c_size_t(256 * MIB)), "memset managed")
    chk(cuda.cuCtxSynchronize(), "sync")
    show("E touched on device (memset)", base, snap())
    chk(cuda.cuMemFree_v2(mp), "free managed")
    show("E freed", base, snap())

    # F: plain heap memory, touched.
    base = snap()
    heap = ctypes.create_string_buffer(256 * MIB)
    ctypes.memset(heap, 1, 256 * MIB)
    show("F heap 256 MiB, touched", base, snap())
    del heap
    show("F freed", base, snap())

    # G: scale. 4 GiB in 64 MiB blocks: does the driver figure still track MemAvailable?
    if meminfo("MemAvailable") - 4096 * MIB > 16 * 1024 * MIB:
        device_allocs("G 64 x 64 MiB", 64, 64 * MIB)
    else:
        print("G skipped: would leave under 16 GiB available")
finally:
    cuda.cuCtxDestroy_v2(ctx)
print("OK")
