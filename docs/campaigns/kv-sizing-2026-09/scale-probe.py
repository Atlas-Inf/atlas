#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Does the driver's per-process figure track MemAvailable at GiB scale? (GB10)

Allocates SIZE_GIB (default 8) of device memory in CHUNK_MIB blocks, twice,
with settle time, and prints which /proc/meminfo fields moved. Control samples
before and between cycles show how much the host drifts on its own.
Refuses unless 16 GiB would stay available; frees everything on exit.
"""
import ctypes, os, sys, time

MIB = 1 << 20
SIZE = int(float(os.environ.get("SIZE_GIB", "8")) * 1024) * MIB
CHUNK = int(os.environ.get("CHUNK_MIB", "64")) * MIB
cuda = ctypes.CDLL("libcuda.so.1")
nv = ctypes.CDLL("libnvidia-ml.so.1")
FIELDS = ("MemAvailable", "MemFree", "Cached", "Shmem", "AnonPages", "Slab", "SUnreclaim",
          "SReclaimable", "PageTables", "KernelStack", "Unevictable", "Mlocked")


class Info(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong),
                ("gi", ctypes.c_uint), ("ci", ctypes.c_uint)]


def chk(status, what):
    if status != 0:
        raise SystemExit(f"{what} failed: status {status}")


def nvml_self():
    chk(nv.nvmlInit_v2(), "nvmlInit")
    try:
        h = ctypes.c_void_p()
        chk(nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)), "handle")
        cnt = ctypes.c_uint(256)
        buf = (Info * 256)()
        chk(nv.nvmlDeviceGetComputeRunningProcesses_v3(h, ctypes.byref(cnt), buf), "procs")
        return sum(buf[j].used for j in range(cnt.value) if buf[j].pid == os.getpid())
    finally:
        nv.nvmlShutdown()


def meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in FIELDS:
            out[k] = int(v.split()[0]) * 1024
    return out


def snap():
    m = meminfo()
    m["driver"] = nvml_self()
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    chk(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    m["cuFree"] = free.value
    return m


def show(label, base, now):
    d = lambda k: (now[k] - base[k]) / MIB
    parts = [f"driver {d('driver'):+8.1f}", f"MemAvailable {d('MemAvailable'):+8.1f}", f"cuFree {d('cuFree'):+8.1f}"]
    parts += [f"{k} {d(k):+.1f}" for k in FIELDS[1:] if abs(d(k)) >= 8]
    print(f"{label:34s} " + "  ".join(parts), flush=True)


chk(cuda.cuInit(0), "cuInit")
dev = ctypes.c_int()
chk(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
chk(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
try:
    if meminfo()["MemAvailable"] - SIZE < 16 * 1024 * MIB:
        raise SystemExit("refused: would leave under 16 GiB available")
    print(f"pid {os.getpid()}: {SIZE / MIB:.0f} MiB in {CHUNK / MIB:.0f} MiB blocks; deltas in MiB since start "
          f"(negative MemAvailable = memory consumed); fields under 8 MiB omitted")
    base = snap()
    for i in range(3):
        time.sleep(2)
        show(f"control {i} (idle 2 s)", base, snap())
    for cycle in (1, 2):
        ptrs = []
        for _ in range(SIZE // CHUNK):
            q = ctypes.c_uint64()
            chk(cuda.cuMemAlloc_v2(ctypes.byref(q), ctypes.c_size_t(CHUNK)), "cuMemAlloc")
            ptrs.append(q)
        show(f"cycle {cycle}: allocated", base, snap())
        time.sleep(2)
        show(f"cycle {cycle}: allocated + 2 s", base, snap())
        for q in ptrs:
            chk(cuda.cuMemFree_v2(q), "cuMemFree")
        show(f"cycle {cycle}: freed", base, snap())
        for wait in (2, 4):
            time.sleep(wait)
            show(f"cycle {cycle}: freed + {wait} s more", base, snap())
finally:
    cuda.cuCtxDestroy_v2(ctx)
time.sleep(2)
m = meminfo()
print(f"context destroyed + 2 s            MemAvailable {(m['MemAvailable'] - base['MemAvailable']) / MIB:+8.1f}")
