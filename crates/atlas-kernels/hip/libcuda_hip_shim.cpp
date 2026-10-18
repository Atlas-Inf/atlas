// SPDX-License-Identifier: AGPL-3.0-only
//
// libcuda → HIP shim. Re-exports the exact 33 CUDA driver-API symbols the
// Atlas spark binary imports (via cudarc) and implements each over HIP/ROCm.
// Built as `libcuda.so` and placed FIRST on the loader path so the unchanged
// cudarc runtime drives AMD GPUs natively — no SCALE.
//
// CUDA driver ABI ↔ HIP type compatibility:
//   CUdeviceptr (u64) ↔ hipDeviceptr_t (void*)   — cast
//   CUstream/CUmodule/CUfunction/CUevent/CUgraph(Exec) ↔ hip* — opaque ptrs
//   CUresult ↔ hipError_t — success==0 matches; error enums differ but cudarc
//   checks success and formats via cuGetErrorString (mapped to hipGetErrorString).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#endif
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

typedef unsigned long long CUdeviceptr;

// Windows device-memory accounting.
//
// hipMemGetInfo is BROKEN on the Windows HIP runtime: it returns
// hipErrorInvalidValue ("invalid argument") standalone, and reports free==0
// under a live context. Atlas sizes its KV cache from cuMemGetInfo, so a bogus
// 0-free makes serve fail with "No memory left for KV cache" even with tens of
// GB genuinely available (measured: 64 GB allocatable via a hipMalloc ladder).
//
// So track what we hand out and synthesise a truthful answer when HIP won't
// give one. Only engages when hipMemGetInfo actually fails or returns zeros,
// so Linux/ROCm behaviour is byte-identical to before.
static std::mutex g_mem_mu;
static std::unordered_map<void *, size_t> g_mem_sizes;
static size_t g_mem_used = 0;

static void atlas_track_alloc(void *p, size_t n) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_mem_mu);
    g_mem_sizes[p] = n;
    g_mem_used += n;
}

static void atlas_track_free(void *p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_mem_mu);
    auto it = g_mem_sizes.find(p);
    if (it == g_mem_sizes.end()) return;
    g_mem_used = (g_mem_used > it->second) ? g_mem_used - it->second : 0;
    g_mem_sizes.erase(it);
}

// Slab sub-allocation for the request sizes hipMalloc rounds up wastefully (#145).
//
// The Linux ROCm runtime commits whole 1 MiB pages for requests above 512 KiB
// and 2 MiB multiples above 1 MiB (measured on gfx1151, ROCm 7.13, Framework
// job f16: 819,200 B -> 1 MiB, 1,310,720 B -> 2 MiB, 3,276,800 B -> 4 MiB).
// Flash-Next's 73,728 routed-expert weights are 819,200 B each, so ~17 GB of
// a 96 GiB carve-out went to rounding and the model could not boot under
// Linux. Such requests are carved from 64 MiB slabs instead, one free list
// per exact (256-aligned) size: the MoE transpose frees each original right
// after allocating its same-sized copy, so freed chunks are reused at once.
// A slab is returned to the driver only once fully free and another slab's
// worth of that size is already free, so an alloc/free cycle cannot thrash;
// at most two empty slabs (128 MiB) per size are kept for reuse.
//
// Accounting stays at the driver level: g_mem_used counts slabs, which is
// what the device actually commits. Default on for Linux; ATLAS_HIP_SUBALLOC=0
// turns it off. Off on Windows unless ATLAS_HIP_SUBALLOC=1 (WDDM's
// granularity differs and the Windows Flash-Next path is validated without it).
static const size_t kSubSlab = 64ull << 20;

static bool atlas_suballoc_on() {
    static const bool on = [] {
        const char* v = getenv("ATLAS_HIP_SUBALLOC");
#ifdef _WIN32
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
#else
        return !(v && (v[0] == '0' || v[0] == 'f' || v[0] == 'F'));
#endif
    }();
    return on;
}

// What the driver commits for an n-byte hipMalloc (Linux measurements above).
static size_t atlas_driver_rounded(size_t n) {
    const size_t mib = 1ull << 20;
    if (n > mib) return (n + 2 * mib - 1) / (2 * mib) * (2 * mib);
    if (n > mib / 2) return mib;
    return (n + 4095) / 4096 * 4096;
}

// Sub-allocate when rounding wastes more than 1/8 of the request, for sizes
// small enough that a 64 MiB slab holds at least eight of them.
static bool atlas_suballoc_eligible(size_t n) {
    if (n <= (256u << 10) || n > (8u << 20)) return false;
    return (atlas_driver_rounded(n) - n) * 8 > n;
}

struct AtlasSubSlab { char* base; size_t sz; unsigned cap; unsigned live; };
struct AtlasSubChunk { void* p; AtlasSubSlab* slab; };
struct AtlasSubClass { std::vector<AtlasSubChunk> free; size_t slabs = 0; };
static std::mutex g_sub_mu;
static std::unordered_map<size_t, AtlasSubClass> g_sub_classes;   // chunk size -> class
static std::unordered_map<void*, AtlasSubSlab*> g_sub_owner;      // live chunk -> slab
static std::map<char*, AtlasSubSlab*> g_sub_slabs;                // slab base -> slab (range lookup)

// The slab whose 64 MiB range holds p, live chunk or not. Caller holds g_sub_mu.
static AtlasSubSlab* atlas_sub_slab_of(void* p) {
    auto it = g_sub_slabs.upper_bound((char*)p);
    if (it == g_sub_slabs.begin()) return nullptr;
    --it;
    return ((char*)p < it->first + kSubSlab) ? it->second : nullptr;
}

// 0 on success; nonzero means "not handled, use plain hipMalloc".
static int atlas_suballoc(void** out, size_t n) {
    const size_t sz = (n + 255) & ~(size_t)255;
    bool first_slab = false;
    unsigned cap = 0;
    {
        std::lock_guard<std::mutex> lk(g_sub_mu);
        AtlasSubClass& c = g_sub_classes[sz];
        if (c.free.empty()) {
            void* base = nullptr;
            if (hipMalloc(&base, kSubSlab) != hipSuccess || !base) return 1;
            atlas_track_alloc(base, kSubSlab);
            auto* s = new AtlasSubSlab{(char*)base, sz, (unsigned)(kSubSlab / sz), 0};
            g_sub_slabs[s->base] = s;
            for (unsigned i = s->cap; i-- > 0;) c.free.push_back({s->base + (size_t)i * sz, s});
            first_slab = c.slabs++ == 0;
            cap = s->cap;
        }
        AtlasSubChunk ch = c.free.back();
        c.free.pop_back();
        ch.slab->live++;
        g_sub_owner[ch.p] = ch.slab;
        *out = ch.p;
    }
    if (first_slab) {
        fprintf(stderr, "[suballoc] %zu B requests -> %u per 64 MiB slab (driver would commit %zu B each)\n",
                n, cap, atlas_driver_rounded(n));
    }
    return 0;
}

// true if p lies in a slab (freed here, or a repeated free of one that is
// already free); false means "not ours, plain hipFree". A slab-interior
// pointer must never reach hipFree.
//
// hipFree (like cuMemFree) synchronizes the device before releasing memory, and
// callers may free a buffer that a kernel still in flight is reading. So the
// chunk leaves g_sub_owner first (a concurrent or repeated free of it now
// finds no owner and is refused), the device is synchronized with no lock
// held, and only then does the chunk go back on its free list for reuse.
static bool atlas_subfree(void* p) {
    AtlasSubSlab* s = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_sub_mu);
        auto it = g_sub_owner.find(p);
        if (it == g_sub_owner.end()) {
            if (!atlas_sub_slab_of(p)) return false;
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr, "[suballoc] free of %p: not a live chunk (repeated free); ignored\n", p);
            }
            return true;
        }
        s = it->second;
        g_sub_owner.erase(it);
    }
    hipError_t se = hipDeviceSynchronize();
    if (se != hipSuccess) {
        fprintf(stderr, "[suballoc] hipDeviceSynchronize before chunk reuse failed: %d\n", (int)se);
    }
    char* release = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_sub_mu);
        AtlasSubClass& c = g_sub_classes[s->sz];
        c.free.push_back({p, s});
        if (--s->live == 0 && c.free.size() >= 2 * (size_t)s->cap) {
            std::vector<AtlasSubChunk> keep;
            keep.reserve(c.free.size() - s->cap);
            for (const AtlasSubChunk& q : c.free) if (q.slab != s) keep.push_back(q);
            c.free.swap(keep);
            c.slabs--;
            g_sub_slabs.erase(s->base);
            atlas_track_free(s->base);
            release = s->base;
            delete s;
        }
    }
    if (release) hipFree(release);
    return true;
}

// ATLAS_TRACE_LAUNCH=1: print every kernel launch (name + grid/block) to
// stderr BEFORE dispatching, so the last line before a 719 names the kernel
// that crashed the context. cuModuleGetFunction maps name→handle; here we keep
// a handle→name table so the opaque launch handle resolves back to a name.
static std::mutex g_fn_mu;
static std::unordered_map<void *, std::string> g_fn_names;
static bool atlas_trace_launch() {
    static const bool on = [] {
        const char *v = getenv("ATLAS_TRACE_LAUNCH");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
    }();
    return on;
}
// ATLAS_TRACE_LAUNCH_SYNC=1 (diagnostic only, requires ATLAS_TRACE_LAUNCH):
// synchronise the stream after every successful launch so an async fault
// (e.g. a sticky 719) is reported against the kernel that caused it instead
// of surfacing at an unrelated later call. Very slow; never enable by default.
static bool atlas_trace_launch_sync() {
    static const bool on = [] {
        const char *v = getenv("ATLAS_TRACE_LAUNCH_SYNC");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
    }();
    return on;
}
// Elapsed-ms wall clock for the launch trace — lets us attribute the TTFT
// setup window to the slow ops (big allocs, first-use module init) rather
// than just counting them.
static double atlas_now_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

extern "C" {

// ── init / context ────────────────────────────────────────────────────
int cuCtxGetCurrent(void** pctx)            { return hipCtxGetCurrent((hipCtx_t*)pctx); }
int cuCtxSetCurrent(void* ctx)              { return hipCtxSetCurrent((hipCtx_t)ctx); }
static void atlas_memprofile_snapshot_once();
int cuCtxCreate_v2(void** pctx, unsigned f, int dev)
                                            { int rc = hipCtxCreate((hipCtx_t*)pctx, f, dev); atlas_memprofile_snapshot_once(); return rc; }
int cuCtxDestroy_v2(void* ctx)              { return hipCtxDestroy((hipCtx_t)ctx); }

// ── errors ────────────────────────────────────────────────────────────
int cuGetErrorName(int err, const char** s)   { *s = hipGetErrorName((hipError_t)err);   return 0; }
int cuGetErrorString(int err, const char** s) { *s = hipGetErrorString((hipError_t)err); return 0; }


// ── [allocfail] diagnostics — diagnostic only, no behaviour change ────────

#ifdef _WIN32
// Win32 commit + DXGI adapter-memory snapshot — shared by [allocfail] and
// [memprofile]. DXGI resolved at runtime; no link change.
static void atlas_mem_win32_diag() {
  MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms); GlobalMemoryStatusEx(&ms);
  PERFORMANCE_INFORMATION pi; pi.cb = sizeof(pi); K32GetPerformanceInfo(&pi, sizeof(pi));
  PROCESS_MEMORY_COUNTERS_EX pmc; pmc.cb = sizeof(pmc);
  K32GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&pmc, sizeof(pmc));
  fprintf(stderr, "[memdiag] TotalPhys=%.1fGiB AvailPhys=%.1fGiB TotalPageFile=%.1fGiB AvailPageFile=%.1fGiB commitTotal=%.1fGiB commitLimit=%.1fGiB commitPeak=%.1fGiB procPrivate=%.1fGiB\n",
    ms.ullTotalPhys/1073741824.0, ms.ullAvailPhys/1073741824.0,
    ms.ullTotalPageFile/1073741824.0, ms.ullAvailPageFile/1073741824.0,
    pi.CommitTotal*pi.PageSize/1073741824.0, pi.CommitLimit*pi.PageSize/1073741824.0,
    pi.CommitPeak*pi.PageSize/1073741824.0, pmc.PrivateUsage/1073741824.0);
  HMODULE dxgi = LoadLibraryA("dxgi.dll");
  if (!dxgi) return;
  typedef HRESULT (WINAPI *CreateFactory1_t)(REFIID, void**);
  CreateFactory1_t cf = (CreateFactory1_t)GetProcAddress(dxgi, "CreateDXGIFactory1");
  if (!cf) return;
  IDXGIFactory1* fac = nullptr;
  if (FAILED(cf(__uuidof(IDXGIFactory1), (void**)&fac)) || !fac) return;
  for (UINT ai = 0; ; ai++) {
    IDXGIAdapter1* ad = nullptr;
    if (FAILED(fac->EnumAdapters1(ai, &ad)) || !ad) break;
    DXGI_ADAPTER_DESC1 dd; memset(&dd, 0, sizeof(dd));
    ad->GetDesc1(&dd);
    if (dd.VendorId == 0x1002) {
      fprintf(stderr, "[memdiag] dxgi: DedicatedVideo=%.1fGiB DedicatedSys=%.1fGiB Shared=%.1fGiB\n",
        dd.DedicatedVideoMemory/1073741824.0, dd.DedicatedSystemMemory/1073741824.0, dd.SharedSystemMemory/1073741824.0);
      IDXGIAdapter3* a3 = nullptr;
      if (SUCCEEDED(ad->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3)) && a3) {
        DXGI_QUERY_VIDEO_MEMORY_INFO vi;
        const struct { DXGI_MEMORY_SEGMENT_GROUP g; const char* n; } grps[2] = {
          {DXGI_MEMORY_SEGMENT_GROUP_LOCAL, "LOCAL"},
          {DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, "NON_LOCAL"}};
        for (auto& g : grps) {
          memset(&vi, 0, sizeof(vi));
          if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, g.g, &vi)))
            fprintf(stderr, "[memdiag] dxgi vmem[%s]: Budget=%.1fGiB Usage=%.1fGiB AvailRes=%.1fGiB CurRes=%.1fGiB\n",
              g.n, vi.Budget/1073741824.0, vi.CurrentUsage/1073741824.0,
              vi.AvailableForReservation/1073741824.0, vi.CurrentReservation/1073741824.0);
        }
        a3->Release();
      }
    }
    ad->Release();
  }
  fac->Release();
}
#endif // _WIN32

static int g_allocfail_count = 0;
static void atlas_alloc_fail_log(const char* fn, size_t n, int r) {
  if (++g_allocfail_count > 8) return;
  size_t used = 0; { std::lock_guard<std::mutex> lk(g_mem_mu); used = g_mem_used; }
  size_t f = 0, t = 0; hipError_t ge = hipMemGetInfo(&f, &t);
  const char* cl = getenv("ATLAS_UMA_COMMIT_LIMIT_GB");
  const char* dr = getenv("ATLAS_UMA_DRIVER_RESERVE_GB");
  fprintf(stderr, "[allocfail #%d] %s n=%zu r=%d g_mem_used=%.2fGiB hipMemGetInfo(e=%d f=%.1fGiB t=%.1fGiB) UMA_COMMIT_LIMIT_GB=%s UMA_DRIVER_RESERVE_GB=%s\n",
          g_allocfail_count, fn, n, r, used/1073741824.0, (int)ge,
          f/1073741824.0, t/1073741824.0,
          cl ? cl : "unset", dr ? dr : "unset");
#ifdef _WIN32
  atlas_mem_win32_diag();
#endif
  fflush(stderr);
}

static void atlas_memprofile_snapshot_once() {
  static bool done = false;
  if (done || !getenv("ATLAS_MEM_PROFILE")) return;
  done = true;
  size_t f = 0, t = 0; hipMemGetInfo(&f, &t);
  fprintf(stderr, "[memprofile] ctx init: hipMemGetInfo f=%.1fGiB t=%.1fGiB\n",
          f/1073741824.0, t/1073741824.0);
#ifdef _WIN32
  atlas_mem_win32_diag();
#endif
  fflush(stderr);
}

// ── memory ────────────────────────────────────────────────────────────
int cuMemAlloc_v2(CUdeviceptr* dptr, size_t n)      {
    double t0 = atlas_now_ms();
    if (dptr && atlas_suballoc_on() && atlas_suballoc_eligible(n) && atlas_suballoc((void**)dptr, n) == 0) {
        if (atlas_trace_launch()) { fprintf(stderr, "[alloc] n=%zu sub dur=%.1fms t=%.1f\n", n, atlas_now_ms()-t0, t0); fflush(stderr); }
        return hipSuccess;
    }
    int r = hipMalloc((void**)dptr, n);
    if (atlas_trace_launch()) { fprintf(stderr, "[alloc] n=%zu dur=%.1fms t=%.1f\n", n, atlas_now_ms()-t0, t0); fflush(stderr); }
    if (r == hipSuccess && dptr) atlas_track_alloc((void*)*dptr, n);
    if (r != hipSuccess) atlas_alloc_fail_log("cuMemAlloc_v2", n, r);
    return r;
}
int cuMemFree_v2(CUdeviceptr dptr)                  {
    if (atlas_subfree((void*)dptr)) return hipSuccess;
    atlas_track_free((void*)dptr);
    return hipFree((void*)dptr);
}
int cuMemAllocHost_v2(void** pp, size_t n)          {
    double t0 = atlas_now_ms();
    int r = hipHostMalloc(pp, n, 0);
    if (atlas_trace_launch()) { fprintf(stderr, "[t=%.0f hostalloc] n=%zu dur=%.1fms\n", t0, n, atlas_now_ms()-t0); fflush(stderr); }
    return r;
}
int cuMemFreeHost(void* p)                          { return hipHostFree(p); }
int cuMemAllocManaged(CUdeviceptr* dptr, size_t n, unsigned flags)
                                                    {
    double t0 = atlas_now_ms();
    int r = hipMallocManaged((void**)dptr, n, flags);
    if (atlas_trace_launch()) { fprintf(stderr, "[t=%.0f allocManaged] n=%zu dur=%.1fms\n", t0, n, atlas_now_ms()-t0); fflush(stderr); }
    if (r == hipSuccess && dptr) atlas_track_alloc((void*)*dptr, n);
    if (r != hipSuccess) atlas_alloc_fail_log("cuMemAllocManaged", n, r);
    return r;
}
// See the g_mem_used comment above: fall back to totalGlobalMem minus tracked
// allocations whenever hipMemGetInfo errors or hands back a zero.
int cuMemGetInfo_v2(size_t* free, size_t* total)    {
    // ATLAS_TRACKED_MEMINFO=1: bypass hipMemGetInfo unconditionally. On the
    // Windows UMA driver it can also return non-zero-but-inflated accounting
    // (~2x real device use — a 20GB checkpoint reads ~44GB "used"), so the
    // tracked-alloc synthesis is the truthful answer, not just a fallback.
    static const bool force_tracked = [] {
        const char* v = getenv("ATLAS_TRACKED_MEMINFO");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
    }();
    // An EXPLICIT ATLAS_UMA_COMMIT_LIMIT_GB is an operator-asserted measured
    // ceiling and is authoritative — including over a successful
    // hipMemGetInfo, which reports only the carve-out aperture: Strix Halo,
    // VGM 32 GB: totalGlobalMem 89.47 GiB while the runtime can back
    // allocations with WDDM shared memory on top of it (hipMalloc +
    // kernel-touch ladder measured 106 GiB, 2026-09-19).
    static const size_t explicit_limit = [] {
        const char* v = getenv("ATLAS_UMA_COMMIT_LIMIT_GB");
        if (!v) return (size_t)0;
        double gb = atof(v);
        return gb > 0.0 ? (size_t)(gb * 1073741824.0) : (size_t)0;
    }();
    size_t f = 0, t = 0;
    hipError_t e = force_tracked ? hipErrorUnknown : hipMemGetInfo(&f, &t);
    if (e == hipSuccess && t != 0 && f != 0) {
        if (explicit_limit) {
            size_t u;
            { std::lock_guard<std::mutex> lk(g_mem_mu); u = g_mem_used; }
            if (free)  *free  = (u < explicit_limit) ? explicit_limit - u : 0;
            if (total) *total = explicit_limit;
        } else {
            if (free)  *free  = f;
            if (total) *total = t;
        }
        return hipSuccess;
    }
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return (int)e;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return (int)e;
    size_t used;
    { std::lock_guard<std::mutex> lk(g_mem_mu); used = g_mem_used; }
    // The UMA driver can RESERVE far more than it can physically COMMIT:
    // totalGlobalMem reports the ~77GB aperture, but writing past the
    // device-addressable window (~63GB on gfx1151) faults the context with
    // hipErrorLaunchFailure. Report the allocatable-commit ceiling as `total`
    // so the KV budget never asks the driver for memory it cannot back. The
    // driver/display holds back a roughly fixed reserve (~14GB here); override
    // with ATLAS_UMA_DRIVER_RESERVE_GB, or cap directly via
    // ATLAS_UMA_COMMIT_LIMIT_GB.
    static const size_t commit_limit = [&] {
        if (explicit_limit) return explicit_limit;
        double reserve_gb = 14.0;
        if (const char* v = getenv("ATLAS_UMA_DRIVER_RESERVE_GB")) {
            double g = atof(v);
            if (g > 0.0) reserve_gb = g;
        }
        const size_t r = (size_t)(reserve_gb * 1073741824.0);
        return (prop.totalGlobalMem > r) ? prop.totalGlobalMem - r
                                       : prop.totalGlobalMem;
    }();
    const size_t tot = explicit_limit
        ? explicit_limit
        : ((prop.totalGlobalMem < commit_limit) ? prop.totalGlobalMem
                                                : commit_limit);
    if (total) *total = tot;
    if (free)  *free  = (used < tot) ? (tot - used) : 0;
    // Diagnostic: report which path answered + the tracked total, so we can
    // tell a real allocation footprint from hipMemGetInfo inflation.
    if (getenv("ATLAS_TRACKED_MEMINFO"))
        fprintf(stderr, "[meminfo] tracked=1 used=%.2fGB total=%.2fGB (cap=%.2fGB)\n",
                used / 1073741824.0, tot / 1073741824.0,
                commit_limit / 1073741824.0);
    return hipSuccess;
}

int cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src, size_t n, void* s) {
    if (atlas_trace_launch()) { fprintf(stderr, "[h2d] dst=%p n=%zu\n", (void*)dst, n); fflush(stderr); }
    return hipMemcpyHtoDAsync((hipDeviceptr_t)dst, (void*)src, n, (hipStream_t)s);
}
int cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src, size_t n, void* s) {
    if (atlas_trace_launch()) { fprintf(stderr, "[d2h] src=%p n=%zu\n", (void*)src, n); fflush(stderr); }
    return hipMemcpyDtoHAsync(dst, (hipDeviceptr_t)src, n, (hipStream_t)s);
}
int cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t n, void* s) {
    if (atlas_trace_launch()) { fprintf(stderr, "[t=%.0f d2d] dst=%p src=%p n=%zu\n", atlas_now_ms(), (void*)dst, (void*)src, n); fflush(stderr); }
    return hipMemcpyDtoDAsync((hipDeviceptr_t)dst, (hipDeviceptr_t)src, n, (hipStream_t)s);
}
int cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t n, void* s) {
    if (atlas_trace_launch()) { fprintf(stderr, "[t=%.0f memset8] dst=%p n=%zu\n", atlas_now_ms(), (void*)dst, n); fflush(stderr); }
    return hipMemsetD8Async((hipDeviceptr_t)dst, uc, n, (hipStream_t)s);
}
int cuMemsetD32Async(CUdeviceptr dst, unsigned int ui, size_t n, void* s) {
    if (atlas_trace_launch()) { fprintf(stderr, "[memset32] dst=%p n=%zu\n", (void*)dst, n); fflush(stderr); }
    return hipMemsetD32Async((hipDeviceptr_t)dst, ui, n, (hipStream_t)s);
}
// Pitched memset — one driver submission zeroes a strided column (e.g. one
// pool slot's slice across every SSM layer), replacing a per-layer
// cuMemsetD8Async loop that dominated WDDM submission overhead.
int cuMemsetD2D8Async(CUdeviceptr dst, size_t pitch, unsigned char uc, size_t w, size_t h, void* s) {
    if (atlas_trace_launch()) { fprintf(stderr, "[t=%.0f memset2d] dst=%p pitch=%zu w=%zu h=%zu\n", atlas_now_ms(), (void*)dst, pitch, w, h); fflush(stderr); }
    return hipMemset2DAsync((hipDeviceptr_t)dst, pitch, uc, w, h, (hipStream_t)s);
}

// ── modules / kernels ─────────────────────────────────────────────────
int cuModuleLoadData(void** m, const void* image)          { return hipModuleLoadData((hipModule_t*)m, image); }
int cuModuleGetFunction(void** f, void* m, const char* nm) {
    int r = hipModuleGetFunction((hipFunction_t*)f, (hipModule_t)m, nm);
    if (r == hipSuccess && f && *f && nm && atlas_trace_launch()) {
        std::lock_guard<std::mutex> lk(g_fn_mu);
        g_fn_names[*f] = nm;
    }
    return r;
}
// Fetch a __constant__/global symbol's device addr+size (registry::device_symbol).
int cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, void* m, const char* nm)
                              { return hipModuleGetGlobal((hipDeviceptr_t*)dptr, bytes, (hipModule_t)m, nm); }
int cuModuleUnload(void* m)                                { return hipModuleUnload((hipModule_t)m); }
// On NVIDIA this opts a module function into >48KB dynamic shared mem. On AMD
// the LDS is sized from hipModuleLaunchKernel's shared-mem arg (up to the
// RDNA 64KB cap), so no opt-in is needed — success no-op.
int cuFuncSetAttribute(void* f, int attr, int val)         { (void)f;(void)attr;(void)val; return 0; }
int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz,
                   unsigned shmem, void* stream, void** params, void** extra) {
  if (atlas_trace_launch()) {
    std::string nm;
    {
      std::lock_guard<std::mutex> lk(g_fn_mu);
      auto it = g_fn_names.find(f);
      nm = (it != g_fn_names.end()) ? it->second : "?";
    }
    // ATLAS_TRACE_LAUNCH_SYNC adds exact GPU time to each launch line:
    // events bracket the launch on the stream, elapsed after the sync.
    // host_before = wall time since the previous launch returned (the
    // caller-side dispatch cost this launch waited behind). Diagnostic
    // only; the events/sync serialize the stream, so timings are for
    // attribution, not steady-state rate.
    const bool sync_mode = atlas_trace_launch_sync();
    static hipEvent_t ev0 = nullptr, ev1 = nullptr;
    static double t_prev_ret = 0.0;
    const double t_entry = atlas_now_ms();
    const double host_before_us = t_prev_ret > 0.0 ? (t_entry - t_prev_ret) * 1000.0 : 0.0;
    if (sync_mode && !ev0) {
      hipEventCreateWithFlags(&ev0, 0); // timing events (default flags)
      hipEventCreateWithFlags(&ev1, 0);
    }
    if (sync_mode) hipEventRecord(ev0, (hipStream_t)stream);
    int r = hipModuleLaunchKernel((hipFunction_t)f, gx, gy, gz, bx, by, bz,
                                  shmem, (hipStream_t)stream, params, extra);
    if (r != hipSuccess) {
      fprintf(stderr, "[launch] %s -> err %d\n", nm.c_str(), (int)r);
      fflush(stderr);
      return r;
    }
    double gpu_us = -1.0;
    if (sync_mode) {
      hipEventRecord(ev1, (hipStream_t)stream);
      int sr = hipStreamSynchronize((hipStream_t)stream);
      if (sr != hipSuccess) {
        fprintf(stderr, "[launch] %s -> ASYNC FAULT err %d (sync after launch)\n",
                nm.c_str(), (int)sr);
        fflush(stderr);
        return sr;
      }
      float ms = 0.f;
      if (hipEventElapsedTime(&ms, ev0, ev1) == hipSuccess) gpu_us = ms * 1000.0;
    }
    t_prev_ret = atlas_now_ms();
    if (sync_mode) {
      fprintf(stderr,
              "[t=%.0f launch] %s grid=%ux%ux%u block=%ux%ux%u shmem=%u gpu=%.1fus host_before=%.1fus\n",
              t_entry, nm.c_str(), gx, gy, gz, bx, by, bz, shmem, gpu_us, host_before_us);
    } else {
      fprintf(stderr, "[t=%.0f launch] %s grid=%ux%ux%u block=%ux%ux%u shmem=%u\n",
              t_entry, nm.c_str(), gx, gy, gz, bx, by, bz, shmem);
    }
    fflush(stderr);
    return r;
  }
  return hipModuleLaunchKernel((hipFunction_t)f, gx, gy, gz, bx, by, bz,
                               shmem, (hipStream_t)stream, params, extra);
}

// CUlaunchConfig (driver API), field for field: grid, block, dynamic shared
// memory, stream, launch attributes. Atlas passes exactly one attribute,
// PROGRAMMATIC_STREAM_SERIALIZATION (PDL, crates/atlas-core/src/registry/
// launch.rs), and the kernels' matching atlas_pdl_enter() compiles to nothing
// on HIP (kernels/gb10/common/atlas_pdl.cuh), so plain stream order is the
// right semantics: drop the attributes and launch normally.
struct AtlasCuLaunchConfig {
  unsigned grid[3];
  unsigned block[3];
  unsigned shared_mem_bytes;
  void* stream;
  void* attrs;
  unsigned num_attrs;
};
// Same layout as the Rust side's #[repr(C)] CuLaunchConfig and CUDA's CUlaunchConfig.
static_assert(offsetof(AtlasCuLaunchConfig, shared_mem_bytes) == 24, "CUlaunchConfig layout");
static_assert(offsetof(AtlasCuLaunchConfig, stream) == 32, "CUlaunchConfig layout");
static_assert(offsetof(AtlasCuLaunchConfig, attrs) == 40, "CUlaunchConfig layout");
static_assert(offsetof(AtlasCuLaunchConfig, num_attrs) == 48, "CUlaunchConfig layout");
// CUlaunchAttribute: a 4-byte id, 4 bytes of padding, a 64-byte value union.
struct AtlasCuLaunchAttribute {
  unsigned id;
  unsigned pad;
  unsigned char value[64];
};
int cuLaunchKernelEx(const AtlasCuLaunchConfig* c, void* f, void** params, void** extra) {
  // Only programmatic stream serialization (id 6) is a no-op here; refuse any
  // other attribute instead of silently changing its semantics.
  const auto* attrs = static_cast<const AtlasCuLaunchAttribute*>(c->attrs);
  for (unsigned i = 0; i < c->num_attrs; ++i) {
    if (attrs[i].id != 6) return 1;  // CUDA_ERROR_INVALID_VALUE
  }
  return cuLaunchKernel(f, c->grid[0], c->grid[1], c->grid[2], c->block[0], c->block[1],
                        c->block[2], c->shared_mem_bytes, c->stream, params, extra);
}

// ── streams ───────────────────────────────────────────────────────────
int cuStreamCreate(void** s, unsigned flags)        { return hipStreamCreateWithFlags((hipStream_t*)s, flags); }
int cuStreamSynchronize(void* s)                    {
    if (atlas_trace_launch()) { fprintf(stderr, "[t=%.0f sync] stream=%p\n", atlas_now_ms(), s); fflush(stderr); }
    int r = hipStreamSynchronize((hipStream_t)s);
    if (atlas_trace_launch() && r != hipSuccess) { fprintf(stderr, "[sync] stream=%p -> err %d\n", s, (int)r); fflush(stderr); }
    return r;
}
// Re-added on the merge to main: main introduced a cuStreamQuery call in
// spark-runtime (copy_d2h_on_stream) after this shim was written.
int cuStreamQuery(void* s)                          { return hipStreamQuery((hipStream_t)s); }
int cuStreamWaitEvent(void* s, void* e, unsigned f) { return hipStreamWaitEvent((hipStream_t)s, (hipEvent_t)e, f); }
int cuStreamBeginCapture(void* s, int mode)         {
    if (atlas_trace_launch()) { fprintf(stderr, "[graph] begin_capture stream=%p mode=%d\n", s, mode); fflush(stderr); }
    int r = hipStreamBeginCapture((hipStream_t)s, (hipStreamCaptureMode)mode);
    if (atlas_trace_launch()) { fprintf(stderr, "[graph] begin_capture -> %d\n", r); fflush(stderr); }
    return r;
}
int cuStreamEndCapture(void* s, void** pgraph)      {
    int r = hipStreamEndCapture((hipStream_t)s, (hipGraph_t*)pgraph);
    if (atlas_trace_launch()) { fprintf(stderr, "[graph] end_capture -> %d graph=%p\n", r, pgraph?*pgraph:nullptr); fflush(stderr); }
    return r;
}
// cudarc's cuStreamIsCapturing(stream, *status) — hip twin writes the same
// hipStreamCaptureStatus enum (NONE=0 .. GLOBAL/THREAD_LOCAL/RELAXED).
int cuStreamIsCapturing(void* s, unsigned* status)  { return hipStreamIsCapturing((hipStream_t)s, (hipStreamCaptureStatus*)status); }

// ── events ────────────────────────────────────────────────────────────
int cuEventCreate(void** e, unsigned flags) { return hipEventCreateWithFlags((hipEvent_t*)e, flags); }
int cuEventDestroy_v2(void* e)              { return hipEventDestroy((hipEvent_t)e); }
int cuEventRecord(void* e, void* s)         { return hipEventRecord((hipEvent_t)e, (hipStream_t)s); }
int cuEventSynchronize(void* e)             { return hipEventSynchronize((hipEvent_t)e); }
int cuEventElapsedTime(float* ms, void* a, void* b) { return hipEventElapsedTime(ms, (hipEvent_t)a, (hipEvent_t)b); }

// ── CUDA graphs ───────────────────────────────────────────────────────
// cudarc's cuGraphInstantiate (legacy arity): (exec*, graph, errNode*, logBuf, bufSize)
int cuGraphInstantiate(void** pexec, void* graph, void** errNode, char* logBuf, size_t bufSize) {
  (void)errNode; (void)logBuf; (void)bufSize;
  int r = hipGraphInstantiate((hipGraphExec_t*)pexec, (hipGraph_t)graph, nullptr, nullptr, 0);
  if (atlas_trace_launch()) { fprintf(stderr, "[graph] instantiate graph=%p -> %d exec=%p\n", graph, r, pexec?*pexec:nullptr); fflush(stderr); }
  return r;
}
int cuGraphLaunch(void* exec, void* s)  {
    if (atlas_trace_launch()) { fprintf(stderr, "[t=%.0f graph] LAUNCH exec=%p stream=%p\n", atlas_now_ms(), exec, s); fflush(stderr); }
    return hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)s);
}
int cuGraphExecDestroy(void* exec)      { return hipGraphExecDestroy((hipGraphExec_t)exec); }
int cuGraphDestroy(void* graph)         { return hipGraphDestroy((hipGraph_t)graph); }

int cuGraphInstantiateWithFlags(void** pexec, void* graph, unsigned long long flags){ return hipGraphInstantiateWithFlags((hipGraphExec_t*)pexec,(hipGraph_t)graph,flags); }

int cuMemcpyHtoD_v2(unsigned long long d,const void*s,size_t n){
  double t0=atlas_now_ms();
  int r=hipMemcpyHtoD((hipDeviceptr_t)d,(void*)s,n);
  if(atlas_trace_launch()){fprintf(stderr,"[t=%.0f h2dSYNC] n=%zu dur=%.1fms\n",t0,n,atlas_now_ms()-t0);fflush(stderr);}
  return r;}
int cuMemcpyDtoH_v2(void*d,unsigned long long s,size_t n){
  double t0=atlas_now_ms();
  int r=hipMemcpyDtoH(d,(hipDeviceptr_t)s,n);
  if(atlas_trace_launch()){fprintf(stderr,"[t=%.0f d2hSYNC] n=%zu dur=%.1fms\n",t0,n,atlas_now_ms()-t0);fflush(stderr);}
  return r;}
int cuMemcpyDtoD_v2(unsigned long long d,unsigned long long s,size_t n){
  double t0=atlas_now_ms();
  int r=hipMemcpyDtoD((hipDeviceptr_t)d,(hipDeviceptr_t)s,n);
  if(atlas_trace_launch()){fprintf(stderr,"[t=%.0f d2dSYNC] n=%zu dur=%.1fms\n",t0,n,atlas_now_ms()-t0);fflush(stderr);}
  return r;}
int cuMemsetD8_v2(unsigned long long d,unsigned char v,size_t n){
  double t0=atlas_now_ms();
  int r=hipMemsetD8((hipDeviceptr_t)d,v,n);
  if(atlas_trace_launch()){fprintf(stderr,"[t=%.0f memset8SYNC] n=%zu dur=%.1fms\n",t0,n,atlas_now_ms()-t0);fflush(stderr);}
  return r;}
int cuMemsetD32_v2(unsigned long long d,unsigned int v,size_t n){
  double t0=atlas_now_ms();
  int r=hipMemsetD32((hipDeviceptr_t)d,v,n);
  if(atlas_trace_launch()){fprintf(stderr,"[t=%.0f memset32SYNC] n=%zu dur=%.1fms\n",t0,n,atlas_now_ms()-t0);fflush(stderr);}
  return r;}
int cuMemHostAlloc(void**p,size_t n,unsigned int f){return hipHostMalloc(p,n,f);}
int cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int f)
                                            { return hipHostGetDevicePointer((void**)pdptr, p, f); }

// --- CudaContext::new path (cuDeviceGetAttribute maps CUDA enum NUMBERS to sane values, bypassing HIP enum mismatch) ---
int cuInit(unsigned int f){return hipInit(f);}
int cuDriverGetVersion(int*v){return hipDriverGetVersion(v);}
int cuDeviceGet(int*d,int o){return hipDeviceGet(d,o);}
int cuDeviceGetCount(int*c){return hipGetDeviceCount(c);}
int cuDeviceTotalMem_v2(size_t*b,int d){return hipDeviceTotalMem(b,d);}
int cuDevicePrimaryCtxRetain(void**c,int d){int rc=hipDevicePrimaryCtxRetain((hipCtx_t*)c,d); atlas_memprofile_snapshot_once(); return rc;}
int cuDevicePrimaryCtxRelease_v2(int d){return hipDevicePrimaryCtxRelease(d);}
int cuCtxSynchronize(void){return hipDeviceSynchronize();}
int cuCtxGetDevice(int*d){return hipGetDevice(d);}
int cuDeviceGetName(char*n,int len,int d){ if(len>0){const char*s="AMD-gfx1151"; int i=0; for(;i<len-1 && s[i];i++) n[i]=s[i]; n[i]=0;} return 0;}
int cuDeviceGetAttribute(int*v,int attr,int dev){ (void)dev; switch(attr){ case 75:*v=12;break; case 76:*v=1;break; case 16:*v=40;break; case 1:*v=1024;break; case 10:*v=32;break; case 8:*v=65536;break; case 18:*v=1;break; case 19:*v=1;break; case 41:*v=1;break; case 36:*v=1500;break; default:*v=0;break;} return 0;}

int cuStreamDestroy_v2(void*s){return hipStreamDestroy((hipStream_t)s);}
int cuStreamDestroy(void*s){return hipStreamDestroy((hipStream_t)s);}

} // extern "C"
