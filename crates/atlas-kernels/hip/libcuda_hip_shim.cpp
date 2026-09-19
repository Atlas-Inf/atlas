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
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

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
int cuCtxCreate_v2(void** pctx, unsigned f, int dev)
                                            { return hipCtxCreate((hipCtx_t*)pctx, f, dev); }
int cuCtxDestroy_v2(void* ctx)              { return hipCtxDestroy((hipCtx_t)ctx); }

// ── errors ────────────────────────────────────────────────────────────
int cuGetErrorName(int err, const char** s)   { *s = hipGetErrorName((hipError_t)err);   return 0; }
int cuGetErrorString(int err, const char** s) { *s = hipGetErrorString((hipError_t)err); return 0; }

// ── memory ────────────────────────────────────────────────────────────
int cuMemAlloc_v2(CUdeviceptr* dptr, size_t n)      {
    double t0 = atlas_now_ms();
    int r = hipMalloc((void**)dptr, n);
    if (atlas_trace_launch()) { fprintf(stderr, "[alloc] n=%zu dur=%.1fms t=%.1f\n", n, atlas_now_ms()-t0, t0); fflush(stderr); }
    if (r == hipSuccess && dptr) atlas_track_alloc((void*)*dptr, n);
    return r;
}
int cuMemFree_v2(CUdeviceptr dptr)                  {
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
    fprintf(stderr, "[t=%.0f launch] %s grid=%ux%ux%u block=%ux%ux%u shmem=%u\n",
            atlas_now_ms(), nm.c_str(), gx, gy, gz, bx, by, bz, shmem);
    fflush(stderr);
    int r = hipModuleLaunchKernel((hipFunction_t)f, gx, gy, gz, bx, by, bz,
                                  shmem, (hipStream_t)stream, params, extra);
    if (r != hipSuccess) {
      fprintf(stderr, "[launch] %s -> err %d\n", nm.c_str(), (int)r);
      fflush(stderr);
    }
    return r;
  }
  return hipModuleLaunchKernel((hipFunction_t)f, gx, gy, gz, bx, by, bz,
                               shmem, (hipStream_t)stream, params, extra);
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
int cuDevicePrimaryCtxRetain(void**c,int d){return hipDevicePrimaryCtxRetain((hipCtx_t*)c,d);}
int cuDevicePrimaryCtxRelease_v2(int d){return hipDevicePrimaryCtxRelease(d);}
int cuCtxSynchronize(void){return hipDeviceSynchronize();}
int cuCtxGetDevice(int*d){return hipGetDevice(d);}
int cuDeviceGetName(char*n,int len,int d){ if(len>0){const char*s="AMD-gfx1151"; int i=0; for(;i<len-1 && s[i];i++) n[i]=s[i]; n[i]=0;} return 0;}
int cuDeviceGetAttribute(int*v,int attr,int dev){ (void)dev; switch(attr){ case 75:*v=12;break; case 76:*v=1;break; case 16:*v=40;break; case 1:*v=1024;break; case 10:*v=32;break; case 8:*v=65536;break; case 18:*v=1;break; case 19:*v=1;break; case 41:*v=1;break; case 36:*v=1500;break; default:*v=0;break;} return 0;}

int cuStreamDestroy_v2(void*s){return hipStreamDestroy((hipStream_t)s);}
int cuStreamDestroy(void*s){return hipStreamDestroy((hipStream_t)s);}

} // extern "C"
