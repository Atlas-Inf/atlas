// SPDX-License-Identifier: AGPL-3.0-only
//
// cuBLASLt entry points for the HIP target, backed by a DYNAMICALLY-LOADED
// hipBLASLt. spark links `-lcublasLt` unconditionally; on strix-hip this shim
// resolves it. The library is dlopen'd (Linux: libhipblaslt.so[.1]) /
// LoadLibraryA'd (Windows: libhipblaslt.dll, found via the amdhip64 dir on
// PATH) once, thread-safe, so the link line and build.rs stay unchanged and
// the shim degrades to today's behaviour — every entry returns nonzero
// (CUBLAS_STATUS_NOT_INITIALIZED=1) — when hipBLASLt is absent or
// ATLAS_HIPBLASLT=0 is set (the A/B lever).
//
// Why real hipBLASLt matters (measured, AzeezStrix gfx1151 ROCm 10,
// 2026-09-20, ~/dp4a-ab/hipblaslt-probe): BF16 GEMM via this exact mapping
// hits 32-34 TFLOPS at M=2048 (~10x the Atlas pipelined WMMA tile) and
// 220 GB/s at M=1 (0.38 ms on the 16384x2560 projection shape).
//
// Enum translations (cuBLASLt int -> hipBLASLt):
//   compute 68 CUBLAS_COMPUTE_32F   -> HIPBLAS_COMPUTE_32F (2)
//   scaleType 0 CUDA_R_32F          -> HIP_R_32F (0)
//   desc attr 3/4 TRANSA/TRANSB     -> HIPBLASLT_MATMUL_DESC_TRANSA/B (0/1),
//     value 0/1 CUBLAS_OP_N/T       -> HIPBLAS_OP_N/T (111/112)
//   desc attr 17/18 A/B_SCALE_POINTER -> HIPBLASLT_.._A/B_SCALE_POINTER (5/6)
//   desc attr 31/32 *_SCALE_MODE    -> REJECTED (nonzero): FP8 block scaling
//     is unsupported on gfx1151 hipBLASLt; the Rust FP8 paths error and their
//     call sites fall back, exactly as they did against the stub.
//   layout dtype 14/0/28            -> HIP_R_16BF / HIP_R_32F / HIP_R_8F_E4M3
//     (identical values; FP8 matmuls get no heuristic algo -> fallback)
//   pref attr 1                     -> HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
//
// Heuristic result ABI: the Rust side passes a 128-byte blob, uses offset 0
// as the algo pointer, and copies nothing else. hipblasLtMatmulAlgo_t is 24
// bytes (16 data + size_t max_workspace_bytes); we write it at offset 0,
// workspaceSize at 64, state at 72, wavesCount at 76 — the documented
// cublasLtMatmulHeuristicResult_t layout, 96-byte stride per result.
//
// extern "C", matched by name at link time.

#if defined(__has_include)
#if __has_include(<hipblaslt/hipblaslt.h>)
#include <hipblaslt/hipblaslt.h>
#define ATLAS_HAVE_HIPBLASLT_HEADER 1
#endif
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// When the hipBLASLt headers are not installed (e.g. a Windows HIP SDK
// without hipBLASLt), supply the handful of types/enums the shim needs. The
// function-pointer typedefs only name opaque pointer types, and the two
// result structs are size-fixed below — a header mismatch on a future ROCm
// would be caught by the static_asserts at first use, not silently misread.
#ifndef ATLAS_HAVE_HIPBLASLT_HEADER
typedef int hipblasStatus_t;
typedef int hipblasOperation_t;
typedef int hipblasComputeType_t;
typedef int hipDataType;
typedef void* hipblasLtHandle_t;
typedef void* hipblasLtMatmulDesc_t;
typedef void* hipblasLtMatrixLayout_t;
typedef void* hipblasLtMatmulPreference_t;
typedef void* hipblasLtMatmulMatrixScale_t;
typedef void* hipStream_t;
typedef struct {
    uint8_t data[16];
    size_t max_workspace_bytes;
} hipblasLtMatmulAlgo_t;
typedef struct {
    hipblasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    hipblasStatus_t state;
    float wavesCount;
    int reserved[4];
} hipblasLtMatmulHeuristicResult_t;
#define HIPBLAS_OP_N 111
#define HIPBLAS_OP_T 112
#define HIPBLAS_COMPUTE_32F 2
#define HIPBLAS_STATUS_SUCCESS 0
#define HIP_R_32F 0
#define HIP_R_16BF 14
#define HIP_R_8F_E4M3 28
#define HIPBLASLT_MATMUL_DESC_TRANSA 0
#define HIPBLASLT_MATMUL_DESC_TRANSB 1
#define HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER 5
#define HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER 6
#define HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES 1
// Prototypes only (never called directly — always through the resolved
// pointers), needed so `decltype(&hipblasLtX)` compiles without the header.
hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t*);
hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t);
hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t*, hipblasComputeType_t,
                                          hipDataType);
hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t);
hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t, int, const void*,
                                                size_t);
hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t*, hipDataType,
                                            uint64_t, uint64_t, int64_t);
hipblasStatus_t hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t);
hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t*);
hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t, int,
                                                      const void*, size_t);
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(hipblasLtMatmulPreference_t);
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t,
    hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
    hipblasLtMatmulPreference_t, int, hipblasLtMatmulHeuristicResult_t*, int*);
hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void*,
                                const void*, hipblasLtMatrixLayout_t, const void*,
                                hipblasLtMatrixLayout_t, const void*, const void*,
                                hipblasLtMatrixLayout_t, void*, hipblasLtMatrixLayout_t,
                                const hipblasLtMatmulAlgo_t*, void*, size_t, hipStream_t);
#endif

static_assert(sizeof(hipblasLtMatmulAlgo_t) == 24, "hipblasLtMatmulAlgo_t must be 24 B");
static_assert(offsetof(hipblasLtMatmulHeuristicResult_t, workspaceSize) == 24,
              "heuristic workspaceSize at offset 24");
static_assert(sizeof(hipblasLtMatmulHeuristicResult_t) == 56,
              "heuristic result must be 56 B");

namespace {

// Resolved hipBLASLt entry points. `state` transitions exactly once:
// 0 = untried, 1 = ready, 2 = unavailable (stub behaviour).
struct HipblasLt {
    decltype(&hipblasLtCreate) create;
    decltype(&hipblasLtDestroy) destroy;
    decltype(&hipblasLtMatmulDescCreate) desc_create;
    decltype(&hipblasLtMatmulDescDestroy) desc_destroy;
    decltype(&hipblasLtMatmulDescSetAttribute) desc_set_attr;
    decltype(&hipblasLtMatrixLayoutCreate) layout_create;
    decltype(&hipblasLtMatrixLayoutDestroy) layout_destroy;
    decltype(&hipblasLtMatmulPreferenceCreate) pref_create;
    decltype(&hipblasLtMatmulPreferenceSetAttribute) pref_set_attr;
    decltype(&hipblasLtMatmulPreferenceDestroy) pref_destroy;
    decltype(&hipblasLtMatmulAlgoGetHeuristic) get_heuristic;
    decltype(&hipblasLtMatmul) matmul;
};

std::atomic<int> g_state{0};
HipblasLt g_lt{};

void* sym(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void load_hipblaslt() {
    int expected = 0;
    if (!g_state.compare_exchange_strong(expected, -1)) {
        // Another thread is loading (or already decided); spin until done.
        while (g_state.load() == -1) {
        }
        return;
    }
#ifdef _WIN32
    if (GetEnvironmentVariableA("ATLAS_HIPBLASLT", nullptr, 0) > 0) {
        char v[8] = {};
        GetEnvironmentVariableA("ATLAS_HIPBLASLT", v, sizeof(v));
        if (v[0] == '0') { g_state.store(2); return; }
    }
    void* lib = reinterpret_cast<void*>(LoadLibraryA("libhipblaslt.dll"));
    if (!lib) lib = reinterpret_cast<void*>(LoadLibraryA("hipblaslt.dll"));
#else
    const char* off = getenv("ATLAS_HIPBLASLT");
    if (off && off[0] == '0' && off[1] == '\0') { g_state.store(2); return; }
    void* lib = dlopen("libhipblaslt.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libhipblaslt.so", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!lib) { g_state.store(2); return; }
    g_lt.create = reinterpret_cast<decltype(g_lt.create)>(sym(lib, "hipblasLtCreate"));
    g_lt.destroy = reinterpret_cast<decltype(g_lt.destroy)>(sym(lib, "hipblasLtDestroy"));
    g_lt.desc_create =
        reinterpret_cast<decltype(g_lt.desc_create)>(sym(lib, "hipblasLtMatmulDescCreate"));
    g_lt.desc_destroy =
        reinterpret_cast<decltype(g_lt.desc_destroy)>(sym(lib, "hipblasLtMatmulDescDestroy"));
    g_lt.desc_set_attr =
        reinterpret_cast<decltype(g_lt.desc_set_attr)>(sym(lib, "hipblasLtMatmulDescSetAttribute"));
    g_lt.layout_create =
        reinterpret_cast<decltype(g_lt.layout_create)>(sym(lib, "hipblasLtMatrixLayoutCreate"));
    g_lt.layout_destroy =
        reinterpret_cast<decltype(g_lt.layout_destroy)>(sym(lib, "hipblasLtMatrixLayoutDestroy"));
    g_lt.pref_create =
        reinterpret_cast<decltype(g_lt.pref_create)>(sym(lib, "hipblasLtMatmulPreferenceCreate"));
    g_lt.pref_set_attr = reinterpret_cast<decltype(g_lt.pref_set_attr)>(
        sym(lib, "hipblasLtMatmulPreferenceSetAttribute"));
    g_lt.pref_destroy = reinterpret_cast<decltype(g_lt.pref_destroy)>(
        sym(lib, "hipblasLtMatmulPreferenceDestroy"));
    g_lt.get_heuristic =
        reinterpret_cast<decltype(g_lt.get_heuristic)>(sym(lib, "hipblasLtMatmulAlgoGetHeuristic"));
    g_lt.matmul = reinterpret_cast<decltype(g_lt.matmul)>(sym(lib, "hipblasLtMatmul"));
    const bool ok = g_lt.create && g_lt.destroy && g_lt.desc_create && g_lt.desc_destroy &&
                    g_lt.desc_set_attr && g_lt.layout_create && g_lt.layout_destroy &&
                    g_lt.pref_create && g_lt.pref_set_attr && g_lt.pref_destroy &&
                    g_lt.get_heuristic && g_lt.matmul;
    g_state.store(ok ? 1 : 2);
}

bool ready() {
    if (g_state.load() == 0) load_hipblaslt();
    return g_state.load() == 1;
}

// cublasLt status codes we emit: 0 success; 1 NOT_INITIALIZED (the stub's
// answer); 7 NOT_SUPPORTED.
constexpr int STUB = 1;
constexpr int UNSUPPORTED = 7;

}  // namespace

extern "C" {

int cublasLtCreate(hipblasLtHandle_t* handle) {
    if (!ready()) return STUB;
    return static_cast<int>(g_lt.create(handle));
}

int cublasLtDestroy(hipblasLtHandle_t handle) {
    if (!ready()) return STUB;
    return static_cast<int>(g_lt.destroy(handle));
}

int cublasLtMatmulDescCreate(hipblasLtMatmulDesc_t* desc, int compute_type, int scale_type) {
    if (!ready()) return STUB;
    if (compute_type != 68 || scale_type != 0) return UNSUPPORTED;  // 32F only
    return static_cast<int>(
        g_lt.desc_create(desc, HIPBLAS_COMPUTE_32F, static_cast<hipDataType>(HIP_R_32F)));
}

int cublasLtMatmulDescDestroy(hipblasLtMatmulDesc_t desc) {
    if (!ready()) return STUB;
    return static_cast<int>(g_lt.desc_destroy(desc));
}

int cublasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t desc, unsigned attr,
                                   const void* buf, size_t size) {
    if (!ready()) return STUB;
    switch (attr) {
        case 3:  // CUBLASLT_MATMUL_DESC_TRANSA
        case 4:  // CUBLASLT_MATMUL_DESC_TRANSB
        {
            if (size != 4 || !buf) return UNSUPPORTED;
            int v = *static_cast<const int*>(buf);
            hipblasOperation_t op;
            if (v == 0) op = HIPBLAS_OP_N;
            else if (v == 1) op = HIPBLAS_OP_T;
            else return UNSUPPORTED;
            return static_cast<int>(g_lt.desc_set_attr(
                desc,
                attr == 3 ? HIPBLASLT_MATMUL_DESC_TRANSA : HIPBLASLT_MATMUL_DESC_TRANSB,
                &op, sizeof(op)));
        }
        case 17:  // CUBLASLT_MATMUL_DESC_A_SCALE_POINTER
            return static_cast<int>(g_lt.desc_set_attr(
                desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, buf, size));
        case 18:  // CUBLASLT_MATMUL_DESC_B_SCALE_POINTER
            return static_cast<int>(g_lt.desc_set_attr(
                desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, buf, size));
        case 31:  // A_SCALE_MODE — FP8 block scaling: unsupported on gfx1151
        case 32:  // B_SCALE_MODE
            return UNSUPPORTED;
        default:
            return UNSUPPORTED;
    }
}

int cublasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* layout, int dtype,
                               uint64_t rows, uint64_t cols, int64_t ld) {
    if (!ready()) return STUB;
    hipDataType t;
    switch (dtype) {
        case 0:  t = HIP_R_32F; break;
        case 14: t = HIP_R_16BF; break;
        case 28: t = HIP_R_8F_E4M3; break;
        default: return UNSUPPORTED;
    }
    return static_cast<int>(g_lt.layout_create(layout, t, rows, cols, ld));
}

int cublasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t layout) {
    if (!ready()) return STUB;
    return static_cast<int>(g_lt.layout_destroy(layout));
}

int cublasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* pref) {
    if (!ready()) return STUB;
    return static_cast<int>(g_lt.pref_create(pref));
}

int cublasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t pref, unsigned attr,
                                         const void* buf, size_t size) {
    if (!ready()) return STUB;
    if (attr != 1) return UNSUPPORTED;  // CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES only
    return static_cast<int>(
        g_lt.pref_set_attr(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, buf, size));
}

int cublasLtMatmulPreferenceDestroy(hipblasLtMatmulPreference_t pref) {
    if (!ready()) return STUB;
    return static_cast<int>(g_lt.pref_destroy(pref));
}

int cublasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
                                   hipblasLtMatrixLayout_t a, hipblasLtMatrixLayout_t b,
                                   hipblasLtMatrixLayout_t c, hipblasLtMatrixLayout_t d,
                                   hipblasLtMatmulPreference_t pref, int requested,
                                   void* results, int* returned) {
    if (!ready()) return STUB;
    if (requested < 1) { *returned = 0; return 0; }
    std::vector<hipblasLtMatmulHeuristicResult_t> hr(static_cast<size_t>(requested));
    int n = 0;
    int st = static_cast<int>(g_lt.get_heuristic(handle, desc, a, b, c, d, pref,
                                                 requested, hr.data(), &n));
    // Repack into the caller's cuBLASLt-layout results: 96-byte stride —
    // algo (24 B) at offset 0, workspaceSize at 64, state at 72, waves at 76.
    for (int i = 0; i < n; i++) {
        auto* dst = static_cast<uint8_t*>(results) + static_cast<size_t>(i) * 96;
        std::memset(dst, 0, 96);
        std::memcpy(dst, &hr[i].algo, sizeof(hr[i].algo));
        std::memcpy(dst + 64, &hr[i].workspaceSize, sizeof(hr[i].workspaceSize));
        std::memcpy(dst + 72, &hr[i].state, sizeof(hr[i].state));
        std::memcpy(dst + 76, &hr[i].wavesCount, sizeof(hr[i].wavesCount));
    }
    *returned = n;
    return st;
}

int cublasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
                   const void* alpha, const void* a, hipblasLtMatrixLayout_t a_desc,
                   const void* b, hipblasLtMatrixLayout_t b_desc, const void* beta,
                   const void* c, hipblasLtMatrixLayout_t c_desc, void* d,
                   hipblasLtMatrixLayout_t d_desc, const void* algo, void* workspace,
                   size_t workspace_size, void* stream) {
    if (!ready()) return STUB;
    return static_cast<int>(g_lt.matmul(
        handle, desc, alpha, a, a_desc, b, b_desc, beta, c, c_desc, d, d_desc,
        static_cast<const hipblasLtMatmulAlgo_t*>(algo), workspace, workspace_size,
        static_cast<hipStream_t>(stream)));
}

}  // extern "C"
