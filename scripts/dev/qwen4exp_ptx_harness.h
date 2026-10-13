// SPDX-License-Identifier: AGPL-3.0-only
// Tiny driver-API harness shared by the qwen4_exp prefill benches.
//
// The benches load kernels the way the server does: from PTX built with the
// production flags (`nvcc --ptx -arch=sm_121f -O3 --fmad=false`), JIT-compiled
// by the driver. That keeps two kernels that share helper names (every QSA
// attention file defines its own `qsa_*_key_off`, BR, BC, HDIM ...) apart --
// each module is its own translation unit, exactly as in serving -- and it
// means a bitwise PASS here is a statement about the PTX the server runs.
//
// Build a module:
//   nvcc --ptx -arch=sm_121f -O3 --fmad=false -o out/<stem>.ptx <file>.cu
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m_ = nullptr; \
    cuGetErrorString(r_, &m_); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, m_ ? m_ : "?"); exit(1); } } while (0)

struct PtxModule {
    CUmodule mod = nullptr;
    // False when the file is missing or the driver refuses the PTX (a tile
    // sweep point over the shared-memory cap, say).
    bool try_load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        std::string src = ss.str();
        return cuModuleLoadData(&mod, src.c_str()) == CUDA_SUCCESS;
    }
    void load(const std::string& path) {
        if (!try_load(path)) { fprintf(stderr, "cannot load %s\n", path.c_str()); exit(1); }
    }
    CUfunction fn(const char* name, unsigned dyn_smem = 0) const {
        CUfunction f;
        CU(cuModuleGetFunction(&f, mod, name));
        if (dyn_smem > 48 * 1024) {
            CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)dyn_smem));
        }
        return f;
    }
};

// Kernel argument pack: every argument is copied into owned storage so the
// pointer array handed to cuLaunchKernel never dangles.
struct Args {
    std::vector<std::vector<unsigned char>> store;
    std::vector<void*> ptrs;
    template <typename T>
    Args& add(T v) {
        store.emplace_back(sizeof(T));
        memcpy(store.back().data(), &v, sizeof(T));
        return *this;
    }
    void** get() {
        ptrs.clear();
        for (auto& s : store) ptrs.push_back(s.data());
        return ptrs.data();
    }
};

inline void launch(CUfunction f, dim3 g, dim3 b, unsigned smem, Args& a, cudaStream_t s = 0) {
    CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, smem, (CUstream)s, a.get(), nullptr));
}

inline void init_driver() {
    CK(cudaFree(0));  // creates the primary context the runtime and driver share
}

template <typename T>
struct Buf {
    T* p = nullptr;
    size_t n = 0;
    void alloc(size_t count) { n = count; CK(cudaMalloc(&p, n * sizeof(T) + 16)); CK(cudaMemset(p, 0, n * sizeof(T))); }
    void put(const std::vector<T>& h) { CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void fill(unsigned char b) { CK(cudaMemset(p, b, n * sizeof(T))); }
    void free_() { if (p) CK(cudaFree(p)); p = nullptr; }
};

inline unsigned short f2bf(float f) {
    unsigned int u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);   // round to nearest even (no NaN inputs here)
    return (unsigned short)(u >> 16);
}

inline float bf2f(unsigned short h) {
    unsigned int u = (unsigned int)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Median of `reps` timings of `body`, each timing `iters` back-to-back calls.
template <typename F>
inline float time_ms(F body, int iters = 5, int reps = 5) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    body();  // warm
    CK(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) body();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        v.push_back(ms / iters);
    }
    std::sort(v.begin(), v.end());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return v[v.size() / 2];
}

// Count of differing bytes between two host buffers of equal size.
template <typename T>
inline size_t diff_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    const unsigned char* x = (const unsigned char*)a.data();
    const unsigned char* y = (const unsigned char*)b.data();
    size_t n = std::min(a.size(), b.size()) * sizeof(T), d = 0;
    for (size_t i = 0; i < n; ++i) d += x[i] != y[i];
    return d;
}
