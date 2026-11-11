// SPDX-License-Identifier: AGPL-3.0-only
// Standalone: cost of one rolling finish-leaf save (ATLAS_GLM_PC_FINISH_LEAF)
// on GB10. One rank's GLM-5.3 SSM state is 34 KDA layers x (32 heads x 128 x
// 128 FP32 h + conv). Timed three ways: the 68 D2D copies
// `run_ssm_state_copies` issues when the destination is not strided, two
// pitched 2-D copies, and a plain copy kernel.
//
//   nvcc -O2 -arch=sm_121a -o leafcopy leafcopy.cu && ./leafcopy
//
// Allocates about 1.6 GiB of device memory and runs for a few seconds.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
// One launch copying L rows of `words` uint4 each between two pitched regions.
__global__ void copy_rows(const uint4* __restrict__ src, unsigned long long src_pitch,
                          uint4* __restrict__ dst, unsigned long long dst_pitch,
                          unsigned long long words) {
    const uint4* s = src + blockIdx.y * src_pitch;
    uint4* d = dst + blockIdx.y * dst_pitch;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += (unsigned long long)gridDim.x * blockDim.x)
        d[i] = s[i];
}
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main() {
    const int L = 34, SLOTS = 5, SNAPS = 16, REPS = 60;
    const size_t H = 32ull * 128 * 128 * 4, C = 3ull * 32 * 128 * 4 * 4;
    char *hs, *cs, *hd, *cd;
    CK(cudaMalloc(&hs, L * SLOTS * H)); CK(cudaMalloc(&cs, L * SLOTS * C));
    CK(cudaMalloc(&hd, L * SNAPS * H)); CK(cudaMalloc(&cd, L * SNAPS * C));
    CK(cudaMemset(hs, 1, L * SLOTS * H)); CK(cudaMemset(cs, 2, L * SLOTS * C));
    CK(cudaMemset(hd, 0, L * SNAPS * H)); CK(cudaMemset(cd, 0, L * SNAPS * C));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    std::vector<double> loop_enq, loop_tot, d2_enq, d2_tot, k_tot;
    for (int r = 0; r < REPS + 5; ++r) {
        int slot = r % SLOTS, snap = r % SNAPS;
        CK(cudaStreamSynchronize(st));
        double t0 = now_ms();
        for (int l = 0; l < L; ++l)
            CK(cudaMemcpyAsync(hd + (l * SNAPS + snap) * H, hs + (l * SLOTS + slot) * H, H, cudaMemcpyDeviceToDevice, st));
        for (int l = 0; l < L; ++l)
            CK(cudaMemcpyAsync(cd + (l * SNAPS + snap) * C, cs + (l * SLOTS + slot) * C, C, cudaMemcpyDeviceToDevice, st));
        double t1 = now_ms();
        CK(cudaStreamSynchronize(st));
        double t2 = now_ms();
        CK(cudaMemcpy2DAsync(hd + snap * H, SNAPS * H, hs + slot * H, SLOTS * H, H, L, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpy2DAsync(cd + snap * C, SNAPS * C, cs + slot * C, SLOTS * C, C, L, cudaMemcpyDeviceToDevice, st));
        double t3 = now_ms();
        CK(cudaStreamSynchronize(st));
        double t4 = now_ms();
        copy_rows<<<dim3(64, L), 256, 0, st>>>((const uint4*)(hs + slot * H), SLOTS * H / 16, (uint4*)(hd + snap * H), SNAPS * H / 16, H / 16);
        copy_rows<<<dim3(8, L), 256, 0, st>>>((const uint4*)(cs + slot * C), SLOTS * C / 16, (uint4*)(cd + snap * C), SNAPS * C / 16, C / 16);
        CK(cudaStreamSynchronize(st));
        double t5 = now_ms();
        if (r >= 5) k_tot.push_back(t5 - t4);
        if (r >= 5) { loop_enq.push_back(t1 - t0); loop_tot.push_back(t2 - t0); d2_enq.push_back(t3 - t2); d2_tot.push_back(t4 - t2); }
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto mn = [](std::vector<double> v) { return *std::min_element(v.begin(), v.end()); };
    double mb = L * (H + C) / 1048576.0;
    printf("state %.1f MiB (h %.1f + conv %.1f)\n", mb, L * H / 1048576.0, L * C / 1048576.0);
    printf("68 x memcpy : enqueue med %.3f ms, total med %.3f min %.3f ms (%.1f GiB/s at min)\n",
           med(loop_enq), med(loop_tot), mn(loop_tot), mb / 1024.0 / (mn(loop_tot) / 1000.0));
    printf("2 x memcpy2D: enqueue med %.3f ms, total med %.3f min %.3f ms (%.1f GiB/s at min)\n",
           med(d2_enq), med(d2_tot), mn(d2_tot), mb / 1024.0 / (mn(d2_tot) / 1000.0));
    printf("2 x kernel  : total med %.3f min %.3f ms (%.1f GiB/s at min)\n", med(k_tot), mn(k_tot), mb / 1024.0 / (mn(k_tot) / 1000.0));
    return 0;
}
