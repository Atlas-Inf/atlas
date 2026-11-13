// KLOOP #70 (Ripple DFlash2 drafter): the drafter's gamma=16 projections and LM head run w4a16_gemv_batch16 on the
// native [N, K/2] layout (dflash_head forward_block.rs / batch_projection.rs), 26.9% of the DFlash2 decode GPU time
// (rocprof, slice 3). batch16 runs 4 outputs per block and every block re-reads all 16 activation rows from global
// (K=5120: 160 KB per block, ~700 MB of L2 traffic per 17408-wide call), so it runs at ~50 GB/s (KLOOP #16).
// Candidate lds16<NB>: NB outputs per block, the 16 x 512 activation tile staged once per K step in LDS, 256/NB
// threads per output each streaming 16-byte weight chunks, the same byte-pair LUT + v_dot2_f32_bf16 products,
// group scale applied per 16 like batch16. Not bit-identical (summation order); the drafter only proposes, so its
// numerics can change acceptance but never the verified output. Reports us, weight GB/s, max rel err vs batch16.
#include "w4a16_gemv.cu"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define L_KS 512
template <int NB>
__global__ __launch_bounds__(256) void w4a16_gemv_lds16(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K) {
    constexpr int TPO = 256 / NB;                 // threads per output
    constexpr int CPS = L_KS / 32 / TPO;          // 32-k chunks per thread per K step
    __shared__ unsigned int s_lut2[256];
    __shared__ __align__(16) unsigned short sA[16][L_KS + 8];
    {
        unsigned int v = threadIdx.x;
        unsigned short lob = (unsigned short)(__float_as_uint(E2M1_LUT[v & 0xFu]) >> 16);
        unsigned short hib = (unsigned short)(__float_as_uint(E2M1_LUT[v >> 4]) >> 16);
        s_lut2[v] = ((unsigned int)hib << 16) | lob;
    }
    const unsigned int lane = threadIdx.x % TPO;
    const unsigned int n = blockIdx.x * NB + threadIdx.x / TPO;
    const bool live = n < N;
    const unsigned int half_K = K / 2, num_groups = K / GROUP_SIZE;
    float acc[16];
    #pragma unroll
    for (int t = 0; t < 16; t++) acc[t] = 0.0f;
    const unsigned short* Au = (const unsigned short*)A;
    for (unsigned int kb = 0; kb < K; kb += L_KS) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; i++) {             // 16 x 512 bf16 = 1024 uint4, 4 per thread
            const unsigned int idx = threadIdx.x + i * 256, r = idx / (L_KS / 8), c = (idx % (L_KS / 8)) * 8;
            uint4 v = (r < M) ? *(const uint4*)&Au[(unsigned long long)r * K + kb + c] : uint4{0, 0, 0, 0};
            *(uint4*)&sA[r][c] = v;
        }
        __syncthreads();
        if (!live) continue;
        #pragma unroll
        for (int j = 0; j < CPS; j++) {
            const unsigned int ch = lane + j * TPO;  // chunk within the step (32 k each)
            const unsigned int kg = kb + ch * 32;
            const uint4 pk = *(const uint4*)(B_packed + (unsigned long long)n * half_K + kg / 2);
            const unsigned short sc2 = *(const unsigned short*)(B_scale + (unsigned long long)n * num_groups + kg / 16);
            const float s0 = scl_fp8((unsigned char)sc2) * scale2, s1 = scl_fp8((unsigned char)(sc2 >> 8)) * scale2;
            const unsigned int pw[4] = {pk.x, pk.y, pk.z, pk.w};
            unsigned int w2[16];
            #pragma unroll
            for (int b = 0; b < 16; b++) w2[b] = s_lut2[(pw[b >> 2] >> ((b & 3) * 8)) & 0xFFu];
            #pragma unroll
            for (int t = 0; t < 16; t++) {
                if ((unsigned int)t >= M) continue;
                const uint4* a4 = (const uint4*)&sA[t][ch * 32];
                const uint4 q0 = a4[0], q1 = a4[1], q2 = a4[2], q3 = a4[3];
                const unsigned int ar[16] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w, q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
                float p0 = 0.0f, p1 = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(p0) : "v"(ar[b]), "v"(w2[b]));
                #pragma unroll
                for (int b = 8; b < 16; b++) asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(p1) : "v"(ar[b]), "v"(w2[b]));
                acc[t] = fmaf(s0, p0, acc[t]);
                acc[t] = fmaf(s1, p1, acc[t]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 16; t++)
        #pragma unroll
        for (int o = TPO / 2; o > 0; o >>= 1) acc[t] += __shfl_xor(acc[t], o, TPO);
    if (live && lane == 0)
        #pragma unroll
        for (int t = 0; t < 16; t++) if ((unsigned int)t < M) C[(unsigned long long)t * N + n] = __float2bfloat16(acc[t]);
}

int main() {
    struct S { const char* name; unsigned int M, N, K; } shapes[] = {
        {"q|gate", 16, 20480, 5120}, {"k|v", 16, 2048, 5120}, {"o", 16, 5120, 10240},
        {"gate/up", 16, 40960, 5120}, {"down", 16, 5120, 20480}, {"fc 5x", 16, 5120, 25600},
        {"lm_head", 16, 100352, 5120}, {"gate/up M9", 9, 40960, 5120}};
    for (auto& s : shapes) {
        const unsigned M = s.M, N = s.N, K = s.K;
        const size_t pb = (size_t)K / 2 * N, sb = (size_t)K / 16 * N, wbytes = pb + sb;
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wbytes + 1);
        std::vector<unsigned char> hp(pb), hs(sb);
        srand(7); for (auto& b : hp) b = rand() & 0x77; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
        std::vector<unsigned char*> P(copies), Q(copies);
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&P[c], pb)); CHECK(hipMemcpy(P[c], hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&Q[c], sb)); CHECK(hipMemcpy(Q[c], hs.data(), sb, hipMemcpyHostToDevice)); }
        std::vector<__nv_bfloat16> hA((size_t)M * K); for (auto& a : hA) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *A, *C, *R; CHECK(hipMalloc(&A, hA.size() * 2)); CHECK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C, (size_t)M * N * 2)); CHECK(hipMalloc(&R, (size_t)M * N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        printf("-- %s M=%u N=%u K=%u (%.1f MB)\n", s.name, M, N, K, wbytes / 1e6);
        hipLaunchKernelGGL(w4a16_gemv_batch16, dim3((N + N_PER_BLOCK - 1) / N_PER_BLOCK), dim3(BLOCK_SIZE), 0, 0, A, P[0], Q[0], 0.01f, R, M, N, K);
        CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> hr((size_t)M * N), hc((size_t)M * N);
        CHECK(hipMemcpy(hr.data(), R, hr.size() * 2, hipMemcpyDeviceToHost));
        double rms = 0; for (auto v : hr) rms += (double)__bfloat162float(v) * __bfloat162float(v); rms = sqrt(rms / hr.size()) + 1e-30;
        for (int v = 0; v < 4; v++) {
            const char* nm[] = {"batch16", "lds16 NB64", "lds16 NB32", "lds16 NB16"};
            auto launch = [&](int c, __nv_bfloat16* out) {
                if (v == 0) hipLaunchKernelGGL(w4a16_gemv_batch16, dim3((N + N_PER_BLOCK - 1) / N_PER_BLOCK), dim3(BLOCK_SIZE), 0, 0, A, P[c], Q[c], 0.01f, out, M, N, K);
                else if (v == 1) hipLaunchKernelGGL(w4a16_gemv_lds16<64>, dim3((N + 63) / 64), dim3(256), 0, 0, A, P[c], Q[c], 0.01f, out, M, N, K);
                else if (v == 2) hipLaunchKernelGGL(w4a16_gemv_lds16<32>, dim3((N + 31) / 32), dim3(256), 0, 0, A, P[c], Q[c], 0.01f, out, M, N, K);
                else hipLaunchKernelGGL(w4a16_gemv_lds16<16>, dim3((N + 15) / 16), dim3(256), 0, 0, A, P[c], Q[c], 0.01f, out, M, N, K);
            };
            launch(0, C); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(hc.data(), C, hc.size() * 2, hipMemcpyDeviceToHost));
            double maxe = 0; for (size_t i = 0; i < hc.size(); i++) maxe = std::max(maxe, (double)fabsf(__bfloat162float(hc[i]) - __bfloat162float(hr[i])));
            for (int i = 0; i < 3; i++) launch(i % copies, C);
            const int it = 30; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) launch(i % copies, C); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            printf("   %-11s %8.1f us  %6.1f GB/s  max|d|/rms %.2e\n", nm[v], 1000 * ms / it, wbytes / 1e9 / (ms / it / 1000), maxe / rms);
        }
        for (int c = 0; c < copies; c++) { hipFree(P[c]); hipFree(Q[c]); }
        hipFree(A); hipFree(C); hipFree(R);
    }
    return 0;
}
