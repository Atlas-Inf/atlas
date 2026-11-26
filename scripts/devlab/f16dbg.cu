// devlab-only: differential test of ink_hip_gemm_tile_f16 vs a host reference.
// Runs the real tile function (copied verbatim below from inkling2_moe_gemm.cu)
// on deterministic inputs and prints max diffs; also an f16 variant whose A
// conversion uses scalar __float2half instead of v_cvt_pkrtz to isolate the
// packed-convert operand order.
#include <hip/hip_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

typedef __bf16 ink_v16bf __attribute__((ext_vector_type(16)));
typedef float ink_v8f __attribute__((ext_vector_type(8)));
typedef _Float16 ink_v16f __attribute__((ext_vector_type(16)));

#define INK_GN 128
#define INK_GK 32
#define INK_PITCH (INK_GK + 8)

__device__ __constant__ float INK_HIP_E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
static const float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float ink_hip_e4m3(unsigned int b) {
    const unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0u) v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
static float host_e4m3(unsigned int b) {
    const unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 0u) v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else { unsigned int u = ((e + 120u) << 23) | (m << 20); v = *(float*)&u; }
    return s ? -v : v;
}

// ---- copied tile (SW=false shape only, NONPKRTZ selects the A conversion) ----
template <bool SW, bool NONPKRTZ>
__device__ __forceinline__ void ink_hip_gemm_tile_f16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ wc,
    const unsigned char* __restrict__ ws, const float g, __nv_bfloat16* __restrict__ C,
    const int m_start, const int m_count, const int m0, const int* __restrict__ rows,
    const unsigned int N, const unsigned int K, const float cg, const float cu
) {
    __shared__ __align__(16) _Float16 As[128][INK_PITCH];
    __shared__ int src[128];
    const unsigned int n0 = blockIdx.x * INK_GN;
    const unsigned int tid = threadIdx.x, lane = tid % 32;
    for (int r = tid; r < 128; r += 256) src[r] = m0 + r < m_count ? (rows ? rows[m_start + m0 + r] : m_start + m0 + r) : -1;
    __syncthreads();

    const unsigned int bn = n0 + tid / 32 * 16 + (lane & 15);
    const bool b_ok = bn < N;
    const unsigned char* wc_row = wc + (unsigned long long)bn * (K / 2);
    const unsigned char* ws_row = ws + (unsigned long long)bn * (K / 16);
    constexpr int A_ITERS = 2;
    const __nv_bfloat16* a_ptr[A_ITERS];
    int a_off[A_ITERS];
    bool a_val[A_ITERS];
#pragma unroll
    for (int it = 0; it < A_ITERS; it++) {
        const int idx = it * 256 + (int)tid, r = idx / 4, col = (idx % 4) * 8;
        const int sr = src[r];
        a_val[it] = sr >= 0;
        a_ptr[it] = A + (unsigned long long)sr * K + col;
        a_off[it] = r * INK_PITCH + col;
    }
    ink_v8f acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = ink_v8f{0, 0, 0, 0, 0, 0, 0, 0};

    uint4 reg_a[A_ITERS];
    uint4 reg_c;
    unsigned int reg_s = 0;
    auto prefetch = [&](const unsigned int k0) {
#pragma unroll
        for (int it = 0; it < A_ITERS; it++)
            reg_a[it] = a_val[it] ? *(const uint4*)(a_ptr[it] + k0) : make_uint4(0, 0, 0, 0);
        if (b_ok) {
            reg_c = *(const uint4*)(wc_row + k0 / 2);
            reg_s = *(const unsigned short*)(ws_row + k0 / 16);
        } else {
            reg_c = make_uint4(0, 0, 0, 0);
            reg_s = 0;
        }
    };
    prefetch(0);
    for (unsigned int k0 = 0; k0 < K; k0 += INK_GK) {
#pragma unroll
        for (int it = 0; it < A_ITERS; it++) {
            const unsigned int u[4] = {reg_a[it].x, reg_a[it].y, reg_a[it].z, reg_a[it].w};
            uint4 out;
            unsigned int* o = &out.x;
            if (NONPKRTZ) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const _Float16 h0 = (_Float16)__bfloat162float(*(const __nv_bfloat16*)&((const unsigned short*)&u[i])[0]);
                    const _Float16 h1 = (_Float16)__bfloat162float(*(const __nv_bfloat16*)&((const unsigned short*)&u[i])[1]);
                    o[i] = *(const unsigned short*)&h0 | ((unsigned int)*(const unsigned short*)&h1 << 16);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float f0 = __bfloat162float(*(const __nv_bfloat16*)&((const unsigned short*)&u[i])[0]);
                    const float f1 = __bfloat162float(*(const __nv_bfloat16*)&((const unsigned short*)&u[i])[1]);
                    asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(o[i]) : "v"(f0), "v"(f1));
                }
            }
            *(uint4*)((_Float16*)As + a_off[it]) = out;
        }
        __syncthreads();
        const uint4 cur_c = reg_c;
        const unsigned int cur_s = reg_s;
        if (k0 + INK_GK < K) prefetch(k0 + INK_GK);
        ink_v16f b_frag[2];
        {
            const unsigned int w[4] = {cur_c.x, cur_c.y, cur_c.z, cur_c.w};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const _Float16 s16h = (_Float16)(ink_hip_e4m3((cur_s >> (8 * h)) & 0xFF) * g);
                const unsigned int s16 = *(const unsigned short*)&s16h;
                const unsigned int scale_pk = s16 | (s16 << 16);
                unsigned int pk[8];
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const unsigned int byte = (w[h * 2 + i / 4] >> (8 * (i % 4))) & 0xFF;
                    const unsigned int sel = ((byte & 7u) << 8) | (((byte >> 4) & 7u) << 24);
                    unsigned int mags;
                    asm("v_perm_b32 %0, %1, %2, %3"
                        : "=v"(mags) : "v"(0x3E3C3800u), "v"(0x46444240u), "v"(sel));
                    const unsigned int sgn = mags | ((byte & 8u) << 12) | ((byte & 0x80u) << 24);
                    asm("v_pk_mul_f16 %0, %1, %2" : "=v"(pk[i]) : "v"(sgn), "v"(scale_pk));
                }
                b_frag[h] = *(ink_v16f*)pk;
            }
        }
        {
            for (int h = 0; h < 2; h++) {
#pragma unroll
                for (int im = 0; im < 8; im++) {
                    if (m0 + im * 16 < m_count) {
                        const ink_v16f a = *(const ink_v16f*)&As[im * 16 + (lane & 15)][h * 16];
                        acc[im] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b_frag[h], acc[im]);
                    }
                }
            }
        }
        __syncthreads();
    }
    const unsigned int c = bn;
    if (c < N) {
#pragma unroll
        for (int im = 0; im < 8; im++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int r = m0 + im * 16 + 2 * e + (int)(lane >> 4);
                if (r < m_count) C[(unsigned long long)(m_start + r) * N + c] = __float2bfloat16(acc[im][e]);
            }
        }
    }
}

// probe 1: dump the dequantized B fragment for one warp column each lane:
// out[lane * 32 + i] = dequantized value for column bn(lane), k = i (16 codes x2 halves).
extern "C" __global__ void dump_b(
    const unsigned char* wc, const unsigned char* ws, float g, float* out, unsigned int K
) {
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int bn = threadIdx.x / 32 * 16 + (lane & 15);
    const unsigned char* wc_row = wc + (unsigned long long)bn * (K / 2);
    const unsigned char* ws_row = ws + (unsigned long long)bn * (K / 16);
    const uint4 reg_c = *(const uint4*)(wc_row);
    const unsigned int reg_s = *(const unsigned short*)(ws_row);
    const unsigned int w[4] = {reg_c.x, reg_c.y, reg_c.z, reg_c.w};
    float* o = out + threadIdx.x * 32;
    for (int h = 0; h < 2; h++) {
        const _Float16 s16h = (_Float16)(ink_hip_e4m3((reg_s >> (8 * h)) & 0xFF) * g);
        const unsigned int s16 = *(const unsigned short*)&s16h;
        const unsigned int scale_pk = s16 | (s16 << 16);
        unsigned int pk[8];
        for (int i = 0; i < 8; i++) {
            const unsigned int byte = (w[h * 2 + i / 4] >> (8 * (i % 4))) & 0xFF;
            const unsigned int sel = ((byte & 7u) << 8) | (((byte >> 4) & 7u) << 24);
            unsigned int mags;
            asm("v_perm_b32 %0, %1, %2, %3" : "=v"(mags) : "v"(0x3E3C3800u), "v"(0x46444240u), "v"(sel));
            const unsigned int sgn = mags | ((byte & 8u) << 12) | ((byte & 0x80u) << 24);
            asm("v_pk_mul_f16 %0, %1, %2" : "=v"(pk[i]) : "v"(sgn), "v"(scale_pk));
        }
        for (int i = 0; i < 8; i++) {
            const _Float16* h2 = (const _Float16*)&pk[i];
            o[h * 16 + 2 * i] = (float)h2[0];
            o[h * 16 + 2 * i + 1] = (float)h2[1];
        }
    }
}

// probe 2: one f16 WMMA on known operands, dump acc. A[r][k], B[n][k] staged as
// the tile does: lane l holds a[i] = A[(l&15)][i], b[i] = B[(l&15)][i].
extern "C" __global__ void wmma_probe(const _Float16* A, const _Float16* B, float* out) {
    const unsigned int lane = threadIdx.x % 32;
    ink_v16f a, b;
    for (int i = 0; i < 16; i++) { a[i] = A[(lane & 15) * 16 + i]; b[i] = B[(lane & 15) * 16 + i]; }
    ink_v8f acc = ink_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    for (int e = 0; e < 8; e++)
        out[(2 * e + (lane >> 4)) * 16 + (lane & 15)] = acc[e];
}

extern "C" __global__ void __launch_bounds__(256) run_tile(
    const __nv_bfloat16* A, const unsigned char* wc, const unsigned char* ws, float g,
    __nv_bfloat16* C, int m_count, unsigned int N, unsigned int K, int nonpkrtz
) {
    if (nonpkrtz)
        ink_hip_gemm_tile_f16<false, true>(A, wc, ws, g, C, 0, m_count, 0, nullptr, N, K, 0, 0);
    else
        ink_hip_gemm_tile_f16<false, false>(A, wc, ws, g, C, 0, m_count, 0, nullptr, N, K, 0, 0);
}

static float bf2f(__nv_bfloat16 b) { unsigned short u = *(unsigned short*)&b; unsigned int f = (unsigned int)u << 16; return *(float*)&f; }
static __nv_bfloat16 f2bf(float f) { unsigned int u = *(unsigned int*)&f; u = (u + 0x8000 + ((u >> 16) & 1)) >> 16; __nv_bfloat16 b; *(unsigned short*)&b = (unsigned short)u; return b; }

int main() {
    const unsigned int N = 128, K = 64; const int M = 100; const float g = 1.5f;
    std::vector<__nv_bfloat16> hA(M * K);
    std::vector<unsigned char> hC(N * (K / 2)), hS(N * (K / 16));
    std::vector<__nv_bfloat16> hO(M * N);
    srand(12345);
    for (auto& v : hA) v = f2bf(((float)(rand() % 2001) - 1000.0f) / 64.0f);
    for (auto& v : hC) v = (unsigned char)(rand() & 0xFF);
    for (auto& v : hS) v = (unsigned char)(0x30 + rand() % 0x40); // mid-range scales
    // one big activation to probe f16 overflow
    hA[3 * K + 7] = f2bf(70000.0f);
    hA[40 * K + 9] = f2bf(-80000.0f);

    __nv_bfloat16 *dA, *dO; unsigned char *dC, *dS;
    hipMalloc(&dA, hA.size() * 2); hipMalloc(&dC, hC.size()); hipMalloc(&dS, hS.size()); hipMalloc(&dO, hO.size() * 2);
    hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), hC.size(), hipMemcpyHostToDevice);
    hipMemcpy(dS, hS.data(), hS.size(), hipMemcpyHostToDevice);

    // ---- probe 1: dequantized B ----
    {
        float* dD; hipMalloc(&dD, 32 * 32 * 4);
        dump_b<<<1, 32>>>(dC, dS, g, dD, K);
        if (hipDeviceSynchronize() != hipSuccess) { printf("dump_b FAIL\n"); return 2; }
        std::vector<float> hd(32 * 32);
        hipMemcpy(hd.data(), dD, hd.size() * 4, hipMemcpyDeviceToHost);
        int bad = 0;
        for (int lane = 0; lane < 32; lane++) {
            const int n = lane & 15;
            for (int k = 0; k < 32; k++) {
                const unsigned char sb = hS[n * (K / 16) + k / 16];
                const _Float16 s16 = (_Float16)(host_e4m3(sb) * g);
                const unsigned char byte = hC[n * (K / 2) + k / 2];
                const int code = (k & 1) ? (byte >> 4) : (byte & 0xF);
                const float want = (float)((_Float16)E2M1[code] * s16);
                const float got = hd[lane * 32 + k];
                if (fabsf(got - want) > 1e-3f) { if (bad++ < 8) printf("B bad lane=%d k=%d n=%d got=%f want=%f (code=%d byte=%02x s=%02x)\n", lane, k, n, got, want, code, byte, sb); }
            }
        }
        printf("probe1 dump_b bad=%d\n", bad);
        hipFree(dD);
    }

    // ---- probe 2: f16 WMMA layout ----
    {
        _Float16* dWA; _Float16* dWB; float* dWO;
        hipMalloc(&dWA, 256 * 2); hipMalloc(&dWB, 256 * 2); hipMalloc(&dWO, 256 * 4);
        std::vector<_Float16> wA(256), wB(256);
        for (auto& v : wA) v = (_Float16)(((float)(rand() % 17) - 8.0f) / 4.0f);
        for (auto& v : wB) v = (_Float16)(((float)(rand() % 17) - 8.0f) / 4.0f);
        hipMemcpy(dWA, wA.data(), 512, hipMemcpyHostToDevice);
        hipMemcpy(dWB, wB.data(), 512, hipMemcpyHostToDevice);
        wmma_probe<<<1, 32>>>(dWA, dWB, dWO);
        if (hipDeviceSynchronize() != hipSuccess) { printf("wmma_probe FAIL\n"); return 2; }
        std::vector<float> wo(256);
        hipMemcpy(wo.data(), dWO, 1024, hipMemcpyDeviceToHost);
        int bad = 0;
        for (int l = 0; l < 32; l++) for (int e = 0; e < 8; e++) {
            const int r = 2 * e + (l >> 4), c = l & 15;
            float want = 0;
            for (int k = 0; k < 16; k++) want += (float)wA[r * 16 + k] * (float)wB[c * 16 + k];
            const float got = wo[r * 16 + c];
            if (fabsf(got - want) > 0.01f) { if (bad++ < 8) printf("W bad r=%d c=%d got=%f want=%f\n", r, c, got, want); }
        }
        printf("probe2 wmma_f16 bad=%d\n", bad);
        hipFree(dWA); hipFree(dWB); hipFree(dWO);
    }

    // host reference: C[m][n] = sum_k f32(f16(A[m,k])) * f32(f16(e2m1(code) * f16(e4m3(scale)*f16? g)))
    // NOTE the kernel rounds e4m3*g to f16 once per 16-k group and the product to f16.
    std::vector<float> ref(M * N, 0.0f);
    for (int n = 0; n < (int)N; n++) {
        for (int k = 0; k < (int)K; k += 16) {
            const unsigned char sb = hS[n * (K / 16) + k / 16];
            const _Float16 s16 = (_Float16)(host_e4m3(sb) * g);
            for (int j = 0; j < 8; j++) {
                const unsigned char byte = hC[n * (K / 2) + k / 2 + j];
                for (int h = 0; h < 2; h++) {
                    const int code = h ? (byte >> 4) : (byte & 0xF);
                    const _Float16 bv = (_Float16)(E2M1[code]) * s16;
                    const int kk = k + j * 2 + h;
                    for (int m = 0; m < M; m++) {
                        const _Float16 av = (_Float16)bf2f(hA[m * K + kk]);
                        ref[m * N + n] += (float)av * (float)bv;
                    }
                }
            }
        }
    }

    for (int np = 0; np < 2; np++) {
        run_tile<<<1, 256>>>(dA, dC, dS, g, dO, M, N, K, np);
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) { printf("np=%d LAUNCH/SYNC FAIL %s\n", np, hipGetErrorString(err)); return 2; }
        hipMemcpy(hO.data(), dO, hO.size() * 2, hipMemcpyDeviceToHost);
        float maxd = 0, maxr = 0; int bad = 0, firstm = -1, firstn = -1;
        for (int m = 0; m < M; m++) for (int n = 0; n < (int)N; n++) {
            const float got = bf2f(hO[m * N + n]), want = bf2f(f2bf(ref[m * N + n]));
            const float d = fabsf(got - want);
            const float rel = d / (fabsf(want) + 1e-6f);
            if (rel > 0.01f && d > 0.5f) { bad++; if (firstm < 0) { firstm = m; firstn = n; printf("np=%d first bad C[%d][%d] got=%f want=%f\n", np, m, n, got, want); } }
            if (d > maxd) maxd = d; if (rel > maxr) maxr = rel;
        }
        printf("np=%d bad=%d maxd=%g maxrel=%g\n", np, bad, maxd, maxr);
    }
    return 0;
}
