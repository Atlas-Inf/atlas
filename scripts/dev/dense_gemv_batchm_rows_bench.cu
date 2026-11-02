// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the exact-row dense_gemv_bf16_batchm body (plain and
// load-ahead) against the runtime-M body it replaced (kept verbatim below as
// `ref_impl`), for the GLM MoE router (batchm / batchm_ahead, N = 288,
// K = 4096) and the KDA b/f_a/g_a triple (batchm_triple_n / batch5_triple_n,
// N = 32 + 128 + 128). Checks every output bit of batchm, batchm_ahead, dual
// and triple_n for M = 0..9, and of the three batch5 entry points at M = 5,
// over odd shapes (short and ragged K, partial CTAs, strided output;
// K % 8 == 0, as the uint4 row loads require) with zero, -0.0, denormal, Inf
// and NaN activations and weights. Then times old and new under PDL launches,
// cycling `copies` weight sets so the weights come from DRAM as in decode
// (2.4 MB per set; 24 sets = 57 MB > 24 MiB L2). It reports the minimum and
// the median over reps: on a GPU shared with other contexts only the minimum
// is clean, and a burst (iters x us) must stay below a ~2 ms time slice.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/dense_gemv_batchm_rows_bench.cu -o batchm_rows_bench
//   ./batchm_rows_bench [copies=24] [iters=48] [reps=60]
// Device memory: ~60 MB at the defaults.
#include "dense_gemv_bf16_batchm.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;

// The body before the restructure (1e84fa7e), unchanged.
template <int ROWS>
__device__ __forceinline__ void ref_impl(
    const bf* __restrict__ A, const bf* __restrict__ B, bf* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride, float* __restrict__ smem
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;
    const unsigned int m = (M > ROWS) ? ROWS : M;
    float acc[ROWS];
    #pragma unroll
    for (int t = 0; t < ROWS; t++) acc[t] = 0.0f;
    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);
    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
        float bf[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 b_lo, b_hi;
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            bf[2 * i] = __bfloat162float(b_lo);
            bf[2 * i + 1] = __bfloat162float(b_hi);
        }
        for (unsigned int t = 0; t < m; t++) {
            const uint4* At_vec = (const uint4*)(A + (unsigned long long)t * K);
            uint4 a_data = At_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            float a = acc[t];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                a += __bfloat162float(a_lo) * bf[2 * i];
                a += __bfloat162float(a_hi) * bf[2 * i + 1];
            }
            acc[t] = a;
        }
    }
    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bfv = __bfloat162float(B_row[k]);
            for (unsigned int t = 0; t < m; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
            }
        }
    }
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }
    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) {
            smem[t * (N_PER_BLOCK * 2) + smem_idx] = acc[t];
        }
    }
    __syncthreads();
    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const unsigned int base = t * (N_PER_BLOCK * 2) + local_out * 2;
            const float r = smem[base] + smem[base + 1];
            C[(unsigned long long)t * out_stride + n] = __float2bfloat16(r);
        }
    }
}

// One reference kernel in the production triple_n signature; the single and
// dual launches are the same body with one or two planes in the grid.
__global__ void ref_planes(const bf* __restrict__ A0, const bf* __restrict__ A1,
    const bf* __restrict__ B0, const bf* __restrict__ B1, const bf* __restrict__ B2,
    bf* __restrict__ C0, bf* __restrict__ C1, bf* __restrict__ C2,
    unsigned int M, unsigned int N0, unsigned int N12, unsigned int K, unsigned int out_stride) {
    atlas_pdl_enter();
    __shared__ float smem[MAX_M * N_PER_BLOCK * 2];
    const unsigned int p = blockIdx.z;
    ref_impl<MAX_M>(p == 1u ? A1 : A0, p == 0u ? B0 : (p == 1u ? B1 : B2), p == 0u ? C0 : (p == 1u ? C1 : C2),
        M, p == 0u ? N0 : N12, K, out_stride, smem);
}

template <typename... Exp, typename... Act>
static void launch(void (*k)(Exp...), dim3 grid, Act... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, k, args...));
}

static unsigned short tobf(float f) { bf b = __float2bfloat16(f); unsigned short u; memcpy(&u, &b, 2); return u; }

struct Dev {
    bf *A, *B, *C[2];  // C[0]: reference outputs, C[1]: production outputs
    size_t c_bytes;
    std::vector<unsigned short> o[2];
};

// Every output bit of the four runtime-M entry points (and, at M = 5, the
// three batch5 ones) against the reference for one [N0 | N12 | N12] x K shape.
static size_t check_shape(Dev& d, unsigned N0, unsigned N12, unsigned K, unsigned stride,
                          size_t& checked, size_t& written) {
    const bf *A0 = d.A, *A1 = A0 + MAX_M * K, *B0 = d.B, *B1 = B0 + (size_t)N0 * K, *B2 = B1 + (size_t)N12 * K;
    const size_t plane = (size_t)MAX_M * std::max(stride, N12);
    const dim3 g1((N0 + 3) / 4, 1, 1), g2((N12 + 3) / 4, 1, 2), g3((std::max(N0, N12) + 3) / 4, 1, 3);
    enum { SINGLE, AHEAD, DUAL, TRIPLE, SINGLE5, DUAL5, TRIPLE5, FORMS };
    size_t diff = 0;
    for (unsigned M = 0; M <= MAX_M + 1; M++)
        for (int form = 0; form < (M == 5 ? FORMS : SINGLE5); form++) {
            for (int s = 0; s < 2; s++) {
                bf *C0 = d.C[s], *C1 = C0 + plane, *C2 = C1 + plane;
                CK(cudaMemset(C0, 0xEE, d.c_bytes));
                if (s == 0) {
                    // The reference body, with the planes and strides of the form.
                    if (form == DUAL || form == DUAL5) {
                        launch(ref_planes, g2, A0, A1, B1, B2, B2, C0, C1, C2, M, N12, N12, K, N12);
                    } else if (form == TRIPLE || form == TRIPLE5) {
                        launch(ref_planes, g1, A0, A0, B0, B1, B2, C0, C1, C2, M, N0, N0, K, N0);
                        launch(ref_planes, dim3(g2.x, 1, 3), A0, A0, B0, B1, B2, C2, C1, C2, M, 0u, N12, K, N12);
                    } else {
                        launch(ref_planes, g1, A0, A1, B0, B1, B2, C0, C1, C2, M, N0, N0, K, stride);
                    }
                    continue;
                }
                switch (form) {
                case SINGLE: launch(dense_gemv_bf16_batchm, g1, A0, B0, C0, M, N0, K, stride); break;
                case AHEAD: launch(dense_gemv_bf16_batchm_ahead, g1, A0, B0, C0, M, N0, K, stride); break;
                case DUAL: launch(dense_gemv_bf16_batchm_dual, g2, A0, A1, B1, B2, C0, C1, M, N12, K); break;
                case TRIPLE: launch(dense_gemv_bf16_batchm_triple_n, g3, A0, B0, B1, B2, C0, C1, C2, M, N0, N12, K); break;
                case SINGLE5: launch(dense_gemv_bf16_batch5, g1, A0, B0, C0, M, N0, K, stride); break;
                case DUAL5: launch(dense_gemv_bf16_batch5_dual, g2, A0, A1, B1, B2, C0, C1, N12, K); break;
                default: launch(dense_gemv_bf16_batch5_triple_n, g3, A0, B0, B1, B2, C0, C1, C2, N0, N12, K);
                }
            }
            CK(cudaDeviceSynchronize());
            for (int s = 0; s < 2; s++) CK(cudaMemcpy(d.o[s].data(), d.C[s], d.c_bytes, cudaMemcpyDeviceToHost));
            // Whole buffers: rows >= M and columns >= N must stay untouched in both.
            for (size_t i = 0; i < d.c_bytes / 2; i++, checked++) {
                diff += d.o[0][i] != d.o[1][i];
                written += d.o[0][i] != 0xEEEE;
            }
        }
    return diff;
}

int main(int argc, char** argv) {
    const int copies = argc > 1 ? atoi(argv[1]) : 24;
    const int iters = argc > 2 ? atoi(argv[2]) : 48;
    const int reps = argc > 3 ? atoi(argv[3]) : 60;
    const unsigned K = 4096, N0 = 32, N12 = 128, N = N0 + 2 * N12;  // router N = 288 = triple planes
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<unsigned short> hA(2 * MAX_M * K), hB((size_t)copies * N * K);
    for (auto& x : hA) x = tobf(nd(rng) * 2.f);
    for (auto& x : hB) x = tobf(nd(rng) * 0.05f);
    Dev d;
    d.c_bytes = 3 * (size_t)MAX_M * 4096 * 2;
    for (auto& o : d.o) o.resize(d.c_bytes / 2);
    CK(cudaMalloc(&d.A, hA.size() * 2));
    CK(cudaMalloc(&d.B, hB.size() * 2));
    for (auto& c : d.C) CK(cudaMalloc(&c, d.c_bytes));
    CK(cudaMemcpy(d.B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));

    size_t diff = 0, checked = 0, written = 0;
    // {N0, N12, K, single-launch output stride}
    const unsigned shapes[][4] = {{32, 128, 4096, 32}, {288, 128, 4096, 300}, {128, 4096, 128, 128},
                                  {132, 64, 104, 140}, {64, 36, 520, 64}, {20, 8, 1536, 20}, {8, 4, 8, 8}};
    for (int special = 0; special < 2; special++)
        for (auto& s : shapes) {
            // Specials, per row stride k: an all-zero row, a +-0.0 row, and
            // denormal / Inf / NaN slots in the activations and the weights.
            const unsigned k = s[2];
            std::vector<unsigned short> a = hA, b(hB.begin(), hB.begin() + 4 * K);
            for (unsigned i = 0; special && i < k; i++) { a[3 * k + i] = 0; a[5 * k + i] = (i & 1) ? 0x8000 : 0; }
            if (special) {
                a[6 * k + 1] = 0x0001; a[6 * k + k / 2] = 0x7F80; a[1 * k + k - 1] = 0x7FC1; a[2 * k + 3] = 0xFF80;
                b[2] = 0x8000; b[k + 5] = 0x0003; b[2 * k + 1] = 0x7F80; b[3 * k + 6] = 0xFFC0;
            }
            CK(cudaMemcpy(d.A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d.B, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
            diff += check_shape(d, s[0], s[1], s[2], s[3], checked, written);
        }
    CK(cudaMemcpy(d.B, hB.data(), 4 * K * 2, cudaMemcpyHostToDevice));
    printf("bitwise: %zu of %zu output words differ (%zu written by the reference)\n", diff, checked, written);

    CK(cudaMemcpy(d.A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    const unsigned Ms[] = {1, 2, 4, 5, 7, 8};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("us/call, min (median) of %d reps x %d PDL launches over %d weight sets\n", reps, iters, copies);
    printf("     batchm old     batchm new   batchm_ahead     triple old     triple new  batch5 triple\n");
    for (unsigned M : Ms) {
        std::vector<float> us[6];
        for (int r = 0; r < reps; r++)
            for (int k = 0; k < 6; k++) {
                if (k == 5 && M != 5) continue;
                auto one = [&](int it) {
                    const bf *B0 = d.B + (size_t)(it % copies) * N * K, *B1 = B0 + N0 * K, *B2 = B1 + N12 * K;
                    bf *C0 = d.C[0], *C1 = C0 + MAX_M * N, *C2 = C1 + MAX_M * N;
                    const bf* A = d.A;
                    const dim3 g1(N / 4, 1, 1), g3(N12 / 4, 1, 3);
                    switch (k) {
                    case 0: launch(ref_planes, g1, A, A, B0, B0, B0, C0, C0, C0, M, N, N, K, N); break;
                    case 1: launch(dense_gemv_bf16_batchm, g1, A, B0, C0, M, N, K, N); break;
                    case 2: launch(dense_gemv_bf16_batchm_ahead, g1, A, B0, C0, M, N, K, N); break;
                    case 3: launch(ref_planes, g3, A, A, B0, B1, B2, C0, C1, C2, M, N0, N12, K, N12); break;
                    case 4: launch(dense_gemv_bf16_batchm_triple_n, g3, A, B0, B1, B2, C0, C1, C2, M, N0, N12, K); break;
                    default: launch(dense_gemv_bf16_batch5_triple_n, g3, A, B0, B1, B2, C0, C1, C2, N0, N12, K);
                    }
                };
                for (int w = 0; w < 10; w++) one(w);
                CK(cudaEventRecord(e0));
                for (int it = 0; it < iters; it++) one(it);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                us[k].push_back(ms * 1000.f / iters);
            }
        printf("M=%u", M);
        for (auto& v : us) {
            if (v.empty()) continue;
            std::sort(v.begin(), v.end());
            printf("  %5.2f (%5.2f)", v[0], v[v.size() / 2]);
        }
        printf("\n");
    }
    return diff == 0 ? 0 : 2;
}
