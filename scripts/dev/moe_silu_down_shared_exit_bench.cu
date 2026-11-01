// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check of where moe_expert_silu_down_shared takes its
// `n1 >= N` early exit: after the cooperative s_act fill and its barrier (the
// file on the include path) versus before them (the old body, kept below).
// A warp only exits when N % 8 is 1..6, in the last x-block; the old body then
// left that warp's share of s_act unwritten for the surviving warps to read.
//
//   N % 8 in {0, 7}: no warp exits, so old and new must agree on every output
//     bit (routed slots, a NULL remote expert, the shared expert, and a NULL
//     shared expert).
//   N % 8 in 1..6: the new kernel must give rows 0..N-1 the bits it gives them
//     when launched with N rounded up to 8 (no exits, same weights), and must
//     agree with the old body on every row outside the last x-block. How often
//     the old body's last-block rows differ is printed for information.
//
// Launch as ops::moe_expert_silu_down_shared: block 128, grid (ceil(N/8),
// top_k + 1, 1), K * 4 bytes of dynamic shared memory. K = 2048, top_k = 8.
// Output buffers start from a different sentinel per variant.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/common \
//        scripts/dev/moe_silu_down_shared_exit_bench.cu -o silu_down_exit_bench
//   ./silu_down_exit_bench [reps=4]
// Device memory: ~70 MB. Exit: 0 when every required comparison is bit-identical.
#include "moe_shared_expert_fused.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;

// ── The kernel before the move (a1708008), verbatim but for the name and the
// long SWIGLU_LIMIT comment ──
extern "C" __global__ void silu_down_shared_before(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                sh_down_out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];

    const float SWIGLU_LIMIT = 10.0f;
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        if (!is_shared) {
            gf = fminf(gf, SWIGLU_LIMIT);
            uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
        }
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3(sb1) * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    __nv_bfloat16* out = is_shared ? sh_down_out : (C + (unsigned long long)expert_slot * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}

typedef void (*Kern)(const bf*, const bf*, const unsigned long long*, const unsigned long long*, const float*, bf*,
                     const unsigned int*, const bf*, const bf*, const unsigned char*, const unsigned char*, float,
                     bf*, unsigned, unsigned, unsigned);

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 4;
    const unsigned K = 2048, top_k = 8, experts = 12, n_max = 4104;
    const unsigned null_expert = 3;
    const unsigned h_idx[top_k] = {0, 5, null_expert, 7, 1, 11, 2, 9};
    const size_t packed_bytes = (size_t)n_max * (K / 2), scale_bytes = (size_t)n_max * (K / GROUP_SIZE);
    std::mt19937 rng(11);
    auto tobf = [](float f) { bf b = __float2bfloat16(f); unsigned short u; memcpy(&u, &b, 2); return u; };
    std::uniform_real_distribution<float> act(-12.f, 12.f);   // past the +-10 clamp on both sides

    // Packed E2M1 nibbles are any byte; E4M3 scales stay finite and moderate.
    std::vector<unsigned char> h_packed((experts + 1) * packed_bytes), h_scale((experts + 1) * scale_bytes);
    for (auto& x : h_packed) x = (unsigned char)rng();
    for (auto& x : h_scale) x = (unsigned char)(0x28 + rng() % 0x20);
    std::vector<unsigned short> h_gate((top_k + 1) * K), h_up((top_k + 1) * K);
    for (auto& x : h_gate) x = tobf(act(rng));
    for (auto& x : h_up) x = tobf(act(rng));

    unsigned char *d_packed, *d_scale;
    bf *d_gate, *d_up, *d_c[2], *d_sh[2];
    unsigned long long *d_pp, *d_sp;
    float* d_s2;
    unsigned int* d_idx;
    const size_t c_elems = (size_t)top_k * n_max;
    CK(cudaMalloc(&d_packed, h_packed.size()));
    CK(cudaMalloc(&d_scale, h_scale.size()));
    CK(cudaMalloc(&d_gate, h_gate.size() * 2));
    CK(cudaMalloc(&d_up, h_up.size() * 2));
    for (auto& p : d_c) CK(cudaMalloc(&p, c_elems * 2));
    for (auto& p : d_sh) CK(cudaMalloc(&p, n_max * 2));
    CK(cudaMalloc(&d_pp, experts * 8));
    CK(cudaMalloc(&d_sp, experts * 8));
    CK(cudaMalloc(&d_s2, experts * 4));
    CK(cudaMalloc(&d_idx, top_k * 4));
    CK(cudaMemcpy(d_packed, h_packed.data(), h_packed.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_scale, h_scale.data(), h_scale.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gate, h_gate.data(), h_gate.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_up, h_up.data(), h_up.size() * 2, cudaMemcpyHostToDevice));
    std::vector<unsigned long long> h_pp(experts), h_sp(experts);
    std::vector<float> h_s2(experts);
    for (unsigned e = 0; e < experts; e++) {
        h_pp[e] = e == null_expert ? 0ull : (unsigned long long)(d_packed + e * packed_bytes);
        h_sp[e] = (unsigned long long)(d_scale + e * scale_bytes);
        h_s2[e] = 0.002f + 0.0005f * e;
    }
    CK(cudaMemcpy(d_pp, h_pp.data(), experts * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sp, h_sp.data(), experts * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s2, h_s2.data(), experts * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_idx, h_idx, top_k * 4, cudaMemcpyHostToDevice));

    const Kern kern[2] = {silu_down_shared_before, moe_expert_silu_down_shared};
    const unsigned char sentinel[2] = {0xEE, 0xDD};
    // Run `k` at N rows into host copies; `shared_null` drops the shared expert.
    auto run = [&](int k, int slot, unsigned N, bool shared_null, std::vector<unsigned short>& c,
                   std::vector<unsigned short>& sh) {
        CK(cudaMemset(d_c[slot], sentinel[slot], c_elems * 2));
        CK(cudaMemset(d_sh[slot], sentinel[slot], n_max * 2));
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3((N + 7) / 8, top_k + 1, 1);
        cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
        cfg.dynamicSmemBytes = K * 4;
        const unsigned char* shp = shared_null ? nullptr : d_packed + experts * packed_bytes;
        CK(cudaLaunchKernelEx(&cfg, kern[k], (const bf*)d_gate, (const bf*)d_up, (const unsigned long long*)d_pp,
                              (const unsigned long long*)d_sp, (const float*)d_s2, d_c[slot],
                              (const unsigned int*)d_idx, (const bf*)(d_gate + top_k * K),
                              (const bf*)(d_up + top_k * K), shp,
                              (const unsigned char*)(d_scale + experts * scale_bytes), 0.0031f, d_sh[slot], N, K,
                              top_k));
        CK(cudaDeviceSynchronize());
        c.resize(c_elems); sh.resize(n_max);
        CK(cudaMemcpy(c.data(), d_c[slot], c_elems * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sh.data(), d_sh[slot], n_max * 2, cudaMemcpyDeviceToHost));
    };

    std::vector<unsigned short> c[2], sh[2];
    size_t same_n = 0, same_diff = 0;          // N % 8 in {0,7}: old vs new
    size_t pad_n = 0, pad_diff = 0;            // N % 8 in 1..6: new(N) vs new(roundup8(N))
    size_t body_n = 0, body_diff = 0;          // N % 8 in 1..6: old vs new outside the last x-block
    size_t tail_n = 0, tail_diff = 0;          // N % 8 in 1..6: old vs new inside it (information)
    const unsigned exact_ns[] = {4096, 4095, 2048, 64, 8, 7};
    const unsigned ragged_ns[] = {4097, 4098, 4099, 4100, 4101, 4102, 2561, 1030, 77, 13, 9, 6, 5, 4, 3, 2, 1};
    for (int rep = 0; rep < reps; rep++) {
        const bool shared_null = rep % 4 == 3;
        for (unsigned N : exact_ns) {
            for (int k = 0; k < 2; k++) run(k, k, N, shared_null, c[k], sh[k]);
            for (unsigned s = 0; s < top_k; s++)
                for (unsigned n = 0; n < N; n++, same_n++)
                    same_diff += c[0][(size_t)s * N + n] != c[1][(size_t)s * N + n];
            for (unsigned n = 0; n < N; n++, same_n++) same_diff += sh[0][n] != sh[1][n];
        }
        for (unsigned N : ragged_ns) {
            const unsigned np = (N + 7) / 8 * 8, body = N / 8 * 8;
            std::vector<unsigned short> cp, shp;
            run(1, 0, np, shared_null, cp, shp);
            for (int k = 0; k < 2; k++) run(k, k, N, shared_null, c[k], sh[k]);
            for (unsigned s = 0; s <= top_k; s++)
                for (unsigned n = 0; n < N; n++) {
                    const unsigned short o = s < top_k ? c[0][(size_t)s * N + n] : sh[0][n];
                    const unsigned short w = s < top_k ? c[1][(size_t)s * N + n] : sh[1][n];
                    const unsigned short p = s < top_k ? cp[(size_t)s * np + n] : shp[n];
                    pad_n++; pad_diff += w != p;
                    if (n < body) { body_n++; body_diff += o != w; }
                    else { tail_n++; tail_diff += o != w; }
                }
        }
    }
    printf("N %% 8 in {0,7} (no warp exits): old vs new, %zu of %zu outputs differ\n", same_diff, same_n);
    printf("N %% 8 in 1..6: new at N vs new at N rounded up to 8, %zu of %zu outputs differ\n", pad_diff, pad_n);
    printf("N %% 8 in 1..6: old vs new outside the last x-block, %zu of %zu outputs differ\n", body_diff, body_n);
    printf("N %% 8 in 1..6: old vs new inside the last x-block, %zu of %zu outputs differ "
           "(old body reads unwritten s_act there)\n", tail_diff, tail_n);
    const bool ok = same_diff == 0 && pad_diff == 0 && body_diff == 0;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
