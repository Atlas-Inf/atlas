// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the GLM routed-MoE routing chain (ATLAS_GLM_MOE_ROUTE_FUSE)
// at a canonical verify's shape (hidden 4096, top-8 of 288 experts):
//
//   chain  dense_gemv_bf16_tc{8,16,32} -> moe_topk_sigmoid_batched ->
//          moe_sort_by_expert_scan -> moe_build_tile_worklist_scan ->
//          quantize_bf16_to_nvfp4
//   fused  glm_moe_router_quant -> glm_moe_route_select
//
// Each layer is `spin` (a latency-bound PDL kernel standing in for the shared
// expert the router waits on, `window` us) -> the arm -> `nop` (standing in
// for the routed gate/up), every kernel launched with PDL as serving does.
// The router weight cycles over `layers` copies (2.4 MB each; L2 is 24 MiB),
// so it is cold as in a step. Prints the median per-layer time of each arm
// and of spin -> nop alone, so "arm" = layer - base is the chain's cost on
// the critical path; then the saving scaled to 42 MoE layers. The Rust GPU
// test route_fuse_test_gpu.rs checks the words.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_moe_route_bench.cu -o glm_moe_route_bench
//   ./glm_moe_route_bench [layers=24] [reps=9] [spin_iters=3000]
#include "glm_moe_route.cu"
#include "../../common/moe_topk_sigmoid.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
typedef unsigned char u8;

static const unsigned H = 4096, E = 288, TOPK = 8;

extern "C" __global__ void spin(unsigned int* tab, unsigned int iters) {
    atlas_pdl_enter();
    unsigned int x = threadIdx.x;
    for (unsigned int i = 0; i < iters; i++) x = tab[(x + i) & 1023u];
    if (x == 0xFFFFFFFFu) tab[0] = 0u;
}

extern "C" __global__ void nop(unsigned int* tab) {
    atlas_pdl_enter();
    if (tab == nullptr) return;
}

extern "C" __global__ void fill_bf16(bf* p, unsigned long long n, float scale, unsigned int seed) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int h = (unsigned int)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = __float2bfloat16(((float)(h & 0xFFFF) / 32768.0f - 1.0f) * scale);
    }
}

template <typename... Args>
static void launch(void (*kern)(Args...), dim3 grid, unsigned int block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, kern, args...));
}

struct Bufs {
    bf *x, *logits;
    float *bias, *wts;
    unsigned int *ids, *worklist;
    int *sorted_tokens, *sorted_experts, *offsets, *perm, *total;
    unsigned long long* ptrs;
    u8* packed;
};

static void chain(const Bufs& b, const bf* gate, unsigned m) {
    auto router = m <= 8 ? dense_gemv_bf16_tc8 : (m <= 16 ? dense_gemv_bf16_tc16 : dense_gemv_bf16_tc32);
    launch(router, dim3((E + 15) / 16), 256u, (const bf*)b.x, gate, b.logits, m, E, H, E);
    launch(moe_topk_sigmoid_batched, dim3(m), 256u, (const bf*)b.logits, (const float*)b.bias, b.ids,
           b.wts, E, TOPK, 1u, 2.5f);
    launch(moe_sort_by_expert_scan, dim3(1), 256u, (const unsigned*)b.ids, b.sorted_tokens,
           b.sorted_experts, b.offsets, b.perm, m * TOPK, E, TOPK);
    launch(moe_build_tile_worklist_scan, dim3(1), 256u, (const int*)b.offsets,
           (const unsigned long long*)b.ptrs, b.worklist + 4, b.total, E, 1u, 64u);
    launch(quantize_bf16_to_nvfp4, dim3(m), 128u, (const bf*)b.x, b.packed, b.packed + m * H / 2,
           1.0f, m, H);
}

static void fused(const Bufs& b, const bf* gate, unsigned m) {
    const unsigned router_ctas = (E + 15) / 16, quant = (m * (H / 16) + 255) / 256;
    launch(glm_moe_router_quant, dim3(router_ctas + quant), 256u, (const bf*)b.x, (const u8*)gate,
           (const u8*)b.bias, b.logits, b.packed, b.packed + m * H / 2, m, E, H, router_ctas, E);
    launch(glm_moe_route_select, dim3(1), 256u, (const bf*)b.logits, (const float*)b.bias, b.ids,
           b.wts, b.sorted_tokens, b.sorted_experts, b.offsets, b.perm,
           (const unsigned long long*)b.ptrs, b.worklist + 4, b.total, m, E, TOPK, 1u, 2.5f, 1u, 64u);
}

int main(int argc, char** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 24;
    const int reps = argc > 2 ? atoi(argv[2]) : 9;
    const unsigned spin_iters = argc > 3 ? (unsigned)atoi(argv[3]) : 3000u;
    std::vector<bf*> gates(layers);
    for (auto& g : gates) {
        CK(cudaMalloc(&g, (size_t)E * H * 2));
        fill_bf16<<<256, 256>>>(g, (unsigned long long)E * H, 0.03f, (unsigned)(&g - gates.data()) * 77u);
    }
    Bufs b;
    CK(cudaMalloc(&b.x, 32 * H * 2));
    fill_bf16<<<64, 256>>>(b.x, 32ull * H, 2.0f, 5u);
    CK(cudaMalloc(&b.logits, 32 * E * 2));
    CK(cudaMalloc(&b.bias, E * 4));
    CK(cudaMemset(b.bias, 0, E * 4));
    CK(cudaMalloc(&b.wts, 32 * TOPK * 4));
    CK(cudaMalloc(&b.ids, 32 * TOPK * 4));
    CK(cudaMalloc(&b.worklist, 16 + 32 * TOPK * 8));
    CK(cudaMalloc(&b.sorted_tokens, 32 * TOPK * 4));
    CK(cudaMalloc(&b.sorted_experts, 32 * TOPK * 4));
    CK(cudaMalloc(&b.offsets, (E + 1) * 4));
    CK(cudaMalloc(&b.perm, 32 * TOPK * 4));
    CK(cudaMalloc(&b.total, 4));
    std::vector<unsigned long long> ptrs(E);
    for (unsigned e = 0; e < E; e++) ptrs[e] = e % 2 ? 0x1000ull + e : 0ull;  // EP: half local
    CK(cudaMalloc(&b.ptrs, E * 8));
    CK(cudaMemcpy(b.ptrs, ptrs.data(), E * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.packed, 32 * H));
    unsigned int* tab;
    CK(cudaMalloc(&tab, 4096));
    std::vector<unsigned> init(1024);
    for (unsigned i = 0; i < 1024; i++) init[i] = (i * 37u + 11u) & 1023u;
    CK(cudaMemcpy(tab, init.data(), 4096, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    // arm 0: spin -> nop; 1: spin -> chain -> nop; 2: spin -> fused -> nop.
    auto run = [&](int arm, unsigned m) {
        std::vector<float> t;
        for (int r = 0; r < reps + 1; r++) {
            spin<<<1, 32>>>(tab, 200000u);  // keep the host ahead of the GPU
            CK(cudaEventRecord(e0));
            for (int l = 0; l < layers; l++) {
                launch(spin, dim3(1), 32u, tab, spin_iters);
                if (arm == 1) chain(b, gates[l], m);
                if (arm == 2) fused(b, gates[l], m);
                launch(nop, dim3(48), 256u, tab);
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (r > 0) t.push_back(ms * 1000.0f / layers);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    printf("rows  base_us  chain_us  fused_us  saved_us/layer  saved_ms/step(42)\n");
    for (unsigned m : {1u, 2u, 3u, 4u, 5u, 8u, 16u, 32u}) {
        const float base = run(0, m), c = run(1, m) - base, f = run(2, m) - base;
        printf("%4u  %7.1f  %8.1f  %8.1f  %14.1f  %17.2f\n", m, base, c, f, c - f, (c - f) * 42 / 1000);
    }
    return 0;
}
