// SPDX-License-Identifier: AGPL-3.0-only
// Ling's 128-row prompt attention against the BR=64 paged kernel it replaced.
//
//   nvcc -std=c++17 -O2 --fmad=false -gencode arch=compute_121a,code=sm_121a \
//     tests/ling/mla_prefill.cu -o /tmp/ling-mla-prefill && /tmp/ling-mla-prefill
//
// The reference is the shared BR=64 kernel built as Ling's partial-LSE
// variant was (HDIM 192 over expanded rows, query-major tiles). For key tiles
// that are wholly visible, the causal tile holding the queries' own keys,
// ragged query and key counts and unaligned offsets, both kernels must write
// the same bytes: FP32 output, softmax (max, sum) and the BF16 output.
// Expanded rows as the reference read them: per-head NoPE channels, then the
// shared RoPE tail at column 32 x 128 for query columns 128..191.
#define ATLAS_KV_ROW_SIZE(heads, dim) ((heads) * 128u + 64u)
#define ATLAS_KV_COLUMN(head, dim, col) ((col) < 128u ? (head) * 128u + (col) : 32u * 128u + (col) - 128u)
#define HDIM 192
#define ATLAS_PREFILL_GROUP_QUERY_TILES
#define PREFILL_PARTIAL_LSE
#define inferspark_prefill_paged reference_prefill
#define KERNEL_EXTRA_PARAMS_OVERRIDE , const float inv_sqrt_d, float* __restrict__ prefill_ml
#include "../../kernels/gb10/common/inferspark_prefill_paged.cu"
#undef BR
#undef BC
#undef HDIM
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_mla_prefill.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
        std::exit(2);
    }
}
template <typename T> static T* allocate(size_t count) {
    T* result;
    check(cudaMallocManaged(&result, count * sizeof(T)));
    check(cudaMemset(result, 0, count * sizeof(T)));
    return result;
}

constexpr unsigned H = 32, QK = 192, V = 128, ROW = H * 128 + 64, BS = 16;

struct Case {
    const char* name;
    unsigned q_len, kv_len, q_offset;
};

static bool run(const Case& c, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.f, 1.f);
    auto* q = allocate<__nv_bfloat16>(size_t(c.q_len) * H * QK);
    auto* k = allocate<__nv_bfloat16>(size_t(c.kv_len) * ROW);
    auto* v = allocate<__nv_bfloat16>(size_t(c.kv_len) * ROW);
    // Sharp enough that the running max moves within a row.
    for (size_t i = 0; i < size_t(c.q_len) * H * QK; ++i) q[i] = __float2bfloat16(2.f * normal(rng));
    for (size_t i = 0; i < size_t(c.kv_len) * ROW; ++i) k[i] = __float2bfloat16(normal(rng));
    for (size_t t = 0; t < c.kv_len; ++t)
        for (unsigned i = 0; i < H * 128; ++i) v[t * ROW + i] = __float2bfloat16(normal(rng));
    const unsigned blocks = (c.kv_len + BS - 1) / BS;
    auto* table = allocate<int>(blocks);
    for (unsigned b = 0; b < blocks; ++b) table[b] = int(b);
    auto* ref_o = allocate<float>(size_t(c.q_len) * H * QK);
    auto* ref_ml = allocate<float>(size_t(c.q_len) * H * 2);
    auto* part_o = allocate<float>(size_t(c.q_len) * H * V);
    auto* part_ml = allocate<float>(size_t(c.q_len) * H * 2);
    auto* out = allocate<__nv_bfloat16>(size_t(c.q_len) * H * V);
    const float scale = 1.f / std::sqrt(float(QK));

    reference_prefill_64<<<dim3(H, (c.q_len + 63) / 64), 256>>>(
        q, k, v, reinterpret_cast<__nv_bfloat16*>(ref_o), table, c.q_len, c.kv_len, c.q_offset, H, H, QK, BS, 0,
        1, scale, ref_ml);
    const dim3 grid(H, (c.q_len + 127) / 128);
    ling_mla_prefill_rows128_part<<<grid, 256>>>(q, k, v, part_o, part_ml, c.q_len, c.kv_len, c.q_offset, scale);
    ling_mla_prefill_rows128<<<grid, 256>>>(q, k, v, out, c.q_len, c.kv_len, c.q_offset, scale);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    size_t bad_o = 0, bad_ml = 0, bad_out = 0;
    for (size_t r = 0; r < c.q_len; ++r)
        for (unsigned h = 0; h < H; ++h) {
            const size_t at = r * H + h;
            for (unsigned d = 0; d < V; ++d) {
                const float want = ref_o[at * QK + d];
                bad_o += std::memcmp(&want, &part_o[at * V + d], 4) != 0;
                const __nv_bfloat16 rounded = __float2bfloat16(want);
                bad_out += std::memcmp(&rounded, &out[at * V + d], 2) != 0;
            }
            bad_ml += std::memcmp(&ref_ml[at * 2], &part_ml[at * 2], 8) != 0;
        }
    std::printf("%-34s q=%5u kv=%5u offset=%5u  output %zu, softmax %zu, bf16 %zu differ\n", c.name, c.q_len,
                c.kv_len, c.q_offset, bad_o, bad_ml, bad_out);
    for (void* p : {(void*)q, (void*)k, (void*)v, (void*)table, (void*)ref_o, (void*)ref_ml, (void*)part_o,
                    (void*)part_ml, (void*)out})
        check(cudaFree(p));
    return bad_o == 0 && bad_ml == 0 && bad_out == 0;
}

int main() {
    std::mt19937 rng(20261126);
    const Case cases[] = {
        {"first tile, whole chunk", 2048, 2048, 0},
        {"first tile, ragged", 1000, 1000, 0},
        {"causal tile after a prefix", 777, 1077, 300},
        {"causal tile, 64-row aligned", 2048, 4096, 2048},
        {"earlier tile, wholly visible", 513, 4100, 5000},
        {"earlier tile, short", 129, 33, 8192},
        {"one row", 1, 4097, 4096},
    };
    bool ok = true;
    for (const Case& c : cases) ok &= run(c, rng);
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
