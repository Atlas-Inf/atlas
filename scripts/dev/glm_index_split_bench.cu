// SPDX-License-Identifier: AGPL-3.0-only
// Standalone check of the ATLAS_GLM_INDEX_SPLIT row split: the production
// prefill selector (glm_index_logits_bf16_wmma_row8_pool32, or the scalar
// row8 scorer, then glm_index_topk_expand) over every row of one piece,
// versus each TP rank selecting only its rows as the Rust split does (rank 0:
// Q0 and Q3 through the 0-3 tail rows; rank 1: Q1+Q2 then the tail rows) and
// then receiving the peer's quarters as the pair exchange would. Reports the
// bitwise logits / token-id comparison (both ranks) and ms per rank. Random
// BF16 pooled keys in a shuffled 16-token block table, random queries/weights.
// The row split follows RiNGSiDE's GLM53_INDEXER_ROW_SPLIT (othexmr,
// github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE; idea
// only); the zigzag Q0+Q3/Q1+Q2 split and the row-subset proof are ours.
//
//   nvcc -arch=sm_121a -O3 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_index_split_bench.cu -o split_bench
//   ./split_bench [rows=4096] [seq_start=57344] [iters=5] [wmma=1]
// Device memory: ~0.9 GB at the defaults.
#include "glm_indexer.cu"
#include "glm_indexer_wmma.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static unsigned short f2bf(float f) { unsigned int u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

struct Piece {
    const __nv_bfloat16 *q, *w, *cache;
    const unsigned* table;
    unsigned seq_start, stride, bs;
    unsigned long long block_bytes;
    bool wmma;
};

// Select rows [row0, row0 + rows) of the piece: logits into `logits` rows,
// token ids into `out` rows (both indexed by absolute piece row).
static void select_rows(const Piece& p, unsigned row0, unsigned rows, float* logits, int* out) {
    const unsigned pools_per_cta = p.wmma ? 32 : 8;
    dim3 grid((p.stride + pools_per_cta - 1) / pools_per_cta, (rows + 7) / 8);
    const __nv_bfloat16* q = p.q + (size_t)row0 * 32 * 128;
    const __nv_bfloat16* w = p.w + (size_t)row0 * 32;
    float* l = logits + (size_t)row0 * p.stride;
    if (p.wmma)
        glm_index_logits_bf16_wmma_row8_pool32<<<grid, 256>>>(q, w, p.cache, l, p.table, rows,
            p.seq_start + row0, p.stride, 32, 128, 4, p.bs, p.block_bytes);
    else
        glm_index_logits_bf16_row8<<<grid, 256, 128 * 8 * 2>>>(q, w, p.cache, l, p.table, rows,
            p.seq_start + row0, p.stride, 32, 128, 4, p.bs, p.block_bytes);
    glm_index_topk_expand<<<rows, 256, 16>>>(l, out + (size_t)row0 * 2051, rows,
        p.seq_start + row0, p.stride, 2048, 4, 2051);
    CK(cudaGetLastError());
}

// First differing element of two device buffers, or -1.
template <class T>
static long long first_diff(const T* a, const T* b, size_t n) {
    std::vector<T> ha(n), hb(n);
    CK(cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++)
        if (memcmp(&ha[i], &hb[i], sizeof(T))) return (long long)i;
    return -1;
}

int main(int argc, char** argv) {
    const unsigned rows = argc > 1 ? atoi(argv[1]) : 4096;
    const unsigned seq_start = argc > 2 ? atoi(argv[2]) : 57344;
    const int iters = argc > 3 ? atoi(argv[3]) : 5;
    const bool wmma = argc > 4 ? atoi(argv[4]) != 0 : true;
    const unsigned bs = 16, tokens = seq_start + rows, stride = (tokens + 3) / 4;
    const unsigned blocks = (tokens + bs - 1) / bs;
    const unsigned long long block_bytes = (bs / 4) * 128 * 2;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<unsigned> table(blocks);
    for (unsigned i = 0; i < blocks; i++) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<unsigned short> cache((size_t)blocks * (bs / 4) * 128), q((size_t)rows * 32 * 128), w((size_t)rows * 32);
    for (auto& v : cache) v = f2bf(nd(rng));
    for (auto& v : q) v = f2bf(nd(rng));
    for (auto& v : w) v = f2bf(nd(rng) * 0.2f);
    // Coarse ties: a few repeated keys so equal logits exercise the tie rule.
    for (size_t pool = 64; pool + 1 < cache.size() / 128; pool += 97)
        memcpy(&cache[(pool + 1) * 128], &cache[pool * 128], 256);

    void *d_cache, *d_q, *d_w, *d_table;
    float *d_logits_rep, *d_logits_split;
    int *d_rep, *d_rank[2];
    const size_t out_n = (size_t)rows * 2051, logit_n = (size_t)rows * stride;
    CK(cudaMalloc(&d_cache, cache.size() * 2));
    CK(cudaMalloc(&d_q, q.size() * 2));
    CK(cudaMalloc(&d_w, w.size() * 2));
    CK(cudaMalloc(&d_table, blocks * 4));
    CK(cudaMalloc(&d_logits_rep, logit_n * 4));
    CK(cudaMalloc(&d_logits_split, logit_n * 4));
    CK(cudaMalloc(&d_rep, out_n * 4));
    CK(cudaMalloc(&d_rank[0], out_n * 4));
    CK(cudaMalloc(&d_rank[1], out_n * 4));
    CK(cudaMemcpy(d_cache, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_table, table.data(), blocks * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_logits_rep, 0x7f, logit_n * 4));
    CK(cudaMemset(d_logits_split, 0x7f, logit_n * 4));

    const Piece p{(const __nv_bfloat16*)d_q, (const __nv_bfloat16*)d_w, (const __nv_bfloat16*)d_cache,
                  (const unsigned*)d_table, seq_start, stride, bs, block_bytes, wmma};
    const unsigned qr = rows / 4;
    // Rank r's passes as (first row, rows), as glm_index_split::passes gives.
    const unsigned pass[2][2][2] = {{{0, qr}, {3 * qr, rows - 3 * qr}}, {{qr, 2 * qr}, {4 * qr, rows - 4 * qr}}};
    auto select_rank = [&](int r) {
        for (int k = 0; k < 2; k++)
            if (pass[r][k][1]) select_rows(p, pass[r][k][0], pass[r][k][1], d_logits_split, d_rank[r]);
    };
    auto receive = [&](int r, unsigned row0, unsigned n) {
        CK(cudaMemcpy(d_rank[r] + (size_t)row0 * 2051, d_rank[1 - r] + (size_t)row0 * 2051,
                      (size_t)n * 2051 * 4, cudaMemcpyDeviceToDevice));
    };

    // Correctness: replicated vs each rank's rows plus the peer's quarters.
    select_rows(p, 0, rows, d_logits_rep, d_rep);
    select_rank(0);
    select_rank(1);
    receive(0, qr, 2 * qr);  // Q1 + Q2 from rank 1
    receive(1, 0, qr);       // Q0 from rank 0
    receive(1, 3 * qr, qr);  // Q3 from rank 0
    CK(cudaDeviceSynchronize());
    const long long dl = first_diff(d_logits_rep, d_logits_split, logit_n);
    const long long di0 = first_diff(d_rep, d_rank[0], out_n);
    const long long di1 = first_diff(d_rep, d_rank[1], out_n);
    const long long di = di0 >= 0 ? di0 : di1;

    // Timing: every row (replicated) vs each rank's half.
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto time = [&](auto&& body) {
        body();
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; i++) body();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return ms / iters;
    };
    const float t_rep = time([&] { select_rows(p, 0, rows, d_logits_rep, d_rep); });
    float t_rank[2];
    for (int r = 0; r < 2; r++)
        t_rank[r] = time([&] { select_rank(r); });
    printf("rows=%u tail=%u seq_start=%u pools=%u scorer=%s\n", rows, rows - 4 * qr, seq_start, stride,
           wmma ? "wmma_row8_pool32" : "row8");
    printf("logits bitwise: %s (first diff %lld)\n", dl < 0 ? "IDENTICAL" : "DIFFER", dl);
    printf("token ids bitwise: %s (first diff rank0 %lld rank1 %lld)\n", di < 0 ? "IDENTICAL" : "DIFFER", di0, di1);
    printf("ms/piece: replicated %.3f  rank0 %.3f  rank1 %.3f  (max rank / replicated %.3f)\n", t_rep,
           t_rank[0], t_rank[1], std::max(t_rank[0], t_rank[1]) / t_rep);
    return dl < 0 && di < 0 ? 0 : 2;
}
