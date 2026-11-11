// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check of how the in-place RoPE kernels tile positions
// over a 128-thread block: rope_forward, rope_forward_strided,
// rope_forward_yarn, rope_forward_yarn_scaled, rope_forward_yarn_interleaved
// and rope_forward_yarn_interleaved_inv (rope.cu on the include path).
//
// A block owns floor(128 / (rotary_dim/2)) positions. When rotary_dim/2 does
// not divide 128 the leftover threads index one position further, which is
// the next block's first row: without the `local_pos < pos_per_block` bound
// two blocks rotate the same pairs in place.
//
// Each case launches the kernel once over all rows, as ops::rope* does (block
// 128, grid (num_q_heads + num_kv_heads, ceil(seq_len / pos_per_block), 1)),
// and compares every element of Q and K with the same kernel launched one row
// at a time (seq_len = 1: a single position per block, so no tiling at all).
//   rotary_dim/2 divides 128 (16..256): the bound never triggers. The FNV-1a
//     checksum of these outputs must be the same before and after the bound
//     is added (build this file against both versions of rope.cu).
//   rotary_dim/2 does not divide 128 (24, 40, 48, 96): needs the bound.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/common \
//        scripts/dev/rope_block_tiling_bench.cu -o rope_tiling_bench
//   ./rope_tiling_bench
// Device memory: < 10 MB. Exit: 0 when every case is bit-identical.
#include "rope.cu"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;

int main() {
    const unsigned nq = 3, nkv = 2, hd = 256, max_seq = 130, q_pad = 24, k_pad = 8;
    const unsigned rotary_dims[] = {256, 128, 64, 32, 16, 96, 48, 40, 24};
    const unsigned seqs[] = {1, 2, 5, 17, max_seq};
    const char* names[6] = {"rope_forward", "rope_forward_strided", "rope_forward_yarn",
                            "rope_forward_yarn_scaled", "rope_forward_yarn_interleaved",
                            "rope_forward_yarn_interleaved_inv"};
    const float theta = 10000000.0f, mscale = 0.85f;
    const size_t q_elems = (size_t)max_seq * (nq * hd + q_pad), k_elems = (size_t)max_seq * (nkv * hd + k_pad);

    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto tobf = [](float f) { bf b = __float2bfloat16(f); unsigned short u; memcpy(&u, &b, 2); return u; };
    std::vector<unsigned short> h_q(q_elems), h_k(k_elems), o_q[2], o_k[2];
    for (auto& x : h_q) x = tobf(nd(rng));
    for (auto& x : h_k) x = tobf(nd(rng));
    std::vector<unsigned int> h_pos(max_seq);
    for (auto& p : h_pos) p = rng() % 60000u;
    for (auto& v : o_q) v.resize(q_elems);
    for (auto& v : o_k) v.resize(k_elems);

    bf *d_q0, *d_k0, *d_q[2], *d_k[2];
    unsigned int* d_pos;
    float* d_freq;
    CK(cudaMalloc(&d_q0, q_elems * 2)); CK(cudaMalloc(&d_k0, k_elems * 2));
    for (auto& p : d_q) CK(cudaMalloc(&p, q_elems * 2));
    for (auto& p : d_k) CK(cudaMalloc(&p, k_elems * 2));
    CK(cudaMalloc(&d_pos, max_seq * 4)); CK(cudaMalloc(&d_freq, 128 * 4));
    CK(cudaMemcpy(d_q0, h_q.data(), q_elems * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_k0, h_k.data(), k_elems * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, h_pos.data(), max_seq * 4, cudaMemcpyHostToDevice));

    size_t diff[6][2] = {}, checked[6][2] = {};   // [kernel][rotary_dim/2 does not divide 128]
    unsigned long long fnv = 0xcbf29ce484222325ull;
    for (unsigned rd : rotary_dims) {
        const unsigned pairs = rd / 2, pos_per_block = 128 / pairs > 0 ? 128 / pairs : 1;
        const int ragged = 128 % pairs != 0;
        std::vector<float> h_freq(pairs);
        for (unsigned i = 0; i < pairs; i++) h_freq[i] = (float)(1.0 / pow(10000.0, (double)(2 * i) / rd));
        CK(cudaMemcpy(d_freq, h_freq.data(), pairs * 4, cudaMemcpyHostToDevice));
        for (int kern = 0; kern < 6; kern++) {
            // Only the strided kernel takes row strides; the rest are packed.
            const unsigned qs = nq * hd + (kern == 1 ? q_pad : 0), ks = nkv * hd + (kern == 1 ? k_pad : 0);
            auto launch = [&](bf* q, bf* k, const unsigned int* pos, unsigned seq) {
                const dim3 grid(nq + nkv, (seq + pos_per_block - 1) / pos_per_block, 1), block(128, 1, 1);
                switch (kern) {
                    case 0: rope_forward<<<grid, block>>>(q, k, pos, seq, nq, nkv, hd, rd, theta); break;
                    case 1: rope_forward_strided<<<grid, block>>>(q, k, pos, seq, nq, nkv, hd, rd, theta, qs, ks); break;
                    case 2: rope_forward_yarn<<<grid, block>>>(q, k, pos, seq, nq, nkv, hd, rd, d_freq, theta); break;
                    case 3: rope_forward_yarn_scaled<<<grid, block>>>(q, k, pos, seq, nq, nkv, hd, rd, d_freq, mscale); break;
                    case 4: rope_forward_yarn_interleaved<<<grid, block>>>(q, k, pos, seq, nq, nkv, hd, rd, d_freq, mscale); break;
                    default: rope_forward_yarn_interleaved_inv<<<grid, block>>>(q, k, pos, seq, nq, nkv, hd, rd, d_freq, mscale);
                }
                CK(cudaGetLastError());
            };
            for (unsigned seq : seqs) {
                for (int v = 0; v < 2; v++) {
                    CK(cudaMemcpy(d_q[v], d_q0, q_elems * 2, cudaMemcpyDeviceToDevice));
                    CK(cudaMemcpy(d_k[v], d_k0, k_elems * 2, cudaMemcpyDeviceToDevice));
                }
                launch(d_q[0], d_k[0], d_pos, seq);
                for (unsigned t = 0; t < seq; t++) launch(d_q[1] + (size_t)t * qs, d_k[1] + (size_t)t * ks, d_pos + t, 1);
                CK(cudaDeviceSynchronize());
                for (int v = 0; v < 2; v++) {
                    CK(cudaMemcpy(o_q[v].data(), d_q[v], q_elems * 2, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(o_k[v].data(), d_k[v], k_elems * 2, cudaMemcpyDeviceToHost));
                }
                // Whole buffers: rows past seq and the stride padding must stay untouched too.
                for (size_t i = 0; i < q_elems; i++) diff[kern][ragged] += o_q[0][i] != o_q[1][i];
                for (size_t i = 0; i < k_elems; i++) diff[kern][ragged] += o_k[0][i] != o_k[1][i];
                checked[kern][ragged] += q_elems + k_elems;
                if (!ragged)
                    for (const auto* o : {&o_q[0], &o_k[0]})
                        for (unsigned short x : *o) {
                            fnv = (fnv ^ (x & 0xff)) * 0x100000001b3ull;
                            fnv = (fnv ^ (x >> 8)) * 0x100000001b3ull;
                        }
            }
        }
    }
    size_t bad = 0;
    for (int kern = 0; kern < 6; kern++) {
        printf("%-34s one launch vs row at a time: dividing rotary_dim %zu of %zu differ, "
               "non-dividing %zu of %zu differ\n", names[kern], diff[kern][0], checked[kern][0], diff[kern][1],
               checked[kern][1]);
        bad += diff[kern][0] + diff[kern][1];
    }
    printf("checksum of the dividing-rotary_dim outputs: %016llx\n", fnv);
    printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
