// SPDX-License-Identifier: AGPL-3.0-only
// Ling MLA over the latent cache, against expanded attention on the CPU.
//
//   nvcc -std=c++17 -O2 --fmad=false -gencode arch=compute_121a,code=sm_121a \
//     tests/ling/mla_latent.cu -o /tmp/ling-mla-latent && /tmp/ling-mla-latent
//
// 1. Absorbed decode (q_absorb, decode, merge, v_up) for rows of lengths 1,
//    17, 300 and 5000 (several splits) on a shuffled block table matches
//    softmax(q . K) V with K = [W_kb c | rope] and V = W_vb c.
// 2. A row's bytes do not depend on how many rows share the launches.
// 3. A prompt tile attended over two key tiles (partial kernel + merge)
//    matches it attended as one tile, and both match the CPU reference.
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_mla_latent.cu"
#undef HDIM
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_mla_prefill_part.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
static float f(__nv_bfloat16 x) { return __bfloat162float(x); }

constexpr int H = 32, R = 512, P = 64, N = 128, BS = 16, ROW = R + P, EXP = H * 128 + 64;

struct Model {
    std::vector<float> w;      // kv_b [H*256][512]
    std::vector<float> cache;  // logical token t: [576]
};

// Expanded attention for one query row over tokens [0, len).
static void reference(const Model& m, const __nv_bfloat16* q, int len, float scale, float* out) {
    for (int h = 0; h < H; ++h) {
        std::vector<double> s(len);
        double top = -1e300;
        for (int t = 0; t < len; ++t) {
            const float* c = &m.cache[size_t(t) * ROW];
            double dot = 0;
            for (int d = 0; d < N; ++d) {
                double k = 0;
                for (int i = 0; i < R; ++i) k += m.w[(h * 256 + d) * R + i] * c[i];
                dot += f(q[h * 192 + d]) * k;
            }
            for (int i = 0; i < P; ++i) dot += f(q[h * 192 + N + i]) * c[R + i];
            s[t] = dot * scale;
            top = std::max(top, s[t]);
        }
        double total = 0;
        for (int t = 0; t < len; ++t) total += (s[t] = std::exp(s[t] - top));
        for (int v = 0; v < 128; ++v) {
            double acc = 0;
            for (int t = 0; t < len; ++t) {
                const float* c = &m.cache[size_t(t) * ROW];
                double val = 0;
                for (int i = 0; i < R; ++i) val += m.w[(h * 256 + 128 + v) * R + i] * c[i];
                acc += s[t] / total * val;
            }
            out[h * 128 + v] = float(acc);
        }
    }
}

int main() {
    check(cudaFuncSetAttribute(ling_latent_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, LING_DECODE_SMEM));
    std::mt19937 rng(576);
    std::uniform_real_distribution<float> draw(-1.f, 1.f);
    const int tokens = 5000, blocks = (tokens + BS - 1) / BS;
    const float scale = 1.f / std::sqrt(192.f);
    Model m;
    m.w.resize(H * 256 * R);
    m.cache.resize(size_t(tokens) * ROW);
    auto w = allocate<__nv_bfloat16>(m.w.size());
    for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = f(w[i] = __float2bfloat16(draw(rng) * 0.05f));
    auto pool = allocate<__nv_bfloat16>(size_t(blocks) * BS * ROW);
    auto table = allocate<int>(blocks);
    std::vector<int> order(blocks);
    for (int i = 0; i < blocks; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    for (int i = 0; i < blocks; ++i) table[i] = order[i];
    for (int t = 0; t < tokens; ++t)
        for (int i = 0; i < ROW; ++i) {
            const __nv_bfloat16 v = __float2bfloat16(draw(rng));
            m.cache[size_t(t) * ROW + i] = f(v);
            pool[(size_t(table[t / BS]) * BS + t % BS) * ROW + i] = v;
        }

    // ---- 1 and 2: absorbed decode ----
    const int lens[4] = {1, 17, 300, 5000};
    const int rows = 4;
    auto q = allocate<__nv_bfloat16>(rows * H * 192);
    for (int i = 0; i < rows * H * 192; ++i) q[i] = __float2bfloat16(draw(rng));
    auto seq = allocate<int>(rows);
    for (int r = 0; r < rows; ++r) seq[r] = lens[r];
    auto q_abs = allocate<__nv_bfloat16>(rows * H * ROW);
    auto part_o = allocate<float>(size_t(rows) * DEC_SPLITS * H * R);
    auto part_ml = allocate<float>(size_t(rows) * DEC_SPLITS * H * 2);
    auto latent = allocate<__nv_bfloat16>(rows * H * R);
    auto out = allocate<__nv_bfloat16>(rows * H * 128);
    auto run = [&](int first, int n) {
        ling_q_absorb<<<dim3(H, (n + 7) / 8), 512>>>(q + first * H * 192, w, q_abs, n);
        ling_latent_decode<<<dim3(DEC_SPLITS, n), 256, LING_DECODE_SMEM>>>(q_abs, pool, table, seq + first, 0, BS, scale,
                                                         part_o, part_ml);
        ling_latent_decode_merge<<<dim3(H, n), 128>>>(part_o, part_ml, seq + first, latent);
        ling_v_up<<<dim3(H, (n + 7) / 8), 256>>>(latent, w, out + first * H * 128, n);
        check(cudaDeviceSynchronize());
    };
    run(0, rows);
    std::vector<__nv_bfloat16> together(out, out + rows * H * 128);
    float worst = 0, peak = 0;
    for (int r = 0; r < rows; ++r) {
        std::vector<float> ref(H * 128);
        reference(m, q + r * H * 192, lens[r], scale, ref.data());
        for (int i = 0; i < H * 128; ++i) {
            worst = std::max(worst, std::abs(f(together[r * H * 128 + i]) - ref[i]));
            peak = std::max(peak, std::abs(ref[i]));
        }
    }
    std::printf("decode: worst %.6g of peak %.6g\n", worst, peak);
    bool ok = worst <= 0.02f * peak;
    for (int r = 0; r < rows; ++r) run(r, 1);
    const bool same = std::equal(together.begin(), together.end(), out,
        [](__nv_bfloat16 a, __nv_bfloat16 b) { return __bfloat16_as_ushort(a) == __bfloat16_as_ushort(b); });
    std::printf("decode rows one at a time vs together: %s\n", same ? "identical" : "DIFFER");
    ok &= same;

    // ---- 3: a prompt tile over two key tiles ----
    const int offset = 700, prompt = 100, total = offset + prompt, split = 512;
    auto k_rows = allocate<__nv_bfloat16>(size_t(total) * EXP);
    auto v_rows = allocate<__nv_bfloat16>(size_t(total) * EXP);
    auto ident = allocate<int>((total + BS - 1) / BS);
    for (int i = 0; i < (total + BS - 1) / BS; ++i) ident[i] = i;
    for (int t = 0; t < total; ++t)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < 128; ++d) {
                double k = 0, v = 0;
                for (int i = 0; i < R; ++i) {
                    k += m.w[(h * 256 + d) * R + i] * m.cache[size_t(t) * ROW + i];
                    v += m.w[(h * 256 + 128 + d) * R + i] * m.cache[size_t(t) * ROW + i];
                }
                k_rows[size_t(t) * EXP + h * 128 + d] = __float2bfloat16(float(k));
                v_rows[size_t(t) * EXP + h * 128 + d] = __float2bfloat16(float(v));
            }
    for (int t = 0; t < total; ++t)
        for (int i = 0; i < P; ++i)
            k_rows[size_t(t) * EXP + H * 128 + i] = __float2bfloat16(m.cache[size_t(t) * ROW + R + i]);
    auto pq = allocate<__nv_bfloat16>(prompt * H * 192);
    for (int i = 0; i < prompt * H * 192; ++i) pq[i] = __float2bfloat16(draw(rng));
    auto single = allocate<__nv_bfloat16>(prompt * H * 192);
    auto merged = allocate<__nv_bfloat16>(prompt * H * 192);
    auto tile_o = allocate<float>(prompt * H * 192);
    auto tile_ml = allocate<float>(prompt * H * 2);
    auto acc_o = allocate<float>(prompt * H * 192);
    auto acc_ml = allocate<float>(prompt * H * 2);
    const dim3 grid(H, (prompt + 63) / 64);
    ling_mla_prefill_part_64<<<grid, 256>>>(pq, k_rows, v_rows, reinterpret_cast<__nv_bfloat16*>(tile_o), ident,
        prompt, total, offset, H, H, 192, BS, 0, 1, scale, tile_ml);
    ling_prompt_tile_merge<<<dim3(H, prompt), 64>>>(tile_o, tile_ml, acc_o, acc_ml, single, 1, 1);
    // Tile 0 = keys [0, split), entirely visible; tile 1 = [split, total).
    for (int tile = 0; tile < 2; ++tile) {
        const int begin = tile ? split : 0, end = tile ? total : split;
        ling_mla_prefill_part_64<<<grid, 256>>>(pq, k_rows + size_t(begin) * EXP, v_rows + size_t(begin) * EXP,
            reinterpret_cast<__nv_bfloat16*>(tile_o), ident, prompt, end - begin, offset - begin, H, H, 192, BS,
            0, 1, scale, tile_ml);
        ling_prompt_tile_merge<<<dim3(H, prompt), 64>>>(tile_o, tile_ml, acc_o, acc_ml, merged, tile == 0, tile == 1);
    }
    check(cudaDeviceSynchronize());
    float tiled = 0, ref_worst = 0, ref_peak = 0;
    for (int r = 0; r < prompt; ++r) {
        for (int h = 0; h < H; ++h)
            for (int v = 0; v < 128; ++v) {
                const int i = (r * H + h) * 192 + v;
                tiled = std::max(tiled, std::abs(f(single[i]) - f(merged[i])));
            }
        if (r % 33) continue;  // the CPU reference is slow; spot-check rows
        std::vector<float> ref(H * 128);
        reference(m, pq + r * H * 192, offset + r + 1, scale, ref.data());
        for (int h = 0; h < H; ++h)
            for (int v = 0; v < 128; ++v) {
                ref_worst = std::max(ref_worst, std::abs(f(merged[(r * H + h) * 192 + v]) - ref[h * 128 + v]));
                ref_peak = std::max(ref_peak, std::abs(ref[h * 128 + v]));
            }
    }
    std::printf("prompt: two tiles vs one worst %.6g; vs reference worst %.6g of peak %.6g\n", tiled, ref_worst,
                ref_peak);
    ok &= tiled <= 0.01f * ref_peak && ref_worst <= 0.02f * ref_peak;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
