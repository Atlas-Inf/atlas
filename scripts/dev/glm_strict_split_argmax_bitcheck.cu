// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check of the strict structured-output TP2 verify head
// (ATLAS_GLM_STRICT_SPEC): each rank's argmax_bf16_value_ban_allow over its
// half of a row, under the row's full-vocabulary grammar bitmask read from
// the rank's first vocabulary bit, then argmax_pair_merge_ban, must pick the
// best ALLOWED token of the whole row, value first then the lower index,
// exactly as a single-GPU masked argmax would; with the min_tokens ban on
// banned rows. Both ranks are emulated on one GPU and both merge views (rank
// 0 and rank 1) must agree. Also: under an all-ones mask the masked kernel
// returns argmax_bf16_value_ban's pairs bit for bit.
//
// Vocabularies cycle through 154,880 (word-aligned split), GLM-5.3's real
// 154,856 (rank 1 starts at 77,428, inside a mask word) and random even sizes
// (random split offsets within a word). Values are multiples of 1/8 in
// [-12.5, 12.5), so ties are common and the tie rule is exercised. Row shapes:
// random density, all ones, a single allowed token (in either shard), allowed
// tokens in one shard only, nothing allowed (both sentinels: token 0), and ban
// rows whose best allowed token is banned.
//
//   nvcc -arch=sm_121a -O3 -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_strict_split_argmax_bitcheck.cu -o glm_strict_split_argmax_bitcheck
//   ./glm_strict_split_argmax_bitcheck [cases=96]
//
// Prints one PASS line, or the first mismatches and exits 1.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include "argmax_bf16.cu"

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(2);                                                             \
        }                                                                        \
    } while (0)

static const unsigned VMAX = 160000, ROWS = 9;

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static unsigned next_u32() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (unsigned)(rng >> 32);
}

struct Shape {
    unsigned v, shard, w, ban[2];
};

static bool bit(const std::vector<unsigned>& m, const Shape& s, unsigned r, unsigned t) {
    return (m[(size_t)r * s.w + t / 32] >> (t % 32)) & 1u;
}
static void set_bit(std::vector<unsigned>& m, const Shape& s, unsigned r, unsigned t, bool on) {
    unsigned& w = m[(size_t)r * s.w + t / 32];
    w = on ? (w | (1u << (t % 32))) : (w & ~(1u << (t % 32)));
}

// Single-GPU reference: best allowed (and unbanned on ban rows) token.
static unsigned reference(const std::vector<float>& v, const std::vector<unsigned>& m,
                          const Shape& s, unsigned r, bool banned_row) {
    float best = -1e30f;
    unsigned idx = 0;
    for (unsigned t = 0; t < s.v; ++t) {
        if (!bit(m, s, r, t)) continue;
        if (banned_row && (t == s.ban[0] || t == s.ban[1])) continue;
        const float x = v[(size_t)r * s.v + t];
        if (x > best || (x == best && t < idx)) {
            best = x;
            idx = t;
        }
    }
    return idx;
}

// Shard-local ban index, as glm_vocab_split's `local_id` computes it.
static unsigned local_ban(unsigned id, const Shape& s, unsigned start) {
    return (id >= start && id - start < s.shard) ? id - start : 0xFFFFFFFFu;
}

int main(int argc, char** argv) {
    const int cases = argc > 1 ? atoi(argv[1]) : 96;
    const size_t wmax = (VMAX + 31) / 32;
    std::vector<float> vals((size_t)ROWS * VMAX);
    std::vector<__nv_bfloat16> logits((size_t)ROWS * VMAX);
    std::vector<unsigned> masks((size_t)ROWS * wmax), ones((size_t)ROWS * wmax, 0xFFFFFFFFu);
    __nv_bfloat16* d_logits;
    unsigned *d_masks, *d_ones, *d_pairs, *d_ref_pairs, *d_out;
    CK(cudaMalloc(&d_logits, logits.size() * 2));
    CK(cudaMalloc(&d_masks, masks.size() * 4));
    CK(cudaMalloc(&d_ones, ones.size() * 4));
    // Masked pairs of rank 0 and 1, then their all-ones-mask pairs.
    CK(cudaMalloc(&d_pairs, 4 * ROWS * 4 * 4));
    CK(cudaMalloc(&d_ref_pairs, 2 * ROWS * 4 * 4));
    CK(cudaMalloc(&d_out, 2 * ROWS * 4));
    CK(cudaMemcpy(d_ones, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
    long checked = 0, bad = 0;
    int unaligned = 0;
    for (int c = 0; c < cases; ++c) {
        Shape s;
        s.v = c % 3 == 0 ? 154880u : c % 3 == 1 ? 154856u : 150000u + 2 * (next_u32() % 5000);
        s.shard = s.v / 2;
        s.w = (s.v + 31) / 32;
        s.ban[0] = s.v - 60;
        s.ban[1] = s.v - 53;
        unaligned += s.shard % 32 != 0;
        const size_t n = (size_t)ROWS * s.v;
        for (size_t i = 0; i < n; ++i) {
            vals[i] = (float)((int)(next_u32() % 200) - 100) / 8.0f;
            logits[i] = __float2bfloat16(vals[i]);
            vals[i] = __bfloat162float(logits[i]);
        }
        for (unsigned r = 0; r < ROWS; ++r) {
            const unsigned shape = (r + c) % 7;
            for (unsigned w = 0; w < s.w; ++w) {
                unsigned x = next_u32() & next_u32();
                if (shape == 1) x = 0xFFFFFFFFu;
                if (shape == 2 || shape == 3 || shape == 6) x = 0;
                masks[(size_t)r * s.w + w] = x;
            }
            // One shard only, cut at the exact (possibly mid-word) split.
            if (shape == 4 || shape == 5)
                for (unsigned t = 0; t < s.v; ++t)
                    if ((shape == 4) == (t >= s.shard)) set_bit(masks, s, r, t, false);
            if (shape == 2 || shape == 3) {
                const unsigned t = (shape == 2 ? 0 : s.shard) + next_u32() % s.shard;
                set_bit(masks, s, r, t, true);
            }
        }
        const bool ban_rows = c % 2 == 1;
        if (ban_rows) {
            // Make a banned token the best allowed one of rows 0..3.
            for (unsigned r = 0; r < 4; ++r) {
                const unsigned t = s.ban[r % 2];
                set_bit(masks, s, r, t, true);
                logits[(size_t)r * s.v + t] = __float2bfloat16(20.0f);
                vals[(size_t)r * s.v + t] = 20.0f;
            }
        }
        const unsigned ban_lo = ban_rows ? 0x0Fu : 0u;
        CK(cudaMemcpy(d_logits, logits.data(), n * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_masks, masks.data(), (size_t)ROWS * s.w * 4, cudaMemcpyHostToDevice));
        for (unsigned rank = 0; rank < 2; ++rank) {
            const unsigned start = rank * s.shard;
            const unsigned b0 = local_ban(s.ban[0], s, start), b1 = local_ban(s.ban[1], s, start);
            argmax_bf16_value_ban_allow<<<ROWS, 1024>>>(
                d_logits + start, d_pairs + rank * ROWS * 4, s.shard, s.v, d_masks, s.w, start,
                b0, b1, 0xFFFFFFFFu, 0xFFFFFFFFu);
            // All-ones mask vs the unmasked kernel: identical pairs.
            argmax_bf16_value_ban<<<ROWS, 1024>>>(
                d_logits + start, d_ref_pairs + rank * ROWS * 4, s.shard, s.v,
                b0, b1, 0xFFFFFFFFu, 0xFFFFFFFFu);
            argmax_bf16_value_ban_allow<<<ROWS, 1024>>>(
                d_logits + start, d_pairs + (2 + rank) * ROWS * 4, s.shard, s.v, d_ones, s.w,
                start, b0, b1, 0xFFFFFFFFu, 0xFFFFFFFFu);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<unsigned> refp(2 * ROWS * 4);
        CK(cudaMemcpy(refp.data(), d_ref_pairs, refp.size() * 4, cudaMemcpyDeviceToHost));
        for (unsigned rank = 0; rank < 2; ++rank) {
            argmax_pair_merge_ban<<<1, 32>>>(
                d_pairs + rank * ROWS * 4, d_pairs + (1 - rank) * ROWS * 4, d_out + rank * ROWS,
                ROWS, s.shard, rank, ban_lo, 0);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<unsigned> out(2 * ROWS);
        CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
        for (unsigned r = 0; r < ROWS; ++r) {
            const unsigned want = reference(vals, masks, s, r, (ban_lo >> r) & 1u);
            for (unsigned rank = 0; rank < 2; ++rank) {
                ++checked;
                if (out[rank * ROWS + r] != want && bad++ < 10)
                    fprintf(stderr, "case %d vocab %u row %u rank %u: got %u want %u\n", c, s.v,
                            r, rank, out[rank * ROWS + r], want);
            }
        }
        std::vector<unsigned> onesp(2 * ROWS * 4);
        CK(cudaMemcpy(onesp.data(), d_pairs + 2 * ROWS * 4, onesp.size() * 4,
                      cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < onesp.size(); ++i) {
            ++checked;
            if (onesp[i] != refp[i] && bad++ < 10)
                fprintf(stderr, "case %d ones-mask pair word %zu: %08x vs unmasked %08x\n", c,
                        i, onesp[i], refp[i]);
        }
    }
    if (bad) {
        printf("FAIL glm_strict_split_argmax_bitcheck: %ld of %ld checks\n", bad, checked);
        return 1;
    }
    printf("PASS glm_strict_split_argmax_bitcheck: %d cases x %u rows (%d with a mid-word "
           "split), %ld checks (masked TP2 argmax == single-GPU masked argmax; all-ones mask "
           "== unmasked)\n",
           cases, ROWS, unaligned, checked);
    return 0;
}
