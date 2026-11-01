// SPDX-License-Identifier: AGPL-3.0-only
// Standalone check of the DFlash2 candidate selector's confidence twin
// (kernels/gb10/glm-5.3-flash/nvfp4/glm_dflash2_selector_conf.cu, the common
// kernel compiled with DF2_SEL_CONF) against the production kernel at the
// GLM-5.3-Flash DFlash2 shape (gamma 8, vocab 154880, rank 256, top-k 16):
//   * every draft token equal, bit for bit, over random blocks with ties,
//     random anchors and min_tokens ban depths with end tokens among the
//     top candidates;
//   * the production kernel writes exactly gamma words, the twin exactly
//     2 * gamma (guard words past both stay untouched);
//   * each confidence equals -log(sum exp(score - best)) of a host replay of
//     the row's scores (same float operation order as the kernel) within
//     1e-4, is <= 0, and the anchor row's word is 0;
//   * a second launch on the same scratch repeats the first (ticket reset).
//
//   nvcc -arch=sm_121a -O3 --fmad=false scripts/dev/dflash2_selector_conf_bitcheck.cu \
//        -o dflash2_selector_conf_bitcheck
//   ./dflash2_selector_conf_bitcheck [blocks=24]
// Prints one line per block and "PASS" or "FAIL"; exit status 0 on PASS.
// Device memory: about 170 MB. Runs in a few seconds.
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace base {
#include "../../kernels/gb10/common/dflash2_candidate_selector.cu"
}
#define DF2_SEL_CONF 1
#define dflash2_candidate_selector dflash2_candidate_selector_conf
namespace conf {
#include "../../kernels/gb10/common/dflash2_candidate_selector.cu"
}
#undef dflash2_candidate_selector

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned GAMMA = 8, VOCAB = 154880, RANK = 256, TOPK = 16, SPLITS = 96 / GAMMA;
static const unsigned GUARD = 0xA5A5A5A5u;

static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;
static unsigned rnd() {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (unsigned)(rng_state >> 33);
}
static float unit() { return (rnd() & 0xFFFFFF) / 8388608.0f - 1.0f; }  // [-1, 1)
static unsigned short to_bf(float f) { bf b = __float2bfloat16(f); unsigned short u; memcpy(&u, &b, 2); return u; }
static float from_bf(unsigned short u) { bf b; memcpy(&b, &u, 2); return __bfloat162float(b); }

// The row's top-k under the kernel's total order: value descending, index ascending.
static void topk(const unsigned short* row, unsigned* ids, float* vals) {
    for (unsigned k = 0; k < TOPK; k++) { ids[k] = 0xFFFFFFFFu; vals[k] = -INFINITY; }
    for (unsigned i = 0; i < VOCAB; i++) {
        const float v = from_bf(row[i]);
        if (!(v > vals[TOPK - 1] || (v == vals[TOPK - 1] && i < ids[TOPK - 1]))) continue;
        unsigned k = TOPK - 1;
        while (k > 0 && (v > vals[k - 1] || (v == vals[k - 1] && i < ids[k - 1]))) {
            vals[k] = vals[k - 1]; ids[k] = ids[k - 1]; k--;
        }
        vals[k] = v; ids[k] = i;
    }
}

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? atoi(argv[1]) : 24;
    std::vector<unsigned short> h_logits((size_t)GAMMA * VOCAB), h_proj((size_t)GAMMA * RANK);
    std::vector<unsigned short> h_pred((size_t)VOCAB * RANK), h_succ((size_t)VOCAB * RANK);
    for (auto& x : h_pred) x = to_bf(unit() * 0.5f);
    for (auto& x : h_succ) x = to_bf(unit() * 0.5f);
    const size_t scratch_bytes = 16 + 2 * 4 * (size_t)GAMMA * 16 * 16;
    bf *d_logits, *d_proj, *d_pred, *d_succ;
    unsigned *d_out[2], *d_anchor, *d_ban, *d_scratch;
    CK(cudaMalloc(&d_logits, h_logits.size() * 2));
    CK(cudaMalloc(&d_proj, h_proj.size() * 2));
    CK(cudaMalloc(&d_pred, h_pred.size() * 2));
    CK(cudaMalloc(&d_succ, h_succ.size() * 2));
    for (auto& o : d_out) CK(cudaMalloc(&o, 3 * GAMMA * 4));
    CK(cudaMalloc(&d_anchor, 4));
    CK(cudaMalloc(&d_ban, 4));
    CK(cudaMalloc(&d_scratch, scratch_bytes));
    CK(cudaMemset(d_scratch, 0, scratch_bytes));
    CK(cudaMemcpy(d_pred, h_pred.data(), h_pred.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_succ, h_succ.data(), h_succ.size() * 2, cudaMemcpyHostToDevice));

    int bad = 0;
    double worst = 0.0;
    for (int block = 0; block < blocks; block++) {
        // Logits: a broad field, plus per row a cluster of near-ties (and one
        // exact tie) so the selector's scores are close and the softmax mass
        // is spread; every third block is peaked instead (a sure pick).
        const bool peaked = block % 3 == 2;
        for (auto& x : h_logits) x = to_bf(unit() * 4.0f);
        unsigned end_ids[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        for (unsigned r = 0; r < GAMMA; r++) {
            unsigned short* row = h_logits.data() + (size_t)r * VOCAB;
            for (int c = 0; c < 20; c++) row[rnd() % VOCAB] = to_bf(9.0f + unit() * (peaked ? 0.25f : 2.0f));
            const unsigned a = rnd() % VOCAB, b = rnd() % VOCAB;
            row[a] = row[b] = to_bf(11.0f);
            if (peaked) row[rnd() % VOCAB] = to_bf(24.0f);
            if (r == 1 || r == 2) end_ids[r - 1] = a;  // an end token at the top of rows 1 and 2
        }
        for (auto& x : h_proj) x = to_bf(unit() * (block % 4 == 0 ? 0.0f : 1.5f));
        const unsigned anchor = block == 1 ? VOCAB + 7 : rnd() % VOCAB;  // out of range clamps to 0
        const unsigned ban = block % 4;                                  // rows 1..=ban skip end tokens
        CK(cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_proj, h_proj.data(), h_proj.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_anchor, &anchor, 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ban, &ban, 4, cudaMemcpyHostToDevice));

        unsigned out[2][2][3 * GAMMA];
        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k < 2; k++) {
                unsigned init[3 * GAMMA];
                for (auto& w : init) w = GUARD;
                CK(cudaMemcpy(d_out[k], init, sizeof init, cudaMemcpyHostToDevice));
                if (k == 0) {
                    base::dflash2_candidate_selector<<<dim3(SPLITS, GAMMA, 1), 512>>>(
                        d_logits, d_proj, d_pred, d_succ, d_out[k], d_anchor, d_ban, GAMMA, VOCAB, RANK, TOPK,
                        end_ids[0], end_ids[1], end_ids[2], end_ids[3], d_scratch);
                } else {
                    conf::dflash2_candidate_selector_conf<<<dim3(SPLITS, GAMMA, 1), 512>>>(
                        d_logits, d_proj, d_pred, d_succ, d_out[k], d_anchor, d_ban, GAMMA, VOCAB, RANK, TOPK,
                        end_ids[0], end_ids[1], end_ids[2], end_ids[3], d_scratch);
                }
                CK(cudaGetLastError());
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(out[pass][k], d_out[k], sizeof init, cudaMemcpyDeviceToHost));
            }
        }
        int fail = 0;
        fail |= memcmp(out[0], out[1], sizeof out[0]) != 0;                         // relaunch repeats
        fail |= memcmp(out[0][0], out[0][1], GAMMA * 4) != 0;                       // same picks
        for (unsigned w = GAMMA; w < 3 * GAMMA; w++) fail |= out[0][0][w] != GUARD;   // base: gamma words
        for (unsigned w = 2 * GAMMA; w < 3 * GAMMA; w++) fail |= out[0][1][w] != GUARD;

        // Host replay of the chain's scores and confidences.
        float got[GAMMA];
        memcpy(got, &out[0][1][GAMMA], sizeof got);
        fail |= got[0] != 0.0f;
        unsigned prev = anchor >= VOCAB ? 0 : anchor;
        double block_worst = 0.0;
        float min_conf = 0.0f;
        for (unsigned r = 1; r < GAMMA; r++) {
            unsigned ids[TOPK];
            float vals[TOPK], score[TOPK];
            topk(h_logits.data() + (size_t)r * VOCAB, ids, vals);
            float best = 0.0f;
            unsigned pick = ids[0];
            for (unsigned c = 0; c < TOPK; c++) {
                float lane[32];
                for (unsigned l = 0; l < 32; l++) {
                    float dot = 0.0f;
                    for (unsigned k = l; k < RANK; k += 32) {
                        // volatile: no fused multiply-add on the host, as --fmad=false on the device.
                        volatile float ctx = from_bf(h_pred[(size_t)prev * RANK + k]) * from_bf(h_proj[(size_t)r * RANK + k]);
                        volatile float term = ctx * from_bf(h_succ[(size_t)ids[c] * RANK + k]);
                        dot += term;
                    }
                    lane[l] = dot;
                }
                for (unsigned off = 16; off > 0; off /= 2)
                    for (unsigned l = 0; l < off; l++) lane[l] += lane[l + off];
                const bool banned = r <= ban && (ids[c] == end_ids[0] || ids[c] == end_ids[1] ||
                                                 ids[c] == end_ids[2] || ids[c] == end_ids[3]);
                score[c] = banned ? -INFINITY : vals[c] + lane[0];
                if (c == 0 || score[c] > best) { best = score[c]; pick = ids[c]; }
            }
            double mass = 0.0;
            for (unsigned c = 0; c < TOPK; c++) mass += exp((double)score[c] - (double)best);
            const double want = -log(mass);
            fail |= pick != out[0][1][r];
            fail |= !(got[r] <= 0.0f) || fabs(got[r] - want) > 1e-4 * (1.0 + fabs(want));
            if (fabs(got[r] - want) > block_worst) block_worst = fabs(got[r] - want);
            if (got[r] < min_conf) min_conf = got[r];
            prev = pick;
        }
        if (block_worst > worst) worst = block_worst;
        bad += fail;
        printf("block %2d ban=%u %s tokens %s  min conf %.4f  max |conf - host| %.2e\n", block, ban,
               peaked ? "peaked" : "spread", fail ? "MISMATCH" : "equal", min_conf, block_worst);
    }
    printf("%s: %d blocks, %d bad, worst |conf - host| %.2e\n", bad ? "FAIL" : "PASS", blocks, bad, worst);
    return bad ? 1 : 0;
}
