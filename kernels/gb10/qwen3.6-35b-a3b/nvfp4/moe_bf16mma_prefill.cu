// SPDX-License-Identifier: AGPL-3.0-only
//
// NVFP4 MoE prefill GEMMs on BF16 tensor cores with register dequant.
//
// Why: the FP8 kernels in moe_w4a16_grouped_gemm.cu dequantize every B tile
// into shared memory between two CTA barriers and then run M64 tiles that are
// half empty at ~33 rows per expert (one score pass of ~1k rows, top-8 of 256).
// The dequant does not amortize and the tensor cores idle. Here:
//
//  * Weights are the MMA A operand (16 output features x k16) and tokens the
//    B operand (k16 x 8 tokens), so tokens are padded to 8, not 64.
//  * Each warp owns its features; every weight value is dequantized once per
//    CTA, straight from shared memory into registers (bit placement + one
//    bf16x2 multiply), and reused across all token tiles of the chunk. Two
//    128-thread halves with their own features share one staged token tile,
//    which halves the token reads from L2.
//  * K order inside a 64-wide stage is permuted so that thread t of a quad
//    owns k = 16t..16t+15 (one NVFP4 scale group) and reads its tokens with
//    two 16-byte loads per token tile. Features are permuted so that a
//    thread's four MMA rows are four adjacent bytes of a weight row. A and B
//    use the same K permutation, so the dot products are unchanged.
//  * Weights are the transposed prefill tables ([K/2, N] e2m1, [K/16, N] e4m3
//    scales, as the FP8 kernels read): the CTAs of one expert stream adjacent
//    columns of the same rows, which the checkpoint [N, K/2] layout cannot do
//    (measured 1.4 vs 2.3 ms for the same bytes at 1k rows).
//
// Exactness: an e2m1 value times an e4m3 block scale has at most 6 significant
// bits, so it is exact in BF16. The bit placement yields e2m1 * 2^-126 (BF16
// subnormal for 0.5); the scale is pre-multiplied by 2^118 so the product is
// e2m1 * scale * 2^-8, a normal BF16, exactly. The FP32 global scale and the
// 2^8 are applied to the FP32 accumulators. Activations stay BF16 (the FP8
// kernels round them to E4M3), so this path is at least as precise.
//
// Grid: (x, y, num_experts), block 256. Feature tiles (128 gate+up features
// or 256 down features) are split evenly over x. An expert's rows are split
// into ceil(M/64) balanced chunks (<= 64 rows); CTA y takes chunks y,
// y + gridDim.y, ...

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Two stages of two feature halves at 2 CTAs/SM (16 warps): at 4k rows,
// 2.22 ms vs 2.81 for 3 stages of one half at 2 CTAs/SM and 2.71 for 3
// stages of two halves at 1 CTA/SM (gate/up, isolated, 256 experts).
#define BM_STAGES 2
#define BM_HV 2        // feature halves per CTA (128 threads each)
#define BM_ROWS 64
#define BM_KS 64
#define BM_TOK_LD 72   // bf16 per token row in smem: 144 B, conflict-free LDS.128

// Weight and scale rows are 128 bytes (eight 16-byte chunks); chunk c of row
// r lives at chunk c ^ bm_swz(r) so that the four quad lanes t, which read
// rows 8t + i (weights) or t (scales), hit four different chunks.
// HV feature halves share one token tile: half h holds features h*FEAT.. .
template <int HV>
struct __align__(16) BmStage {
    __nv_bfloat16 tok[BM_ROWS][BM_TOK_LD];      // 9216 B
    unsigned char w[HV][BM_KS / 2][128];        // 4096 B each: [k pair][feature]
    unsigned char s[HV][BM_KS / 16][128];       //  512 B each: [group][feature]
};

__device__ __forceinline__ unsigned int bm_swz_w(unsigned int kp) { return ((kp >> 3) & 3) << 1; }
__device__ __forceinline__ unsigned int bm_swz_s(unsigned int gr) { return (gr & 3) << 1; }

__device__ __forceinline__ void bm_cp16(void* dst, const void* src, bool pred) {
    unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;"
                 :: "r"(d), "l"(src), "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void bm_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void bm_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// Four e4m3 block scales -> bf16x2 (both halves) of scale * 2^118. Exact.
__device__ __forceinline__ void bm_scales(unsigned int s4, unsigned int out[4]) {
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)((s4 >> (16 * i)) & 0xFFFF), __NV_E4M3);
        const float2 v = __half22float2(__half2(h));
        const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x * 0x1p118f, v.x * 0x1p118f);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(v.y * 0x1p118f, v.y * 0x1p118f);
        out[2 * i] = *(const unsigned int*)&lo;
        out[2 * i + 1] = *(const unsigned int*)&hi;
    }
}

__device__ __forceinline__ unsigned int bm_mul_bf16x2(unsigned int a, unsigned int s) {
    unsigned int d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(s));
    return d;
}

// Four e2m1 pairs (bytes B0..B3 of w, low nibble = even k) -> four bf16x2
// (low half = even k) of value * 2^-126, byte i times scale s[i].
__device__ __forceinline__ void bm_dequant4(unsigned int w, const unsigned int s[4], unsigned int out[4]) {
    const unsigned int v = w << 4;
    const unsigned int z01 = __byte_perm(w, v, 0x0145);
    const unsigned int z23 = __byte_perm(w, v, 0x2367);
    const unsigned int S = 0x80008000u, M = 0x01C001C0u;
    out[0] = bm_mul_bf16x2((z01 & S) | ((z01 >> 6) & M), s[0]);
    out[1] = bm_mul_bf16x2(((z01 << 8) & S) | ((z01 << 2) & M), s[1]);
    out[2] = bm_mul_bf16x2((z23 & S) | ((z23 >> 6) & M), s[2]);
    out[3] = bm_mul_bf16x2(((z23 << 8) & S) | ((z23 << 2) & M), s[3]);
}

__device__ __forceinline__ void bm_mma(float c[4], unsigned int a0, unsigned int a1,
                                       unsigned int a2, unsigned int a3,
                                       unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The four weight bytes (one per MMA row slot) of thread (w, g) at row `r` of
// a 128-byte staged row: FUSED stages [gate 64 | up 64] and gives
// gate[16w + 2g + {0,1}], up[16w + 2g + {0,1}]; otherwise features
// 32w + 4g + {0..3}. Slot order: frag0 row g, frag0 row g+8, frag1 row g,
// frag1 row g+8.
template <bool FUSED>
__device__ __forceinline__ unsigned int bm_slots(const unsigned char* row, unsigned int swz,
                                                 unsigned int warp, unsigned int g) {
    if (FUSED) {
        const unsigned int o = 16 * warp + 2 * g;
        const unsigned int gate = *(const unsigned short*)&row[((o >> 4) ^ swz) * 16 + (o & 15)];
        const unsigned int up = *(const unsigned short*)&row[((4 + (o >> 4)) ^ swz) * 16 + (o & 15)];
        return gate | (up << 16);
    }
    const unsigned int o = 32 * warp + 4 * g;
    return *(const unsigned int*)&row[((o >> 4) ^ swz) * 16 + (o & 15)];
}

// One 64-wide K stage for TT token tiles into acc[f][0..TT).
template <bool FUSED, int TT>
__device__ __forceinline__ void bm_stage(const __nv_bfloat16 (*tok)[BM_TOK_LD],
                                         const unsigned char (*w)[128], const unsigned char (*sg)[128],
                                         float (&acc)[2][8][4],
                                         unsigned int warp, unsigned int g, unsigned int t) {
    unsigned int sc[4];
    bm_scales(bm_slots<FUSED>(sg[t], bm_swz_s(t), warp, g), sc);
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        // k16 steps j = 2h, 2h+1 use k pairs 8t + 2j (slots 2t, 2t+1) and
        // 8t + 2j + 1 (slots 2t+8, 2t+9).
        uint4 b[TT];
        #pragma unroll
        for (int tt = 0; tt < TT; tt++)
            b[tt] = *(const uint4*)&tok[tt * 8 + g][16 * t + 8 * h];
        unsigned int d[4][4];   // [k pair 8t + 4h + i][slot]
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned int kp = 8 * t + 4 * h + i;
            bm_dequant4(bm_slots<FUSED>(w[kp], bm_swz_w(kp), warp, g), sc, d[i]);
        }
        #pragma unroll
        for (int tt = 0; tt < TT; tt++) {
            #pragma unroll
            for (int f = 0; f < 2; f++) {
                bm_mma(acc[f][tt], d[0][2 * f], d[0][2 * f + 1], d[1][2 * f], d[1][2 * f + 1], b[tt].x, b[tt].y);
                bm_mma(acc[f][tt], d[2][2 * f], d[2][2 * f + 1], d[3][2 * f], d[3][2 * f + 1], b[tt].z, b[tt].w);
            }
        }
    }
}

// Per half, FUSED: 64 features of gate and of up, output silu(gate)*up,
// N = inter, one tile per CTA. Otherwise 128 features of one matrix; a CTA
// covers N / (128 * HV * gridDim.x) such tiles back to back for the same rows,
// on one cp.async pipeline (the grid picks it: fewer, longer CTAs amortize the
// pipeline fill when K is short).
template <bool FUSED, int HV>
__device__ __forceinline__ void bm_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ p0, const unsigned long long* __restrict__ s0,
    const float* __restrict__ g0,
    const unsigned long long* __restrict__ p1, const unsigned long long* __restrict__ s1,
    const float* __restrict__ g1,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K)
{
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e];
    const int M = expert_offsets[e + 1] - m_start;
    if (M <= 0) return;
    const int nchunks = (M + BM_ROWS - 1) / BM_ROWS;
    if ((int)blockIdx.y >= nchunks) return;
    const int cs = (M + nchunks - 1) / nchunks;

    const unsigned char* P0 = (const unsigned char*)p0[e];
    const unsigned char* S0 = (const unsigned char*)s0[e];
    const unsigned char* P1 = FUSED ? (const unsigned char*)p1[e] : P0;
    const unsigned char* S1 = FUSED ? (const unsigned char*)s1[e] : S0;
    if (P0 == nullptr || (FUSED && P1 == nullptr)) return;

    constexpr unsigned int FEAT = FUSED ? 64 : 128;      // features per half
    constexpr unsigned int NT = 128 * HV;                // threads
    const unsigned int xt = N / (FEAT * HV * gridDim.x);
    const unsigned int n0 = blockIdx.x * FEAT * HV * xt;
    const unsigned int tid = threadIdx.x;
    // Each 128-thread half loads and multiplies its own FEAT features.
    const unsigned int half = tid >> 7, ht = tid & 127;
    const unsigned int warp = ht >> 5, lane = tid & 31;
    const unsigned int g = lane >> 2, t = lane & 3;
    const float k0 = g0[e] * 256.0f;
    const float k1 = FUSED ? g1[e] * 256.0f : k0;

    __shared__ BmStage<HV> st[BM_STAGES];

    // Weight loader: two 16-byte chunks of k-pair row `wkp`; scale loader
    // (threads 0..31): one chunk of group row `sgr`. FUSED chunks 0..3 are
    // gate, 4..7 up.
    const unsigned int nh = n0 + half * FEAT;
    const unsigned int wkp = FUSED ? (ht & 63) >> 1 : ht >> 2;
    const unsigned int wc = FUSED ? (ht >> 6) * 4 + (ht & 1) * 2 : (ht & 3) * 2;
    const unsigned char* w_src = ((FUSED && wc >= 4) ? P1 : P0) + nh + (wc & (FUSED ? 3 : 7)) * 16
                                 + (unsigned long long)wkp * N;
    const unsigned int sgr = (ht >> 3) & 3, sc_c = ht & 7;
    const unsigned char* s_src = ((FUSED && sc_c >= 4) ? S1 : S0) + nh + (sc_c & (FUSED ? 3 : 7)) * 16
                                 + (unsigned long long)sgr * N;
    const int ksteps = K / BM_KS;

    // A short grid strides the chunks; every chunk is computed by exactly one
    // CTA with the same K order, so the grid size does not change results.
    for (int chunk = blockIdx.y; chunk < nchunks; chunk += gridDim.y) {
        const int r0 = chunk * cs;
        const int rows = min(cs, M - r0);
        const int tt_valid = (rows + 7) >> 3;
        __syncthreads();   // the previous chunk's last stage is still being read

        // Token loader: row tid / TPR, 8 / TPR chunks of its 128-byte stage
        // slice each.
        constexpr int TPR = 2 * HV, CPT = 8 / TPR;
        const int lr = tid / TPR;
        const bool tok_live = lr < tt_valid * 8;
        const bool tok_valid = lr < rows;
        long long tok_row = m_start + r0 + (tok_valid ? lr : 0);
        if (sorted_token_ids != nullptr) tok_row = sorted_token_ids[tok_row];
        const unsigned int tc0 = (tid % TPR) * CPT * 8;
        const __nv_bfloat16* a_src = A + tok_row * (long long)K + tc0;

        // Stage s: feature tile s / ksteps, k = (s % ksteps) * 64.
        auto issue = [&](int buf, int s) {
            const unsigned int x = s / ksteps, kb = (s % ksteps) * BM_KS;
            if (tok_live) {
                #pragma unroll
                for (int c = 0; c < CPT; c++)
                    bm_cp16(&st[buf].tok[lr][tc0 + c * 8], a_src + kb + c * 8, tok_valid);
            }
            const unsigned char* ws = w_src + x * FEAT * HV + (unsigned long long)(kb / 2) * N;
            const unsigned int sw = bm_swz_w(wkp);
            bm_cp16(&st[buf].w[half][wkp][(wc ^ sw) * 16], ws, true);
            bm_cp16(&st[buf].w[half][wkp][((wc + 1) ^ sw) * 16], ws + 16, true);
            if (ht < 32)
                bm_cp16(&st[buf].s[half][sgr][(sc_c ^ bm_swz_s(sgr)) * 16],
                        s_src + x * FEAT * HV + (unsigned long long)(kb / 16) * N, true);
        };

        float acc[2][8][4];
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int i = 0; i < 8; i++)
                acc[f][i][0] = acc[f][i][1] = acc[f][i][2] = acc[f][i][3] = 0.0f;

        const int nstages = xt * ksteps;
        #pragma unroll
        for (int s = 0; s < BM_STAGES - 1; s++) {
            if (s < nstages) issue(s, s);
            bm_commit();
        }

        for (int s = 0; s < nstages; s++) {
            bm_wait<BM_STAGES - 2>();
            __syncthreads();
            {
                const int ns = s + BM_STAGES - 1;
                if (ns < nstages) issue(ns % BM_STAGES, ns);
                bm_commit();
            }
            // Token-tile count as a template argument: with the tiles' loads
            // unconditional, they issue ahead of the MMAs instead of one
            // dependent load per tile behind a branch.
            const BmStage<HV>& sb = st[s % BM_STAGES];
            const auto* sw = sb.w[half];
            const auto* ss = sb.s[half];
            switch (tt_valid) {
                case 1: bm_stage<FUSED, 1>(sb.tok, sw, ss, acc, warp, g, t); break;
                case 2: bm_stage<FUSED, 2>(sb.tok, sw, ss, acc, warp, g, t); break;
                case 3: bm_stage<FUSED, 3>(sb.tok, sw, ss, acc, warp, g, t); break;
                case 4: bm_stage<FUSED, 4>(sb.tok, sw, ss, acc, warp, g, t); break;
                case 5: bm_stage<FUSED, 5>(sb.tok, sw, ss, acc, warp, g, t); break;
                case 6: bm_stage<FUSED, 6>(sb.tok, sw, ss, acc, warp, g, t); break;
                case 7: bm_stage<FUSED, 7>(sb.tok, sw, ss, acc, warp, g, t); break;
                default: bm_stage<FUSED, 8>(sb.tok, sw, ss, acc, warp, g, t); break;
            }
            if (FUSED || (s + 1) % ksteps != 0) continue;

            // Down tile done: store from registers. Thread (g, t) holds tokens
            // 2t, 2t+1 of each token tile, features 32w + 4g + {0..3} of each.
            const unsigned int nx = n0 + (s / ksteps) * FEAT * HV + half * FEAT;
            #pragma unroll
            for (int tt = 0; tt < 8; tt++) {
                #pragma unroll
                for (int p = 0; p < 2; p++) {
                    const int r = tt * 8 + 2 * t + p;
                    if (tt < tt_valid && r < rows) {
                        __nv_bfloat162 lo = __floats2bfloat162_rn(acc[0][tt][p] * k0, acc[0][tt][2 + p] * k0);
                        __nv_bfloat162 hi = __floats2bfloat162_rn(acc[1][tt][p] * k1, acc[1][tt][2 + p] * k1);
                        *(uint2*)&C[(unsigned long long)(m_start + r0 + r) * N + nx + 32 * warp + 4 * g] =
                            make_uint2(*(unsigned int*)&lo, *(unsigned int*)&hi);
                    }
                }
                #pragma unroll
                for (int f = 0; f < 2; f++)
                    acc[f][tt][0] = acc[f][tt][1] = acc[f][tt][2] = acc[f][tt][3] = 0.0f;
            }
        }
        bm_wait<0>();
        if (!FUSED) continue;

        // gate/up (one feature tile, xt == 1): SiLU·mul staged through shared
        // memory as [token][feature], then whole 256-byte rows. Measured faster
        // than 4-byte register stores (1.27 vs 1.34 ms at 1k rows).
        __syncthreads();
        constexpr unsigned int OLD = FEAT * HV + 8;
        __nv_bfloat16* os = (__nv_bfloat16*)&st[0];
        #pragma unroll
        for (int tt = 0; tt < 8; tt++) {
            if (tt < tt_valid) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const unsigned int tok = tt * 8 + 2 * t + (i & 1);
                    // The SiLU reads BF16 gate/up, as the unfused chain does.
                    const float gv = __bfloat162float(__float2bfloat16(acc[0][tt][i] * k0));
                    const float uv = __bfloat162float(__float2bfloat16(acc[1][tt][i] * k1));
                    os[tok * OLD + half * FEAT + 16 * warp + 2 * g + (i >> 1)] =
                        __float2bfloat16(gv * (1.0f / (1.0f + __expf(-gv))) * uv);
                }
            }
        }
        __syncthreads();
        for (unsigned int i = tid; i < (unsigned int)rows * (FEAT * HV / 8); i += NT) {
            const unsigned int r = i / (FEAT * HV / 8), c = i % (FEAT * HV / 8);
            *(uint4*)&C[(unsigned long long)(m_start + r0 + r) * N + n0 + c * 8] =
                *(const uint4*)&os[r * OLD + c * 8];
        }
    }
}

// gate+up+SiLU: A token-major [T, K] gathered by sorted_token_ids; weights the
// transposed [K/2, N] tables; C sorted [total_expanded, N] = silu(gate) * up.
// Grid x = N / 128 (one feature tile per CTA), block 256.
extern "C" __global__ void __launch_bounds__(128 * BM_HV, 2)
moe_bf16mma_gate_up_silu(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K)
{
    bm_body<true, BM_HV>(A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals,
                  up_packed_ptrs, up_scale_ptrs, up_scale2_vals, C,
                  expert_offsets, sorted_token_ids, num_experts, N, K);
}

// down: A sorted [total_expanded, K]; weights the transposed [K/2, N] table;
// C sorted [total_expanded, N]. Grid x divides N / 256, block 256.
extern "C" __global__ void __launch_bounds__(128 * BM_HV, 2)
moe_bf16mma_down(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    unsigned int num_experts, unsigned int N, unsigned int K)
{
    bm_body<false, BM_HV>(A, packed_ptrs, scale_ptrs, scale2_vals,
                   nullptr, nullptr, nullptr, C,
                   expert_offsets, nullptr, num_experts, N, K);
}
