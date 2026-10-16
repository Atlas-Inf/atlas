// SPDX-License-Identifier: AGPL-3.0-only
//
// Block-aligned W8A16 GEMV row compute for strix-hip — ONE shared device
// function used by w8a16_gemv (M=1) and w8a16_gemv_batch{4,16}.
//
// Tiling: 64 threads per output row. Each pass covers 16 FP8 scale blocks
// (2048 k-elements): lane l handles TWO consecutive k16 chunks inside one
// 128-wide scale block, so the block scale is loaded once per lane/pass
// instead of once per 16 elements:
//
//   j   = (l >> 2) + pass * (threads_per_out >> 2)   // scale block index
//   k16 = j * 8 + (l & 3) * 2                        // lane's 2 chunks in j
//
// Per pass the lane loads 2x16B of B (32 FP8 weights) plus 4x16B of A per
// row, applies scale, FMA-accumulates in increasing-k order, and adds the
// pass partial into the per-row accumulator. Lane idle when j >= k_blocks.
//
// The per-lane k-ownership, per-row accumulation order and FMA usage are
// identical across M=1 and the batch twins (twin rule). NOT bit-identical
// to the legacy strided kernel — allowed under the accuracy gate.

#ifndef ATLAS_W8A16_ROW_TILED_CUH
#define ATLAS_W8A16_ROW_TILED_CUH

__device__ __constant__ float E4M3_LUT_T[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};

// Decode+scale 32 FP8 bytes into wf[32] (w = LUT[b] * scale).
__device__ __forceinline__ void w8a16_row_dequant32(
    const uint4 b0,
    const uint4 b1,
    const float scale,
    const float* __restrict__ s_lut,
    float* __restrict__ wf
) {
    const unsigned int b_raw[8] = {b0.x, b0.y, b0.z, b0.w,
                                   b1.x, b1.y, b1.z, b1.w};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned int w32 = b_raw[i];
        wf[i * 4 + 0] = s_lut[(w32      ) & 0xFF] * scale;
        wf[i * 4 + 1] = s_lut[(w32 >>  8) & 0xFF] * scale;
        wf[i * 4 + 2] = s_lut[(w32 >> 16) & 0xFF] * scale;
        wf[i * 4 + 3] = s_lut[(w32 >> 24) & 0xFF] * scale;
    }
}

// MAC 32 weights against A row t at k16 chunks {k16a, k16a+1}: two uint4
// loads, 16 BF16 pairs each, FMA-accumulate in order.
__device__ __forceinline__ void w8a16_row_mac32(
    const __nv_bfloat16* __restrict__ At,
    const float* __restrict__ wf,
    const unsigned int k16a,
    float& acc
) {
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        const uint4 a_lo = ((const uint4*)At)[(k16a + c) * 2];
        const uint4 a_hi = ((const uint4*)At)[(k16a + c) * 2 + 1];
        const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                    a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            __nv_bfloat16 lo, hi;
            *(unsigned short*)&lo = (unsigned short)(ar[j] & 0xFFFF);
            *(unsigned short*)&hi = (unsigned short)(ar[j] >> 16);
            acc = fmaf(__bfloat162float(lo), wf[c * 16 + j * 2], acc);
            acc = fmaf(__bfloat162float(hi), wf[c * 16 + j * 2 + 1], acc);
        }
    }
}

// Whole-row partial for one output n: all passes, per-lane accumulator.
// A: [M,K] bf16 (M<=MAX_M); B: [N,K] FP8; block_scale [ceil(N/128), ceil(K/128)].
// K must be a multiple of 32 (asserted by callers' shapes).
template <int MAX_M>
__device__ __forceinline__ void w8a16_row_tiled(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    const float* __restrict__ s_lut,
    const unsigned int n,
    const unsigned int n_block,
    const unsigned int k_blocks,
    const unsigned int M,
    const unsigned int K,
    const unsigned int lane,
    const unsigned int lanes_per_out,
    float* __restrict__ acc            // [MAX_M] per-row accumulators
) {
    const unsigned int blocks_per_pass = lanes_per_out >> 2;      // 16
    const unsigned int passes = (k_blocks + blocks_per_pass - 1) / blocks_per_pass;
    const unsigned int lane_blk = lane >> 2;
    const unsigned int lane_off = (lane & 3u) * 2;

    // Process passes in pairs: both weight loads + scale loads issue
    // before any arithmetic (ILP), and the second block's k-range is
    // contiguously after the first — acc sees the identical per-row order.
    for (unsigned int p = 0; p < passes; p += 2) {
        const unsigned int ja = lane_blk + p * blocks_per_pass;
        const unsigned int jb = lane_blk + (p + 1) * blocks_per_pass;
        if (ja >= k_blocks) break;
        const unsigned int ka = ja * 8 + lane_off;
        const float scale_a = block_scale[n_block * k_blocks + ja];
        const uint4 ba0 = ((const uint4*)(B + (unsigned long long)n * K))[ka];
        const uint4 ba1 = ((const uint4*)(B + (unsigned long long)n * K))[ka + 1];
        const bool have_b = jb < k_blocks;
        const unsigned int kb = jb * 8 + lane_off;
        float scale_b = 0.0f;
        uint4 bb0 = make_uint4(0, 0, 0, 0), bb1 = bb0;
        if (have_b) {
            scale_b = block_scale[n_block * k_blocks + jb];
            bb0 = ((const uint4*)(B + (unsigned long long)n * K))[kb];
            bb1 = ((const uint4*)(B + (unsigned long long)n * K))[kb + 1];
        }
        float wfa[32];
        w8a16_row_dequant32(ba0, ba1, scale_a, s_lut, wfa);
        float wfb[32];
        if (have_b) w8a16_row_dequant32(bb0, bb1, scale_b, s_lut, wfb);
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            w8a16_row_mac32(A + (unsigned long long)t * K, wfa, ka, acc[t]);
            if (have_b) w8a16_row_mac32(A + (unsigned long long)t * K, wfb, kb, acc[t]);
        }
    }
}

#endif // ATLAS_W8A16_ROW_TILED_CUH
