// SPDX-License-Identifier: AGPL-3.0-only
//
// W4A8 integer-DP4A decode GEMV for AMD gfx1151 (RDNA3.5 / Strix Halo).
//
// This is the strix-hip-only DP4A decode path. It is ADDITIVE: the float
// E2M1-LUT path in w4a16_gemv.cu is untouched and remains the gb10/NVIDIA
// default. Dispatch selects these kernels only on strix-hip behind a flag.
//
// Why: Atlas decode GEMV currently dequantizes 4-bit weights to float and
// accumulates in FP32 against BF16 activations (one FMA per weight). On the
// bandwidth-bound LPDDR5X (~256 GB/s) gfx1151 part the win comes from (a) cutting
// activation traffic (int8 not bf16) and (b) replacing 4 FP32 FMAs with one
// hardware `v_dot4` (`__builtin_amdgcn_sudot4`) — 4 int8xint8 MACs/instruction.
//
// FAITHFULNESS to the float path: the NVFP4 weight codebook E2M1_LUT =
// {0,.5,1,1.5,2,3,4,6} is exactly representable as half-integers; multiply by 2
// to get the integer grid {0,1,2,3,4,6,8,12} (+ negatives) and fold the x0.5 into
// the per-group scale. Weights are therefore EXACT in int8; the ONLY new error vs
// W4A16 is the int8 activation quantization (block-q8_1 style, d = amax/127). This
// is the same accuracy/speed trade the rocmfp4-llama reference ships at BFCL parity.
//
//   float path : acc += bf16(a_i) * (E2M1_LUT[nib_i] * wscale_g * scale2)
//   dp4a  path : acc += a_d_g * (wscale_g * 0.5 * scale2) * SUM_i( aq_i * wint_i )
//                where aq_i = round(a_i / a_d_g), a_d_g = amax_g/127,
//                      wint_i = round(E2M1_LUT[nib_i] * 2)   (exact integer)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Standard E4M3 (1-4-3, bias 7) decode via bit-math — IDENTICAL to w4a16_gemv.cu's
// scl_fp8 (SSOT: the block-scale decode must match the encoder; gfx1151's builtin
// __nv_fp8_e4m3 cast is a non-standard narrow format and corrupts scales).
__device__ __forceinline__ float dp4a_scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;                 // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                                    // NaN -> 0
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

// Integer NVFP4 codebook = E2M1_LUT * 2 (exact). Index = 4-bit nibble (sign in bit3).
__device__ __constant__ signed char DP4A_CODEBOOK[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

#define DP4A_BLOCK_SIZE 256
#define DP4A_N_PER_BLOCK 4
#define DP4A_WARP_SIZE 32
#define DP4A_GROUP_SIZE 16   // one weight scale + one act scale per 16 elements

#if defined(__HIP_PLATFORM_AMD__) || defined(__SCALE__)
#define DP4A_DOT(a, b, c) __builtin_amdgcn_sudot4(true, (a), true, (b), (c), false)
#else
#define DP4A_DOT(a, b, c) __dp4a((a), (b), (c))   // NVIDIA fallback (parity / portability)
#endif

// ── Codebook expansion: 8 packed weight bytes (16 nibbles) → 4 int32 DP4A
// operands holding the SIGNED integer codebook value for each element, in
// Atlas's consecutive-pair layout (element 2b = byte b low nibble, 2b+1 = high).
//
// On AMD (gfx1151) this uses the branchless v_perm expansion grabbed from
// rocmfp4-llama (ggml/rocmfp4/rocmfp4_hip_codebook.cuh) — NO shared-memory
// codebook, NO __syncthreads on the GEMV hot path. The four constants encode the
// SAME grid as DP4A_CODEBOOK ({0,1,2,3,4,6,8,12} + negatives); proven byte-exact
// vs the portable loop for all inputs on gfx1151 (perm_equiv_test.cu). The two
// encodings are an unavoidable consequence of the perm op and are test-locked.
__device__ __forceinline__ unsigned int dp4a_perm_codebook(unsigned int q) {
    const unsigned int values0 = 0x03020100u; // [ 0, 1, 2, 3]
    const unsigned int values1 = 0x0c080604u; // [ 4, 6, 8,12]
    const unsigned int values2 = 0xfdfeff00u; // [ 0,-1,-2,-3]
    const unsigned int values3 = 0xf4f8fafcu; // [-4,-6,-8,-12]
    unsigned int vl = __builtin_amdgcn_perm(values1, values0, q & 0x07070707u);
    unsigned int vh = __builtin_amdgcn_perm(values3, values2, q & 0x07070707u);
    unsigned int m  = 0x03020100u | ((q & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(vh, vl, m);
}

__device__ __forceinline__ void dp4a_expand_codebook(unsigned long long packed8, int wint[4]) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__SCALE__)
    unsigned int w0 = (unsigned int)(packed8 & 0xFFFFFFFFull);          // bytes 0-3
    unsigned int w1 = (unsigned int)((packed8 >> 32) & 0xFFFFFFFFull);  // bytes 4-7
    // deinterleave consecutive nibbles into one magnitude index per byte
    unsigned int na = (w0 & 0xF) | ((w0 & 0xF0) << 4) | ((w0 & 0xF00) << 8) | ((w0 & 0xF000) << 12);
    unsigned int nb = ((w0 >> 16) & 0xF) | (((w0 >> 16) & 0xF0) << 4) | (((w0 >> 16) & 0xF00) << 8) | (((w0 >> 16) & 0xF000) << 12);
    unsigned int nc = (w1 & 0xF) | ((w1 & 0xF0) << 4) | ((w1 & 0xF00) << 8) | ((w1 & 0xF000) << 12);
    unsigned int nd = ((w1 >> 16) & 0xF) | (((w1 >> 16) & 0xF0) << 4) | (((w1 >> 16) & 0xF00) << 8) | (((w1 >> 16) & 0xF000) << 12);
    wint[0] = (int)dp4a_perm_codebook(na);
    wint[1] = (int)dp4a_perm_codebook(nb);
    wint[2] = (int)dp4a_perm_codebook(nc);
    wint[3] = (int)dp4a_perm_codebook(nd);
#else
    // Portable fallback (NVIDIA parity): per-element codebook lookup from constant mem.
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned int packed = 0;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int elem = j * 4 + e;
            int b = elem >> 1;
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            unsigned char nib = (elem & 1) ? (byte_val >> 4) : (byte_val & 0xF);
            packed |= ((unsigned int)(unsigned char)DP4A_CODEBOOK[nib]) << (e * 8);
        }
        wint[j] = (int)packed;
    }
#endif
}

__device__ __forceinline__ int2 dp4a_expand_codebook_d4(unsigned int packed) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__SCALE__)
    return make_int2(
        (int)dp4a_perm_codebook(packed),
        (int)dp4a_perm_codebook(packed >> 4));
#else
    unsigned int even = 0;
    unsigned int odd = 0;
    #pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        const unsigned char pair = (unsigned char)(packed >> (byte * 8));
        even |= ((unsigned int)(unsigned char)DP4A_CODEBOOK[pair & 0x0f]) << (byte * 8);
        odd |= ((unsigned int)(unsigned char)DP4A_CODEBOOK[pair >> 4]) << (byte * 8);
    }
    return make_int2((int)even, (int)odd);
#endif
}

// ── Activation int8 quantizer (once per layer, NOT per GEMV) ──────────────
// Quantizes one BF16 activation row [1,K] to int8 [1,K] with per-16-group symmetric
// scales [K/16]. Grid: (K/16) blocks, block: 16 threads (one per group element).
extern "C" __global__ void quantize_act_int8_g16(
    const __nv_bfloat16* __restrict__ A,   // [1, K] BF16
    signed char* __restrict__ a_q,          // [1, K] int8
    float* __restrict__ a_scale,            // [K/16] f32 per-group scale (= amax/127)
    unsigned int K
) {
    const unsigned int g = blockIdx.x;
    const unsigned int i = g * DP4A_GROUP_SIZE + threadIdx.x;
    if (i >= K) return;

    float a = __bfloat162float(A[i]);
    float aa = fabsf(a);

    // amax over the 16-element group (warp-style reduction over 16 lanes via smem).
    __shared__ float s_amax[DP4A_GROUP_SIZE];
    s_amax[threadIdx.x] = aa;
    __syncthreads();
    #pragma unroll
    for (unsigned int off = DP4A_GROUP_SIZE / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            float o = s_amax[threadIdx.x + off];
            if (o > s_amax[threadIdx.x]) s_amax[threadIdx.x] = o;
        }
        __syncthreads();
    }
    float amax = s_amax[0];
    float d = amax * (1.0f / 127.0f);
    float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;

    int q = (int)rintf(a * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    a_q[i] = (signed char)q;
    if (threadIdx.x == 0) a_scale[g] = d;
}

extern "C" __global__ void quantize_act_int8_g16_batch4_d4(
    const __nv_bfloat16* __restrict__ A,
    signed char* __restrict__ a_q,
    float* __restrict__ a_scale,
    unsigned int M,
    unsigned int K
) {
    const unsigned int group = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int local = threadIdx.x;
    const unsigned int index = group * DP4A_GROUP_SIZE + local;
    if (row >= M || index >= K) return;

    const float value = __bfloat162float(A[(unsigned long long)row * K + index]);
    __shared__ float amax_values[DP4A_GROUP_SIZE];
    amax_values[local] = fabsf(value);
    __syncthreads();
    #pragma unroll
    for (unsigned int step = DP4A_GROUP_SIZE / 2; step > 0; step >>= 1) {
        if (local < step)
            amax_values[local] = fmaxf(amax_values[local], amax_values[local + step]);
        __syncthreads();
    }
    const float scale = amax_values[0] * (1.0f / 127.0f);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    int quantized = (int)rintf(value * inverse);
    quantized = quantized < -127 ? -127 : (quantized > 127 ? 127 : quantized);
    const unsigned int within_half = local & 7u;
    const unsigned int mapped = (local >> 3) * 8 + (within_half & 1u) * 4 + (within_half >> 1);
    a_q[(unsigned long long)row * K + group * DP4A_GROUP_SIZE + mapped] =
        (signed char)quantized;
    if (local == 0)
        a_scale[(unsigned long long)row * (K / DP4A_GROUP_SIZE) + group] = scale;
}

// ── Fused SiLU(gate)*up → int8 activation quantizer (down-proj input prep) ──
// The float FFN fuses silu(gate)*up INTO the down-proj GEMV (w4a16_gemv_silu_input).
// The DP4A down-proj needs its activation as int8 + per-16-group scales, so we
// materialize that activation here ONCE per layer: h_i = silu(gate_i)*up_i, then
// the SAME symmetric block-q8_1 quant (d = amax_g/127) used by quantize_act_int8_g16.
// SSOT: the silu*mul math is bit-identical to w4a16_gemv_silu_input's inline form;
// the quant is identical to quantize_act_int8_g16. Grid: (K/16) blocks, 16 threads.
extern "C" __global__ void silu_mul_quant_int8_g16(
    const __nv_bfloat16* __restrict__ gate,  // [1, K] BF16 gate proj output
    const __nv_bfloat16* __restrict__ up,    // [1, K] BF16 up proj output
    signed char* __restrict__ a_q,            // [1, K] int8
    float* __restrict__ a_scale,              // [K/16] f32 per-group scale (= amax/127)
    unsigned int K
) {
    const unsigned int g = blockIdx.x;
    const unsigned int i = g * DP4A_GROUP_SIZE + threadIdx.x;
    if (i >= K) return;

    // silu(gate)*up — identical to w4a16_gemv_silu_input's per-element activation.
    float gf = __bfloat162float(gate[i]);
    float uf = __bfloat162float(up[i]);
    float h  = (gf / (1.0f + __expf(-gf))) * uf;
    float ha = fabsf(h);

    __shared__ float s_amax[DP4A_GROUP_SIZE];
    s_amax[threadIdx.x] = ha;
    __syncthreads();
    #pragma unroll
    for (unsigned int off = DP4A_GROUP_SIZE / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            float o = s_amax[threadIdx.x + off];
            if (o > s_amax[threadIdx.x]) s_amax[threadIdx.x] = o;
        }
        __syncthreads();
    }
    float amax = s_amax[0];
    float d = amax * (1.0f / 127.0f);
    float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;

    int q = (int)rintf(h * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    a_q[i] = (signed char)q;
    if (threadIdx.x == 0) a_scale[g] = d;
}

// ── DP4A GEMV (correctness v1: element-order codebook expansion) ──────────
// out[n] = SUM_g a_scale[g] * (wscale_g * 0.5 * scale2) * dot(aq[g], wint[g])
extern "C" __global__ void w4a16_gemv_dp4a(
    const signed char* __restrict__ a_q,    // [1, K] int8 activations
    const float* __restrict__ a_scale,      // [K/16] f32 act scales
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8 (2 nibbles/byte)
    const unsigned char* __restrict__ B_scale,    // [N, K/16] FP8-E4M3 weight scales
    const float scale2,
    __nv_bfloat16* __restrict__ C,           // [1, N] BF16 out
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = DP4A_BLOCK_SIZE / DP4A_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * DP4A_N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / DP4A_GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float smem[DP4A_N_PER_BLOCK * 2];

    float acc = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        // 16 int8 activations = one int4 (16 bytes), in element order.
        int4 aq4 = *(const int4*)(a_q + base_k);
        const int aq[4] = {aq4.x, aq4.y, aq4.z, aq4.w};

        // 8 packed weight bytes (16 nibbles).
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);

        // Expand to 16 signed codebook values in ELEMENT order so they align lane-wise
        // with aq[] (element 2b = byte b low nibble, 2b+1 = high; matches float w4a16_gemv).
        // Branchless v_perm on AMD; portable loop on NVIDIA. (see dp4a_expand_codebook)
        int wint[4];
        dp4a_expand_codebook(packed8, wint);

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) sumi = DP4A_DOT(aq[j], wint[j], sumi);

        unsigned int scale_group = base_k / DP4A_GROUP_SIZE;
        float wscale = dp4a_scl_fp8(B_scale[(unsigned long long)n * num_groups + scale_group]);
        float a_d = a_scale[k16];
        acc += (float)sumi * a_d * (wscale * 0.5f * scale2);
    }

    const unsigned int warp_lane = threadIdx.x % DP4A_WARP_SIZE;
    #pragma unroll
    for (int offset = DP4A_WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) smem[local_out * 2 + (lane / DP4A_WARP_SIZE)] = acc;
    __syncthreads();
    if (lane == 0) C[n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
}

extern "C" __global__ void w4a16_gemv_dp4a_batch4_d4(
    const signed char* __restrict__ a_q,
    const float* __restrict__ a_scale,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = DP4A_BLOCK_SIZE / DP4A_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * DP4A_N_PER_BLOCK + local_out;
    const bool valid_n = n < N;
    const unsigned int half_k = K / 2;
    const unsigned int groups = K / DP4A_GROUP_SIZE;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (valid_n) {
        for (unsigned int group = lane; group < groups; group += threads_per_out) {
            const unsigned long long packed = *(const unsigned long long*)(
                B_packed + (unsigned long long)n * half_k + group * 8);
            const int2 first = dp4a_expand_codebook_d4((unsigned int)packed);
            const int2 second = dp4a_expand_codebook_d4((unsigned int)(packed >> 32));
            const int weight[4] = {first.x, first.y, second.x, second.y};
            const float weight_scale = dp4a_scl_fp8(
                B_scale[(unsigned long long)n * groups + group]) * (0.5f * scale2);
            #pragma unroll
            for (int row = 0; row < 4; ++row) {
                if ((unsigned int)row >= M) continue;
                const int4 activation = *(const int4*)(
                    a_q + (unsigned long long)row * K + group * DP4A_GROUP_SIZE);
                const int values[4] = {
                    activation.x, activation.y, activation.z, activation.w
                };
                int dot = 0;
                #pragma unroll
                for (int index = 0; index < 4; ++index)
                    dot = DP4A_DOT(values[index], weight[index], dot);
                acc[row] += (float)dot *
                    a_scale[(unsigned long long)row * groups + group] * weight_scale;
            }
        }
    }

    __shared__ float partial[4][DP4A_N_PER_BLOCK * 2];
    const unsigned int warp_lane = lane % DP4A_WARP_SIZE;
    const unsigned int warp_in_out = lane / DP4A_WARP_SIZE;
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if ((unsigned int)row >= M) continue;
        float result = acc[row];
        #pragma unroll
        for (int offset = DP4A_WARP_SIZE / 2; offset > 0; offset >>= 1)
            result += __shfl_down_sync(0xffffffffu, result, offset);
        if (warp_lane == 0)
            partial[row][local_out * 2 + warp_in_out] = result;
    }
    __syncthreads();

    if (valid_n && lane == 0) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            if ((unsigned int)row < M)
                C[(unsigned long long)row * N + n] = __float2bfloat16(
                    partial[row][local_out * 2] + partial[row][local_out * 2 + 1]);
        }
    }
}

extern "C" __global__ void w4a16_gemv_dp4a_dual_batch4_d4(
    const signed char* __restrict__ a_q,
    const float* __restrict__ a_scale,
    const unsigned char* __restrict__ B0_packed,
    const unsigned char* __restrict__ B0_scale,
    const float B0_scale2,
    __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float B1_scale2,
    __nv_bfloat16* __restrict__ C1,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = DP4A_BLOCK_SIZE / DP4A_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * DP4A_N_PER_BLOCK + local_out;
    const bool valid_n = n < N;
    const unsigned int half_k = K / 2;
    const unsigned int groups = K / DP4A_GROUP_SIZE;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (valid_n) {
        for (unsigned int group = lane; group < groups; group += threads_per_out) {
            const unsigned long long packed0 = *(const unsigned long long*)(
                B0_packed + (unsigned long long)n * half_k + group * 8);
            const unsigned long long packed1 = *(const unsigned long long*)(
                B1_packed + (unsigned long long)n * half_k + group * 8);
            const int2 first0 = dp4a_expand_codebook_d4((unsigned int)packed0);
            const int2 second0 = dp4a_expand_codebook_d4((unsigned int)(packed0 >> 32));
            const int2 first1 = dp4a_expand_codebook_d4((unsigned int)packed1);
            const int2 second1 = dp4a_expand_codebook_d4((unsigned int)(packed1 >> 32));
            const int weight0[4] = {first0.x, first0.y, second0.x, second0.y};
            const int weight1[4] = {first1.x, first1.y, second1.x, second1.y};
            const float weight_scale0 = dp4a_scl_fp8(
                B0_scale[(unsigned long long)n * groups + group]) * (0.5f * B0_scale2);
            const float weight_scale1 = dp4a_scl_fp8(
                B1_scale[(unsigned long long)n * groups + group]) * (0.5f * B1_scale2);
            #pragma unroll
            for (int row = 0; row < 4; ++row) {
                if ((unsigned int)row >= M) continue;
                const int4 activation = *(const int4*)(
                    a_q + (unsigned long long)row * K + group * DP4A_GROUP_SIZE);
                const int values[4] = {
                    activation.x, activation.y, activation.z, activation.w
                };
                int dot0 = 0;
                int dot1 = 0;
                #pragma unroll
                for (int index = 0; index < 4; ++index) {
                    dot0 = DP4A_DOT(values[index], weight0[index], dot0);
                    dot1 = DP4A_DOT(values[index], weight1[index], dot1);
                }
                const float activation_scale =
                    a_scale[(unsigned long long)row * groups + group];
                acc0[row] += (float)dot0 * activation_scale * weight_scale0;
                acc1[row] += (float)dot1 * activation_scale * weight_scale1;
            }
        }
    }

    __shared__ float partial0[4][DP4A_N_PER_BLOCK * 2];
    __shared__ float partial1[4][DP4A_N_PER_BLOCK * 2];
    const unsigned int warp_lane = lane % DP4A_WARP_SIZE;
    const unsigned int warp_in_out = lane / DP4A_WARP_SIZE;
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        if ((unsigned int)row >= M) continue;
        float value0 = acc0[row];
        float value1 = acc1[row];
        #pragma unroll
        for (int offset = DP4A_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            value0 += __shfl_down_sync(0xffffffffu, value0, offset);
            value1 += __shfl_down_sync(0xffffffffu, value1, offset);
        }
        if (warp_lane == 0) {
            partial0[row][local_out * 2 + warp_in_out] = value0;
            partial1[row][local_out * 2 + warp_in_out] = value1;
        }
    }
    __syncthreads();

    if (valid_n && lane == 0) {
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            if ((unsigned int)row < M) {
                C0[(unsigned long long)row * N + n] = __float2bfloat16(
                    partial0[row][local_out * 2] + partial0[row][local_out * 2 + 1]);
                C1[(unsigned long long)row * N + n] = __float2bfloat16(
                    partial1[row][local_out * 2] + partial1[row][local_out * 2 + 1]);
            }
        }
    }
}
