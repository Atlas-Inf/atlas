// SPDX-License-Identifier: AGPL-3.0-only
//
// The expert-unit plan and helpers shared by qwen4exp_moe_c8.cu (exact,
// ATLAS_QWEN4EXP_MOE_UNITS) and qwen4exp_moe_c8_tc.cu (tensor-core
// prototype): see qwen4exp_moe_c8.cu for the workspace layout.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define C8_RMAX 16u
#define C8_SLOTS_MAX 1024u
#define C8_ROWS_MAX 64u
#define C8_UNITS_MAX (C8_SLOTS_MAX + C8_ROWS_MAX)
#define C8_WS_UNITS 16u
#define C8_WS_ROWS (C8_WS_UNITS + 3u * C8_UNITS_MAX)
#define C8_SHARED 0xFFFFFFFFu
#define C8_H 2560u
#define C8_I 640u

__device__ __forceinline__ float c8_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = b;
    return (float)f;
}

// ── plan: one block of 1024 threads ──
// Entries ranked by (expert, q); a unit head every C8_RMAX entries of one
// expert; shared units of C8_RMAX tokens; units placed by (rows desc, index).
__device__ __forceinline__ void c8_plan_body(
    const unsigned int* __restrict__ expert_indices, unsigned int* __restrict__ ws,
    unsigned int top_k, unsigned int rows
) {
    __shared__ unsigned s_idx[C8_SLOTS_MAX];
    __shared__ unsigned s_e[C8_SLOTS_MAX];
    __shared__ unsigned s_cnt[C8_UNITS_MAX];
    __shared__ unsigned s_units;
    const unsigned slots = rows * top_k;
    if (slots > C8_SLOTS_MAX || rows > C8_ROWS_MAX) __trap();
    const unsigned n_sh = (rows + C8_RMAX - 1) / C8_RMAX;
    const unsigned cands = slots + n_sh;
    if (threadIdx.x == 0) s_units = 0;
    for (unsigned t = threadIdx.x; t < slots; t += blockDim.x) s_idx[t] = expert_indices[t];
    __syncthreads();
    for (unsigned t = threadIdx.x; t < slots; t += blockDim.x) {
        const unsigned e = s_idx[t];
        unsigned rank = 0;
        for (unsigned j = 0; j < slots; j++) {
            const unsigned f = s_idx[j];
            rank += (f < e) || (f == e && j < t);
        }
        s_e[rank] = e;
        ws[C8_WS_ROWS + rank] = t;
    }
    __syncthreads();
    // Candidate c < slots: the unit headed at rank c (0 rows: not a head);
    // c >= slots: shared unit c - slots.
    for (unsigned c = threadIdx.x; c < cands; c += blockDim.x) {
        unsigned cnt;
        if (c < slots) {
            const unsigned e = s_e[c];
            unsigned start = c;
            while (start > 0 && s_e[start - 1] == e) start--;
            unsigned end = c + 1;
            while (end < slots && s_e[end] == e) end++;
            cnt = (c - start) % C8_RMAX == 0 ? min(C8_RMAX, end - c) : 0u;
        } else {
            cnt = min(C8_RMAX, rows - (c - slots) * C8_RMAX);
        }
        s_cnt[c] = cnt;
        if (cnt) atomicAdd(&s_units, 1u);
    }
    __syncthreads();
    for (unsigned c = threadIdx.x; c < cands; c += blockDim.x) {
        const unsigned cnt = s_cnt[c];
        if (!cnt) continue;
        unsigned pos = 0;
        for (unsigned d = 0; d < cands; d++) {
            const unsigned o = s_cnt[d];
            pos += (o > cnt) || (o == cnt && d < c);
        }
        unsigned* u = ws + C8_WS_UNITS + 3 * pos;
        if (c < slots) { u[0] = s_e[c]; u[1] = c; }
        else           { u[0] = C8_SHARED; u[1] = (c - slots) * C8_RMAX; }
        u[2] = cnt;
    }
    if (threadIdx.x == 0) ws[0] = s_units;
}

struct C8Unit {
    unsigned expert, start, n;
    __device__ __forceinline__ bool shared() const { return expert == C8_SHARED; }
};
__device__ __forceinline__ C8Unit c8_unit_at(const unsigned* __restrict__ ws, unsigned y) {
    const unsigned* p = ws + C8_WS_UNITS + 3 * y;
    C8Unit u;
    u.expert = p[0]; u.start = p[1]; u.n = p[2];
    return u;
}
__device__ __forceinline__ bool c8_unit(const unsigned* __restrict__ ws, C8Unit& u) {
    if (blockIdx.y >= ws[0]) return false;
    const unsigned* p = ws + C8_WS_UNITS + 3 * blockIdx.y;
    u.expert = p[0]; u.start = p[1]; u.n = p[2];
    return true;
}
// Unit row i's slot q (routed) or token (shared).
__device__ __forceinline__ unsigned c8_slot(const unsigned* ws, const C8Unit& u, unsigned i) {
    return u.shared() ? u.start + i : ws[C8_WS_ROWS + u.start + i];
}

__device__ __forceinline__ void c8_bf16_row(const __nv_bfloat16* p, unsigned k16, float (&f)[16]) {
    const uint4* q = (const uint4*)p + k16 * 2;
    const uint4 lo = q[0], hi = q[1];
    const unsigned w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int b = 0; b < 8; b++) {
        f[2 * b] = __uint_as_float(w[b] << 16);
        f[2 * b + 1] = __uint_as_float(w[b] & 0xFFFF0000u);
    }
}

__device__ __forceinline__ void c8_f32_row(const float* p, unsigned k16, float (&f)[16]) {
#pragma unroll
    for (int v = 0; v < 4; v++) {
        const float4 x = ((const float4*)p)[k16 * 4 + v];
        f[4 * v] = x.x; f[4 * v + 1] = x.y; f[4 * v + 2] = x.z; f[4 * v + 3] = x.w;
    }
}

#define C8_GU_ARGS                                                             \
    const __nv_bfloat16* __restrict__ A,                                       \
    const unsigned long long* __restrict__ gate_packed_ptrs,                   \
    const unsigned long long* __restrict__ gate_scale_ptrs,                    \
    const float* __restrict__ gate_scale2_vals,                                \
    const unsigned long long* __restrict__ up_packed_ptrs,                     \
    const unsigned long long* __restrict__ up_scale_ptrs,                      \
    const float* __restrict__ up_scale2_vals,                                  \
    const unsigned char* __restrict__ sh_gate_packed,                          \
    const unsigned char* __restrict__ sh_gate_scale, float sh_gate_s2,         \
    const unsigned char* __restrict__ sh_up_packed,                            \
    const unsigned char* __restrict__ sh_up_scale, float sh_up_s2,             \
    const unsigned int* __restrict__ ws,                                       \
    __nv_bfloat16* __restrict__ gate_out, __nv_bfloat16* __restrict__ up_out,  \
    __nv_bfloat16* __restrict__ sh_gate_out,                                   \
    __nv_bfloat16* __restrict__ sh_up_out, float* __restrict__ act,            \
    unsigned int top_k, unsigned int rows
#define C8_GU_PASS                                                             \
    A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs,    \
    up_scale_ptrs, up_scale2_vals, sh_gate_packed, sh_gate_scale, sh_gate_s2,  \
    sh_up_packed, sh_up_scale, sh_up_s2, ws, gate_out, up_out, sh_gate_out,    \
    sh_up_out, act, top_k, rows
#define C8_SD_PASS                                                             \
    act, packed_ptrs, scale_ptrs, scale2_vals, sh_down_packed, sh_down_scale,  \
    sh_down_s2, ws, C, sh_down_out, top_k, rows
#define C8_SD_ARGS                                                             \
    const float* __restrict__ act,                                             \
    const unsigned long long* __restrict__ packed_ptrs,                        \
    const unsigned long long* __restrict__ scale_ptrs,                         \
    const float* __restrict__ scale2_vals,                                     \
    const unsigned char* __restrict__ sh_down_packed,                          \
    const unsigned char* __restrict__ sh_down_scale, float sh_down_s2,         \
    const unsigned int* __restrict__ ws,                                       \
    __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ sh_down_out,    \
    unsigned int top_k, unsigned int rows

__device__ __forceinline__ void c8_cp16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
// `bytes` (a multiple of 16) from global to shared by the whole CTA: 16-byte
// cp.async when both ends allow, else bytewise. The caller commits and waits.
__device__ __forceinline__ void c8_stage(unsigned char* dst, const unsigned char* src, unsigned bytes) {
    if ((((size_t)src | (size_t)dst) & 15) == 0) {
        for (unsigned i = threadIdx.x; i < bytes / 16; i += blockDim.x) c8_cp16(dst + 16 * i, src + 16 * i);
    } else {
        for (unsigned i = threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
    }
}
// The 16 weights of one packed step, w[i] = lut[nibble i] * sc, without the
// LUT: nibbles i and i + 4 of a 32-bit word become an f16x2 whose halves are
// lut / 2^14 exactly (E2M1's 2 exponent bits and 1 mantissa bit land on
// f16's low exponent bits and top mantissa bit; 0 and 0.5 on f16's
// subnormals), widened exactly to f32 and multiplied by sc * 2^14 (exact:
// a power-of-two scaling). The real product is lut * sc's, so the rounded
// f32 is the LUT form's bit for bit.
__device__ __forceinline__ void c8_dq16(unsigned long long p, float sc, float (&w)[16]) {
    const float scx = sc * 16384.0f;
#pragma unroll
    for (unsigned h = 0; h < 2; h++) {
        const unsigned q = (unsigned)(p >> (32 * h));
#pragma unroll
        for (unsigned s = 0; s < 16; s += 4) {
            const unsigned a = s <= 9 ? q << (9 - s) : q >> (s - 9);
            const unsigned b = q << (12 - s);
            const unsigned x = (a & 0x0E000E00u) | (b & 0x80008000u);
            const __half2 hx = *(const __half2*)&x;
            // nibble s / 4 of the word's low half, and nibble s / 4 + 4.
            w[8 * h + s / 4] = __low2float(hx) * scx;
            w[8 * h + s / 4 + 4] = __high2float(hx) * scx;
        }
    }
}

