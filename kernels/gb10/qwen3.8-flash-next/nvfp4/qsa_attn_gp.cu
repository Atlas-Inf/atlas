// SPDX-License-Identifier: AGPL-3.0-only

// QSA prefill attention at TP2, exact: `qsa_prefill_attn_g`'s arithmetic,
// rescheduled. Opt-in: ATLAS_QWEN4EXP_PREFILL_QSA_GP=1.
//
// WHY. At TP=EP=2 a rank holds one kv head, tc2 cannot serve, and the scalar
// `qsa_prefill_attn_g` (qsa_indexer.cu) is the default: ~1.5 s of a 16K cold
// prefill on the pair, the largest kernel in it. It is latency-bound: every
// key's K/V address is a chain of two dependent global loads (the row's block
// list, then the block table) before the K/V loads, only the next key's loads
// are in flight, and each key runs four serial five-level shuffle chains.
//
// WHAT CHANGES -- schedule, not arithmetic:
//  * the row's cached-token slots are resolved once, up front, by all 256
//    threads into shared memory, so a key's address is one shared load;
//  * each warp streams its keys' K and V through a QSA_GP_DEPTH-deep cp.async
//    ring (each lane copies, then reads back, only its own 16 bytes); ring
//    and slot table alias the merge buffer, so two CTAs still share an SM;
//  * the four heads' butterflies run as one: level 16 has lanes 0..15 finish
//    heads 0/1 and lanes 16..31 heads 2/3, level 8 halves that again, levels
//    4/2/1 run inside the 8-lane group that owns one head -- 6 shuffles a key
//    instead of 20, every add one of the butterfly's own (x + its xor
//    partner at that level; FP addition commutes);
//  * that group keeps its head's running max and sum, forms the head's
//    rescale and weight once and broadcasts them;
//  * two keys a step: both keys' scores, then the two softmax steps in order.
//
// EXACTNESS. Bit-identical to `qsa_prefill_attn_g`, element by element: the
// same grid, block and head grouping (QSA_PA_G heads a CTA), the same
// warp-striped key order (warp w takes t = w, w + 8, ...), the same BF16 ->
// float widening, the same per-lane product sum and butterfly tree, the same
// online-softmax expressions and the same cross-warp merge, all in order.
// One shortcut, exact: when no head's running max moves on a key, every
// rescale is `__expf(0.0f)` == 1.0f and the `x * 1.0f` products are skipped
// (a float times 1.0f is that float). Built with `--fmad=false` like the
// rest of this target, so no product is fused into an add.
// `scripts/dev/qwen4exp_qsa_gp_bench.cu` compares every output byte against
// `_g`: GB10, 2048 rows at position 14000, 12 q / 1 kv: 21.1 -> 9.1 ms.
//
// Grid: (rows, nq / QSA_PA_G, 1), Block: (256, 1, 1), dynamic shared memory
// `qsa_prefill_attn_gp_smem` (crates/spark-model/src/layers/ops/qsa_prefill_attn.rs).
// Requires hd == 256 (one 16-byte K and V segment a lane) and
// QSA_PA_G | nq / nkv.

#include <cuda_bf16.h>

#define QSA_PA_WARPS 8
#define QSA_PA_G 4
// The host sizes the ring's dynamic shared memory from its own copy of
// QSA_GP_DEPTH (`layers/ops/qwen4exp_prefill.rs`): change both, never this
// one alone through -D (the launch would be under-sized).
#ifndef QSA_GP_DEPTH
#define QSA_GP_DEPTH 4             // keys in flight a warp (3: -2%, 5: +0%, 6: -50%)
#endif
static_assert(QSA_GP_DEPTH >= 3, "an odd tail key is staged by the step before it");

__device__ __forceinline__ void qsa_gp_cp16(void* dst_smem, const void* src_gmem, bool pred) {
    const unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_smem);
    const unsigned int n = pred ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src_gmem), "r"(n)
                 : "memory");
}

extern "C" __global__ void __launch_bounds__(256, 2) qsa_prefill_attn_gp(
    const __nv_bfloat16* __restrict__ q,        // [rows, nq, hd] (roped)
    const __nv_bfloat16* __restrict__ k_cache,  // paged NHD
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ lists,              // [rows, topk] block ids
    __nv_bfloat16* __restrict__ attn_out,       // [rows, nq, hd]
    const unsigned int first_pos,
    const unsigned int topk,
    const unsigned int ratio,
    const unsigned int block_size,
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int hd,
    const float inv_sqrt_d
) {
    const unsigned int r = blockIdx.x;
    const unsigned int qh0 = blockIdx.y * QSA_PA_G;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int topk_ratio = topk * ratio;
    const unsigned int n_tok = topk_ratio + tail;
    const unsigned int kvh = qh0 / (nq / nkv);
    const unsigned int row_elems = nkv * hd;

    extern __shared__ __align__(16) float smem[];
    // Key loop: [WARPS][DEPTH] x {K, V} x 256 BF16, then the slot table.
    __nv_bfloat16* ring = reinterpret_cast<__nv_bfloat16*>(smem);
    unsigned int* slots = reinterpret_cast<unsigned int*>(
        ring + (size_t)QSA_PA_WARPS * QSA_GP_DEPTH * 2 * 256);
    // Merge (after the loop, aliasing both): `_g`'s [WARPS][G][hd], m, l.
    float* acc_w = smem;
    float* m_w = smem + QSA_PA_WARPS * QSA_PA_G * hd;
    float* l_w = m_w + QSA_PA_WARPS * QSA_PA_G;

    // Every key's cached-token slot, as `_g`'s qsa_key_off forms its offset.
    for (unsigned int t = tid; t < n_tok; t += blockDim.x) {
        const unsigned int tok = t < topk_ratio
            ? (unsigned int)lists[(size_t)r * topk + t / ratio] * ratio + t % ratio
            : complete * ratio + (t - topk_ratio);
        slots[t] = (unsigned int)block_table[tok / block_size] * block_size + tok % block_size;
    }

    float qreg[QSA_PA_G][8], acc[QSA_PA_G][8];
    #pragma unroll
    for (unsigned int g = 0; g < QSA_PA_G; ++g) {
        const __nv_bfloat16* qrow = q + ((size_t)r * nq + qh0 + g) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            qreg[g][e] = (float)qrow[lane * 8 + e];
            acc[g][e] = 0.0f;
        }
    }
    // Running max and sum of head lane / 8, kept by that head's lane group.
    float m_own = -1e30f, l_own = 0.0f;
    __syncthreads();   // slot table complete

    __nv_bfloat16* my_ring = ring + (size_t)warp * QSA_GP_DEPTH * 2 * 256 + lane * 8;
    const unsigned long long head_off = (unsigned long long)kvh * hd + lane * 8;
    // Stage this warp's i-th key (t = warp + 8 i) into ring slot i % DEPTH:
    // one commit group a call, empty past the end, so wait counts stay uniform.
    auto issue = [&](unsigned int i) {
        const unsigned int t = warp + i * QSA_PA_WARPS;
        const bool live = t < n_tok;
        const unsigned long long off =
            live ? (unsigned long long)slots[t] * row_elems + head_off : 0ull;
        __nv_bfloat16* dst = my_ring + (size_t)(i % QSA_GP_DEPTH) * 2 * 256;
        qsa_gp_cp16(dst, k_cache + off, live);
        qsa_gp_cp16(dst + 256, v_cache + off, live);
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // Key i's K and V, widened.
    auto widen = [&](unsigned int i, float (&kreg)[8], float (&vreg)[8]) {
        const __nv_bfloat16* src = my_ring + (size_t)(i % QSA_GP_DEPTH) * 2 * 256;
        const uint4 kraw = *reinterpret_cast<const uint4*>(src);
        const uint4 vraw = *reinterpret_cast<const uint4*>(src + 256);
        const __nv_bfloat16* kp = reinterpret_cast<const __nv_bfloat16*>(&kraw);
        const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&vraw);
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            kreg[e] = (float)kp[e];
            vreg[e] = (float)vp[e];
        }
    };
    // The scaled score of this lane's head (lane / 8) for one key.
    auto score = [&](const float (&kreg)[8]) {
        float dot[QSA_PA_G];
        #pragma unroll
        for (unsigned int g = 0; g < QSA_PA_G; ++g) {
            dot[g] = 0.0f;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) dot[g] += qreg[g][e] * kreg[e];
        }
        const bool b4 = (lane & 16u) != 0u, b3 = (lane & 8u) != 0u;
        const float x0 = (b4 ? dot[2] : dot[0]) + __shfl_xor_sync(0xFFFFFFFFu, b4 ? dot[0] : dot[2], 16);
        const float x1 = (b4 ? dot[3] : dot[1]) + __shfl_xor_sync(0xFFFFFFFFu, b4 ? dot[1] : dot[3], 16);
        float y = (b3 ? x1 : x0) + __shfl_xor_sync(0xFFFFFFFFu, b3 ? x0 : x1, 8);
        y += __shfl_xor_sync(0xFFFFFFFFu, y, 4);
        y += __shfl_xor_sync(0xFFFFFFFFu, y, 2);
        y += __shfl_xor_sync(0xFFFFFFFFu, y, 1);
        return y * inv_sqrt_d;
    };
    // `_g`'s online-softmax step for one key.
    auto update = [&](float dot, const float (&vreg)[8]) {
        const float m_new = fmaxf(m_own, dot);
        const float scale = __expf(m_own - m_new);
        const float p = __expf(dot - m_new);
        l_own = l_own * scale + p;
        m_own = m_new;
        float pg[QSA_PA_G];
        #pragma unroll
        for (unsigned int g = 0; g < QSA_PA_G; ++g) pg[g] = __shfl_sync(0xFFFFFFFFu, p, g * 8);
        if (__all_sync(0xFFFFFFFFu, scale == 1.0f)) {
            #pragma unroll
            for (unsigned int g = 0; g < QSA_PA_G; ++g) {
                #pragma unroll
                for (unsigned int e = 0; e < 8; ++e) acc[g][e] = acc[g][e] + pg[g] * vreg[e];
            }
            return;
        }
        #pragma unroll
        for (unsigned int g = 0; g < QSA_PA_G; ++g) {
            const float sg = __shfl_sync(0xFFFFFFFFu, scale, g * 8);
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) acc[g][e] = acc[g][e] * sg + pg[g] * vreg[e];
        }
    };

    // Keys this warp walks: t = warp, warp + 8, ... < n_tok.
    const unsigned int n_mine = n_tok > warp ? (n_tok - warp + QSA_PA_WARPS - 1) / QSA_PA_WARPS : 0u;
    #pragma unroll
    for (unsigned int i = 0; i < QSA_GP_DEPTH - 2; ++i) issue(i);
    unsigned int i = 0;
    for (; i + 1 < n_mine; i += 2) {
        issue(i + QSA_GP_DEPTH - 2);
        issue(i + QSA_GP_DEPTH - 1);
        asm volatile("cp.async.wait_group %0;" ::"n"(QSA_GP_DEPTH - 2) : "memory");
        float ka[8], va[8], kb[8], vb[8];
        widen(i, ka, va);
        widen(i + 1, kb, vb);
        const float da = score(ka);
        const float db = score(kb);
        update(da, va);
        update(db, vb);
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (i < n_mine) {
        float ka[8], va[8];
        widen(i, ka, va);
        update(score(ka), va);
    }
    __syncthreads();   // every warp is done with the ring before the merge reuses it

    #pragma unroll
    for (unsigned int g = 0; g < QSA_PA_G; ++g) {
        float* dst = acc_w + ((size_t)warp * QSA_PA_G + g) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) dst[lane * 8 + e] = acc[g][e];
        if (lane == g * 8) {
            m_w[warp * QSA_PA_G + g] = m_own;
            l_w[warp * QSA_PA_G + g] = l_own;
        }
    }
    __syncthreads();

    // `_g`'s merge: one warp per head, warps 0..WARPS-1 in order.
    if (warp < QSA_PA_G) {
        const unsigned int g = warp;
        float m_tot = -1e30f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) m_tot = fmaxf(m_tot, m_w[w * QSA_PA_G + g]);
        float l_tot = 0.0f;
        float out[8];
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) out[e] = 0.0f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) {
            const float s = __expf(m_w[w * QSA_PA_G + g] - m_tot);
            l_tot += l_w[w * QSA_PA_G + g] * s;
            const float* srcw = acc_w + ((size_t)w * QSA_PA_G + g) * hd;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) out[e] += srcw[lane * 8 + e] * s;
        }
        const float inv_l = (l_tot > 0.0f) ? 1.0f / l_tot : 0.0f;
        __nv_bfloat16* orow = attn_out + ((size_t)r * nq + qh0 + g) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) orow[lane * 8 + e] = __float2bfloat16(out[e] * inv_l);
    }
}
