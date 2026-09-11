// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN prefill — FLA-style MULTI-KERNEL decomposition, HIP port for
// gfx1151 (strix-hip). Same three-kernel contract and buffer ABI as
// kernels/gb10/common/gated_delta_rule_fla.cu, with the NVIDIA-only pieces
// removed:
//   * mma.sync.m16n8k16 PTX  → scalar FP32 Gram/dot loops. The Gram matrices
//     are small (64x64x128 and 64x128x128 per chunk) and parallel over the
//     whole block, so they stay off the critical path behind the solves.
//   * cp.async / TMA / mbarrier → synchronous shared-memory staging.
//   * chunk_fwd_o's 97KB smem layout exceeds RDNA3.5's 64KB LDS cap — this
//     version drops the S_c smem staging and the uc copy (read from global,
//     L2-resident) and keeps sq + sk/o1 + kq + gc/egc ≈ 48.5KB.
// The serial spine (chunk_delta_h_vfused) is gb10's cdh_vtile_core<2,1> taken
// verbatim — it was already scalar-register + __shfl_xor_sync only, i.e.
// portable. Masks are written as full 64-bit literals for wave64.
// Math parity is inherited from the same formulation the CPU oracle covers
// (crates/spark-runtime/tests/gdn_chunk64_oracle.rs :: fla_decomposed_ref).

#include <cuda_bf16.h>

// AMD WMMA 16x16x16 BF16 fragment types (gfx1151, wave32 — same contract as
// dense_gemm_tc.cu): a[i]=A[lane&15][i], b[k]=B[k][n_base+lane&15],
// acc[e]→C[2e+(lane>>4)][lane&15].
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64
// Floor for the linear gate before log-space cumsum — see the gb10 source;
// log(0)=-inf → exp(gc_i-gc_l) NaNs in the chunked log-space form.
#define GATE_FLOOR 1e-30f

// Per-stream prefill geometry — identical to the gb10 macro (varlen reads
// cu_seqlens/cu_chunks; uniform reduces to b*seq_len / b*num_chunks).
struct GdnGeom { unsigned int seqlen, nchunks, choff; unsigned long long tokoff; };
#define GDN_GEOM(g)                                                            \
    GdnGeom g;                                                                 \
    (void)cu_chunks;                                                          \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                       \
        g.seqlen  = (unsigned int)cu_seqlens[b + 1] - _s0;                    \
        g.tokoff  = (unsigned long long)_s0;                                  \
        unsigned int _co = 0;                                                \
        for (unsigned int _i = 0; _i < b; _i++)                               \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])       \
                    + CHUNK - 1) / CHUNK;                                      \
        g.choff   = _co;                                                       \
        g.nchunks = (g.seqlen + CHUNK - 1) / CHUNK;                           \
    } else {                                                                  \
        g.seqlen  = seq_len;                                                   \
        g.tokoff  = (unsigned long long)b * seq_len;                          \
        g.choff   = b * num_chunks;                                            \
        g.nchunks = num_chunks;                                                \
    }

// ── KERNEL 1: recompute_w_u ──────────────────────────────────────────────
// Grid: (NT, num_v_heads, batch)  Block: (256,1,1).  One CTA per (chunk, head).
//   W_out: [.. ][CHUNK][K_DIM]   = T·(β·exp(gc)·K)
//   U_out: [.. ][CHUNK][V_DIM]   = T·(βV)
// where T=(I+L)⁻¹ applied by right-looking blocked forward-substitution.
// gb10 used mma_gram for the KᵀK Gram; here the same kk[l][i] = <k_l,k_i> is a
// scalar loop over all 256 threads — 64×64 outputs × 128-deep dots is only
// ~2k FMA/thread, far under the forward-substitution cost it feeds.
// smem: sk_bf(16K) + kk/L(16K f32) + gc(256) ≈ 32.25KB — under the 64KB cap.
#define RL_BLK 16

extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_recompute_wu(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out,
    __nv_bfloat16* __restrict__ U_out,
    float* __restrict__ gc_out,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

    key   += g.tokoff * qk_stride;
    value += g.tokoff * v_stride;
    gate  += g.tokoff * gb_stride;
    beta  += g.tokoff * gb_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sk = (__nv_bfloat16*)smem_raw;       // [CHUNK*K_DIM] bf16
    float* kk = (float*)(sk + CHUNK * K_DIM);           // [CHUNK*CHUNK] f32 Gram
    float* L = kk;                                     // aliased strict-lower (see gb10)
    float* gc = L + CHUNK * CHUNK;                      // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += blockDim.x) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        sk[i * K_DIM + j] = (i < ce)
            ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + j]
            : __float2bfloat16(0.0f);
    }
    // ORDER-STABLE gc scan — identical to gb10 (parallel loads + logf, serial
    // 64-add on one thread; reassociation here is a known accuracy-gate risk).
    for (unsigned int idx = tid; idx < CHUNK; idx += blockDim.x) {
        gc[idx] = (idx < ce)
            ? logf(fmaxf(gate[(unsigned long long)(cs + idx) * gb_stride + vh], GATE_FLOOR))
            : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            acc += gc[i];
            gc[i] = acc;
        }
    }
    __syncthreads();
    for (unsigned int idx = tid; idx < CHUNK; idx += blockDim.x) {
        if (idx >= ce) {
            gc[idx] = 0.0f;
        } else {
            gc_out[base * CHUNK + idx] = gc[idx];
        }
    }
    __syncthreads();

    // Gram: kk[l][i] = <k_l, k_i> — sk[64,128]·sk[64,128]ᵀ on WMMA
    // (replaces gb10's mma_gram<8,CHUNK,false>). 16 tiles of 16×16 over 8
    // waves; fp32 accumulation, fp32 out to smem. Symmetric output, all
    // entries — the consumer only reads the strict lower triangle.
    {
        const unsigned int warp = tid / 32, lane = tid % 32;
        for (unsigned int tile = warp; tile < 16; tile += 8) {
            const unsigned int tr = (tile / 4) * 16, tc = (tile % 4) * 16;
            v8f acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int kb = 0; kb < K_DIM; kb += 16) {
                v16bf a, bfrag;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)sk[(tr + (lane & 15)) * K_DIM + kb + i];
                #pragma unroll
                for (int k = 0; k < 16; k++)
                    bfrag[k] = (__bf16)(float)sk[(tc + (lane & 15)) * K_DIM + kb + k];
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bfrag, acc);
            }
            #pragma unroll
            for (int e = 0; e < 8; e++)
                kk[(tr + 2 * e + (lane >> 4)) * CHUNK + tc + (lane & 15)] = acc[e];
        }
    }
    __syncthreads();

    // L[i][l] = β_i·exp(gc_i-gc_l)·<k_l,k_i>  for l<i ; strict-lower only.
    for (unsigned int p = tid; p < CHUNK * CHUNK; p += blockDim.x) {
        unsigned int i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l < i) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            L[i * CHUNK + l] = bi * expf(gc[i] - gc[l]) * kk[l * CHUNK + i];
        }
    }
    __syncthreads();

    // Two independent right-looking forward-subs on disjoint halves — identical
    // to gb10 (RL_BLK must stay compile-time or xb spills to local).
    if (tid < v_dim) {
        float acc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            acc[i] = bi * (float)value[(unsigned long long)(cs + i) * v_stride + vh * v_dim + tid];
        }
        for (unsigned int jb = 0; jb < ce; jb += RL_BLK) {
            float xb[RL_BLK];
            #pragma unroll
            for (unsigned int r = 0; r < RL_BLK; r++) {
                if (jb + r >= ce) continue;
                float x = acc[jb + r];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (q < r) x -= L[(jb + r) * CHUNK + jb + q] * xb[q];
                }
                xb[r] = x;
                acc[jb + r] = x;
                U_out[base * CHUNK * V_DIM + (jb + r) * v_dim + tid] = __float2bfloat16(x);
            }
            for (unsigned int i = jb + RL_BLK; i < ce; i++) {
                float a = acc[i];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (jb + q < ce) a -= L[i * CHUNK + jb + q] * xb[q];
                }
                acc[i] = a;
            }
        }
    }
    const unsigned int wtid = tid - 128u;
    if (tid >= 128u && wtid < k_dim) {
        float acc[CHUNK];
        for (unsigned int i = 0; i < ce; i++) {
            float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
            acc[i] = bi * expf(gc[i]) * (float)sk[i * K_DIM + wtid];
        }
        for (unsigned int jb = 0; jb < ce; jb += RL_BLK) {
            float xb[RL_BLK];
            #pragma unroll
            for (unsigned int r = 0; r < RL_BLK; r++) {
                if (jb + r >= ce) continue;
                float x = acc[jb + r];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (q < r) x -= L[(jb + r) * CHUNK + jb + q] * xb[q];
                }
                xb[r] = x;
                acc[jb + r] = x;
                W_out[base * CHUNK * K_DIM + (jb + r) * k_dim + wtid] = __float2bfloat16(x);
            }
            for (unsigned int i = jb + RL_BLK; i < ce; i++) {
                float a = acc[i];
                #pragma unroll
                for (unsigned int q = 0; q < RL_BLK; q++) {
                    if (jb + q < ce) a -= L[i * CHUNK + jb + q] * xb[q];
                }
                acc[i] = a;
            }
        }
    }
}

// ── KERNEL 2: chunk_delta_h_vfused — gb10's cdh_vtile_core<2,1> verbatim ──
// Scalar register-S spine; the only NVIDIA-ism in the original was the warp
// butterfly, which HIP supports (__shfl_xor_sync; the build's widen pass turns
// the mask 64-bit — written here already widened for clarity on wave64).
// smem: W(16K)+K(16K)+U(16K) bf16 single-buffered + decs[(C+1)] = 49,412 B —
// the launcher's smem_fused formula — under the 64KB cap.
extern "C" __global__ void __launch_bounds__(256, 1)
gated_delta_rule_chunk_delta_h_vfused(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    constexpr int SPLIT = 2, VT = 1;
    constexpr int KH = K_DIM / SPLIT;          // per-thread slice of state column
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    GDN_GEOM(g);
    const unsigned int t = threadIdx.x;        // 0 .. (V_DIM/VT)*SPLIT - 1
    const unsigned int v0 = (t / SPLIT) * VT;
    const unsigned int sub = t % SPLIT;
    const unsigned int k0 = sub * KH;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ char smem_raw_vt[];
    __nv_bfloat16* Wp = (__nv_bfloat16*)smem_raw_vt;   // [CHUNK*K_DIM]
    __nv_bfloat16* Kp = Wp + CHUNK * K_DIM;            // [CHUNK*K_DIM]
    __nv_bfloat16* Up = Kp + CHUNK * K_DIM;            // [CHUNK*V_DIM]
    float* decs = (float*)(Up + CHUNK * V_DIM);        // [CHUNK+1], [0]=exp(gc_last)

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);
    float Sold[KH][VT];
    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = H[(k0 + kk) * V_DIM + v0 + vt];

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    const unsigned int nthr = blockDim.x;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();
        const __nv_bfloat16* Wsrc = W_in + base * CHUNK * K_DIM;
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) Wp[e] = Wsrc[e];
        for (unsigned int e = t; e < CHUNK * K_DIM; e += nthr) {
            unsigned int i = e / K_DIM, kx = e % K_DIM;
            Kp[e] = (i < ce)
                ? key_b[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + kx]
                : __float2bfloat16(0.0f);
        }
        {
            const __nv_bfloat16* Usrc0 = U_in + base * CHUNK * V_DIM;
            for (unsigned int e = t; e < CHUNK * V_DIM; e += nthr) Up[e] = Usrc0[e];
        }
        if (t == 0) {
            float dl = gc_in[base * CHUNK + ce - 1];
            decs[0] = expf(dl);
            for (unsigned int i = 0; i < ce; i++) decs[1 + i] = expf(dl - gc_in[base * CHUNK + i]);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                S_out[base * K_DIM * V_DIM + (k0 + kk) * V_DIM + v0 + vt] =
                    __float2bfloat16(Sold[kk][vt]);

        const float edl = decs[0];
        float Snew[KH][VT];
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Snew[kk][vt] = edl * Sold[kk][vt];

        for (unsigned int i = 0; i < ce; i++) {
            float wsp[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) wsp[vt] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float w = (float)Wp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) wsp[vt] += w * Sold[kk][vt];
            }
            #pragma unroll
            for (int vt = 0; vt < VT; vt++)
                #pragma unroll
                for (int s = 1; s < SPLIT; s <<= 1)
                    wsp[vt] += __shfl_xor_sync(0xffffffffffffffffULL, wsp[vt], s);
            const float dc = decs[1 + i];
            float d[VT];
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) {
                const float uci = (float)Up[i * V_DIM + v0 + vt] - wsp[vt];
                if (sub == 0) uc_out[base * CHUNK * V_DIM + i * v_dim + v0 + vt] = __float2bfloat16(uci);
                d[vt] = dc * uci;
            }
            #pragma unroll
            for (int kk = 0; kk < KH; kk++) {
                const float kv = (float)Kp[i * K_DIM + k0 + kk];
                #pragma unroll
                for (int vt = 0; vt < VT; vt++) Snew[kk][vt] += d[vt] * kv;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KH; kk++)
            #pragma unroll
            for (int vt = 0; vt < VT; vt++) Sold[kk][vt] = Snew[kk][vt];
    }

    #pragma unroll
    for (int kk = 0; kk < KH; kk++)
        #pragma unroll
        for (int vt = 0; vt < VT; vt++) H[(k0 + kk) * V_DIM + v0 + vt] = Sold[kk][vt];
}

// ── KERNEL 3: chunk_fwd_o — scalar, slim-smem gfx1151 variant ────────────
// Same math as gb10: O_i = (exp(gc_i)·<S_c[:,v],q_i> + Σ_{l<=i} exp(gc_i-gc_l)·
// <k_l,q_i>·uc_l[v])·rsqrt(d). The two mma_gram calls become scalar dot loops;
// S_c and uc are read from global (each CTA streams them once — L2-resident),
// which is what drops smem under the 64KB cap:
//   sq(16K bf16) + sk/o1(16K bf16, aliased like gb10) + kq(16K f32) + gc/egc(0.5K)
//   ≈ 48.5KB. Grid (NT, nv, batch), 512 threads.
extern "C" __global__ void __launch_bounds__(512, 1)
gated_delta_rule_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate,
    const float* __restrict__ gc_in,
    const __nv_bfloat16* __restrict__ S_in,
    const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output,
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_chunks,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int gb_stride,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * CHUNK;
    const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    const unsigned long long out_base = (g.tokoff * num_v_heads + vh) * v_dim;
    query += g.tokoff * qk_stride;
    key   += g.tokoff * qk_stride;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sq = (__nv_bfloat16*)smem_raw;          // [CHUNK*K_DIM]
    __nv_bfloat16* sk = sq + CHUNK * K_DIM;                // [CHUNK*K_DIM]
    float* kq = (float*)(sk + CHUNK * K_DIM);              // [CHUNK*CHUNK] f32
    float* gc = kq + CHUNK * CHUNK;                        // [CHUNK]
    float* egc = gc + CHUNK;                               // [CHUNK]

    for (unsigned int idx = tid; idx < CHUNK * k_dim; idx += blockDim.x) {
        unsigned int i = idx / k_dim, j = idx % k_dim;
        if (i < ce) {
            unsigned long long off = (unsigned long long)(cs + i) * qk_stride + kh * k_dim + j;
            sq[i * K_DIM + j] = query[off];
            sk[i * K_DIM + j] = key[off];
        } else {
            sq[i * K_DIM + j] = __float2bfloat16(0.0f);
            sk[i * K_DIM + j] = __float2bfloat16(0.0f);
        }
    }
    for (unsigned int i = tid; i < ce; i += blockDim.x) {
        float g = gc_in[base * CHUNK + i];
        gc[i] = g;
        egc[i] = expf(g);
    }
    __syncthreads();

    // Gram 1: kq[i][l] = exp(gc_i-gc_l)·<q_i,k_l> for l<=i (0 elsewhere so the
    // t2 GEMM below reads a clean lower-triangular matrix — the scalar version
    // left the upper triangle unread). sq·skᵀ on WMMA: 16 tiles / 16 waves.
    {
        const unsigned int warp = tid / 32, lane = tid % 32;
        for (unsigned int tile = warp; tile < 16; tile += 16) {
            const unsigned int tr = (tile / 4) * 16, tc = (tile % 4) * 16;
            v8f acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int kb = 0; kb < K_DIM; kb += 16) {
                v16bf a, bfrag;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)sq[(tr + (lane & 15)) * K_DIM + kb + i];
                #pragma unroll
                for (int k = 0; k < 16; k++)
                    bfrag[k] = (__bf16)(float)sk[(tc + (lane & 15)) * K_DIM + kb + k];
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bfrag, acc);
            }
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int r = tr + 2 * e + (lane >> 4);
                const unsigned int cl = tc + (lane & 15);
                kq[r * CHUNK + cl] =
                    (r < ce && cl <= r) ? expf(gc[r] - gc[cl]) * acc[e] : 0.0f;
            }
        }
    }
    __syncthreads();

    // Gram 2: o1[i][v] = <q_i, S_c[:,v]> — sq·S_c on WMMA, bf16 out into the
    // freed sk region (S_c read straight from global, transposed access,
    // L2-resident). 32 tiles / 16 waves.
    __nv_bfloat16* o1 = sk;                               // [CHUNK*V_DIM] bf16
    const __nv_bfloat16* Sb = S_in + base * K_DIM * V_DIM; // S_c[k][v]
    {
        const unsigned int warp = tid / 32, lane = tid % 32;
        for (unsigned int tile = warp; tile < 32; tile += 16) {
            const unsigned int tr = (tile / 8) * 16, tc = (tile % 8) * 16;
            v8f acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int kb = 0; kb < K_DIM; kb += 16) {
                v16bf a, bfrag;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)sq[(tr + (lane & 15)) * K_DIM + kb + i];
                #pragma unroll
                for (int k = 0; k < 16; k++)
                    bfrag[k] = (__bf16)(float)Sb[(kb + k) * V_DIM + tc + (lane & 15)];
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bfrag, acc);
            }
            #pragma unroll
            for (int e = 0; e < 8; e++)
                o1[(tr + 2 * e + (lane >> 4)) * V_DIM + tc + (lane & 15)] =
                    __float2bfloat16(acc[e]);
        }
    }
    __syncthreads();

    // Output: out[i][v] = (egc_i·o1[i][v] + Σ_l kq[i][l]·uc[l][v])·rsqrt(d) —
    // the masked accumulate is a plain GEMM since kq's upper triangle is 0.
    // A = kq (f32 smem → bf16 at fragment load), B = uc (global). 32 tiles.
    {
        const unsigned int warp = tid / 32, lane = tid % 32;
        for (unsigned int tile = warp; tile < 32; tile += 16) {
            const unsigned int tr = (tile / 8) * 16, tc = (tile % 8) * 16;
            v8f acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int kb = 0; kb < CHUNK; kb += 16) {
                v16bf a, bfrag;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)kq[(tr + (lane & 15)) * CHUNK + kb + i];
                #pragma unroll
                for (int k = 0; k < 16; k++)
                    bfrag[k] = (__bf16)(float)
                        uc_in[base * CHUNK * V_DIM + (kb + k) * v_dim + tc + (lane & 15)];
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bfrag, acc);
            }
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int r = tr + 2 * e + (lane >> 4);
                const unsigned int cl = tc + (lane & 15);
                if (r < ce)
                    output[out_base + (unsigned long long)(cs + r) * num_v_heads * v_dim + cl] =
                        __float2bfloat16(
                            (egc[r] * (float)o1[r * V_DIM + cl] + acc[e]) * inv_sqrt_d);
            }
        }
    }
}
