// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise A/B for the conv_state owner change in the token-parallel
// causal conv1d kernels: causal_conv1d_update_prefill_tp and its GLM verify
// twin causal_conv1d_update_prefill_tp_snap (the file on the include path),
// versus the bodies as they were before the change (kept below), where the
// owner of the LAST token tile wrote conv_state while the t0 == 0 thread read
// it. Every case also runs the serial causal_conv1d_update_prefill (one thread
// per channel, so race-free) and an exact host model of the final state and of
// the rollback snapshots (both are pure copies of inputs / incoming state).
//
// Launches as production does: block (32,8,1), grid (ceil(dim/32),
// ceil(seq_len/64), 1); the _snap twin under the programmatic-serialization
// attribute (it is on the runtime's PDL list). Random BF16 inputs (1 in 64 a
// raw bit pattern: NaN, Inf, denormal), random BF16 weights, random FP32
// initial state; bias is NULL on even chunks (GLM) and random FP32 on odd
// ones. Each variant carries its own conv_state through every chunk of every
// seq_len, so state written by one launch is what the next one reads.
// Outputs and snapshot slabs start from a different sentinel per variant, so
// an element that nobody wrote also counts as a difference.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/conv1d_tp_state_owner_bench.cu -o conv1d_owner_bench
//   (common copy, which has no _snap twin:
//        -I kernels/gb10/common -DCONV1D_BENCH_NO_SNAP)
//   ./conv1d_owner_bench [chunks=3] [snap_max_seq_at_full_dim=100]
// Dims: 24576 channels (GLM: 3 * 64 * 128; strides as production) and 1001
// channels (partial last warp, token stride dim + 7, snapshot stride + 7).
// Device memory: ~1.8 GB. Exit: 0 when the kernel under test matches the
// serial kernel and the exact model in every case and the old body wherever
// that one is race-free (seq_len <= 8, one thread per channel); 1 otherwise.
// Past 8 tokens the old _snap body does race (its snapshots of rows 0..2 pick
// up the new state), which the old/exact columns show.
#include "causal_conv1d.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;

// ── The kernels before the owner change (a1708008), verbatim but for the name ──
extern "C" __global__ void __launch_bounds__(256, 4)
conv1d_tp_before(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,
    unsigned int input_stride,
    unsigned int output_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    const unsigned int t0 = (blockIdx.y * blockDim.y + threadIdx.y) * 8u;
    if (t0 >= seq_len) return;

    const float* state = conv_state + (unsigned long long)ch * d_conv;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (unsigned int k = 0; k < 4; k++)
        if (k < d_conv) w_reg[k] = __bfloat162float(w[k]);

    auto xin = [&] (long long t) -> float {
        if (t >= 0) {
            return (t < (long long)seq_len)
                ? __bfloat162float(input[(unsigned long long)t * input_stride + ch])
                : 0.0f;
        }
        const long long idx = (long long)d_conv + t;   // -1 -> d_conv-1
        return (idx >= 0) ? state[idx] : 0.0f;
    };

    float s0 = xin((long long)t0 - 3);
    float s1 = xin((long long)t0 - 2);
    float s2 = xin((long long)t0 - 1);
    #pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        const unsigned int t = t0 + i;
        if (t >= seq_len) break;
        const float s3 = xin((long long)t);
        const float acc = b_val + s0 * w_reg[0] + s1 * w_reg[1] + s2 * w_reg[2] + s3 * w_reg[3];
        const float sig = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sig);
        s0 = s1; s1 = s2; s2 = s3;
    }

    if (t0 + 8u >= seq_len) {
        float* st = conv_state + (unsigned long long)ch * d_conv;
        #pragma unroll
        for (unsigned int k = 0; k < 4; k++)
            if (k < d_conv)
                st[k] = xin((long long)seq_len - (long long)d_conv + (long long)k);
    }
}

#ifndef CONV1D_BENCH_NO_SNAP
extern "C" __global__ void __launch_bounds__(256, 4)
conv1d_tp_snap_before(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ state_inter,
    unsigned long long inter_stride,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,
    unsigned int input_stride,
    unsigned int output_stride
) {
    atlas_pdl_enter();
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    const unsigned int t0 = (blockIdx.y * blockDim.y + threadIdx.y) * 8u;
    if (t0 >= seq_len) return;

    const float* state = conv_state + (unsigned long long)ch * d_conv;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (unsigned int k = 0; k < 4; k++)
        if (k < d_conv) w_reg[k] = __bfloat162float(w[k]);

    auto xin = [&] (long long t) -> float {
        if (t >= 0) {
            return (t < (long long)seq_len)
                ? __bfloat162float(input[(unsigned long long)t * input_stride + ch])
                : 0.0f;
        }
        const long long idx = (long long)d_conv + t;
        return (idx >= 0) ? state[idx] : 0.0f;
    };

    float s0 = xin((long long)t0 - 3);
    float s1 = xin((long long)t0 - 2);
    float s2 = xin((long long)t0 - 1);
    #pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        const unsigned int t = t0 + i;
        if (t >= seq_len) break;
        const float s3 = xin((long long)t);
        const float acc = b_val + s0 * w_reg[0] + s1 * w_reg[1] + s2 * w_reg[2] + s3 * w_reg[3];
        const float sig = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sig);
        s0 = s1; s1 = s2; s2 = s3;
    }

    if (t0 == 0u) {
        for (unsigned int t = 0; t + 1u < seq_len; ++t) {
            float* snapshot = state_inter + (unsigned long long)t * inter_stride
                            + (unsigned long long)ch * d_conv;
            #pragma unroll
            for (unsigned int k = 0; k < 4; ++k) {
                if (k < d_conv) {
                    snapshot[k] = xin((long long)t + 1ll - (long long)d_conv + (long long)k);
                }
            }
        }
    }

    if (t0 + 8u >= seq_len) {
        float* st = conv_state + (unsigned long long)ch * d_conv;
        #pragma unroll
        for (unsigned int k = 0; k < 4; k++)
            if (k < d_conv)
                st[k] = xin((long long)seq_len - (long long)d_conv + (long long)k);
    }
}
#endif

// ── Test data ──
__device__ unsigned int mix(unsigned long long i, unsigned int seed) {
    unsigned long long z = (i + 1ull) * 0x9E3779B97F4A7C15ull + (unsigned long long)seed * 0xD6E8FEB86659FD93ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (unsigned int)((z ^ (z >> 31)) >> 24);
}

// Uniform in [-scale, scale); every `raw_every`-th draw (0 = never) a raw bit pattern.
__global__ void fill_bf16(unsigned short* p, unsigned long long n, unsigned int seed, float scale,
                          unsigned int raw_every) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int h = mix(i, seed);
    if (raw_every != 0u && (h % raw_every) == 0u) { p[i] = (unsigned short)(h >> 12); return; }
    const __nv_bfloat16 b = __float2bfloat16(((float)(h >> 8) / 8388608.0f - 1.0f) * scale);
    p[i] = *(const unsigned short*)&b;
}

__global__ void fill_f32(unsigned int* p, unsigned long long n, unsigned int seed, float scale,
                         unsigned int raw_every) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int h = mix(i, seed);
    if (raw_every != 0u && (h % raw_every) == 0u) { p[i] = mix(i, seed ^ 0x5bd1e995u) << 8 | (h >> 24); return; }
    p[i] = __float_as_uint(((float)(h >> 8) / 8388608.0f - 1.0f) * scale);
}

// FP32 bits of every BF16 pattern as the device converts it, for the host model.
__global__ void bf16_to_f32_bits(unsigned int* lut) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536u) return;
    __nv_bfloat16 b;
    *(unsigned short*)&b = (unsigned short)i;
    lut[i] = __float_as_uint(__bfloat162float(b));
}

struct Tally {
    size_t n = 0, diff = 0;
    void add(bool differs) { n++; diff += differs; }
    void merge(const Tally& o) { n += o.n; diff += o.diff; }
};

// One line per (dim, kernel, seq_len), summed over the chunks.
struct Row {
    Tally out_old_new, out_new_serial, st_old_new, st_new_serial, st_new_exact;
    Tally snap_old_new, snap_new_exact, snap_old_exact, pad;
    void merge(const Row& o) {
        out_old_new.merge(o.out_old_new); out_new_serial.merge(o.out_new_serial);
        st_old_new.merge(o.st_old_new); st_new_serial.merge(o.st_new_serial);
        st_new_exact.merge(o.st_new_exact); snap_old_new.merge(o.snap_old_new);
        snap_new_exact.merge(o.snap_new_exact); snap_old_exact.merge(o.snap_old_exact);
        pad.merge(o.pad);
    }
};

typedef void (*TpK)(float*, const bf*, const bf*, const float*, bf*, unsigned, unsigned, unsigned, unsigned, unsigned);
typedef void (*SnapK)(float*, const bf*, const bf*, const float*, bf*, float*, unsigned long long,
                      unsigned, unsigned, unsigned, unsigned, unsigned);

int main(int argc, char** argv) {
    const int chunks = argc > 1 ? atoi(argv[1]) : 3;
    const unsigned snap_max_full = argc > 2 ? (unsigned)atoi(argv[2]) : 100u;
    const unsigned d_conv = 4;
    const unsigned seqs[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 63, 64, 65, 100, 4096, 8196};
    const unsigned seq_max = 8196;
    struct Cfg { unsigned dim, pad, snap_max; };
    const Cfg cfgs[] = {{24576, 0, snap_max_full}, {1001, 7, seq_max}};
    enum { OLD = 0, NEW = 1, SERIAL = 2 };
    const unsigned char sentinel[3] = {0xEE, 0xDD, 0xCC};
#ifdef CONV1D_BENCH_NO_SNAP
    const int kinds = 1;
#else
    const int kinds = 2;
#endif

    unsigned int* d_lut;
    std::vector<unsigned int> lut(65536);
    CK(cudaMalloc(&d_lut, 65536 * 4));
    bf16_to_f32_bits<<<256, 256>>>(d_lut);
    CK(cudaMemcpy(lut.data(), d_lut, 65536 * 4, cudaMemcpyDeviceToHost));

    unsigned int seed = 1;
    auto blocks = [](size_t n) { return (unsigned)((n + 255) / 256); };
    Row total[2][2];   // [kernel][seq_len > 8]
    int shown = 0;
    for (const Cfg& cfg : cfgs) {
        const unsigned dim = cfg.dim, stride = dim + cfg.pad;
        const size_t inter_stride = (size_t)dim * d_conv + cfg.pad;
        const size_t io_elems = (size_t)seq_max * stride, st_elems = (size_t)dim * d_conv;
        const size_t inter_elems = kinds == 2 ? (size_t)(cfg.snap_max - 1) * inter_stride : 1;
        bf *d_in, *d_w, *d_out[3];
        float *d_bias, *d_state[3], *d_inter[2];
        CK(cudaMalloc(&d_in, io_elems * 2));
        CK(cudaMalloc(&d_w, st_elems * 2));
        CK(cudaMalloc(&d_bias, dim * 4));
        for (auto& p : d_out) CK(cudaMalloc(&p, io_elems * 2));
        for (auto& p : d_state) CK(cudaMalloc(&p, st_elems * 4));
        for (auto& p : d_inter) CK(cudaMalloc(&p, inter_elems * 4));
        fill_bf16<<<blocks(st_elems), 256>>>((unsigned short*)d_w, st_elems, seed++, 1.0f, 0);
        fill_f32<<<blocks(dim), 256>>>((unsigned int*)d_bias, dim, seed++, 1.0f, 0);
        fill_f32<<<blocks(st_elems), 256>>>((unsigned int*)d_state[OLD], st_elems, seed++, 4.0f, 64);
        for (int v = 1; v < 3; v++) CK(cudaMemcpy(d_state[v], d_state[OLD], st_elems * 4, cudaMemcpyDeviceToDevice));

        std::vector<unsigned short> h_in(io_elems), h_out[3];
        std::vector<unsigned int> h_prev(st_elems), h_state[3], h_inter[2];
        for (auto& v : h_out) v.resize(io_elems);
        for (auto& v : h_state) v.resize(st_elems);
        for (auto& v : h_inter) v.resize(inter_elems);

        for (int kind = 0; kind < kinds; kind++) {
            for (unsigned seq : seqs) {
                if (kind == 1 && seq > cfg.snap_max) continue;
                Row row;
                for (int c = 0; c < chunks; c++) {
                    const size_t live = (size_t)seq * stride, snaps = (size_t)(seq - 1) * inter_stride;
                    const float* bias = (c & 1) ? d_bias : nullptr;
                    CK(cudaMemcpy(h_prev.data(), d_state[NEW], st_elems * 4, cudaMemcpyDeviceToHost));
                    fill_bf16<<<blocks(live), 256>>>((unsigned short*)d_in, live, seed++, 4.0f, 64);
                    for (int v = 0; v < 3; v++) CK(cudaMemset(d_out[v], sentinel[v], live * 2));
                    if (kind == 1)
                        for (int v = 0; v < 2; v++) CK(cudaMemset(d_inter[v], sentinel[v], snaps * 4));

                    cudaLaunchConfig_t lc = {};
                    lc.gridDim = dim3((dim + 31) / 32, (seq + 63) / 64, 1);
                    lc.blockDim = dim3(32, 8, 1);
                    cudaLaunchAttribute at;
                    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    at.val.programmaticStreamSerializationAllowed = 1;
                    if (kind == 1) { lc.attrs = &at; lc.numAttrs = 1; }
                    for (int v = 0; v < 2; v++) {
                        if (kind == 0) {
                            const TpK k = v == OLD ? conv1d_tp_before : causal_conv1d_update_prefill_tp;
                            CK(cudaLaunchKernelEx(&lc, k, d_state[v], (const bf*)d_in, (const bf*)d_w, bias,
                                                  d_out[v], dim, d_conv, seq, stride, stride));
                        }
#ifndef CONV1D_BENCH_NO_SNAP
                        else {
                            const SnapK k = v == OLD ? conv1d_tp_snap_before : causal_conv1d_update_prefill_tp_snap;
                            CK(cudaLaunchKernelEx(&lc, k, d_state[v], (const bf*)d_in, (const bf*)d_w, bias,
                                                  d_out[v], d_inter[v], (unsigned long long)inter_stride,
                                                  dim, d_conv, seq, stride, stride));
                        }
#endif
                    }
                    causal_conv1d_update_prefill<<<(dim + 255) / 256, 256>>>(
                        d_state[SERIAL], d_in, d_w, bias, d_out[SERIAL], dim, d_conv, seq, stride, stride);
                    CK(cudaDeviceSynchronize());

                    CK(cudaMemcpy(h_in.data(), d_in, live * 2, cudaMemcpyDeviceToHost));
                    for (int v = 0; v < 3; v++) {
                        CK(cudaMemcpy(h_out[v].data(), d_out[v], live * 2, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(h_state[v].data(), d_state[v], st_elems * 4, cudaMemcpyDeviceToHost));
                    }
                    if (kind == 1)
                        for (int v = 0; v < 2; v++)
                            CK(cudaMemcpy(h_inter[v].data(), d_inter[v], snaps * 4, cudaMemcpyDeviceToHost));

                    // x(t): the token stream for t >= 0, the incoming state window below it.
                    auto x = [&](long long t, unsigned ch) -> unsigned int {
                        if (t >= 0) return lut[h_in[(size_t)t * stride + ch]];
                        const long long idx = (long long)d_conv + t;
                        return idx >= 0 ? h_prev[(size_t)ch * d_conv + idx] : 0u;
                    };
                    // First few failures; the old body's own races (seq > 8) are only tallied.
                    const bool old_is_reference = seq <= 8;
                    auto report = [&](const char* what, size_t a, size_t b, unsigned va, unsigned vb) {
                        if (shown++ < 12)
                            printf("  DIFF %s dim=%u %s seq=%u chunk=%d at [%zu][%zu]: %08x vs %08x\n", what, dim,
                                   kind ? "snap" : "tp", seq, c, a, b, va, vb);
                    };
                    for (size_t t = 0; t < seq; t++) {
                        const size_t base = t * stride;
                        for (unsigned ch = 0; ch < dim; ch++) {
                            const unsigned short o = h_out[OLD][base + ch], n = h_out[NEW][base + ch];
                            if (old_is_reference && o != n) report("out old/new", t, ch, o, n);
                            if (n != h_out[SERIAL][base + ch]) report("out new/serial", t, ch, n, h_out[SERIAL][base + ch]);
                            row.out_old_new.add(o != n);
                            row.out_new_serial.add(n != h_out[SERIAL][base + ch]);
                        }
                        for (unsigned ch = dim; ch < stride; ch++)
                            for (int v = 0; v < 3; v++)
                                row.pad.add(h_out[v][base + ch] != (unsigned short)(sentinel[v] * 0x0101u));
                    }
                    for (unsigned ch = 0; ch < dim; ch++)
                        for (unsigned k = 0; k < d_conv; k++) {
                            const size_t i = (size_t)ch * d_conv + k;
                            const unsigned int n = h_state[NEW][i], e = x((long long)seq - d_conv + k, ch);
                            if (old_is_reference && h_state[OLD][i] != n)
                                report("state old/new", ch, k, h_state[OLD][i], n);
                            if (n != e) report("state new/exact", ch, k, n, e);
                            row.st_old_new.add(h_state[OLD][i] != n);
                            row.st_new_serial.add(n != h_state[SERIAL][i]);
                            row.st_new_exact.add(n != e);
                        }
                    if (kind == 1)
                        for (size_t t = 0; t + 1 < seq; t++) {
                            const size_t base = t * inter_stride;
                            for (unsigned ch = 0; ch < dim; ch++)
                                for (unsigned k = 0; k < d_conv; k++) {
                                    const size_t i = base + (size_t)ch * d_conv + k;
                                    const unsigned int o = h_inter[OLD][i], n = h_inter[NEW][i];
                                    const unsigned int e = x((long long)t + 1 - d_conv + k, ch);
                                    if (n != e) report("snapshot new/exact", t, (size_t)ch * d_conv + k, n, e);
                                    else if (old_is_reference && o != n)
                                        report("snapshot old/new", t, (size_t)ch * d_conv + k, o, n);
                                    row.snap_old_new.add(o != n);
                                    row.snap_new_exact.add(n != e);
                                    row.snap_old_exact.add(o != e);
                                }
                            for (size_t i = base + st_elems; i < base + inter_stride; i++)
                                for (int v = 0; v < 2; v++)
                                    row.pad.add(h_inter[v][i] != sentinel[v] * 0x01010101u);
                        }
                }
                printf("dim=%-5u %-4s seq=%-4u out old/new %zu new/serial %zu of %zu | state old/new %zu "
                       "new/serial %zu new/exact %zu of %zu",
                       dim, kind ? "snap" : "tp", seq, row.out_old_new.diff, row.out_new_serial.diff,
                       row.out_old_new.n, row.st_old_new.diff, row.st_new_serial.diff, row.st_new_exact.diff,
                       row.st_old_new.n);
                if (kind == 1)
                    printf(" | snapshot old/new %zu new/exact %zu old/exact %zu of %zu", row.snap_old_new.diff,
                           row.snap_new_exact.diff, row.snap_old_exact.diff, row.snap_old_new.n);
                printf(" | pad clobbered %zu of %zu\n", row.pad.diff, row.pad.n);
                total[kind][seq > 8].merge(row);
            }
        }
        CK(cudaFree(d_in)); CK(cudaFree(d_w)); CK(cudaFree(d_bias));
        for (auto p : d_out) CK(cudaFree(p));
        for (auto p : d_state) CK(cudaFree(p));
        for (auto p : d_inter) CK(cudaFree(p));
    }

    // Pass: the kernel under test matches the serial kernel and the exact model
    // everywhere, and matches the old body wherever the old body's state reader
    // and writer were the same thread (seq_len <= 8). Past that the old body is
    // the racy one, so where it differs it is reported, not counted.
    size_t bad = 0;
    for (int kind = 0; kind < kinds; kind++) {
        const char* name = kind ? "causal_conv1d_update_prefill_tp_snap" : "causal_conv1d_update_prefill_tp";
        Row all = total[kind][0];
        all.merge(total[kind][1]);
        printf("TOTAL %s, %d chunks per seq_len\n", name, chunks);
        printf("  vs serial kernel / exact model: outputs %zu of %zu, final state %zu + %zu of %zu",
               all.out_new_serial.diff, all.out_new_serial.n, all.st_new_serial.diff, all.st_new_exact.diff,
               all.st_new_serial.n);
        if (kind == 1) printf(", snapshots %zu of %zu", all.snap_new_exact.diff, all.snap_new_exact.n);
        printf("; pad clobbered %zu of %zu\n", all.pad.diff, all.pad.n);
        for (int wide = 0; wide < 2; wide++) {
            const Row& r = total[kind][wide];
            printf("  vs old body, seq_len %s: outputs %zu of %zu, final state %zu of %zu",
                   wide ? "> 8 (old body races)" : "<= 8", r.out_old_new.diff, r.out_old_new.n,
                   r.st_old_new.diff, r.st_old_new.n);
            if (kind == 1)
                printf(", snapshots %zu of %zu (old body vs exact model: %zu)", r.snap_old_new.diff,
                       r.snap_old_new.n, r.snap_old_exact.diff);
            printf("\n");
        }
        const Row& narrow = total[kind][0];
        bad += all.out_new_serial.diff + all.st_new_serial.diff + all.st_new_exact.diff + all.snap_new_exact.diff
             + all.pad.diff + narrow.out_old_new.diff + narrow.st_old_new.diff + narrow.snap_old_new.diff;
    }
    printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
