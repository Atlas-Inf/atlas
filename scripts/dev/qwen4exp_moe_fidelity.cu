// SPDX-License-Identifier: AGPL-3.0-only
// How close each qwen4_exp routed-MoE path comes to the exact math, on the
// real checkpoint's experts and (with a dump) real decode-time MoE inputs.
//
//   (a) reference  float64 on the CPU: w = lut[nibble] * e4m3(scale) * scale2
//                  exactly, x = the BF16 input, gate/up, SiLU * up, down, no
//                  rounding anywhere; twice: with the decode kernels' routed
//                  clamp (g <= 10, |u| <= 10) and without (the checkpoint's
//                  swiglu_limit is null)
//   (b) decode     qwen4exp_moe_rows_{plan,gate_up,silu_down} (today's verify
//                  rows; bit-identical to serial decode's single-row kernels)
//   (c) TC         qwen4exp_moe_c8_tc (ATLAS_QWEN4EXP_MOE_TC)
//   (d) prefill    moe_prefill_q38's W2 chain (ATLAS_QWEN4EXP_PREFILL_MOE[_W2]):
//                  A and SiLU*up as E4M3, W as e4m3(lut * scale * scale2)
// For every (row, slot) the expert's [2560] output (each path's BF16), and
// each row's routed sum sum_s w_s y_s (float64 over the paths' outputs):
// relative L2 error ||y - ref|| / ||ref|| and max |y - ref| over the layer's
// rows, plus the floor any BF16 output has (ref rounded to BF16).
//
// Inputs: DUMP=<file>.bin from ATLAS_QWEN4EXP_MOE_ROUTE_DUMP (rows, routing,
// weights as served); without it, synthetic rows (N(0, SIGMA=1)) and uniform
// routing -- weights still the checkpoint's. LAYERS="0 24 47" (dump layer
// order = layer order), RECS=<records a layer> (default 4).
//
// SUMMARY=1: one line a layer (all 48 cheaply). EMU=1: float64 emulations of
// prefill's three E4M3 steps alone and with per-row scales.
//
// Build/run (repo root, GB10): scripts/dev/qwen4exp_moe_fidelity.sh <model dir>
//
// Preliminary (2026-10-06, a GB10, real decode inputs (prose),
// C8, 40 rows a layer, all 48 layers): expert-output rel L2 vs float64
// -- BF16 floor 1.66e-3, (b) 2.8-3.1e-3, (c) 3.1-3.6e-3, (d) 5.7-8.2e-2;
// prefill's error is mostly W -> E4M3 (5.4-6.6e-2 alone); the decode clamp
// fires only in layer 47 (0.002% of values) and costs 1.3e-2 there.
#include "qwen4exp_moe_c8_bench.h"
#include <map>
#include <set>

// ── the checkpoint: index -> shard -> tensor bytes ──
struct Ckpt {
    std::string dir, index;
    std::map<std::string, std::string> headers;
    std::map<std::string, size_t> hlen;
    explicit Ckpt(const std::string& d) : dir(d) {
        std::ifstream f(d + "/model.safetensors.index.json");
        std::stringstream ss; ss << f.rdbuf(); index = ss.str();
    }
    std::string shard(const std::string& name) {
        std::string key = "\"" + name + "\":\"";
        size_t p = index.find(key);
        if (p == std::string::npos) { key = "\"" + name + "\": \""; p = index.find(key); }
        if (p == std::string::npos) { fprintf(stderr, "no tensor %s\n", name.c_str()); exit(1); }
        p += key.size();
        return index.substr(p, index.find('"', p) - p);
    }
    std::vector<unsigned char> get(const std::string& name) {
        const std::string file = dir + "/" + shard(name);
        if (!headers.count(file)) {
            FILE* fp = fopen(file.c_str(), "rb");
            unsigned long long n = 0;
            if (fread(&n, 8, 1, fp) != 1) exit(1);
            std::string h(n, ' ');
            if (fread(&h[0], 1, n, fp) != n) exit(1);
            fclose(fp);
            headers[file] = h; hlen[file] = n;
        }
        const std::string& h = headers[file];
        size_t p = h.find("\"" + name + "\":");
        p = h.find("\"data_offsets\":[", p) + 16;
        const size_t a = std::stoull(h.substr(p)), b = std::stoull(h.substr(h.find(',', p) + 1));
        std::vector<unsigned char> v(b - a);
        FILE* fp = fopen(file.c_str(), "rb");
        fseeko(fp, (off_t)(8 + hlen[file] + a), SEEK_SET);
        if (fread(v.data(), 1, v.size(), fp) != v.size()) exit(1);
        fclose(fp);
        return v;
    }
};

static double e4m3(unsigned char b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return s ? -v : v;
}
static const double LUT[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};
static double bf(unsigned short u) { unsigned w = (unsigned)u << 16; float f; memcpy(&f, &w, 4); return f; }
static double bf_round(double x) { return bf(tobf((float)x)); }  // the floor any BF16 output has

// One projection of one expert as served: [N, K/2] packed, [N, K/16] E4M3, scale2.
struct HostProj { std::vector<unsigned char> p, s; float s2; unsigned n, k; };
struct Expert { HostProj g, u, d; };
static HostProj load_proj(Ckpt& c, const std::string& base, unsigned n, unsigned k) {
    HostProj h{c.get(base + ".weight"), c.get(base + ".weight_scale"), 0.f, n, k};
    auto s2 = c.get(base + ".weight_scale_2");
    memcpy(&h.s2, s2.data(), 4);
    if (h.p.size() != (size_t)n * k / 2 || h.s.size() != (size_t)n * k / 16) { fprintf(stderr, "%s shape\n", base.c_str()); exit(1); }
    return h;
}
// y[n] = sum_k W[n][k] x[k], W exact in float64.
static void gemv64(const HostProj& w, const double* x, double* y) {
    for (unsigned n = 0; n < w.n; n++) {
        double acc = 0;
        for (unsigned g = 0; g < w.k / 16; g++) {
            const double sc = e4m3(w.s[(size_t)n * (w.k / 16) + g]) * (double)w.s2;
            double part = 0;
            for (unsigned i = 0; i < 8; i++) {
                const unsigned char b = w.p[(size_t)n * (w.k / 2) + g * 8 + i];
                part += LUT[b & 15] * x[g * 16 + 2 * i] + LUT[b >> 4] * x[g * 16 + 2 * i + 1];
            }
            acc += part * sc;
        }
        y[n] = acc;
    }
}

struct Rec { unsigned layer, rows; std::vector<unsigned> ids; std::vector<float> w; std::vector<unsigned short> x; };
static std::vector<Rec> read_dump(const char* path) {
    std::vector<Rec> out;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no %s\n", path); exit(1); }
    unsigned hd[5];
    while (fread(hd, 4, 5, f) == 5) {
        if (hd[0] != 0x31444D51u || hd[3] != TOPK || hd[4] != H) { fprintf(stderr, "bad record\n"); exit(1); }
        Rec r{hd[1], hd[2], std::vector<unsigned>(hd[2] * TOPK), std::vector<float>(hd[2] * TOPK),
              std::vector<unsigned short>((size_t)hd[2] * H)};
        if (fread(r.ids.data(), 4, r.ids.size(), f) != r.ids.size() || fread(r.w.data(), 4, r.w.size(), f) != r.w.size() ||
            fread(r.x.data(), 2, r.x.size(), f) != r.x.size()) break;
        out.push_back(std::move(r));
    }
    fclose(f);
    return out;
}

// Error accumulator against one reference.
struct Err { double num = 0, den = 0, mx = 0; void add(double y, double r) { num += (y - r) * (y - r); den += r * r; mx = std::max(mx, fabs(y - r)); }
             void print() const { printf(" %9.2e %9.2e |", sqrt(num / den), mx); } };

#include "qwen4exp_moe_fidelity_run.h"
