// SPDX-License-Identifier: AGPL-3.0-only
// The kernels scripts/dev/qwen4exp_moe_c8_bench.cu compares with the
// production rows pair.
#pragma once
#include "qwen4exp_moe_c8_bench.h"

// Serial decode's kernels, one launch pair per row (moe_shared_expert_fused.cu).
static Variant per_row_loop() {
    CUfunction gu = load("moe_shared_expert_fused", "moe_expert_gate_up_shared");
    CUfunction sd = load("moe_shared_expert_fused",
                         nc() ? "moe_expert_silu_down_shared_noclamp" : "moe_expert_silu_down_shared");
    Variant v;
    v.name = "per-row loop (serial)";
    v.plan = [](Pool&, Bufs&, unsigned) {};
    v.gate_up = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned n = I, k = H, topk = TOPK;
        for (unsigned r = 0; r < rows; r++) {
            void* A = (char*)b.A + (size_t)r * H * 2;
            void* go = (char*)b.gate + (size_t)r * TOPK * I * 2;
            void* uo = (char*)b.up + (size_t)r * TOPK * I * 2;
            void* ids = (char*)b.ids + (size_t)r * TOPK * 4;
            void* shg = (char*)b.shg + (size_t)r * I * 2;
            void* shu = (char*)b.shu + (size_t)r * I * 2;
            launch(gu, dim3(I / 8, TOPK + 1, 2), dim3(128),
                   {&A, &p.gp, &p.gs, &p.g2, &go, &p.upk, &p.us, &p.u2, &uo, &ids, &p.sg.packed,
                    &p.sg.scale, &p.sg.s2, &shg, &p.su.packed, &p.su.scale, &p.su.s2, &shu, &n, &k, &topk});
        }
    };
    v.silu_down = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned n = H, k = I, topk = TOPK;
        for (unsigned r = 0; r < rows; r++) {
            void* go = (char*)b.gate + (size_t)r * TOPK * I * 2;
            void* uo = (char*)b.up + (size_t)r * TOPK * I * 2;
            void* ids = (char*)b.ids + (size_t)r * TOPK * 4;
            void* shg = (char*)b.shg + (size_t)r * I * 2;
            void* shu = (char*)b.shu + (size_t)r * I * 2;
            void* dn = (char*)b.down + (size_t)r * TOPK * H * 2;
            void* shd = (char*)b.shd + (size_t)r * H * 2;
            launch(sd, dim3(H / 8, TOPK + 1, 1), dim3(128),
                   {&go, &uo, &p.dp, &p.ds, &p.d2, &dn, &ids, &shg, &shu, &p.sd.packed, &p.sd.scale,
                    &p.sd.s2, &shd, &n, &k, &topk}, I * 4);
        }
    };
    return v;
}

// The production rows pair compiled with other unit shapes (module
// qwen4exp_moe_rows_<tag>, built by the .sh with -DQU_RMAX/-DQU_SD_*).
static Variant rows_shape(const char* tag, unsigned rmax, unsigned tile, unsigned warps, unsigned rc) {
    static std::vector<std::string> names;
    names.push_back(std::string("qwen4exp_moe_rows_") + tag);
    const char* M = names.back().c_str();
    CUfunction pl = load(M, "qwen4exp_moe_rows_plan"), gu = load(M, "qwen4exp_moe_rows_gate_up"),
               sd = load(M, "qwen4exp_moe_rows_silu_down");
    Variant v = production();
    v.name = std::string("rows ") + tag;
    v.plan = [=](Pool&, Bufs& b, unsigned rows) {
        unsigned slots = rows * TOPK;
        launch(pl, dim3(1), dim3(256), {&b.ids, &b.order, &slots});
    };
    v.gate_up = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned n = I, k = H, topk = TOPK, R = rows;
        launch(gu, dim3(I / 8, rows * TOPK + rows, 2), dim3(128),
               {&b.A, &p.gp, &p.gs, &p.g2, &b.gate, &p.upk, &p.us, &p.u2, &b.up, &b.ids, &b.order,
                &p.sg.packed, &p.sg.scale, &p.sg.s2, &b.shg, &p.su.packed, &p.su.scale, &p.su.s2,
                &b.shu, &n, &k, &topk, &R});
    };
    v.silu_down = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned n = H, k = I, topk = TOPK, R = rows;
        launch(sd, dim3(H / tile, (rows + rmax - 1) / rmax + rows * TOPK, 1), dim3(warps * 32),
               {&b.gate, &b.up, &p.dp, &p.ds, &p.d2, &b.down, &b.ids, &b.order, &b.shg, &b.shu,
                &p.sd.packed, &p.sd.scale, &p.sd.s2, &b.shd, &n, &k, &topk, &R},
               tile * (I / 2 + I / 16) + rc * I * 4);
    };
    return v;
}

// Shapes the .sh builds: tag, RMAX, TILE, WARPS, RC.
static const struct { const char* tag; unsigned rmax, tile, warps, rc; } ROWS_SHAPES[] = {
    {"r8c4", 8, 64, 8, 4}, {"r8c8", 8, 64, 8, 8}, {"r16c4", 16, 64, 8, 4}, {"r32c4", 32, 64, 8, 4},
    {"r32c8", 32, 64, 8, 8}, {"r16t32c4", 16, 32, 4, 4},
};

// qwen4exp_moe_c8.cu (ATLAS_QWEN4EXP_MOE_UNITS) as ops::Qwen4ExpMoeRows::units
// launches it: plan, unit gate/up + SiLU (8 outputs a CTA), unit down (64).
// `fused`: gate/up BF16 rows not stored, as serving runs it; else stored for
// the check.
static Variant c8_units(bool fused, bool tc = false, bool v1 = false) {
    // tc: qwen4exp_moe_c8_tc.cu (ATLAS_QWEN4EXP_MOE_TC, contract (b): not the
    // rows pair's bytes; checked for row invariance by tc-units-check).
    const char* M = tc ? "qwen4exp_moe_c8_tc" : "qwen4exp_moe_c8";
    const std::string sfx = v1 ? "_v1" : "";
    const std::string p = tc ? "qwen4exp_moe_c8_tc_" : "qwen4exp_moe_c8_";
    CUfunction pl = load(M, (p + "plan" + (tc ? sfx : "")).c_str()), gu = load(M, (p + (nc() ? "gate_up_nc" : "gate_up") + sfx).c_str()),
               sd = load(M, (p + "down" + sfx).c_str());
    Variant v;
    v.name = std::string(tc ? (v1 ? "tc v1 " : "tc ") : "") + (fused ? "units fused" : "units");
    v.plan = [=](Pool&, Bufs& b, unsigned rows) {
        unsigned topk = TOPK, R = rows;
        launch(pl, dim3(1), dim3(1024), {&b.ids, &b.ws, &topk, &R});
    };
    // TC v2 strides the units: gridDim.y = min(bound, TC_Y) (default 64).
    const unsigned ycap = tc && !v1 ? (getenv("TC_Y") ? atoi(getenv("TC_Y")) : 64) : 1u << 30;
    const auto units = [=](unsigned rows) { return std::min(ycap, rows * TOPK + (rows + 15) / 16); };
    v.gate_up = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned topk = TOPK, R = rows;
        void* null = nullptr;
        void** go = fused ? &null : &b.gate;
        void** uo = fused ? &null : &b.up;
        launch(gu, dim3(I / 8, units(rows)), dim3(256),
               {&b.A, &p.gp, &p.gs, &p.g2, &p.upk, &p.us, &p.u2, &p.sg.packed, &p.sg.scale, &p.sg.s2,
                &p.su.packed, &p.su.scale, &p.su.s2, &b.ws, go, uo, &b.shg, &b.shu, &b.act, &topk, &R});
    };
    v.silu_down = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned topk = TOPK, R = rows;
        launch(sd, dim3(H / 64, units(rows)), dim3(256),
               {&b.act, &p.dp, &p.ds, &p.d2, &p.sd.packed, &p.sd.scale, &p.sd.s2, &b.ws, &b.down,
                &b.shd, &topk, &R});
    };
    return v;
}
// qwen4exp_moe_c8_tc3.cu: the v3 plan + gate/up and down in one persistent
// launch (TC3_CTAS CTAs, default 2 a SM). Its time shows as gate_up, and only
// the layer column is real: a gate_up rerun without the plan finds the claim
// counter spent (LAYER_PASSES times layers only).
static Variant tc3_units(const std::string& tag = "") {
    static std::vector<std::string> mods;
    mods.push_back(tag.empty() ? "qwen4exp_moe_c8_tc3" : "qwen4exp_moe_c8_tc3_" + tag);
    const char* M = mods.back().c_str();
    CUfunction pl = load(M, "qwen4exp_moe_c8_tc3_plan"), fu = load(M, nc() ? "qwen4exp_moe_c8_tc3_nc" : "qwen4exp_moe_c8_tc3");
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    // A tag "x<n>..." launches n CTAs.
    const unsigned ctas = tag.size() > 1 && tag[0] == 'x' ? atoi(tag.c_str() + 1)
                          : getenv("TC3_CTAS")         ? atoi(getenv("TC3_CTAS"))
                                                       : 2 * sms;
    Variant v;
    v.name = "tc3 " + (tag.empty() ? std::string("") : tag + " ") + "fused";
    v.plan = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned topk = TOPK, R = rows;
        launch(pl, dim3(1), dim3(1024), {&b.ids, &p.gp, &b.ws, &topk, &R});
    };
    v.gate_up = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned topk = TOPK, R = rows;
        launch(fu, dim3(ctas), dim3(256),
               {&b.A, &p.gp, &p.gs, &p.g2, &p.upk, &p.us, &p.u2, &p.dp, &p.ds, &p.d2, &p.sg.packed,
                &p.sg.scale, &p.sg.s2, &p.su.packed, &p.su.scale, &p.su.s2, &p.sd.packed, &p.sd.scale,
                &p.sd.s2, &b.ws, &b.act, &b.down, &b.shd, &topk, &R});
    };
    v.silu_down = [](Pool&, Bufs&, unsigned) {};
    return v;
}

std::vector<Variant> variants() {
    std::vector<Variant> v;
    v.push_back(per_row_loop());
    if (getenv("ROWS_SHAPES"))
        for (auto& s : ROWS_SHAPES) v.push_back(rows_shape(s.tag, s.rmax, s.tile, s.warps, s.rc));
    v.push_back(c8_units(false));
    v.push_back(c8_units(true));
    v.push_back(c8_units(true, true, true));
    v.push_back(c8_units(true, true));
    // NO_TC3=1: without tc3; TC3_TAGS=a,b: extra builds of it, modules
    // qwen4exp_moe_c8_tc3_<tag>.ptx (a tag "x<n>..." launches n CTAs).
    if (!getenv("NO_TC3") || !atoi(getenv("NO_TC3"))) v.push_back(tc3_units());
    if (const char* tags = getenv("TC3_TAGS")) {
        std::stringstream ss(tags);
        std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(tc3_units(t));
    }
    return v;
}
