// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from vcruz305/exllamav3 @ 047ce72 (fork of turboderp-org/exllamav3, MIT,
// Copyright (c) 2025 Turboderp); see ../NOTICE.md.
// Source: exllamav3_ext/quant/exl3_moe_coop.cuh. Modifications: kept only the activation/launch
// defines and struct MoeCoopParams (verbatim); dropped the ATen host API; added CEIL_DIVIDE (util.h)
// and layout static_asserts pinned by the Rust mirror (layers/ops/exl3_coopmk.rs).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

#ifndef CEIL_DIVIDE
#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#endif

#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3

#define MOE_COOP_THREADS 512
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp (32-column groups per block)
#define MOE_COOP_COLS (MOE_COOP_WNT * 16)

struct MoeCoopParams
{
    // Inputs
    const half* x;              // (bsz, H), row stride x_stride
    int x_stride;
    const int64_t* sel;         // (bsz, topk) global expert indices
    const half* rw;             // (bsz, topk) routing weights
    int bsz;
    int topk;
    int H;                      // x width (the experts' input width before padding)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: no expert-range filtering; else sel in [min, max) is local
    int max_expert;

    // Per-expert pointer tables (int64 device arrays indexed by local expert)
    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    const int64_t* g_bias;      // nullable
    const int64_t* u_bias;
    const int64_t* d_bias;
    int act;                    // MOE_COOP_ACT_*
    float act_limit;
    bool gated;                 // false: no gate projection, activation is relu(u) * u

    // Scratch (leading dim >= bsz * topk)
    half* had_g;                // (slots, Hi) rotated gate input (bsz > 1, written by the rot kernel)
    half* had_u;                // (slots, Hi) rotated up input
    bool a_global;              // true: the GEMV reads had_g / had_u; false (bsz 1): rotated in-block
    void* gu_g;                 // (slots, I) gate GEMV output, half or float (gu_f32)
    void* gu_u;                 // (slots, I)
    bool gu_f32;
    half* act_out;              // (slots, I) activation, rotated for the down projection
    float* d_out;               // (slots, Ho) down GEMV output before the output rotation
    int* ctr_a;                 // (slots_max, I / 128) completion counters of the gate/up stage
    int* ctr_b;                 // (MAX_BSZN, Ho / 128) completion counters of the down stage
    int ctr_a_len;
    int ctr_b_len;
    int ksplit_a;               // split-k blocks per chunk per stage (launcher; partial rows at slot + q * slots)
    int ksplit_b;
    int dbg;                    // diagnostics (EXL3_MOE_COOP_DBG): 1 skip A epilogue, 2 skip B reduction
    int* runs;                  // run table (bsz > 1): [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    int slots_max;
    int rows_max;               // token rows the output scratch holds
    int n_local;                // entries in the pointer tables
    int sh_gate_n;              // shared gate weight length (checked against H per call)

    // Output
    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;

    // Shared expert (nullable): out += gate * sh_out, gate = sigmoid(x . sh_gate_w) or 1
    const float* sh_out;        // (bsz, H), row stride H
    const half* sh_gate_w;      // (H,) or null
};

// Layout pinned against the Rust #[repr(C)] mirror (sizes/offsets on LP64).
static_assert(sizeof(MoeCoopParams) == 344, "MoeCoopParams size changed: update the Rust mirror");
static_assert(offsetof(MoeCoopParams, g_trellis) == 72, "MoeCoopParams layout");
static_assert(offsetof(MoeCoopParams, act) == 168, "MoeCoopParams layout");
static_assert(offsetof(MoeCoopParams, had_g) == 184, "MoeCoopParams layout");
static_assert(offsetof(MoeCoopParams, gu_g) == 208, "MoeCoopParams layout");
static_assert(offsetof(MoeCoopParams, act_out) == 232, "MoeCoopParams layout");
static_assert(offsetof(MoeCoopParams, runs) == 288, "MoeCoopParams layout");
static_assert(offsetof(MoeCoopParams, out) == 312, "MoeCoopParams layout");
static_assert(offsetof(MoeCoopParams, sh_out) == 328, "MoeCoopParams layout");
