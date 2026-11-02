// SPDX-License-Identifier: AGPL-3.0-only

// tc2's LEAN twin for TP1 (two kv heads per rank): the QK warps' Q fragments
// live in registers instead of a 17 KB shared tile, and K is single-buffered
// (tc2 issues the K[i+1] prefetch after the mid-iteration barrier, so the
// second buffer never overlapped anything). Shared memory falls from ~69 KB
// to ~36 KB, so two CTAs share an SM instead of one.
//
// Same source, same per-element arithmetic: every output byte equals
// `qsa_prefill_attn_tc2`'s (`scripts/dev/qwen4exp_qsa_tc2r_bench.cu`).
// Measured on GB10, 2048 rows at position 14000, 24 q / 2 kv: 20.9 -> 12.0 ms.
// Opt-in: `ATLAS_QWEN4EXP_PREFILL_QSA_LEAN=1`, where tc2 is the arm in force.
//
// Grid: (1, rows, 1), Block: (128, 1, 1).

#define QSA_TC2_LEAN 1
#define ATLAS_PREFILL_ENTRY qsa_prefill_attn_tc2l
#include "qsa_attn_tc2.cu"
