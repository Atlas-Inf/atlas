// SPDX-License-Identifier: AGPL-3.0-only

// QSA prefill attention on tensor cores at TP2: tc2's kernel with its two
// M-tile halves holding two ROWS of a one-kv-head rank instead of the two kv
// heads of one row.
//
// WHY. tc2 is the NVIDIA default at TP1, and it needs `num_kv_heads == 2`: it
// packs kv head 0's twelve q-heads into rows 0..11 of a BR=32 tile and kv head
// 1's into rows 16..27. At TP=EP=2 each rank holds ONE kv head (24 q / 2 kv
// split by heads), `qsa_prefill_attn_tc2_ok` fails and prefill falls all the
// way back to the scalar `qsa_prefill_attn_g` -- 1.49 s, 16.7% of a 16K cold
// prefill on the pair (nsys rank 0, 2026-10-05).
//
// HOW. Neighbouring rows select DIFFERENT blocks (QSA selects per row), so the
// second half cannot share the first half's keys -- but tc2's halves never
// shared keys either: each half already has its own K tile
// (`smem_K[buf][half]`) and V tile (`smem_V[half]`), and only the ADDRESS of
// each cached token differed (`kvh * head_dim`). Here the half index picks the
// row's block list instead. Everything a warp computes -- QK^T, the masked
// online softmax, the rescale, P@V, the final 1/l -- is tc2's code, compiled
// from the same source (`qsa_attn_tc2.cu` under `QSA_TC2_ROWPAIR`).
//
// EXACTNESS. Bit-identical, per (row, head), to TP1's default tc2 for the same
// Q / K / V / list: each output element is the same sequence of
// `mma.sync.m16n8k16` and FP32 operations over the same 16-key blocks in the
// same order. The two rows of a pair can differ by one 16-key block (their
// tails differ); the shorter half sits that block out warp-uniformly rather
// than running a fully masked block, so not even a `* 1.0f` is added.
// `scripts/dev/qwen4exp_qsa_tc2r_bench.cu` compares every output byte against
// tc2 run on two-kv-head data whose kv head 0 is this data.
//
// It is NOT bit-identical to `qsa_prefill_attn_g` -- the default at TP2 today
// -- for the reason tc2 is not (a different summation tree), so it is opt-in:
// `ATLAS_QWEN4EXP_PREFILL_QSA_TC2R=1`.
//
// It is also LEAN (`QSA_TC2_LEAN`, see `qsa_attn_tc2l.cu`): Q fragments in
// registers and one K buffer, ~36 KB of shared instead of ~69 KB, so two CTAs
// share an SM. Measured on GB10 (2048 rows at position 14000, 12 q / 1 kv):
// `_g` 21.2 ms, tc2r without LEAN 12.7 ms, with it 5.85 ms.
//
// Grid: (1, ceil(rows / 2), 1), Block: (128, 1, 1).

#define QSA_TC2_ROWPAIR 1
#define QSA_TC2_LEAN 1
#define ATLAS_PREFILL_ENTRY qsa_prefill_attn_tc2r
#include "qsa_attn_tc2.cu"
