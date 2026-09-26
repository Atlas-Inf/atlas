// SPDX-License-Identifier: AGPL-3.0-only

// Capability marker for the persistent-m-tile MoE prefill grid. Never launched.
//
// `moe_w4a16_grouped_gemm_ptrtable_t_k64` and `moe_w4a16_fused_gate_up_t_k64`
// stride `blockIdx.y` over m-tiles, so a grid sized by the AVERAGE expert still
// computes every row instead of dropping the ones past it. That saved 0.87 s of
// a 18.45 s prefill at 29,671 tokens, byte-identical output.
//
// The striding lives in `moe_w4a16_grouped_gemm.cu`, which is a SYMLINK shared
// by seven model targets -- so its presence cannot say who was measured. This
// marker is a real file in ONE model directory, and the host shortens the grid
// only where it resolves (`try_kernel` gives handle 0 elsewhere). The other six
// keep today's grid, where the striding loop runs exactly once per CTA and is
// bit-identical to what they ran before; they can opt in behind
// ATLAS_MOE_PREFILL_PERSIST_TILES once someone measures them.
extern "C" __global__ void moe_w4a16_k64_strides_m_tiles(void) {}
