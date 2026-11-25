// SPDX-License-Identifier: AGPL-3.0-only

//! Ling-3.2-air (Bailing-Moe-V3-VL) family driver — the decode-path wiring
//! that turns the `BailingWeightLoader`'s validated checkpoint plan plus the
//! four Ling kernel files (`kernels/gb10/ling-3.2-air/nvfp4/`) into a
//! serving-capable layer stack. GLM-5 (`weight_loader/glm5/layers.rs` +
//! `layers/glm5_kda`) is the operative precedent; the house pattern keeps
//! family logic beside `model/` and the loader in `weight_loader/bailing/`.
//!
//! Scope (campaign slice `family-driver`): DECODE-ONLY, TEXT-ONLY,
//! CONSERVATIVE — `spark serve` loads the checkpoint and generates correct
//! greedy/sampled tokens for single- and multi-turn decode with ISL
//! 4096-class prompts. Out of scope, seams left documented:
//!
//! * MTP drafting is wired for the single-NEXTN shape (the MTP-wiring
//!   campaign slice): `weight_loader::bailing::load_ling_mtp_module` builds
//!   layer 42's body via this driver, `layers::ling_mtp::LingMtpHead`
//!   implements the house [`crate::speculative::DraftProposer`] contract on
//!   it, and the family's layers serve the K = 2 verify
//!   (`kda_recurrent_bf16_verify_snap` rollback snapshots on KDA,
//!   multi-row paged attention on gated MLA). `load_mtp_weights` stays
//!   `None`: the Qwen-shaped `MtpWeights` does not fit the NEXTN module,
//!   same as GLM-5's and V4's dedicated loaders. Deeper draft counts
//!   (`verify_max_drafts`), the KDA slot pool and captured graphs remain
//!   out of scope.
//! * Slot-pool KDA decode (`ling_kda_recurrent_indexed` + the GLM
//!   `SsmStatePool` multi-seq machinery). This driver uses the GLM bring-up
//!   pattern instead: per-sequence `SsmLayerState` from `alloc_state` on the
//!   non-pool path; wiring the pool kernels is the MTP/verify follow-up.
//! * CUDA-graph decode: both layer kinds report `decode_graph_unsupported`
//!   so the scheduler stays eager (correct before fast; the pool machinery
//!   above is the prerequisite for capture).
//! * Vision (the loader skips `model.visual.*` / `linear_proj.*`).
//!
//! Decode step sequence, per layer in plan order
//! (embed is the generic model's; norms are `rms_norm_vanilla`; the residual
//! add is `bf16_residual_add`; the final norm + lm_head are the generic
//! model loop's):
//!
//! 1. `hidden` row(s) — RMS-norm by `input_layernorm`.
//! 2. Attention half:
//!    * KDA (35 layers): GEMV/GEMM projections `q/k/v/b/f/g/o_proj` (common
//!      `gemv`/`gemm` modules), causal short conv (`causal_conv1d`) fused
//!      SiLU, `kda_recurrent_bf16` (module `kda`) advancing the sequence's
//!      FP32 `[32, 128, 128]` state, `kda_sigmoid_gated_rms_norm` o_norm
//!      gated by `g_proj`, `o_proj` GEMV/GEMM.
//!    * gated MLA (7 layers): GEMV/GEMM `q_proj`/`kv_a_proj_with_mqa`/
//!      `g_proj`, `ling_mla_kv_a_norm_rope_cache` writing the paged K cache
//!      `[latent 512 | rope 64]` and the layer-owned V latent `[512]` pool,
//!      `ling_mla_q_absorb_rope` with the load-time-transposed `w_uk`,
//!      `ling_mla_paged_attn` (absorbed, 192^-0.5 scale),
//!      `ling_mla_v_extract_gated` (head-wise sigmoid gate),
//!      `dense` (o_proj) GEMV/GEMM.
//! 3. Residual add; RMS-norm by `post_attention_layernorm`.
//! 4. FFN half: dense layers 0-1 via the MoE shared-expert kernels
//!    (`ling_moe_shared_*`, limit 0); MoE layers via `ling_moe_route` ->
//!    `ling_moe_gate_up_silu` -> `ling_moe_down` -> `ling_moe_shared_*`
//!    (per-layer SwiGLU limits from the checkpoint lists) ->
//!    `ling_moe_unpermute`.
//! 5. Residual add.
//!
//! Module-to-kernel-name table (the launch contract this driver relies on):
//!
//! | module (file stem) | kernels used |
//! |---|---|
//! | `kda` | `kda_recurrent_bf16`, `kda_sigmoid_gated_rms_norm` |
//! | `gated_mla` | `ling_mla_kv_a_norm_rope_cache`, `ling_mla_q_absorb_rope`, `ling_mla_paged_attn`, `ling_mla_v_extract_gated`, `ling_mla_prefill_attn` |
//! | `moe_decode` | `ling_moe_route`, `ling_moe_gate_up_silu`, `ling_moe_down`, `ling_moe_shared_gate_up_r{1,4,8,16,32}`, `ling_moe_shared_down_r{1,4,8,16,32}`, `ling_moe_unpermute` |
//! | `causal_conv1d` | `causal_conv1d_update`, `causal_conv1d_update_prefill` |
//! | `gemv` | `dense_gemv_bf16` |
//! | `gemm` | `dense_gemm_bf16` |
//! | `rms_norm_vanilla` | `rms_norm_vanilla` |
//! | `residual_add` | `bf16_residual_add` |
//! | `moe_transpose_batched` | `moe_transpose_u8_batched` (load-time expert repack) |
//!
//! The drafter side's kernel table (the NEXTN-chain glue;
//! `layers/ling_mtp.rs` resolves and owns these handles):
//!
//! | module (file stem) | kernels used |
//! |---|---|
//! | `ling_mtp_eh` | `ling_mtp_eh_norm`, `ling_mtp_draft_pick` |
//!
//! On the VERIFY side the KDA layers add
//! `kda_recurrent_bf16_verify_snap` (module `kda`) for the in-place K-row
//! walk with per-row H rollback snapshots.

mod ffn;
mod kda;
mod kernels;
mod mla;
mod scratch;
mod spec;

#[cfg(test)]
#[path = "bailing_c1/kda_parity_gpu_tests.rs"]
mod kda_parity_gpu_tests;

pub(crate) use ffn::{LingDenseWeights, LingExpertTable, LingFfn, LingMoeWeights};
pub(crate) use kda::{LingKdaLayer, LingKdaWeights};
pub(crate) use kernels::LingKernels;
pub(crate) use mla::{LingMlaLayer, LingMlaWeights};
pub(crate) use scratch::{LingScratch, MLA_MAX_ROWS_PAGED};
pub(crate) use spec::{FfnKind, KDA_HEAD_DIM, KDA_HEADS, LayerSpec, LingSpec, PrefillPlan};

#[cfg(test)]
#[path = "bailing_c1_tests.rs"]
mod tests;
