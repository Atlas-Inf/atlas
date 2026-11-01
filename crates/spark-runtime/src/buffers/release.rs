// SPDX-License-Identifier: AGPL-3.0-only

//! `BufferArena` teardown (`ModelResource::release`). Split from `buffers.rs`
//! (500-LoC cap).

use super::BufferArena;
use crate::gpu::{DevicePtr, GpuBackend};

/// Release every buffer this arena owns.
///
/// The destructure below is **exhaustive on purpose — no `..`**. A buffer added
/// to `BufferArena` without a matching free is a leak that only shows up as the
/// next model failing to fit, so the compiler is made to refuse the addition
/// instead. If this line stops compiling, the fix is to free the new field, not
/// to add a wildcard.
impl atlas_core::scope::ModelResource<dyn GpuBackend> for BufferArena {
    fn label(&self) -> &'static str {
        "buffer arena"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let Self {
            // Not allocations — named rather than wildcarded so the
            // exhaustiveness check above keeps its teeth.
            sizes: _,
            max_batch_tokens: _,
            dirty_rows: _,
            // Layout, not an allocation — derived from `--max-batch-size`.
            decode_meta: _,
            hidden_states,
            residual,
            norm_output,
            qkv_output,
            attn_output,
            gate_logits,
            gate_logits_f32,
            moe_router_in_f32,
            moe_output,
            logits,
            ssm_qkvz,
            ssm_ba,
            ssm_deinterleaved,
            ssm_gates,
            ssm_conv_out_f32,
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            splitk_workspace,
            o_latent,
            norm_unit_w,
            hc_streams,
            hc_post,
            hc_comb,
            hc_lowrank_scratch,
            qsa_select_scratch,
            gdn_fla_scratch,
            ssd_scratch,
            token_ids,
            ffn_act_q8,
            ffn_act_a,
            ffn_act_scale,
            fp8_act,
            fp8_act_scale,
            lora_xa,
            lora_delta,
            lora_hact,
            lora_seq_slot,
            q2_dequant_scratch,
            q2_act_q8,
            glm_latent_bf16,
            glm_identity_table,
            glm_latent_tokens: _,
            verify_masks,
            verify_masks_bytes: _,
        } = self;
        // Every pointer, then NULL it: `release` must be idempotent because a
        // `Drop` backstop may call it again, and `free` already no-ops on NULL.
        let owned = [
            *hidden_states,
            *residual,
            *norm_output,
            *qkv_output,
            *attn_output,
            *gate_logits,
            *gate_logits_f32,
            *moe_router_in_f32,
            *moe_output,
            *logits,
            *ssm_qkvz,
            *ssm_ba,
            *ssm_deinterleaved,
            *ssm_gates,
            *ssm_conv_out_f32,
            *scratch,
            *expert_gate_out,
            *expert_up_out,
            *expert_down_out,
            *splitk_workspace,
            *o_latent,
            *norm_unit_w,
            *hc_streams,
            *hc_lowrank_scratch,
            *qsa_select_scratch,
            *hc_post,
            *hc_comb,
            *gdn_fla_scratch,
            *ssd_scratch,
            *token_ids,
            *ffn_act_q8,
            *ffn_act_a,
            *ffn_act_scale,
            *fp8_act,
            *fp8_act_scale,
            *lora_xa,
            *lora_delta,
            *lora_hact,
            *lora_seq_slot,
            *q2_dequant_scratch,
            *q2_act_q8,
            *glm_latent_bf16,
            *glm_identity_table,
            *verify_masks,
        ];
        let mut first_error = None;
        for ptr in owned {
            if let Err(e) = gpu.free(ptr)
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        *hidden_states = DevicePtr::NULL;
        *residual = DevicePtr::NULL;
        *norm_output = DevicePtr::NULL;
        *qkv_output = DevicePtr::NULL;
        *attn_output = DevicePtr::NULL;
        *gate_logits = DevicePtr::NULL;
        *gate_logits_f32 = DevicePtr::NULL;
        *moe_router_in_f32 = DevicePtr::NULL;
        *moe_output = DevicePtr::NULL;
        *logits = DevicePtr::NULL;
        *ssm_qkvz = DevicePtr::NULL;
        *ssm_ba = DevicePtr::NULL;
        *ssm_deinterleaved = DevicePtr::NULL;
        *ssm_gates = DevicePtr::NULL;
        *ssm_conv_out_f32 = DevicePtr::NULL;
        *scratch = DevicePtr::NULL;
        *expert_gate_out = DevicePtr::NULL;
        *expert_up_out = DevicePtr::NULL;
        *expert_down_out = DevicePtr::NULL;
        *splitk_workspace = DevicePtr::NULL;
        *o_latent = DevicePtr::NULL;
        *norm_unit_w = DevicePtr::NULL;
        *hc_streams = DevicePtr::NULL;
        *hc_lowrank_scratch = DevicePtr::NULL;
        *qsa_select_scratch = DevicePtr::NULL;
        *hc_post = DevicePtr::NULL;
        *hc_comb = DevicePtr::NULL;
        *gdn_fla_scratch = DevicePtr::NULL;
        *ssd_scratch = DevicePtr::NULL;
        *token_ids = DevicePtr::NULL;
        *ffn_act_q8 = DevicePtr::NULL;
        *ffn_act_a = DevicePtr::NULL;
        *ffn_act_scale = DevicePtr::NULL;
        *fp8_act = DevicePtr::NULL;
        *fp8_act_scale = DevicePtr::NULL;
        *lora_xa = DevicePtr::NULL;
        *lora_delta = DevicePtr::NULL;
        *lora_hact = DevicePtr::NULL;
        *lora_seq_slot = DevicePtr::NULL;
        *q2_dequant_scratch = DevicePtr::NULL;
        *q2_act_q8 = DevicePtr::NULL;
        *glm_latent_bf16 = DevicePtr::NULL;
        *glm_identity_table = DevicePtr::NULL;
        *verify_masks = DevicePtr::NULL;
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
