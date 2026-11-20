// SPDX-License-Identifier: AGPL-3.0-only

//! The arena of a model whose layers keep their own working buffers.
//!
//! DeepSeek-V4.1's layers allocate what they work in (`deepseek_v41::shape`)
//! and read of the arena only a batch's embeddings, its streams and their
//! mix, its token ids and, with the driver, a few normed rows, the logits
//! and the metadata scratch. The members below are the working buffers of
//! the other families' attention, state-space and MoE layers: sized for
//! this model's shape they are 3.4 GB of a 4.3 GB arena at 8,200 tokens,
//! and nothing on its paths reads or writes one. They are kept at the
//! arena's smallest size, not zero, so each stays an allocation of its own
//! address.

use atlas_core::config::ModelConfig;

use super::BufferSizes;

/// The smallest size the arena gives a member it does not use.
const UNUSED: usize = 256;

impl BufferSizes {
    /// These sizes as `config`'s model needs them.
    pub(super) fn for_model(mut self, config: &ModelConfig) -> Self {
        if config.model_type != "deepseek_v41" {
            return self;
        }
        for member in [
            &mut self.qkv_output,
            &mut self.attn_output,
            &mut self.ssm_deinterleaved,
            &mut self.ssm_conv_out_f32,
            &mut self.ssm_qkvz,
            &mut self.ssm_ba,
            &mut self.gate_logits,
            &mut self.gate_logits_f32,
            &mut self.moe_router_in_f32,
            &mut self.expert_gate_out,
            &mut self.expert_up_out,
            &mut self.expert_down_out,
            &mut self.moe_output,
            &mut self.o_latent,
            &mut self.fp8_act,
            &mut self.fp8_act_scale,
        ] {
            *member = (*member).min(UNUSED);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use atlas_core::config::ModelConfig;

    use super::super::BufferSizes;

    /// A config of DeepSeek-V4.1's shape where the arena reads it.
    fn shaped(model_type: &str) -> ModelConfig {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.model_type = model_type.into();
        c.hidden_size = 5120;
        c.hc_mult = 4;
        c
    }

    #[test]
    fn deepseek_v41_keeps_what_it_reads_and_no_scratch_of_another_family() {
        let rows = 8200;
        let of =
            |model_type: &str| BufferSizes::from_config(&shaped(model_type), rows, 34_000, 16, 1);
        let (own, other) = (of("deepseek_v41"), of("deepseek_v4"));
        // What its layers and the driver read is as any model's.
        assert_eq!(own.hidden_states, other.hidden_states);
        assert_eq!(own.residual, other.residual);
        assert_eq!(own.norm_output, other.norm_output);
        assert_eq!(own.hc_streams, other.hc_streams);
        assert_eq!(own.hc_post, other.hc_post);
        assert_eq!(own.token_ids, other.token_ids);
        assert_eq!(own.logits, other.logits);
        assert_eq!(own.scratch, other.scratch);
        assert!(own.hidden_states >= rows * 5120 * 2 && own.hc_post >= rows * 4 * 4);
        // The other families' working buffers are not sized for a batch,
        // and none is empty.
        for (name, mine, theirs) in [
            ("qkv_output", own.qkv_output, other.qkv_output),
            ("attn_output", own.attn_output, other.attn_output),
            (
                "ssm_deinterleaved",
                own.ssm_deinterleaved,
                other.ssm_deinterleaved,
            ),
            (
                "expert_gate_out",
                own.expert_gate_out,
                other.expert_gate_out,
            ),
            ("expert_up_out", own.expert_up_out, other.expert_up_out),
            (
                "expert_down_out",
                own.expert_down_out,
                other.expert_down_out,
            ),
            ("moe_output", own.moe_output, other.moe_output),
            ("fp8_act", own.fp8_act, other.fp8_act),
        ] {
            assert!(theirs > rows, "{name}: {theirs} bytes for another model");
            assert!(mine > 0 && mine <= super::UNUSED, "{name}: {mine} bytes");
        }
    }
}
