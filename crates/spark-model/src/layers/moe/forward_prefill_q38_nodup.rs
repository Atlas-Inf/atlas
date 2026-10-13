// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_MOE_NODUP=1` (with `ATLAS_QWEN4EXP_PREFILL_MOE=1`;
//! both ranks must agree, `startup_parity`): Qwen3.8-Flash-Next's routed-MoE
//! prefill without the K-major duplicate of the routed experts.
//!
//! The startup transpose pass (`factory::m2_setup`) builds a `[K/2, N]` /
//! `[K/16, N]` copy of every local routed expert for the prefill GEMMs and
//! keeps the checkpoint's `[N, K/2]` / `[N, K/16]` planes for decode: 31.6 GiB
//! a rank at TP=EP=2 (256 experts x 48 layers x 2.76 MB), 37.2 GiB of free
//! memory as the per-expert FULL tier allocates it. Under the switch the pass
//! transposes only the shared expert, and the q38 routed chain reads the
//! checkpoint planes through the `moe_q38{,w}n_*` twins
//! (`moe_prefill_q38.cu`, NM in `q38_tile`): every FP8 weight value, every
//! MMA and its order are the K-major kernels', so the activation and down
//! bytes are identical (`scripts/dev/qwen4exp_moe_nodup_bench.sh`, which also
//! checks row invariance: a chunk's first third alone gives the whole
//! chunk's rows). Decode never read the copy.
//!
//! GB10, 256 local experts, W2, gate_up+SiLU + down (a->e4m3 0.53 ms aside):
//!
//! | chunk tokens | K-major | N-major |
//! |---|---|---|
//! | 16000 | 13.67 ms | 13.82 ms (+1.1%) |
//! | 8192 | 8.06 | 8.39 (+4%) |
//! | 2048 | 4.14 | 4.85 (+17%) |
//! | 512 | 3.72 | 4.17 (+12%) |
//!
//! Below ~4K tokens the chain is weight-bandwidth bound, and a CTA's
//! scattered 64-byte reads of 128 checkpoint rows reach ~170 GB/s where the
//! K-major tiles' neighbouring reads reach ~190.
//!
//! `ATLAS_QWEN4EXP_PREFILL_MOE_CHECK` compares against the default chain on
//! the K-major copy, so it does not run under the switch.

use super::*;

/// `ATLAS_QWEN4EXP_PREFILL_MOE_NODUP=1`, read once.
pub(crate) fn nodup_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::forward_prefill_routed::env_flag("ATLAS_QWEN4EXP_PREFILL_MOE_NODUP"))
}

/// Whether the startup transpose pass leaves `config`'s routed experts in
/// their checkpoint layout: the switch, the q38 prefill arm that reads them,
/// and this model.
pub(crate) fn skips_routed_transpose(config: &atlas_core::config::ModelConfig) -> bool {
    nodup_requested() && super::q38_requested() && config.model_type == "qwen4_exp"
}

/// The q38 routed entry `name` (after the W2 choice), or its N-major twin.
pub(super) fn routed_entry(name: &'static str, nm: bool) -> &'static str {
    match (nm, name) {
        (false, _) => name,
        (true, "moe_q38_gate_up_silu") => "moe_q38n_gate_up_silu",
        (true, "moe_q38w_gate_up_silu") => "moe_q38wn_gate_up_silu",
        (true, "moe_q38_down") => "moe_q38n_down",
        (true, "moe_q38w_down") => "moe_q38wn_down",
        (true, other) => other,
    }
}

impl MoeLayer {
    /// The (gate, up, down) tables the q38 routed chain reads, and whether
    /// they are the checkpoint's N-major planes: the K-major copy when the
    /// transpose pass built it, else the originals under the switch. The
    /// N-major kernels load 64-byte packed and 8-byte scale runs of four K
    /// steps, so they need both K (hidden and `inter`) to be multiples of
    /// 128 (the startup pass checked the planes' alignment).
    pub(super) fn q38_routed_tables(
        &self,
        h: u32,
        inter: u32,
    ) -> Option<([&ExpertPtrTable; 3], bool)> {
        match (&self.gate_ptrs_t, &self.up_ptrs_t, &self.down_ptrs_t) {
            (Some(g), Some(u), Some(d)) => Some(([g, u, d], false)),
            (None, None, None)
                if nodup_requested() && h.is_multiple_of(128) && inter.is_multiple_of(128) =>
            {
                Some(([&self.gate_ptrs, &self.up_ptrs, &self.down_ptrs], true))
            }
            _ => None,
        }
    }

    /// The startup check behind [`Self::q38_routed_tables`]' N-major arm:
    /// every local expert's packed planes 16-byte and scale planes 8-byte
    /// aligned (the kernels' cp.async sizes), NVFP4 scales.
    pub(super) fn check_q38_nodup_planes(&self) -> Result<()> {
        anyhow::ensure!(
            self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4,
            "ATLAS_QWEN4EXP_PREFILL_MOE_NODUP: routed experts are not NVFP4"
        );
        for (e, expert) in self.weights.experts.iter().enumerate() {
            if expert.gate_proj.is_null() {
                continue;
            }
            for w in [&expert.gate_proj, &expert.up_proj, &expert.down_proj] {
                anyhow::ensure!(
                    w.weight.0 % 16 == 0 && w.weight_scale.0 % 8 == 0,
                    "ATLAS_QWEN4EXP_PREFILL_MOE_NODUP: expert {e} planes misaligned \
                     (packed {:#x}, scales {:#x})",
                    w.weight.0,
                    w.weight_scale.0
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::routed_entry;

    #[test]
    fn routed_entries_map_to_their_n_major_twins() {
        assert_eq!(routed_entry("moe_q38w_down", false), "moe_q38w_down");
        assert_eq!(
            routed_entry("moe_q38_gate_up_silu", true),
            "moe_q38n_gate_up_silu"
        );
        assert_eq!(
            routed_entry("moe_q38w_gate_up_silu", true),
            "moe_q38wn_gate_up_silu"
        );
        assert_eq!(routed_entry("moe_q38_down", true), "moe_q38n_down");
        assert_eq!(routed_entry("moe_q38w_down", true), "moe_q38wn_down");
    }
}
