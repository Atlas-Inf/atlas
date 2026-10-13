// SPDX-License-Identifier: AGPL-3.0-only

//! Which tensors [`SafetensorsLoader`] does NOT load: EP-remote experts and
//! the opt-in skips (appended layer, MTP head, activation scales).

use super::{SafetensorsLoader, parse_expert_index};

impl SafetensorsLoader {
    /// Check if a tensor should be skipped under EP.
    /// Skips `*.experts.{E}.*` tensors where E is not in local range.
    /// MTP head experts are never skipped (small, fully replicated).
    pub(super) fn should_skip_tensor(&self, name: &str) -> bool {
        if self
            .skip_layer_prefix
            .as_ref()
            .is_some_and(|prefix| name.starts_with(prefix))
        {
            return true;
        }
        // MTP head weights for a model whose loader does not build one.
        if self.skip_mtp && name.starts_with("mtp.") {
            return true;
        }
        // W4A4 activation scales: never read on the w4a16 path (the NVFP4
        // loader falls back to `DevicePtr::NULL`), and 4-byte allocations are
        // almost pure granule padding at expert scale.
        if self.skip_activation_scales && name.ends_with(".input_scale") {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        // MTP head experts are small — always replicate, never shard.
        if name.starts_with("mtp.") {
            return false;
        }
        // Parse expert index from patterns like "*.experts.42.gate_proj*"
        if let Some(idx) = parse_expert_index(name) {
            if self
                .replicated_expert_prefix
                .as_ref()
                .is_some_and(|prefix| name.contains(prefix))
            {
                return false;
            }
            if self
                .rank0_only_expert_prefix
                .as_ref()
                .is_some_and(|prefix| name.contains(prefix))
            {
                return self.ep_rank != 0;
            }
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false // Non-expert tensors are always loaded (replicated)
        }
    }
}
