// SPDX-License-Identifier: AGPL-3.0-only

//! Which tensors the fast loader does NOT upload.
//!
//! Four independent rules, worth reading together because each one withholds
//! bytes a downstream loader might expect:
//!
//!   1. **`demand_paged_patterns`** — tensors a model declares it will read by
//!      ROW at use time and never make resident (qwen4_exp's 47.7 GiB n-gram
//!      table). Model-declared, so it is checked FIRST and independently of
//!      expert parallelism.
//!   2. **EP sharding** — remote experts belong to another rank.
//!   3. **`skip_activation_scales`** — W4A4 `*.input_scale`, opt-in.
//!   4. **`skip_mtp`** — `mtp.*` for a loader that builds no MTP head, opt-in.
//!
//! Rules 3 and 4 default OFF and are allow-listed per model, because
//! withholding a tensor a loader DOES read is invisible until the output is
//! subtly wrong. Rule 2 is structural and always active under EP, and rule 1
//! is only ever populated by a loader that has its own row reader.

use super::FastSafetensorsLoader;
use crate::weights::parse_expert_index;

impl FastSafetensorsLoader {
    pub(super) fn should_skip_tensor(&self, name: &str) -> bool {
        // Demand-paged first: the model has declared it reads these by row at
        // use time, which is independent of expert parallelism and must hold
        // on a single rank where the EP check below returns early.
        //
        // NOTE this rule is NOT what withholds the qwen4_exp n-gram table, and
        // must not become it. The shard loop checks `is_ngram_table` BEFORE it
        // consults this function and takes those tensors down the DEFERRED
        // path, which records each one's file and offset — which is what
        // `NgramRowCache` reads rows through. A plain skip records nothing, so
        // if that predicate ever stopped matching the PLE shards this rule
        // would quietly take over and the PLE loader would fail its own
        // "no shard was deferred" check at load. The overlap is deliberate
        // (this rule is the general mechanism, the deferral is the one with
        // offsets) and `name_utils::ngram_table_predicate_matches_the_qwen_ple_shards`
        // is what keeps the precedence honest.
        if self
            .demand_paged_patterns
            .iter()
            .any(|pattern| name.contains(pattern.as_str()))
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
        if name.starts_with("mtp.") {
            return false;
        }
        if let Some(idx) = parse_expert_index(name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false
        }
    }
}
