// SPDX-License-Identifier: AGPL-3.0-only
//
// Pure shard planning for the Windows streaming loader (`stream_shard_windows`).
// Split out so the keep/defer/skip decision and the slab layout are unit-testable
// on Linux — the streaming path itself is cfg(windows).

use anyhow::{Context, Result};
use std::collections::HashMap;

use crate::weights::WeightDtype;

/// One tensor that will be uploaded from this shard, in index order.
pub(super) struct PlannedTensor {
    pub name: String,
    /// Byte range relative to the shard's data section start.
    pub begin: u64,
    pub len: usize,
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
}

/// One tensor deferred to demand-paging (the n-gram tables): name, absolute
/// file offset, shape, dtype.
pub(super) struct PlannedDeferred {
    pub name: String,
    pub offset: u64,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

/// The per-shard plan: kept tensors in `tensor_names` order, deferred n-gram
/// tables, skipped count, and the largest kept byte length (staging buffer).
pub(super) struct ShardPlan {
    pub kept: Vec<PlannedTensor>,
    pub deferred: Vec<PlannedDeferred>,
    pub skipped: usize,
    pub max_len: usize,
}

/// Plan one shard: apply the skip predicate and the n-gram deferral, preserve
/// the caller's tensor order, and size the pinned staging buffer.
pub(super) fn plan_shard(
    tensor_names: &[String],
    infos: &HashMap<String, safetensors::tensor::TensorInfo>,
    skip_fn: &dyn Fn(&str) -> bool,
    data_base: u64,
) -> Result<ShardPlan> {
    let mut plan = ShardPlan {
        kept: Vec::new(),
        deferred: Vec::new(),
        skipped: 0,
        max_len: 0,
    };
    for name in tensor_names {
        if skip_fn(name) {
            plan.skipped += 1;
            continue;
        }
        let info = infos
            .get(name.as_str())
            .with_context(|| format!("tensor {name} missing from shard metadata"))?;
        let (begin, end) = info.data_offsets;
        if crate::weights::is_ngram_table(name) {
            plan.deferred.push(PlannedDeferred {
                name: name.clone(),
                offset: data_base + begin as u64,
                shape: info.shape.clone(),
                dtype: WeightDtype::from_safetensors(info.dtype)?,
            });
            plan.skipped += 1;
            continue;
        }
        let len = end.saturating_sub(begin);
        plan.max_len = plan.max_len.max(len);
        plan.kept.push(PlannedTensor {
            name: name.clone(),
            begin: begin as u64,
            len,
            dtype: info.dtype,
            shape: info.shape.clone(),
        });
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(len: usize, dtype: safetensors::Dtype) -> safetensors::tensor::TensorInfo {
        safetensors::tensor::TensorInfo {
            dtype,
            shape: vec![len],
            data_offsets: (0, len),
        }
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn plan_keeps_order_and_skips() {
        let infos: HashMap<_, _> = [
            ("a.weight".to_string(), info(100, safetensors::Dtype::BF16)),
            (
                "skip_me.weight".to_string(),
                info(50, safetensors::Dtype::BF16),
            ),
            ("b.weight".to_string(), info(200, safetensors::Dtype::F16)),
        ]
        .into_iter()
        .collect();
        let tn = names(&["a.weight", "skip_me.weight", "b.weight"]);
        let p = plan_shard(&tn, &infos, &|n| n.starts_with("skip_"), 8).unwrap();
        assert_eq!(p.skipped, 1);
        assert_eq!(
            p.kept
                .iter()
                .map(|t| (t.name.as_str(), t.begin, t.len, t.dtype, t.shape.clone()))
                .collect::<Vec<_>>(),
            vec![
                ("a.weight", 0u64, 100, safetensors::Dtype::BF16, vec![100]),
                ("b.weight", 0u64, 200, safetensors::Dtype::F16, vec![200]),
            ]
        );
        assert_eq!(p.max_len, 200);
        assert!(p.deferred.is_empty());
    }

    #[test]
    fn plan_defers_ngram_tables() {
        let ngram = "ple.ple_embedding.ngram_embedding.trellis";
        let infos: HashMap<_, _> = [
            (ngram.to_string(), info(64, safetensors::Dtype::I16)),
            ("x.weight".to_string(), info(32, safetensors::Dtype::BF16)),
        ]
        .into_iter()
        .collect();
        let tn = names(&[ngram, "x.weight"]);
        let p = plan_shard(&tn, &infos, &|_| false, 4096).unwrap();
        assert_eq!(p.deferred.len(), 1);
        assert_eq!(p.deferred[0].offset, 4096); // data_base + begin
        assert_eq!(p.deferred[0].name, ngram);
        assert_eq!(p.deferred[0].shape, vec![64]);
        assert_eq!(p.deferred[0].dtype, WeightDtype::Int16);
        assert_eq!(p.skipped, 1);
        assert_eq!(p.kept.len(), 1);
    }

    #[test]
    fn plan_missing_tensor_errors() {
        let infos: HashMap<String, _> = HashMap::new();
        let tn = names(&["gone.weight"]);
        assert!(plan_shard(&tn, &infos, &|_| false, 0).is_err());
    }

    #[test]
    fn slab_layout_over_plan_aligns_and_reserves_zero() {
        let infos: HashMap<_, _> = [
            ("a.weight".to_string(), info(1, safetensors::Dtype::BF16)),
            ("b.weight".to_string(), info(4097, safetensors::Dtype::BF16)),
        ]
        .into_iter()
        .collect();
        let tn = names(&["a.weight", "b.weight"]);
        let p = plan_shard(&tn, &infos, &|_| false, 0).unwrap();
        let (offsets, total) = crate::weights::slab_layout(p.kept.iter().map(|t| t.len));
        assert_eq!(offsets, vec![4096, 8192]);
        assert_eq!(total, 16384);
        assert!(offsets.iter().all(|o| o % 4096 == 0 && *o > 0));
    }

    #[test]
    fn slab_layout_empty_plan_is_zero() {
        let infos: HashMap<String, _> = HashMap::new();
        let tn = names(&[]);
        let p = plan_shard(&tn, &infos, &|_| false, 0).unwrap();
        let (offsets, total) = crate::weights::slab_layout(p.kept.iter().map(|t| t.len));
        assert!(offsets.is_empty() && total == 0);
    }
}
