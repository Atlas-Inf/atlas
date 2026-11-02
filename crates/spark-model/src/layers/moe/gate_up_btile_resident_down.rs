// SPDX-License-Identifier: AGPL-3.0-only
//! Seal extents from the actual compact down allocations, not pointer claims.
use super::*;
use crate::weight_loader::glm5::retirement::RetirementLog;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

impl Ready {
    pub(in crate::layers::moe::gate_up_repack) fn complete_down(
        &mut self,
        layer: &MoeLayer,
        values: Vec<QuantizedWeight>,
        config: &ModelConfig,
        log: &RetirementLog<'_>,
        gpu: &dyn GpuBackend,
        scratch: Span,
    ) -> Result<()> {
        anyhow::ensure!(
            self.down.is_none() && values.len() == 288,
            "invalid down completion"
        );
        let first = values
            .iter()
            .find(|q| !q.is_null())
            .ok_or_else(|| anyhow::anyhow!("no local down allocation"))?;
        let packed = Span::new(first.weight, 144 * 4_194_304, 16)?;
        let scales = Span::new(first.weight_scale, 144 * 524_288, 16)?;
        let mut local = 0;
        for (e, q) in values.iter().enumerate() {
            if config.is_local_expert(e) {
                anyhow::ensure!(
                    q.weight == packed.ptr.offset(local * 4_194_304)
                        && q.weight_scale == scales.ptr.offset(local * 524_288),
                    "changed compact down allocation"
                );
                local += 1;
            } else {
                anyhow::ensure!(
                    q.is_null() && q.weight_scale.is_null(),
                    "remote down allocation"
                );
            }
        }
        anyhow::ensure!(local == 144, "local down count");
        let table = layer
            .down_ptrs_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing completed down table"))?;
        let regions = table.owned_regions(gpu, 288)?;
        let shared = layer
            .shared_down_t
            .ok_or_else(|| anyhow::anyhow!("missing shared down transform"))?;
        let spans = [
            packed,
            scales,
            Span::new(regions[0].0, regions[0].1, 8)?,
            Span::new(regions[1].0, regions[1].1, 8)?,
            Span::new(regions[2].0, regions[2].1, 4)?,
            Span::new(shared.weight, 4_194_304, 16)?,
            Span::new(shared.weight_scale, 524_288, 16)?,
        ];
        for (i, &span) in spans.iter().enumerate() {
            log.disjoint_live(span.ptr, span.bytes, gpu)?;
            anyhow::ensure!(
                scratch.disjoint(span)
                    && spans[..i].iter().all(|s| s.disjoint(span))
                    && self.table_spans.iter().all(|s| s.disjoint(span))
                    && self.shared.1.iter().all(|s| s.disjoint(span)),
                "completed down allocation alias"
            );
        }
        self.down = Some(DownOwnership { spans, shared });
        Ok(())
    }
}

pub(super) fn same_weight(a: QuantizedWeight, b: QuantizedWeight) -> bool {
    a.weight == b.weight
        && a.weight_scale == b.weight_scale
        && a.weight_scale_2.to_bits() == b.weight_scale_2.to_bits()
        && a.input_scale == b.input_scale
        && a.weight_scale_2_vec == b.weight_scale_2_vec
}
