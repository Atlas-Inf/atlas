// SPDX-License-Identifier: AGPL-3.0-only
//! Construction-only lease. No raw table getter or serving publication.
use super::{
    UnpublishedBTileLayer,
    kernels::KernelFamily,
    native_source::{NativeGateUpLayer, Span},
};
use crate::layers::moe::MoeLayer;
use anyhow::{Result, ensure};

pub(super) struct Lease<'a, 's, 'g> {
    pub(super) unpublished: &'a UnpublishedBTileLayer<'s, 'g>,
    pub(super) family: &'a KernelFamily<'g>,
    pub(super) layer: &'a mut MoeLayer,
    pub(super) tables: [Span; 6],
    pub(super) shared: Option<([crate::weight_map::QuantizedWeight; 2], [Span; 4])>,
}
impl<'a, 's, 'g> Lease<'a, 's, 'g> {
    pub(super) fn bind(
        unpublished: &'a UnpublishedBTileLayer<'s, 'g>,
        family: &'a KernelFamily<'g>,
        layer: &'a mut MoeLayer,
    ) -> Result<Self> {
        let checked = ValidatedTables::read(&unpublished.source, family, layer)?;
        Ok(Self {
            unpublished,
            family,
            layer,
            tables: checked.tables,
            shared: checked.shared,
        })
    }
}

/// Sealed result of reading the actual table allocations, not caller spans.
pub(super) struct ValidatedTables {
    pub(super) tables: [Span; 6],
    pub(super) shared: Option<([crate::weight_map::QuantizedWeight; 2], [Span; 4])>,
}
impl ValidatedTables {
    pub(super) fn read(
        source: &NativeGateUpLayer<'_, '_>,
        family: &KernelFamily<'_>,
        layer: &MoeLayer,
    ) -> Result<Self> {
        ensure!(
            std::ptr::addr_eq(source.gpu(), family.gpu),
            "lease backend mismatch"
        );
        ensure!(
            !family.gpu.stream_is_capturing(source.stream()),
            "lease during capture"
        );
        ensure!(
            layer.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
                && !layer.nvfp4_mmq_layout
                && !layer.unified_layout
                && !layer.hybrid_layout
                && layer.gate_ptrs_t.is_none()
                && layer.up_ptrs_t.is_none()
                && layer.lora.is_none()
                && layer.weights.experts.len() == 288,
            "incompatible layer ownership/layout"
        );
        let gate = layer.gate_ptrs.owned_regions(family.gpu, 288)?;
        let up = layer.up_ptrs.owned_regions(family.gpu, 288)?;
        let mut tables = [Span {
            ptr: spark_runtime::gpu::DevicePtr::NULL,
            bytes: 1,
        }; 6];
        for (index, (ptr, bytes)) in gate.into_iter().chain(up).enumerate() {
            ensure!(
                bytes == if index % 3 == 2 { 1152 } else { 2304 },
                "table extent mismatch"
            );
            let span = Span::new(ptr, bytes, if index % 3 == 2 { 4 } else { 8 })?;
            ensure!(
                source.scratch_is_disjoint(span)
                    && tables[..index].iter().all(|s| s.disjoint(span)),
                "table owner alias"
            );
            tables[index] = span;
        }
        let shared = Self::shared(source, family, layer, &tables)?;
        // Every capacity and live-owner check precedes the first D2H.
        for (index, span) in tables.iter().enumerate() {
            let mut data = vec![0u8; span.bytes];
            family
                .gpu
                .copy_d2h_on_stream(span.ptr, &mut data, source.stream())?;
            let is_up = index >= 3;
            let column = index % 3;
            let width = if column == 2 { 4 } else { 8 };
            for expert in 0..288 {
                let projection = source
                    .projections()
                    .iter()
                    .find(|p| p.expert == expert && p.is_up == is_up);
                let expected = projection.map_or(0, |p| match column {
                    0 => p.packed.ptr.0,
                    1 => p.scales.ptr.0,
                    _ => u64::from(p.scalar_bits),
                });
                ensure!(
                    data[expert * width..(expert + 1) * width] == expected.to_le_bytes()[..width],
                    "table payload mismatch expert {expert} column {index}"
                );
            }
        }
        Ok(Self { tables, shared })
    }
    pub(super) fn with_shared(
        mut self,
        source: &NativeGateUpLayer<'_, '_>,
        family: &KernelFamily<'_>,
        layer: &MoeLayer,
    ) -> Result<Self> {
        self.shared = Self::shared(source, family, layer, &self.tables)?;
        ensure!(self.shared.is_some(), "missing active shared transform");
        Ok(self)
    }
    fn shared(
        source: &NativeGateUpLayer<'_, '_>,
        family: &KernelFamily<'_>,
        layer: &MoeLayer,
        tables: &[Span; 6],
    ) -> Result<Option<([crate::weight_map::QuantizedWeight; 2], [Span; 4])>> {
        Ok(
            match (
                layer.shared_gate_t,
                layer.shared_up_t,
                &layer.shared_gate_up_receipt,
            ) {
                (None, None, None) => None,
                (Some(gate), Some(up), Some(receipt)) => {
                    ensure!(
                        layer.shared_experts_scale_kind
                            == crate::weight_map::WeightQuantFormat::Nvfp4,
                        "shared quant format changed"
                    );
                    let regions = receipt.validate(family.gpu, gate, up)?;
                    let mut spans = [tables[0]; 4];
                    for (i, (ptr, bytes)) in regions.into_iter().enumerate() {
                        let span = Span::new(ptr, bytes, 16)?;
                        ensure!(
                            source.scratch_is_disjoint(span)
                                && tables.iter().all(|s| s.disjoint(span))
                                && spans[..i].iter().all(|s| s.disjoint(span)),
                            "shared transform owner alias"
                        );
                        spans[i] = span;
                    }
                    Some(([gate, up], spans))
                }
                _ => anyhow::bail!("missing or incomplete shared transform authority"),
            },
        )
    }
}
