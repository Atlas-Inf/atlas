// SPDX-License-Identifier: AGPL-3.0-only
//! Extents come from actual arena owners; device routing contents are not read.
use super::{binding::Lease, call::LaunchLease, kernels::KernelFamily, native_source::Span};
use crate::layer::ForwardContext;
use anyhow::{Result, ensure};
use spark_runtime::{buffers::BufferArena, gpu::DevicePtr};

pub(super) struct CheckedArena<'b, 'g> {
    pub(super) lease: LaunchLease<'g>,
    pub(super) arena: &'b BufferArena,
    owners: [Span; 10],
}
impl<'a, 's, 'g> Lease<'a, 's, 'g> {
    pub(super) fn check_arena<'b>(
        &'b self,
        ctx: &ForwardContext<'b>,
    ) -> Result<CheckedArena<'b, 'g>> {
        let source = &self.unpublished.source;
        ensure!(
            std::ptr::addr_eq(ctx.gpu, self.family.gpu),
            "arena context backend mismatch"
        );
        ensure!(
            !ctx.graph_capture && !ctx.gpu.stream_is_capturing(source.stream()),
            "arena binding during capture"
        );
        super::kernels::validate_profile(ctx.config)?;
        ensure!(
            ctx.routed_lora_layers.is_none()
                && !matches!(ctx.moe_lora_route, crate::layer::MoeLoraRoute::Refuse),
            "adapted arena context"
        );
        ensure!(
            source
                .projections()
                .iter()
                .all(|p| ctx.config.is_local_expert(p.expert)),
            "arena context rank mismatch"
        );
        let arena = ctx.buffers;
        ensure!(
            (1..=1088).contains(&arena.max_batch_tokens()),
            "arena owner row bound"
        );
        let owners = owner_spans(arena)?;
        for (i, &span) in owners.iter().enumerate() {
            ensure!(
                source.scratch_is_disjoint(span)
                    && self.tables.iter().all(|t| t.disjoint(span))
                    && self
                        .shared
                        .as_ref()
                        .is_none_or(|(_, shared)| shared.iter().all(|s| s.disjoint(span)))
                    && owners[..i].iter().all(|s| s.disjoint(span)),
                "arena allocation owner alias"
            );
        }
        Ok(CheckedArena {
            lease: LaunchLease {
                family: KernelFamily {
                    gpu: self.family.gpu,
                    handles: self.family.handles,
                },
                tables: self.tables,
                shared: self.shared,
                stream: source.stream(),
            },
            arena,
            owners,
        })
    }
}

pub(super) fn slice(
    ptr: DevicePtr,
    capacity: usize,
    offset: usize,
    bytes: usize,
    alignment: u64,
) -> Result<Span> {
    ensure!(
        offset.checked_add(bytes).is_some_and(|end| end <= capacity),
        "B-tile arena capacity"
    );
    Span::new(
        DevicePtr(
            ptr.0
                .checked_add(offset as u64)
                .ok_or_else(|| anyhow::anyhow!("arena offset overflow"))?,
        ),
        bytes,
        alignment,
    )
}
pub(super) fn owner_spans(arena: &BufferArena) -> Result<[Span; 10]> {
    let s = arena.sizes();
    let raw = [
        (arena.norm_output(), s.norm_output),
        (arena.scratch(), s.scratch),
        (arena.expert_gate_out(), s.expert_gate_out),
        (arena.expert_up_out(), s.expert_up_out),
        (arena.expert_down_out(), s.expert_down_out),
        (arena.gate_logits(), s.gate_logits),
        (arena.moe_router_in_f32(), s.moe_router_in_f32),
        (arena.ssm_deinterleaved(), s.ssm_deinterleaved),
        (arena.ssm_qkvz(), s.ssm_qkvz),
        (arena.logits(), s.logits),
    ];
    let mut result = [Span {
        ptr: DevicePtr::NULL,
        bytes: 0,
    }; 10];
    for (i, (ptr, bytes)) in raw.into_iter().enumerate() {
        result[i] = Span::new(ptr, bytes, 16)?;
    }
    Ok(result)
}
impl<'b, 'g> CheckedArena<'b, 'g> {
    pub(super) fn resident(
        lease: LaunchLease<'g>,
        arena: &'b BufferArena,
        owners: [Span; 10],
    ) -> Self {
        Self {
            lease,
            arena,
            owners,
        }
    }
}
impl CheckedArena<'_, '_> {
    pub(super) fn rows(&self, arena: &BufferArena, rows: usize) -> Result<()> {
        ensure!(
            std::ptr::eq(arena, self.arena) && rows > 0 && rows <= arena.max_batch_tokens(),
            "B-tile rows outside bound arena"
        );
        Ok(())
    }
    pub(super) fn disjoint_live(&self, spans: &[Span]) -> Result<()> {
        for (index, span) in spans.iter().enumerate() {
            ensure!(
                self.owners.iter().any(|owner| span.ptr.0 >= owner.ptr.0
                    && span.ptr.0 + span.bytes as u64 <= owner.ptr.0 + owner.bytes as u64),
                "unbound arena subrange"
            );
            ensure!(
                spans[..index].iter().all(|s| s.disjoint(*span)),
                "simultaneous live arena alias"
            );
        }
        Ok(())
    }
}
