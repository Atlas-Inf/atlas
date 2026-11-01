// SPDX-License-Identifier: AGPL-3.0-only
//! Exclusive routed-GU publication; matrix allocations remain in WeightStore.
//! Publication is minted only by the private validated load transaction.
use super::{
    UnpublishedBTileLayer, binding::ValidatedTables, kernels::KernelFamily, native_source::Span,
};
use crate::layers::moe::{ExpertPtrTable, MoeLayer};
use crate::weight_map::QuantizedWeight;
use anyhow::Result;
use spark_runtime::gpu::KernelHandle;
#[path = "gate_up_btile_resident_arena.rs"]
mod arena;
#[path = "gate_up_btile_resident_down.rs"]
mod down;
#[path = "gate_up_btile_resident_grouped.rs"]
mod grouped;

/// No raw routed table getter. Only private checked launches can read Ready.
pub(in crate::layers::moe) enum Storage {
    Legacy,
    Constructing,
    Failed,
    Published(Box<Ready>),
}

pub(in crate::layers::moe) struct Ready {
    backend: usize,
    handles: [KernelHandle; 16],
    tables: [ExpertPtrTable; 2],
    table_spans: [Span; 6],
    projections: Vec<Projection>,
    retained: Vec<super::native_source::CheckpointIdentity>,
    shared: ([QuantizedWeight; 2], [Span; 4]),
    shared_native: [QuantizedWeight; 3],
    down: Option<DownOwnership>,
    rank: usize,
    arena: Option<ArenaStamp>,
}

struct Projection {
    expert: usize,
    is_up: bool,
    packed: Span,
    scales: Span,
    scalar_bits: u32,
}

struct ArenaStamp {
    rows: usize,
    owners: [Span; 10],
}
struct DownOwnership {
    spans: [Span; 7],
    shared: QuantizedWeight,
}

impl Storage {
    pub(in crate::layers::moe) fn is_published(&self) -> bool {
        matches!(self, Self::Published(_))
    }
    pub(in crate::layers::moe) fn require_legacy(&self) -> Result<()> {
        anyhow::ensure!(
            matches!(self, Self::Legacy),
            "reader requires native GU storage"
        );
        Ok(())
    }
}

/// Consumes an actual successful transaction; never accepts caller-made tables.
/// Moves only the owners admitted by the real validated construction lease.
pub(super) fn seal(
    unpublished: UnpublishedBTileLayer<'_, '_>,
    family: &KernelFamily<'_>,
    layer: &mut MoeLayer,
    checked: ValidatedTables,
    rank: usize,
) -> Result<Box<Ready>> {
    anyhow::ensure!(
        matches!(layer.btile_storage, Storage::Constructing),
        "publication outside exclusive construction"
    );
    let shared = checked
        .shared
        .ok_or_else(|| anyhow::anyhow!("publication requires active shared-T authority"))?;
    let projections = unpublished
        .source
        .projections()
        .iter()
        .map(|p| Projection {
            expert: p.expert,
            is_up: p.is_up,
            packed: p.packed,
            scales: p.scales,
            scalar_bits: p.scalar_bits,
        })
        .collect();
    let retained = unpublished.source.retained().to_vec();
    let empty = || ExpertPtrTable {
        allocation: None,
        packed_ptrs: spark_runtime::gpu::DevicePtr::NULL,
        scale_ptrs: spark_runtime::gpu::DevicePtr::NULL,
        scale2_vals: spark_runtime::gpu::DevicePtr::NULL,
    };
    let tables = [
        std::mem::replace(&mut layer.gate_ptrs, empty()),
        std::mem::replace(&mut layer.up_ptrs, empty()),
    ];
    for expert in &mut layer.weights.experts {
        expert.gate_proj = QuantizedWeight::null();
        expert.up_proj = QuantizedWeight::null();
    }
    // Consume/drop the construction store borrow before native down retirement.
    drop(unpublished);
    Ok(Box::new(Ready {
        backend: family.gpu as *const dyn spark_runtime::gpu::GpuBackend as *const () as usize,
        handles: family.handles,
        tables,
        table_spans: checked.tables,
        projections,
        retained,
        shared,
        shared_native: [
            layer.weights.shared_expert.gate_proj,
            layer.weights.shared_expert.up_proj,
            layer.weights.shared_expert.down_proj,
        ],
        down: None,
        rank,
        arena: None,
    }))
}
