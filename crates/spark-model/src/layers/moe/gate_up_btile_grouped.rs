// SPDX-License-Identifier: AGPL-3.0-only
use super::arena::CheckedArena;
use super::arena::slice;
use anyhow::Result;
use spark_runtime::buffers::BufferArena;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};
#[derive(Clone, Copy)]
pub(super) enum ScalePolicy {
    Scalar,
    Vector,
}
#[derive(Clone, Copy)]
pub(super) enum InputLayout {
    Gathered,
    RouteMajor,
}
#[derive(Clone, Copy)]
pub(super) enum GroupedMode {
    Dense,
    SeparateCompact,
    Fused { prefer_small: bool },
}
pub(super) struct GroupedPlan {
    pub(super) a: DevicePtr,
    pub(super) a_scale: DevicePtr,
    pub(super) work: DevicePtr,
    pub(super) total: DevicePtr,
    pub(super) offsets: DevicePtr,
    pub(super) max_tiles: usize,
    gate: DevicePtr,
    up: DevicePtr,
    ids: DevicePtr,
    rows: usize,
    gathered: bool,
    mode: GroupedMode,
}
impl CheckedArena<'_, '_> {
    pub(super) fn grouped(
        &self,
        arena: &BufferArena,
        rows: usize,
        scale: ScalePolicy,
        layout: InputLayout,
        mode: GroupedMode,
    ) -> Result<()> {
        self.run_grouped(self.plan_grouped(arena, rows, layout, mode)?, scale)
    }
    pub(super) fn plan_grouped(
        &self,
        arena: &BufferArena,
        rows: usize,
        layout: InputLayout,
        mode: GroupedMode,
    ) -> Result<GroupedPlan> {
        self.rows(arena, rows)?;
        let sizes = arena.sizes();
        let expanded = rows * 8;
        let gathered = matches!(layout, InputLayout::Gathered);
        let a_rows = if gathered { rows } else { expanded };
        let a = slice(
            arena.expert_down_out(),
            sizes.expert_down_out,
            0,
            a_rows * 2048,
            16,
        )?;
        let a_scale = slice(
            arena.expert_down_out(),
            sizes.expert_down_out,
            a_rows * 2048,
            a_rows * 256,
            16,
        )?;
        let gate = slice(
            arena.expert_gate_out(),
            sizes.expert_gate_out,
            0,
            expanded * 4096,
            16,
        )?;
        let up = slice(
            arena.expert_up_out(),
            sizes.expert_up_out,
            0,
            expanded * 4096,
            16,
        )?;
        let offsets = slice(
            arena.gate_logits(),
            sizes.gate_logits,
            expanded * 8,
            289 * 4,
            4,
        )?;
        let mut spans = [a, a_scale, gate, up, offsets, a, a, a];
        let mut span_count = 5;
        let ids = if gathered {
            let ids = slice(arena.gate_logits(), sizes.gate_logits, 0, expanded * 4, 4)?;
            spans[span_count] = ids;
            span_count += 1;
            ids.ptr
        } else {
            DevicePtr::NULL
        };
        let max_tiles = expanded * 16;
        let compact = !matches!(mode, GroupedMode::Dense);
        let (work, total) = if compact {
            let total = slice(arena.moe_router_in_f32(), sizes.moe_router_in_f32, 0, 4, 4)?;
            let work = slice(
                arena.moe_router_in_f32(),
                sizes.moe_router_in_f32,
                16,
                max_tiles * 8,
                8,
            )?;
            spans[span_count] = total;
            spans[span_count + 1] = work;
            span_count += 2;
            (work.ptr, total.ptr)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        self.disjoint_live(&spans[..span_count])?;
        Ok(GroupedPlan {
            a: a.ptr,
            a_scale: a_scale.ptr,
            gate: gate.ptr,
            up: up.ptr,
            offsets: offsets.ptr,
            ids,
            work,
            total,
            max_tiles,
            rows,
            gathered,
            mode,
        })
    }
    pub(super) fn run_grouped(&self, plan: GroupedPlan, scale: ScalePolicy) -> Result<()> {
        let GroupedPlan {
            a,
            a_scale,
            gate,
            up,
            offsets,
            ids,
            work,
            total,
            max_tiles,
            rows,
            gathered,
            mode,
        } = plan;
        let compact = !matches!(mode, GroupedMode::Dense);
        let vector = usize::from(matches!(scale, ScalePolicy::Vector));
        let stream = self.lease.stream;
        match mode {
            GroupedMode::Fused { prefer_small } => {
                let small = prefer_small && gathered && rows <= 5;
                KernelLaunch::new(
                    self.lease.family.gpu,
                    self.lease.family.handles[if small { 8 + vector } else { 10 + vector }],
                )
                .grid([max_tiles as u32, 2, 1])
                .block([128, 1, 1])
                .arg_ptr(a)
                .arg_ptr(a_scale)
                .arg_ptr(self.lease.tables[0].ptr)
                .arg_ptr(self.lease.tables[1].ptr)
                .arg_ptr(self.lease.tables[2].ptr)
                .arg_ptr(gate)
                .arg_ptr(self.lease.tables[3].ptr)
                .arg_ptr(self.lease.tables[4].ptr)
                .arg_ptr(self.lease.tables[5].ptr)
                .arg_ptr(up)
                .arg_ptr(offsets)
                .arg_ptr(ids)
                .arg_u32(288)
                .arg_u32(2048)
                .arg_u32(4096)
                .arg_ptr(work)
                .arg_ptr(total)
                .arg_u32(max_tiles as u32)
                .launch(stream)
            }
            GroupedMode::Dense | GroupedMode::SeparateCompact => {
                // Validate the entire pair before its first launch. A backend
                // failure propagates immediately; never submit a fallback.
                for (base, output) in [(0, gate), (3, up)] {
                    let launch = KernelLaunch::new(
                        self.lease.family.gpu,
                        self.lease.family.handles[if compact { 14 + vector } else { 12 + vector }],
                    )
                    .grid(if compact {
                        [max_tiles as u32, 1, 1]
                    } else {
                        [16, rows.div_ceil(64) as u32, 288]
                    })
                    .block([128, 1, 1])
                    .arg_ptr(a)
                    .arg_ptr(a_scale)
                    .arg_ptr(self.lease.tables[base].ptr)
                    .arg_ptr(self.lease.tables[base + 1].ptr)
                    .arg_ptr(self.lease.tables[base + 2].ptr)
                    .arg_ptr(output)
                    .arg_ptr(offsets)
                    .arg_ptr(ids)
                    .arg_u32(288)
                    .arg_u32(2048)
                    .arg_u32(4096);
                    if compact {
                        launch
                            .arg_ptr(work)
                            .arg_ptr(total)
                            .arg_u32(max_tiles as u32)
                            .launch(stream)?;
                    } else {
                        launch.launch(stream)?;
                    }
                }
                Ok(())
            }
        }
    }
}
