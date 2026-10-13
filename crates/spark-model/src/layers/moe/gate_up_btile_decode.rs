// SPDX-License-Identifier: AGPL-3.0-only
use super::arena::CheckedArena;
use super::arena::slice;
use crate::weight_map::QuantizedWeight;
use anyhow::{Result, ensure};
use spark_runtime::buffers::BufferArena;
use spark_runtime::kernel_args::KernelLaunch;

#[derive(Clone, Copy)]
pub(super) enum WordPolicy {
    Word,
    Vector,
}
#[derive(Clone, Copy)]
pub(super) enum SharedMode {
    Active,
    ActiveLogits,
    RoutedOnly,
}
#[derive(Clone, Copy)]
pub(super) struct DecodeRows {
    pub count: usize,
    pub input: usize,
    pub routes: usize,
    pub output: usize,
}
impl CheckedArena<'_, '_> {
    pub(super) fn decode(
        &self,
        arena: &BufferArena,
        range: DecodeRows,
        word: WordPolicy,
        shared: SharedMode,
    ) -> Result<()> {
        let rows = range.count;
        self.rows(arena, rows)?;
        ensure!(
            rows <= 3
                && [range.input, range.routes, range.output]
                    .iter()
                    .all(|start| start
                        .checked_add(rows)
                        .is_some_and(|n| n <= arena.max_batch_tokens())),
            "decode row range"
        );
        let [sg, su] = match shared {
            SharedMode::Active | SharedMode::ActiveLogits => {
                self.lease
                    .shared
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!("active shared-T requires construction receipt")
                    })?
                    .0
            }
            SharedMode::RoutedOnly => [QuantizedWeight::null(); 2],
        };
        let sizes = arena.sizes();
        let input = slice(
            arena.norm_output(),
            sizes.norm_output,
            range.input * 8192,
            rows * 8192,
            16,
        )?;
        let ids = slice(
            arena.scratch(),
            sizes.scratch,
            range.routes * 32,
            rows * 32,
            4,
        )?;
        let gate = slice(
            arena.expert_gate_out(),
            sizes.expert_gate_out,
            range.output * 32768,
            rows * 32768,
            16,
        )?;
        let up = slice(
            arena.expert_up_out(),
            sizes.expert_up_out,
            range.output * 32768,
            rows * 32768,
            16,
        )?;
        // Routed-only still writes zero to both shared outputs. These are
        // dedicated dead-down-buffer slices, never precomputed shared results.
        // The next down phase must overwrite these before reading; grouped A
        // staging uses this owner in a different, non-overlapping phase.
        let (shared_gate, shared_up) = match shared {
            SharedMode::Active | SharedMode::ActiveLogits => (
                slice(
                    if matches!(shared, SharedMode::ActiveLogits) {
                        arena.logits()
                    } else {
                        arena.ssm_deinterleaved()
                    },
                    if matches!(shared, SharedMode::ActiveLogits) {
                        sizes.logits
                    } else {
                        sizes.ssm_deinterleaved
                    },
                    range.output * 4096,
                    rows * 4096,
                    16,
                )?,
                slice(
                    arena.ssm_qkvz(),
                    sizes.ssm_qkvz,
                    range.output * 4096,
                    rows * 4096,
                    16,
                )?,
            ),
            SharedMode::RoutedOnly => (
                slice(
                    arena.expert_down_out(),
                    sizes.expert_down_out,
                    0,
                    rows * 4096,
                    16,
                )?,
                slice(
                    arena.expert_down_out(),
                    sizes.expert_down_out,
                    rows * 4096,
                    rows * 4096,
                    16,
                )?,
            ),
        };
        self.disjoint_live(&[input, ids, gate, up, shared_gate, shared_up])?;
        let offset = match word {
            WordPolicy::Word => 2,
            WordPolicy::Vector => 5,
        };
        KernelLaunch::new(
            self.lease.family.gpu,
            self.lease.family.handles[offset + rows - 1],
        )
        .grid([64, rows as u32 * 9, 2])
        .block([32, 1, 1])
        .arg_ptr(input.ptr)
        .arg_ptr(self.lease.tables[0].ptr)
        .arg_ptr(self.lease.tables[1].ptr)
        .arg_ptr(self.lease.tables[2].ptr)
        .arg_ptr(gate.ptr)
        .arg_ptr(self.lease.tables[3].ptr)
        .arg_ptr(self.lease.tables[4].ptr)
        .arg_ptr(self.lease.tables[5].ptr)
        .arg_ptr(up.ptr)
        .arg_ptr(ids.ptr)
        .arg_ptr(sg.weight)
        .arg_ptr(sg.weight_scale)
        .arg_f32(sg.weight_scale_2)
        .arg_ptr(shared_gate.ptr)
        .arg_ptr(su.weight)
        .arg_ptr(su.weight_scale)
        .arg_f32(su.weight_scale_2)
        .arg_ptr(shared_up.ptr)
        .arg_u32(2048)
        .arg_u32(4096)
        .arg_u32(8)
        .launch(self.lease.stream)
    }
}
