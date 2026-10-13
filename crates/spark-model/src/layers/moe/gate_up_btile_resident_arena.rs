// SPDX-License-Identifier: AGPL-3.0-only
//! Bind once to real arena owners; launches use the live caller context/stream.
use super::super::{
    arena::{CheckedArena, owner_spans},
    call::LaunchLease,
};
use super::*;
use crate::layer::ForwardContext;
use atlas_core::config::ModelConfig;
use spark_runtime::{
    buffers::BufferArena,
    gpu::{DevicePtr, GpuBackend},
    weights::WeightStore,
};

impl Ready {
    fn check_backend(&self, gpu: &dyn GpuBackend) -> Result<()> {
        anyhow::ensure!(
            self.backend == gpu as *const dyn GpuBackend as *const () as usize,
            "resident backend mismatch"
        );
        for table in &self.tables {
            table.owned_regions(gpu, 288)?;
        }
        Ok(())
    }
    pub(super) fn checked<'a>(
        &self,
        ctx: &ForwardContext<'a>,
        stream: u64,
    ) -> Result<CheckedArena<'a, 'a>> {
        self.check_backend(ctx.gpu)?;
        super::super::kernels::validate_profile(ctx.config)?;
        anyhow::ensure!(
            ctx.config.ep_rank == self.rank
                && ctx.routed_lora_layers.is_none()
                && !matches!(ctx.moe_lora_route, crate::layer::MoeLoraRoute::Refuse),
            "adapted or foreign resident context"
        );
        let stamp = self
            .arena
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("resident arena not bound"))?;
        anyhow::ensure!(
            stamp.rows == ctx.buffers.max_batch_tokens()
                && stamp.owners == owner_spans(ctx.buffers)?,
            "resident arena owner changed"
        );
        Ok(CheckedArena::resident(
            LaunchLease {
                family: KernelFamily {
                    gpu: ctx.gpu,
                    handles: self.handles,
                },
                tables: self.table_spans,
                shared: Some(self.shared),
                stream,
            },
            ctx.buffers,
            stamp.owners,
        ))
    }
}
impl MoeLayer {
    pub(crate) fn bind_btile_arena_if_resident(
        &mut self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        arena: &BufferArena,
        stream: u64,
    ) -> Result<()> {
        if matches!(self.btile_storage, Storage::Legacy) {
            return Ok(());
        }
        self.bind_btile_arena(store, config, gpu, arena, stream)
    }
    pub(crate) fn invalidate_btile_before_release(&mut self) {
        if !matches!(self.btile_storage, Storage::Legacy) {
            self.btile_storage = Storage::Failed;
        }
    }
    pub(in crate::layers::moe) fn use_btile_or_t_decode(&self) -> bool {
        self.btile_storage.is_published() || self.use_t_layout_for_decode()
    }
    pub(in crate::layers::moe) fn use_btile_or_t_prefill(&self) -> bool {
        self.btile_storage.is_published() || self.use_t_layout_for_prefill()
    }
    pub(in crate::layers::moe) fn btile_input_guard(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.btile_forward_guard(ctx, stream)?;
        if self.btile_storage.is_published() {
            let delta = input
                .0
                .checked_sub(ctx.buffers.norm_output().0)
                .ok_or_else(|| anyhow::anyhow!("input below bound arena"))?;
            anyhow::ensure!(
                delta.is_multiple_of(8192)
                    && rows > 0
                    && rows <= 1088
                    && (delta / 8192)
                        .checked_add(rows as u64)
                        .is_some_and(|n| n <= ctx.buffers.max_batch_tokens() as u64),
                "input rows outside bound arena"
            );
        }
        Ok(())
    }
    pub(in crate::layers::moe) fn btile_forward_guard(
        &self,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match &self.btile_storage {
            Storage::Legacy => Ok(()),
            Storage::Constructing | Storage::Failed => {
                anyhow::bail!("B-tile construction incomplete or failed")
            }
            Storage::Published(ready) => {
                anyhow::ensure!(
                    self.lora.is_none()
                        && self.gate_ptrs.packed_ptrs.is_null()
                        && self.up_ptrs.packed_ptrs.is_null()
                        && self.gate_ptrs_t.is_none()
                        && self.up_ptrs_t.is_none()
                        && !self.unified_layout
                        && !self.hybrid_layout
                        && !self.nvfp4_mmq_layout
                        && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
                        && self.shared_experts_scale_kind
                            == crate::weight_map::WeightQuantFormat::Nvfp4,
                    "resident layout or adapter changed"
                );
                anyhow::ensure!(
                    self.bf16_gate_weight_ptrs.is_none()
                        && self.fp8_gate_weight_ptrs.is_none()
                        && self.bf16_shared_expert.is_none()
                        && self.fp8_shared_expert.is_none()
                        && self.gate_fp8.is_none()
                        && self.shared_gate_fp8.is_none()
                        && self.shared_up_fp8.is_none()
                        && self.shared_down_fp8.is_none()
                        && self.gate_nvfp4.is_none()
                        && self.pre_expert_norm.is_none()
                        && self.tid2eid_dev.is_none()
                        && !self.gelu_activation
                        && self.nvfp4_prequant_moe
                        && self.down_t_scratch_packed.is_none()
                        && self.down_t_scratch_scale.is_none(),
                    "resident reader profile changed"
                );
                let native = [
                    self.weights.shared_expert.gate_proj,
                    self.weights.shared_expert.up_proj,
                    self.weights.shared_expert.down_proj,
                ];
                anyhow::ensure!(
                    native
                        .into_iter()
                        .zip(ready.shared_native)
                        .all(|(a, b)| super::down::same_weight(a, b)),
                    "native shared view changed"
                );
                let down = ready
                    .down
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("down publication incomplete"))?;
                let regions = self
                    .down_ptrs_t
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("resident down table missing"))?
                    .owned_regions(ctx.gpu, 288)?;
                anyhow::ensure!(
                    regions
                        .into_iter()
                        .zip(&down.spans[2..5])
                        .all(|((ptr, bytes), s)| ptr == s.ptr && bytes == s.bytes)
                        && self
                            .shared_down_t
                            .is_some_and(|q| super::down::same_weight(q, down.shared)),
                    "resident down view changed"
                );
                self.shared_gate_up_receipt
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("resident shared receipt missing"))?
                    .validate(
                        ctx.gpu,
                        self.shared_gate_t
                            .ok_or_else(|| anyhow::anyhow!("shared gate missing"))?,
                        self.shared_up_t
                            .ok_or_else(|| anyhow::anyhow!("shared up missing"))?,
                    )?;
                ready.checked(ctx, stream)?;
                Ok(())
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::moe) fn dispatch_btile_decode(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        gate: DevicePtr,
        up: DevicePtr,
        ids: DevicePtr,
        shared: Option<(DevicePtr, DevicePtr)>,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        self.btile_forward_guard(ctx, stream)?;
        let Storage::Published(ready) = &self.btile_storage else {
            anyhow::bail!("B-tile reader requires publication");
        };
        fn origin(ptr: DevicePtr, base: DevicePtr, width: u64) -> Result<usize> {
            let delta = ptr
                .0
                .checked_sub(base.0)
                .ok_or_else(|| anyhow::anyhow!("reader pointer below arena"))?;
            anyhow::ensure!(delta.is_multiple_of(width), "reader row alignment");
            usize::try_from(delta / width).map_err(Into::into)
        }
        let arena = ctx.buffers;
        let output = origin(gate, arena.expert_gate_out(), 32768)?;
        anyhow::ensure!(
            output == origin(up, arena.expert_up_out(), 32768)?,
            "different GU output origins"
        );
        let shared_mode = if let Some((sg, su)) = shared {
            anyhow::ensure!(
                output == origin(sg, arena.logits(), 4096)?
                    && output == origin(su, arena.ssm_qkvz(), 4096)?,
                "shared output origins"
            );
            super::super::decode::SharedMode::ActiveLogits
        } else {
            super::super::decode::SharedMode::RoutedOnly
        };
        ready.checked(ctx, stream)?.decode(
            arena,
            super::super::decode::DecodeRows {
                count: rows,
                input: origin(input, arena.norm_output(), 8192)?,
                routes: origin(ids, arena.scratch(), 32)?,
                output,
            },
            if self.nvfp4_vecscale {
                super::super::decode::WordPolicy::Vector
            } else {
                super::super::decode::WordPolicy::Word
            },
            shared_mode,
        )
    }
    pub(in crate::layers::moe) fn bind_btile_arena(
        &mut self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        arena: &BufferArena,
        stream: u64,
    ) -> Result<()> {
        let Storage::Published(ready) = &mut self.btile_storage else {
            anyhow::bail!("arena requires published B-tile owner");
        };
        ready.check_backend(gpu)?;
        super::super::kernels::validate_profile(config)?;
        anyhow::ensure!(
            config.ep_rank == ready.rank
                && ready.arena.is_none()
                && !gpu.stream_is_capturing(stream)
                && (1..=1088).contains(&arena.max_batch_tokens()),
            "invalid resident arena construction"
        );
        let owners = owner_spans(arena)?;
        for span in &owners {
            anyhow::ensure!(
                ready
                    .projections
                    .iter()
                    .all(|p| p.packed.disjoint(*span) && p.scales.disjoint(*span))
                    && ready
                        .retained
                        .iter()
                        .all(|identity| identity.span.disjoint(*span)),
                "resident arena aliases sealed GU source"
            );
        }
        for identity in &ready.retained {
            identity.validate(store)?;
        }
        // Retained GU identities have been checked independently of the supplied
        // rebuilt map. This whole-owner index is construction-only, never a
        // launch-time scan or a generic authentication of unrelated store keys.
        let live = crate::weight_loader::glm5::retirement::RetirementLog::new(store, gpu)?;
        let down = ready
            .down
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("down publication incomplete"))?;
        for (i, span) in owners.iter().enumerate() {
            live.disjoint_live(span.ptr, span.bytes, gpu)?;
            anyhow::ensure!(
                owners[..i].iter().all(|s| s.disjoint(*span))
                    && ready.table_spans.iter().all(|s| s.disjoint(*span))
                    && ready.shared.1.iter().all(|s| s.disjoint(*span))
                    && down.spans.iter().all(|s| s.disjoint(*span)),
                "resident arena allocation alias"
            );
        }
        ready.arena = Some(ArenaStamp {
            rows: arena.max_batch_tokens(),
            owners,
        });
        Ok(())
    }
}
