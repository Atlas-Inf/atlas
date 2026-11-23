// SPDX-License-Identifier: AGPL-3.0-only

//! Direct-mapped original-FP8 Ling decode.
//!
//! The strict checkpoint keeps routed matrices in immutable safetensor
//! shards.  On GB10, pageable reads through host page tables let the CUDA
//! kernels index those mmap ranges directly.  This module builds the three
//! small device pointer tables once and retains every [`LingPage`] owner until
//! the layer is torn down.  Unsupported backends simply use `stage_projection`.

use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::ling_paging::{LingPage, LingPageKind, LingPagingReader};

use super::{EXPERTS, GROUP, HIDDEN, INTERMEDIATE, LingMoeFfn, LingProjection, TOP_K, ops};
use crate::layer::ForwardContext;

struct StrictFp8Table {
    weight_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
}

/// Resident metadata for the direct-mapped strict path.
pub(super) struct StrictFastTables {
    gate: StrictFp8Table,
    up: StrictFp8Table,
    down: StrictFp8Table,
    /// `[gate | up]` shared-expert BF16 outputs.
    shared_gate_up: DevicePtr,
    /// The original mmap owners must outlive every kernel launch.
    _pages: Vec<LingPage>,
    /// Keep the widened block scales alive; pointers in `*_scale_ptrs` index
    /// into these three allocations.
    _scale_storage: [DevicePtr; 3],
}

impl StrictFastTables {
    pub(super) fn try_build(
        gpu: &dyn GpuBackend,
        reader: &Arc<Mutex<LingPagingReader>>,
        routed: &[[LingProjection; 3]],
    ) -> Result<Self> {
        ensure!(routed.len() == EXPERTS, "Ling strict expert count mismatch");
        ensure!(
            gpu.pageable_memory_access_uses_host_page_tables()?,
            "device does not expose pageable host-page-table access"
        );

        let scale_elements = (INTERMEDIATE / GROUP) * (HIDDEN / GROUP);
        let mut weight_ptrs: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::with_capacity(EXPERTS));
        let mut scale_bytes: [Vec<u8>; 3] = std::array::from_fn(|_| {
            Vec::with_capacity(EXPERTS * scale_elements * std::mem::size_of::<f32>())
        });
        let mut pages = Vec::with_capacity(EXPERTS * 3 * 2);
        let mut guard = reader
            .lock()
            .map_err(|_| anyhow::anyhow!("Ling paging reader poisoned"))?;

        // Probe every mapped weight before allocating any persistent table.
        // A single unsupported range must leave the caller on the safe staged
        // path rather than creating a partially direct-mapped layer.
        for expert in routed {
            for (projection, (weights, scales)) in expert
                .iter()
                .zip(weight_ptrs.iter_mut().zip(scale_bytes.iter_mut()))
            {
                let LingProjection::Paged {
                    weight,
                    scale,
                    n,
                    k,
                } = projection
                else {
                    anyhow::bail!("Ling strict fast path requires deferred FP8 experts")
                };
                ensure!(*n == INTERMEDIATE || *n == HIDDEN);
                ensure!(*k == HIDDEN || *k == INTERMEDIATE);
                let weight_page = guard.e4m3_weight(weight)?;
                ensure!(weight_page.len() == n * k);
                let device_weight = gpu.pageable_host_ptr_to_device(weight_page.host_ptr())?;
                ensure!(!device_weight.is_null(), "pageable weight alias is null");
                weights.push(device_weight.0);

                let scale_page = guard.page(scale)?;
                ensure!(scale_page.len() == scale_elements * scale_page.kind().element_bytes());
                append_scale_f32(scales, &scale_page)?;
                pages.push(weight_page);
                pages.push(scale_page);
            }
        }
        drop(guard);

        let scale_storage = [
            upload_scales(gpu, &scale_bytes[0])?,
            upload_scales(gpu, &scale_bytes[1])?,
            upload_scales(gpu, &scale_bytes[2])?,
        ];
        let tables = [
            StrictFp8Table {
                weight_ptrs: upload_ptrs(gpu, &weight_ptrs[0])?,
                scale_ptrs: scale_table(gpu, scale_storage[0], scale_elements)?,
            },
            StrictFp8Table {
                weight_ptrs: upload_ptrs(gpu, &weight_ptrs[1])?,
                scale_ptrs: scale_table(gpu, scale_storage[1], scale_elements)?,
            },
            StrictFp8Table {
                weight_ptrs: upload_ptrs(gpu, &weight_ptrs[2])?,
                scale_ptrs: scale_table(gpu, scale_storage[2], scale_elements)?,
            },
        ];
        let shared_gate_up = gpu.alloc(INTERMEDIATE * 2 * std::mem::size_of::<u16>())?;
        Ok(Self {
            gate: tables[0].clone_for_move(),
            up: tables[1].clone_for_move(),
            down: tables[2].clone_for_move(),
            shared_gate_up,
            _pages: pages,
            _scale_storage: scale_storage,
        })
    }
}

impl StrictFp8Table {
    fn clone_for_move(&self) -> Self {
        Self {
            weight_ptrs: self.weight_ptrs,
            scale_ptrs: self.scale_ptrs,
        }
    }
}

fn upload_ptrs(gpu: &dyn GpuBackend, pointers: &[u64]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = pointers.iter().flat_map(|p| p.to_le_bytes()).collect();
    let table = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, table)?;
    Ok(table)
}

fn upload_scales(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let storage = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, storage)?;
    Ok(storage)
}

fn scale_table(gpu: &dyn GpuBackend, storage: DevicePtr, elements: usize) -> Result<DevicePtr> {
    let pointers: Vec<u64> = (0..EXPERTS)
        .map(|expert| {
            storage
                .offset(expert * elements * std::mem::size_of::<f32>())
                .0
        })
        .collect();
    upload_ptrs(gpu, &pointers)
}

fn append_scale_f32(output: &mut Vec<u8>, page: &LingPage) -> Result<()> {
    match page.kind() {
        LingPageKind::Bf16Scale => {
            for chunk in page.bytes().chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                output.extend_from_slice(&f32::from_bits(u32::from(bits) << 16).to_le_bytes());
            }
        }
        LingPageKind::F32Scale => output.extend_from_slice(page.bytes()),
        LingPageKind::E4M3Weight => anyhow::bail!("Ling weight page used as a scale"),
    }
    ensure!(page.bytes().len() % page.kind().element_bytes() == 0);
    Ok(())
}

impl LingMoeFfn {
    pub(super) fn forward_strict_fast(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(ctx.config.model_type == "bailing_moe_v3_vl");
        ensure!(
            !ctx.graph_capture,
            "Ling native FFN does not support graph capture yet"
        );
        ensure!(ctx.config.hidden_size == self.hidden);
        let fast = self
            .strict_fast
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ling strict fast tables are unavailable"))?;
        let shared_gate = self.shared_gate.resident().expect("validated shared gate");
        let shared_up = self.shared_up.resident().expect("validated shared up");
        let shared_down = self.shared_down.resident().expect("validated shared down");
        let ids = ctx.buffers.scratch();
        let weights = ids.offset(TOP_K * std::mem::size_of::<i32>());
        ensure!(ctx.buffers.scratch_bytes() >= TOP_K * 8 * std::mem::size_of::<u32>());

        ops::dense_gemv_bf16_fp32out(
            ctx.gpu,
            self.kernels.router_gemv,
            input,
            &self.router,
            ctx.buffers.gate_logits_f32(),
            EXPERTS as u32,
            HIDDEN as u32,
            stream,
        )?;
        ops::ling_grouped_top8(
            ctx.gpu,
            self.kernels.router,
            ctx.buffers.gate_logits_f32(),
            self.expert_bias,
            ids,
            weights,
            stream,
        )?;
        let shared_up_out = fast.shared_gate_up.offset(INTERMEDIATE * 2);
        ops::moe_expert_gate_up_shared_fp8(
            ctx.gpu,
            self.kernels
                .fp8_gate_up
                .expect("strict fast gate/up kernel"),
            input,
            fast.gate.weight_ptrs,
            fast.gate.scale_ptrs,
            ctx.buffers.expert_gate_out(),
            fast.up.weight_ptrs,
            fast.up.scale_ptrs,
            ctx.buffers.expert_up_out(),
            ids,
            &shared_gate,
            fast.shared_gate_up,
            &shared_up,
            shared_up_out,
            INTERMEDIATE as u32,
            HIDDEN as u32,
            TOP_K as u32,
            stream,
        )?;
        ops::ling_silu_down_shared_fp8(
            ctx.gpu,
            self.kernels.ling_down_fp8.expect("strict fast down kernel"),
            ctx.buffers.expert_gate_out(),
            ctx.buffers.expert_up_out(),
            fast.down.weight_ptrs,
            fast.down.scale_ptrs,
            ctx.buffers.expert_down_out(),
            ids,
            fast.shared_gate_up,
            shared_up_out,
            &shared_down,
            ctx.buffers.attn_output(),
            HIDDEN as u32,
            INTERMEDIATE as u32,
            TOP_K as u32,
            super::routed_swiglu_limit(self.layer_idx),
            super::shared_swiglu_limit(self.layer_idx),
            stream,
        )?;
        ops::ling_weighted_sum(
            ctx.gpu,
            self.kernels.weighted_sum,
            ctx.buffers.expert_down_out(),
            weights,
            ctx.buffers.moe_output(),
            HIDDEN as u32,
            TOP_K as u32,
            stream,
        )?;
        ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            ctx.buffers.moe_output(),
            ctx.buffers.attn_output(),
            HIDDEN as u32,
            stream,
        )?;
        Ok(ctx.buffers.moe_output())
    }
}

#[cfg(test)]
mod tests {
    use super::append_scale_f32;
    use spark_runtime::weights::ling_paging::{LingPage, LingPageKind};

    // The conversion is intentionally kept in this module rather than using
    // a GPU widen launch: table construction must finish before any decode
    // stream can observe a pointer table.
    #[test]
    fn bf16_scale_widen_is_exact() {
        // Constructing a LingPage needs a mapped shard, so exercise the same
        // bit operation directly without coupling this unit test to mmap I/O.
        let bits = 0x3f80u16;
        assert_eq!(f32::from_bits(u32::from(bits) << 16), 1.0);
        let _ = append_scale_f32;
        let _ = LingPageKind::Bf16Scale;
        let _ = std::mem::size_of::<LingPage>();
    }
}
