// SPDX-License-Identifier: AGPL-3.0-only

//! Routed NVFP4 MoE prefill on BF16 tensor cores (`moe_bf16mma_prefill.cu`,
//! lever `ATLAS_MOE_PREFILL_BF16MMA=1`): the gate/up GEMM with SiLU·mul in its
//! epilogue, then the down GEMM. Both read the transposed `[K/2, N]` prefill
//! tables the FP8 kernels read (`gate_ptrs_t`/`up_ptrs_t`/`down_ptrs_t`).
//! Weights dequantize exactly to BF16 and activations stay BF16, where the
//! FP8 kernels round both to E4M3.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

const MODULE: &str = "moe_bf16mma_prefill";
/// 128-feature down tiles per CTA. The down K is short (one 512-wide
/// intermediate, 8 pipeline stages per tile), so chaining 4 tiles on one
/// pipeline amortizes its fill: 0.95 -> 0.89 ms at 1k rows, 1.81 -> 1.70 ms
/// at 4k (isolated, 256 experts, hidden 2048).
const DOWN_TILES_PER_CTA: u32 = 4;

/// `[gate_up_silu, down]`; handle 0 where the target does not ship them.
pub(super) fn kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 2] {
    [
        super::super::try_kernel(gpu, MODULE, "moe_bf16mma_gate_up_silu"),
        super::super::try_kernel(gpu, MODULE, "moe_bf16mma_down"),
    ]
}

impl MoeLayer {
    /// Whether this routed prefill takes the BF16-MMA kernels: lever on,
    /// kernels shipped, plain NVFP4 experts with transposed tables, and none
    /// of the paths that need the separate gate/up outputs or another weight
    /// layout.
    pub(super) fn bf16mma_serves(&self, h: u32, inter: u32, ctx: &ForwardContext) -> bool {
        ctx.levers.moe_prefill_bf16mma
            && self.bf16mma_k.iter().all(|k| k.0 != 0)
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.grouped_decode_ok()
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && !self.btile_storage.is_published()
            && !self.nvfp4_mmq_layout
            && !self.nvfp4_vecscale
            && !self.prequant_moe()
            && !self.routed_scales_released
            && !self.gelu_activation
            && self.lora.is_none()
            && ctx.comm.is_none()
            && h.is_multiple_of(128 * DOWN_TILES_PER_CTA)
            && inter.is_multiple_of(64)
    }

    /// Gate/up GEMM + SiLU·mul: token-major `input` gathered by
    /// `sorted_token_ids` → sorted `[rows, inter]` in `expert_gate_out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bf16mma_gate_up(
        &self,
        input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_experts: u32,
        h: u32,
        inter: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (Some(g), Some(u)) = (&self.gate_ptrs_t, &self.up_ptrs_t) else {
            anyhow::bail!("BF16-MMA MoE gate/up needs the transposed tables");
        };
        let rows = Rows {
            offsets: expert_offsets,
            sorted_token_ids,
            num_experts,
            max_m_tiles,
        };
        launch_gate_up(
            ctx.gpu,
            self.bf16mma_k[0],
            input,
            [table(g), table(u)],
            ctx.buffers.expert_gate_out(),
            rows,
            [h, inter],
            stream,
        )
    }

    /// Down GEMM: sorted `expert_gate_out` `[rows, inter]` → sorted
    /// `[rows, h]` in `expert_down_out`.
    pub(super) fn bf16mma_down(
        &self,
        expert_offsets: DevicePtr,
        num_experts: u32,
        h: u32,
        inter: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(d) = &self.down_ptrs_t else {
            anyhow::bail!("BF16-MMA MoE down needs the transposed table");
        };
        let rows = Rows {
            offsets: expert_offsets,
            sorted_token_ids: DevicePtr::NULL,
            num_experts,
            max_m_tiles,
        };
        launch_down(
            ctx.gpu,
            self.bf16mma_k[1],
            ctx.buffers.expert_gate_out(),
            table(d),
            ctx.buffers.expert_down_out(),
            rows,
            [h, inter],
            stream,
        )
    }
}

/// One projection's per-expert device arrays: packed `[K/2, N]` e2m1 and
/// `[K/16, N]` e4m3 scale pointers, and the FP32 global scales.
pub(super) type Table = [DevicePtr; 3];

fn table(t: &ExpertPtrTable) -> Table {
    [t.packed_ptrs, t.scale_ptrs, t.scale2_vals]
}

/// The expert-sorted rows: `[num_experts + 1]` offsets, the token of each
/// sorted row (gate/up only), and the hottest expert's 64-row tile count.
#[derive(Clone, Copy)]
pub(super) struct Rows {
    pub offsets: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub num_experts: u32,
    pub max_m_tiles: u32,
}

/// `moe_bf16mma_gate_up_silu` over `[gate, up]`; `[h, inter]` dims.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_gate_up(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    [g, u]: [Table; 2],
    out: DevicePtr,
    rows: Rows,
    [h, inter]: [u32; 2],
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([inter / 64, rows.max_m_tiles, rows.num_experts])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(g[0])
        .arg_ptr(g[1])
        .arg_ptr(g[2])
        .arg_ptr(u[0])
        .arg_ptr(u[1])
        .arg_ptr(u[2])
        .arg_ptr(out)
        .arg_ptr(rows.offsets)
        .arg_ptr(rows.sorted_token_ids)
        .arg_u32(rows.num_experts)
        .arg_u32(inter)
        .arg_u32(h)
        .launch(stream)
}

/// `moe_bf16mma_down`: sorted `[rows, inter]` → sorted `[rows, h]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_down(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    d: Table,
    out: DevicePtr,
    rows: Rows,
    [h, inter]: [u32; 2],
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([
            h / (128 * DOWN_TILES_PER_CTA),
            rows.max_m_tiles,
            rows.num_experts,
        ])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(d[0])
        .arg_ptr(d[1])
        .arg_ptr(d[2])
        .arg_ptr(out)
        .arg_ptr(rows.offsets)
        .arg_u32(rows.num_experts)
        .arg_u32(h)
        .arg_u32(inter)
        .launch(stream)
}

#[cfg(test)]
#[path = "forward_prefill_bf16mma_test_gpu.rs"]
mod gpu_tests;
