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
/// Threads per CTA: two 128-thread feature halves (`BM_HV` in the kernel).
const THREADS: u32 = 256;
/// Gate/up features per CTA (64 of gate and of up per half).
const GATE_UP_FEATURES: u32 = 128;
/// Down features per CTA (128 per half).
const DOWN_FEATURES: u32 = 256;
/// Down CTAs per expert along y cover 4x the average expert's 64-row chunks;
/// the kernel strides the rest, so the hottest expert's tile count is not
/// launched as empty CTAs for every expert (short K: 8 stages per CTA). Down
/// per layer, one score pass: 1011 -> 949 us at 2k rows, 1567 -> 1465 at 4k.
/// Gate/up keeps the full grid: there the hot experts' serialized chunks cost
/// more (1111 -> 1237 us at 1k rows).
const DOWN_GRID_AVG_FACTOR: u32 = 4;

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
            && h.is_multiple_of(DOWN_FEATURES)
            && inter.is_multiple_of(GATE_UP_FEATURES)
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
            grid_y: max_m_tiles,
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
    /// `[max_m_tiles, avg_rows]`: the hottest expert's 64-row tiles and the
    /// average expert's rows.
    pub(super) fn bf16mma_down(
        &self,
        expert_offsets: DevicePtr,
        num_experts: u32,
        [h, inter]: [u32; 2],
        [max_m_tiles, avg_rows]: [u32; 2],
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
            grid_y: max_m_tiles.min((DOWN_GRID_AVG_FACTOR * avg_rows).div_ceil(64).max(1)),
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
/// sorted row (gate/up only), and the CTAs per expert along y (any count:
/// the kernels stride an expert's 64-row chunks).
#[derive(Clone, Copy)]
pub(super) struct Rows {
    pub offsets: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub num_experts: u32,
    pub grid_y: u32,
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
        .grid([inter / GATE_UP_FEATURES, rows.grid_y, rows.num_experts])
        .block([THREADS, 1, 1])
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
        .grid([h / DOWN_FEATURES, rows.grid_y, rows.num_experts])
        .block([THREADS, 1, 1])
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
