// SPDX-License-Identifier: AGPL-3.0-only

//! Native-FP4 grouped MoE launcher. Split from `moe_grouped_a.rs` (500-LoC cap).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// Grouped W4A4 expert UP GEMM with fused relu^2 (native FP4 tensor cores).
/// A is the pre-quantized NVFP4 latent (packed E2M1 + per-16 E4M3 scales);
/// B comes from the per-expert NVFP4 pointer tables unchanged.
/// Grid: (ceil(n_out/128), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_relu2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_sf: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_sf)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Native-FP4 grouped GEMM over an activation quantized once outside the
/// output-column grid. A uses the common row-major NVFP4 packed/scale layout.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    threads: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([threads, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}

/// Row-tile (M64) prefix of the experts with local weights: `prefix[e]` =
/// row tiles before expert `e`, `prefix[num_experts]` = the total. One block;
/// `num_experts <= 1024`. Launch on the stream of the GEMMs that read it.
pub fn moe_mtile_prefix(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_offsets: DevicePtr,
    b_packed_ptrs: DevicePtr,
    prefix: DevicePtr,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(expert_offsets)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(prefix)
        .arg_u32(num_experts)
        .launch(stream)
}

/// A K128W kernel: the row-tile `grid` kernel and its `_persist` twin
/// (`KernelHandle(0)` unless ATLAS_GLM_MOE_PREFILL_PERSIST resolved it). The
/// twin takes one more argument, so only the schedule picks between them.
#[derive(Clone, Copy, Debug)]
pub struct K128wKernel {
    pub grid: KernelHandle,
    pub persist: KernelHandle,
}

/// Which row tiles a K128W launch covers: a grid over `bound` >= the local
/// experts' row tiles (`moe_mtile_prefix`), or `ctas` CTAs of the kernel's
/// `_persist` twin claiming its work items from the `next_work` counter,
/// zeroed before each launch (ATLAS_GLM_MOE_PREFILL_PERSIST), or `ctas` CTAs
/// of a kernel that strides over its work items itself, with the grid
/// kernel's arguments (the GLM decode persistent twins,
/// ATLAS_GLM_MOE_DECODE_PERSIST). Same bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum K128wSchedule {
    Grid { bound: u32 },
    Persistent { ctas: u32, next_work: DevicePtr },
    Stride { ctas: u32 },
}

impl K128wSchedule {
    /// Launch grid of a kernel with `n_tiles` column tiles.
    pub fn grid(self, n_tiles: u32) -> [u32; 3] {
        match self {
            Self::Grid { bound } => [n_tiles, bound.max(1), 1],
            Self::Persistent { ctas, .. } | Self::Stride { ctas } => [ctas.max(1), 1, 1],
        }
    }

    fn launch<'a>(
        self,
        gpu: &'a dyn GpuBackend,
        kernel: K128wKernel,
        n_tiles: u32,
        stream: u64,
    ) -> Result<KernelLaunch<'a>> {
        let handle = match self {
            Self::Grid { .. } | Self::Stride { .. } => kernel.grid,
            Self::Persistent { next_work, .. } => {
                anyhow::ensure!(kernel.persist.0 != 0, "persistent K128W kernel not loaded");
                gpu.memset_async(next_work, 0, 4, stream)?;
                kernel.persist
            }
        };
        Ok(KernelLaunch::new(gpu, handle)
            .grid(self.grid(n_tiles))
            .block([256, 1, 1]))
    }

    fn finish(self, launch: KernelLaunch<'_>, stream: u64) -> Result<()> {
        match self {
            Self::Grid { .. } | Self::Stride { .. } => launch.launch(stream),
            Self::Persistent { next_work, .. } => launch.arg_ptr(next_work).launch(stream),
        }
    }
}

/// Native-FP4 prequant GEMM with 64 x 256 tiles over only the row tiles the
/// local experts have (`moe_mtile_prefix`), as `schedule` covers them.
/// Outputs match `moe_w4a4_grouped_gemm_prequant_n128` with the K128
/// kernel bit for bit. Requires `n_out % 256 == 0`, `k % 128 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_k128w(
    gpu: &dyn GpuBackend,
    kernel: K128wKernel,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    prefix: DevicePtr,
    schedule: K128wSchedule,
    stream: u64,
) -> Result<()> {
    let launch = schedule
        .launch(gpu, kernel, n_out / 256, stream)?
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(prefix);
    schedule.finish(launch, stream)
}

/// K128W gate and up projections of `n_out` intermediate columns with the
/// DeepSeek-clamped SiLU·mul and NVFP4 quantization of `silu_mul_quant_nvfp4`
/// in the epilogue: writes its packed `[rows, n_out/2]` E2M1 and
/// `[rows, n_out/16]` E4M3 bytes for the local experts' rows. Tables are
/// `[packed, scales, scale2]` pointers. Requires `n_out % 128 == 0`,
/// `k % 128 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
    gpu: &dyn GpuBackend,
    kernel: K128wKernel,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    [gate_packed, gate_scale, gate_scale2]: [DevicePtr; 3],
    [up_packed, up_scale, up_scale2]: [DevicePtr; 3],
    out_packed: DevicePtr,
    out_scale: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    prefix: DevicePtr,
    schedule: K128wSchedule,
    stream: u64,
) -> Result<()> {
    let launch = schedule
        .launch(gpu, kernel, n_out / 128, stream)?
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(gate_packed)
        .arg_ptr(gate_scale)
        .arg_ptr(gate_scale2)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(prefix)
        .arg_ptr(up_packed)
        .arg_ptr(up_scale)
        .arg_ptr(up_scale2)
        .arg_ptr(out_packed)
        .arg_ptr(out_scale);
    schedule.finish(launch, stream)
}

/// Native-FP4 prequant GEMM over a compact `(expert, m_tile, n_tile)`
/// worklist. A conservative work-item bound replaces the dense expert grid;
/// excess CTAs exit before entering the unchanged per-tile MMA implementation.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_compact_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_tiles.max(1), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .arg_u32(max_tiles)
        .launch(stream)
}

/// Compact native-FP4 gate and up projections in one projection-multiplexed
/// launch. Grid y selects gate/up; grid x indexes the common device worklist.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_prequant_compact_gate_up_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_output: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([max_tiles.max(1), 2, 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_output)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .arg_u32(max_tiles)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// Grid kernel 1, `_persist` twin 2.
    const PAIR: K128wKernel = K128wKernel {
        grid: KernelHandle(1),
        persist: KernelHandle(2),
    };

    fn launch_down(
        gpu: &MockGpuBackend,
        kernel: K128wKernel,
        schedule: K128wSchedule,
    ) -> Result<()> {
        let p = DevicePtr(0x100);
        moe_w4a4_grouped_gemm_prequant_k128w(
            gpu,
            kernel,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            DevicePtr::NULL,
            288,
            4096,
            2048,
            p,
            schedule,
            0,
        )
    }

    #[test]
    fn k128w_grid_launch_takes_the_grid_kernel_over_the_row_tile_bound() {
        let gpu = MockGpuBackend::new();
        launch_down(&gpu, PAIR, K128wSchedule::Grid { bound: 1312 }).unwrap();
        assert_eq!(gpu.memset_count(), 0);
        let launch = &gpu.launches_snapshot()[0];
        assert_eq!(launch.func, 1);
        assert_eq!(launch.args, 12);
        assert_eq!((launch.grid, launch.block), ([16, 1312, 1], [256, 1, 1]));
    }

    #[test]
    fn k128w_persistent_launch_takes_the_twin_and_zeroes_its_counter_first() {
        let gpu = MockGpuBackend::new();
        let next_work = gpu.alloc(4).unwrap();
        gpu.memset(next_work, 0xff, 4).unwrap();
        let schedule = K128wSchedule::Persistent {
            ctas: 96,
            next_work,
        };
        launch_down(&gpu, PAIR, schedule).unwrap();
        assert_eq!(gpu.memset_count(), 1);
        assert_eq!(gpu.read_alloc(next_work).unwrap(), vec![0; 4]);
        let launch = &gpu.launches_snapshot()[0];
        assert_eq!(launch.func, 2);
        assert_eq!(launch.args, 13);
        assert_eq!((launch.grid, launch.block), ([96, 1, 1], [256, 1, 1]));
    }

    #[test]
    fn k128w_persistent_schedule_without_the_twin_launches_nothing() {
        let gpu = MockGpuBackend::new();
        let grid_only = K128wKernel {
            persist: KernelHandle(0),
            ..PAIR
        };
        let schedule = K128wSchedule::Persistent {
            ctas: 96,
            next_work: DevicePtr(0x200),
        };
        assert!(launch_down(&gpu, grid_only, schedule).is_err());
        assert_eq!((gpu.memset_count(), gpu.launch_count()), (0, 0));
    }
}
