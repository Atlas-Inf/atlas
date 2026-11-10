// SPDX-License-Identifier: AGPL-3.0-only

//! Small-M W4A16 row-GEMV (`w4a16_rowgemv_m{9,17}`, strix-hip
//! `qwen3.6-27b/nvfp4/w4a16_gemm.cu`) for DFlash verify rows on the transposed
//! NVFP4 layout `w4a16_gemm_t_m128` reads. Each weight is dequantized once for
//! all rows instead of m128's 128-row tile (7/8 zero rows at M = 17). Not
//! bit-identical to the WMMA kernels (fp32 summation order), so a model takes
//! it by default only once its accuracy gate passes with it (Ripple-65B on
//! strix-hip); elsewhere it is opt-in (`ATLAS_FFN_ROWGEMV=1`), and
//! `ATLAS_FFN_ROWGEMV=0` turns it off everywhere.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::kernel_pick::try_kernel_gated;
use crate::weight_map::QuantizedWeight;

/// Largest M the kernels take (the `_m17` instantiation).
pub const ROWGEMV_MAX_M: u32 = 17;
/// K splits across blocks for narrow N (fills the 40 CUs of gfx1151).
const ROWGEMV_KS_NARROW: u32 = 4;
/// N below which the K split is used: N/128 blocks fill the part above it.
const ROWGEMV_SPLIT_BELOW_N: u32 = 16384;
/// The fp32 split-K workspace `[KS][M][N]`, sized for the widest split shape.
pub const ROWGEMV_WS_BYTES: usize =
    (ROWGEMV_KS_NARROW * ROWGEMV_MAX_M * ROWGEMV_SPLIT_BELOW_N) as usize * 4;

/// `ATLAS_FFN_ROWGEMV=0`: off even where a model takes it by default.
pub fn rowgemv_opted_out() -> bool {
    std::env::var("ATLAS_FFN_ROWGEMV").as_deref() == Ok("0")
}

/// `[m9, m17, reduce]`; all 0 on targets without the kernels.
#[derive(Clone, Copy, Debug)]
pub struct RowGemvKernels {
    pub m9: KernelHandle,
    pub m17: KernelHandle,
    pub reduce: KernelHandle,
    /// Shared split-K workspace (one per backend: layers run in stream order).
    pub ws: DevicePtr,
}

impl RowGemvKernels {
    /// Looked up only on strix-hip with the lever set: the kernels exist in
    /// the qwen3.6-27b gfx1151 tree alone, and a lookup that is never issued
    /// leaves no failed row in the fail-closed boot audit on other targets.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Self::resolve_if(gpu, super::ModelLevers::from_env().ffn_rowgemv)
    }

    /// [`Self::resolve`] for a model that takes the kernels by default (`on`),
    /// unless `ATLAS_FFN_ROWGEMV=0`.
    pub fn resolve_default(gpu: &dyn GpuBackend) -> Result<Self> {
        Self::resolve_if(gpu, !rowgemv_opted_out())
    }

    fn resolve_if(gpu: &dyn GpuBackend, on: bool) -> Result<Self> {
        let on = cfg!(atlas_hip) && on;
        let m9 = try_kernel_gated(on, gpu, "w4a16", "w4a16_rowgemv_m9");
        let m17 = try_kernel_gated(on, gpu, "w4a16", "w4a16_rowgemv_m17");
        let reduce = try_kernel_gated(on, gpu, "w4a16", "w4a16_rowgemv_reduce");
        let ws = if m9.0 != 0 && m17.0 != 0 && reduce.0 != 0 {
            // Allocated at load, never inside a captured forward.
            gpu.op_cache()
                .scratch(gpu, "w4a16_rowgemv_ws", ROWGEMV_WS_BYTES)?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            m9,
            m17,
            reduce,
            ws,
        })
    }

    fn ks(n: u32) -> u32 {
        if n < ROWGEMV_SPLIT_BELOW_N {
            ROWGEMV_KS_NARROW
        } else {
            1
        }
    }

    /// Whether `[m, n, k]` is a shape the kernels take.
    pub fn fits(&self, m: u32, n: u32, k: u32) -> bool {
        let ks = Self::ks(n);
        !self.ws.is_null()
            && (1..=ROWGEMV_MAX_M).contains(&m)
            && n.is_multiple_of(128)
            && (k / 2).is_multiple_of(16 * ks)
            && k.is_multiple_of(2)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight_t: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let ks = Self::ks(n);
        let kernel = if m <= 9 { self.m9 } else { self.m17 };
        KernelLaunch::new(gpu, kernel)
            .grid([n / 128, ks, 1])
            .block([128, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight_t.weight)
            .arg_ptr(weight_t.weight_scale)
            .arg_f32(weight_t.weight_scale_2)
            .arg_ptr(output)
            .arg_ptr(self.ws)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        if ks > 1 {
            KernelLaunch::new(gpu, self.reduce)
                .grid([div_ceil(m * n, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.ws)
                .arg_ptr(output)
                .arg_u32(m * n)
                .arg_u32(ks)
                .launch(stream)?;
        }
        Ok(())
    }
}
