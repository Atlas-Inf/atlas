// SPDX-License-Identifier: AGPL-3.0-only

//! TC MoE v3 (`ATLAS_QWEN4EXP_MOE_TC=1`, the default TC kernels):
//! `kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_moe_c8_tc3.cu`'s plan and
//! ONE persistent launch for gate/up + SiLU and down -- the TC v2 kernels'
//! bytes (`scripts/dev/qwen4exp_moe_c8_bench.sh tc-ident`), each expert's
//! weights read once, the next work item's weights in flight under the
//! current one's math. `ATLAS_QWEN4EXP_MOE_TC_V2=1` keeps v2 for A/B.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::Qwen4ExpMoeRows;
use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

const M: &str = "qwen4exp_moe_c8_tc3";
/// An expert table: packed, scale and scale2 pointers.
type Ptrs = (DevicePtr, DevicePtr, DevicePtr);
/// Persistent CTAs per SM (the kernel's `__launch_bounds__(256, 2)`).
const CTAS_PER_SM: u32 = 2;

impl Qwen4ExpMoeRows {
    /// v3's (plan, fused) handles and its grid width, 2 CTAs a SM.
    pub(super) fn tc3_kernels(gpu: &dyn GpuBackend, nc: bool) -> (KernelHandle, KernelHandle, u32) {
        let fused = try_kernel(
            gpu,
            M,
            if nc {
                "qwen4exp_moe_c8_tc3_nc"
            } else {
                "qwen4exp_moe_c8_tc3"
            },
        );
        let ctas = gpu.sm_count().unwrap_or(48).max(1) * CTAS_PER_SM;
        (try_kernel(gpu, M, "qwen4exp_moe_c8_tc3_plan"), fused, ctas)
    }

    /// The requested (tc, units) switches, both off with one warning when the
    /// model has more experts than every units plan takes (NEXP = 512 in the
    /// .cu files: a larger id traps).
    pub(super) fn units_fit(
        config: &atlas_core::config::ModelConfig,
        tc: bool,
        units: bool,
    ) -> (bool, bool) {
        static WARNED: std::sync::Once = std::sync::Once::new();
        let fit = config.num_experts <= 512;
        if units && !fit {
            WARNED.call_once(|| {
                tracing::warn!(
                    "qwen4_exp MoE units off: {} experts > 512",
                    config.num_experts
                )
            });
        }
        (tc && fit, units && fit)
    }

    /// [`Self::units`] on v3: the plan into `ws`, then the fused launch
    /// (BF16 activations in `act`).
    pub(super) fn units_tc3(
        &self,
        gpu: &dyn GpuBackend,
        (input, expert_indices, ws, act): (DevicePtr, DevicePtr, DevicePtr, DevicePtr),
        (gate_ptrs, up_ptrs, down_ptrs): (Ptrs, Ptrs, Ptrs),
        (sh_gate, sh_up, sh_down): (&QuantizedWeight, &QuantizedWeight, &QuantizedWeight),
        (output, sh_down_out): (DevicePtr, DevicePtr),
        (top_k, rows): (u32, u32),
        stream: u64,
    ) -> Result<()> {
        // v3 always runs the shared unit (v1/v2 write zeros without one).
        anyhow::ensure!(
            [sh_gate, sh_up, sh_down].iter().all(|w| w.weight.0 != 0),
            "qwen4exp MoE TC v3: no shared expert"
        );
        KernelLaunch::new(gpu, self.units_plan)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(expert_indices)
            .arg_ptr(gate_ptrs.0)
            .arg_ptr(ws)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)?;
        let mut l = KernelLaunch::new(gpu, self.units_fused)
            .grid([self.units_ctas, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input);
        for t in [gate_ptrs, up_ptrs, down_ptrs] {
            l = l.arg_ptr(t.0).arg_ptr(t.1).arg_ptr(t.2);
        }
        for w in [sh_gate, sh_up, sh_down] {
            l = l
                .arg_ptr(w.weight)
                .arg_ptr(w.weight_scale)
                .arg_f32(w.weight_scale_2);
        }
        l.arg_ptr(ws)
            .arg_ptr(act)
            .arg_ptr(output)
            .arg_ptr(sh_down_out)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)
    }
}
