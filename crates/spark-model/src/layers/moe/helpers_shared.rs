// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-native separate shared expert (BF16 or MXFP8) installed beside
//! the fused routed-expert kernels.

use super::*;

impl MoeLayer {
    /// Install checkpoint-native BF16 shared-expert weights independently of
    /// routed-expert precision.
    pub fn set_bf16_shared_expert(
        &mut self,
        gate_proj: DenseWeight,
        up_proj: DenseWeight,
        down_proj: DenseWeight,
    ) -> Result<()> {
        self.btile_storage.require_legacy()?;
        self.separate_shared_expert = Some(SeparateSharedExpert::Bf16(Bf16SharedExpert::new(
            gate_proj, up_proj, down_proj,
        )?));
        Ok(())
    }

    /// Install FP4-Local's checkpoint-native MXFP8 shared expert: the fused
    /// routed kernels get a NULL shared expert and this chain projects the
    /// E4M3 bytes through `mxfp8_proj.cu` — no BF16 expansion, no 4-bit
    /// requant of an 8-bit weight.
    pub(crate) fn set_mxfp8_shared_expert(
        &mut self,
        gate_proj: Mxfp8Proj,
        up_proj: Mxfp8Proj,
        down_proj: Mxfp8Proj,
        gate_up: Option<Mxfp8GateUp>,
        kernels: ops::Mxfp8ProjKernels,
        scratch: ops::Mxfp8Scratch,
    ) -> Result<()> {
        self.btile_storage.require_legacy()?;
        anyhow::ensure!(
            !gate_proj.w.data.is_null() && !up_proj.w.data.is_null() && !down_proj.w.data.is_null(),
            "MXFP8 shared expert requires non-null gate/up/down weights"
        );
        self.separate_shared_expert = Some(SeparateSharedExpert::Mxfp8 {
            gate_proj,
            up_proj,
            down_proj,
            gate_up,
            kernels,
            scratch,
        });
        Ok(())
    }

    /// Whether a checkpoint-native shared expert must overwrite the
    /// contribution produced by a quantized fused routed-expert kernel.
    pub(super) fn has_separate_shared_expert(&self) -> bool {
        self.separate_shared_expert.is_some() && self.bf16_gate_weight_ptrs.is_none()
    }

    /// Evaluate a checkpoint-native shared expert into `down_out`.
    ///
    /// Callers supply scratch buffers because the safe aliases differ between
    /// decode and prefill. Returns `true` when a separate expert was installed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_separate_shared_expert(
        &self,
        input: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        shared_intermediate: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(shared) = self.separate_shared_expert else {
            return Ok(false);
        };
        anyhow::ensure!(
            num_tokens > 0 && shared_intermediate > 0,
            "separate shared expert requires non-zero token and intermediate dimensions"
        );

        if let SeparateSharedExpert::Mxfp8 {
            gate_proj,
            up_proj,
            down_proj,
            gate_up,
            kernels,
            scratch,
        } = shared
        {
            let m = num_tokens as usize;
            if let Some(gu) = gate_up
                && m <= ops::MXFP8_PROJ_MAX_CHUNKED_ROWS
                && gu.silu_mul.0 != 0
            {
                // One projection yields `[m, 2*inter]` gate|up; fused_silu_mul
                // halves it in place-compatible scratch before the down pass.
                ops::mxfp8_proj(
                    ctx.gpu,
                    &kernels,
                    &gu.proj,
                    input,
                    gu.gate_up_out,
                    m,
                    scratch,
                    stream,
                )?;
                ops::fused_silu_mul(
                    ctx.gpu,
                    gu.silu_mul,
                    gu.gate_up_out,
                    up_out,
                    num_tokens,
                    shared_intermediate,
                    stream,
                )?;
                ops::mxfp8_proj(
                    ctx.gpu, &kernels, &down_proj, up_out, down_out, m, scratch, stream,
                )?;
                return Ok(true);
            }
            ops::mxfp8_proj(
                ctx.gpu, &kernels, &gate_proj, input, gate_out, m, scratch, stream,
            )?;
            ops::mxfp8_proj(
                ctx.gpu, &kernels, &up_proj, input, up_out, m, scratch, stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                gate_out,
                up_out,
                gate_out,
                num_tokens * shared_intermediate,
                stream,
            )?;
            ops::mxfp8_proj(
                ctx.gpu, &kernels, &down_proj, gate_out, down_out, m, scratch, stream,
            )?;
            return Ok(true);
        }
        let SeparateSharedExpert::Bf16(shared) = shared else {
            unreachable!()
        };

        let project = |activation: DevicePtr,
                       weight: &DenseWeight,
                       output: DevicePtr,
                       n: u32,
                       k: u32|
         -> Result<()> {
            if num_tokens == 1 {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv,
                    activation,
                    weight,
                    output,
                    n,
                    k,
                    stream,
                )
            } else if ctx.dispatch.cublas_gemm {
                // Multi-token shared expert: cuBLASLt BF16 beats the hand-written
                // mma.sync GEMM. The single-token arm above stays on the GEMV —
                // decode-sized shapes do not repay cuBLAS heuristic overhead.
                ops::cublas_bf16_proj_dense(
                    activation,
                    weight.weight,
                    output,
                    num_tokens,
                    n,
                    k,
                    stream,
                )
            } else {
                ops::dense_gemm_prefill(
                    ctx.gpu,
                    self.dense_gemm,
                    self.dense_gemm_pipelined,
                    activation,
                    weight,
                    output,
                    num_tokens,
                    n,
                    k,
                    stream,
                )
            }
        };

        project(
            input,
            &shared.gate_proj,
            gate_out,
            shared_intermediate,
            hidden_size,
        )?;
        project(
            input,
            &shared.up_proj,
            up_out,
            shared_intermediate,
            hidden_size,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            num_tokens * shared_intermediate,
            stream,
        )?;
        project(
            gate_out,
            &shared.down_proj,
            down_out,
            hidden_size,
            shared_intermediate,
        )?;
        Ok(true)
    }
}
