// SPDX-License-Identifier: AGPL-3.0-only

//! C3 grouped verify helpers: batch-three shared expert and router logits.

use super::*;

impl MoeLayer {
    /// Same batch-three shared projections as forward_k3, with destinations
    /// owned by the grouped pipeline until its final post-EP shared blend.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn c3_shared_expert(
        &self,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        h: u32,
        inter: u32,
        rows: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Every W4A16 batch tier runs the same per-row FMA chain (bit-identical
        // to batch3 and to M x w4a16_gemv, per w4a16_batch_bitparity_microtest),
        // so several owners' shared-expert rows share one weight read while each
        // row keeps the single-owner K3 arithmetic.
        anyhow::ensure!(
            rows.is_multiple_of(3) && rows > 0,
            "C3 shared expert rows {rows}"
        );
        let (h, inter) = (h as usize, inter as usize);
        let mut row = 0usize;
        while row < rows as usize {
            let left = rows as usize - row;
            let wide = if left <= 8 {
                left
            } else if left <= 12 {
                6
            } else {
                8
            };
            let kernel = self.w4a16_batchm.kernel(wide as u32);
            let take = if wide == 3 || kernel.0 == 0 { 3 } else { wide };
            let (x, g, u, d) = (
                input.offset(row * h * 2),
                gate_out.offset(row * inter * 2),
                up_out.offset(row * inter * 2),
                down_out.offset(row * h * 2),
            );
            let gemv =
                |x: DevicePtr, weight: &QuantizedWeight, out: DevicePtr, n: usize, k: usize| {
                    if take == 3 {
                        ops::w4a16_gemv_batch3(
                            ctx.gpu,
                            self.w4a16_gemv_batch3,
                            x,
                            weight,
                            out,
                            n as u32,
                            k as u32,
                            stream,
                        )
                    } else {
                        ops::w4a16_gemv_batchm(
                            ctx.gpu,
                            kernel,
                            x,
                            weight,
                            out,
                            take as u32,
                            n as u32,
                            k as u32,
                            stream,
                        )
                    }
                };
            gemv(x, &self.weights.shared_expert.gate_proj, g, inter, h)?;
            gemv(x, &self.weights.shared_expert.up_proj, u, inter, h)?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_silu_mul,
                g,
                u,
                g,
                (take * inter) as u32,
                stream,
            )?;
            gemv(g, &self.weights.shared_expert.down_proj, d, h, inter)?;
            row += take;
        }
        Ok(())
    }

    /// C3 router logits in scalar `dense_gemm` numerics. Short batches use
    /// the bit-identical row-parallel kernel (`ATLAS_GLM_ROUTER_ROWS=0`
    /// reverts; `ATLAS_GLM_ROUTER_ROWS_CHECK=1` compares both byte for byte).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn c3_router_logits(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        rows: u32,
        num_experts: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        static MODE: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
        let (enabled, check) = *MODE.get_or_init(|| {
            (
                std::env::var("ATLAS_GLM_ROUTER_ROWS").as_deref() != Ok("0"),
                std::env::var("ATLAS_GLM_ROUTER_ROWS_CHECK").as_deref() == Ok("1"),
            )
        });
        let scalar = |out: DevicePtr| {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                out,
                rows,
                num_experts,
                h,
                stream,
            )
        };
        if !enabled || rows > 32 || self.dense_gemm_router_rows.0 == 0 {
            return scalar(gate_logits);
        }
        ops::dense_gemm_router_rows(
            ctx.gpu,
            self.dense_gemm_router_rows,
            router_in,
            &self.weights.gate,
            gate_logits,
            rows,
            num_experts,
            h,
            stream,
        )?;
        if check {
            let bytes = (rows * num_experts) as usize * 2;
            let mut fast = vec![0u8; bytes];
            ctx.gpu.copy_d2h(gate_logits, &mut fast)?;
            scalar(gate_logits)?;
            let mut reference = vec![0u8; bytes];
            ctx.gpu.copy_d2h(gate_logits, &mut reference)?;
            let diff = fast.iter().zip(&reference).filter(|(a, b)| a != b).count();
            anyhow::ensure!(
                diff == 0,
                "router rows kernel differs from dense_gemm in {diff}/{bytes} bytes (rows={rows})"
            );
        }
        Ok(())
    }
}
