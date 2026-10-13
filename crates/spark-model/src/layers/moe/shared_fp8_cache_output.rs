// SPDX-License-Identifier: AGPL-3.0-only
//! Validated FP8 M64 dispatch, retained-T large prefill, and eager K5 diagnostic.
use super::*;
use std::sync::atomic::Ordering;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_fp8_cache(
        &self,
        projection: usize,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = &self.shared_fp8_cache;
        if state.layer.is_none() {
            return ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                weight,
                output,
                rows,
                n,
                k,
                stream,
            );
        }
        anyhow::ensure!(
            !state.verify || (!ctx.graph_capture && !ctx.gpu.stream_is_capturing(stream)),
            "shared FP8 output VERIFY requires eager stream, including previously checked projections"
        );
        // Validate the ordinal before its diagnostic bit shift, including
        // ordinary and large-row calls that never enter the K5 oracle.
        let (pn, pk, owner, capacity, cached, original) = match projection {
            0 => (
                2048,
                4096,
                ctx.buffers.ssm_deinterleaved(),
                ctx.buffers.sizes().ssm_deinterleaved,
                self.shared_gate_fp8,
                self.shared_gate_t,
            ),
            1 => (
                2048,
                4096,
                ctx.buffers.ssm_qkvz(),
                ctx.buffers.sizes().ssm_qkvz,
                self.shared_up_fp8,
                self.shared_up_t,
            ),
            2 => (
                4096,
                2048,
                ctx.buffers.attn_output(),
                ctx.buffers.sizes().attn_output,
                self.shared_down_fp8,
                self.shared_down_t,
            ),
            _ => anyhow::bail!("shared FP8 projection ordinal"),
        };
        anyhow::ensure!(
            rows > 0 && rows as usize <= ctx.buffers.max_batch_tokens(),
            "shared FP8 output row capacity exceeded"
        );
        anyhow::ensure!(
            (n, k) == (pn, pk) && output == owner && cached == Some(weight),
            "shared FP8 output geometry/owner/cached-weight binding"
        );
        let bf16_bytes = |width: u32| {
            (rows as usize)
                .checked_mul(width as usize)
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("shared FP8 output byte capacity overflow"))
        };
        let bytes = bf16_bytes(n)?;
        anyhow::ensure!(
            capacity >= bytes,
            "shared FP8 output geometry/owner/capacity"
        );
        let original = original
            .ok_or_else(|| anyhow::anyhow!("shared FP8 output missing retained T weight"))?;
        anyhow::ensure!(
            self.w4a16_gemm_t.0 != 0
                && original.weight_scale_2.is_finite()
                && !original.has_per_row_scale2(),
            "shared FP8 retained T handle/scalar contract"
        );
        let out = super::m5_projections::span(output, bytes, 2)?;
        let sources = [
            (input, bf16_bytes(k)?),
            (weight, n as usize * k as usize),
            (original.weight, n as usize * k as usize / 2),
            (original.weight_scale, n as usize * k as usize / 16),
        ];
        let mut spans = [0..0, 0..0, 0..0, 0..0];
        for (index, (ptr, len)) in sources.into_iter().enumerate() {
            spans[index] = super::m5_projections::span(ptr, len, 16)?;
            super::m5_projections::disjoint(&out, &spans[index])?;
            for previous in &spans[..index] {
                super::m5_projections::disjoint(previous, &spans[index])?;
            }
        }
        let reference = || {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                input,
                &original,
                output,
                rows,
                n,
                k,
                stream,
            )
        };
        // A configured 1024-token chunk can legitimately arrive as a solo
        // 1025-row pass in a 1025-row arena. The original T projection already
        // supports it; keep that existing precision/ABI without extending the
        // separately validated FP8 kernel envelope or fabricating oracle passes.
        if rows > 1024 {
            if ops::bf16_gemm_cutlass_rows(rows) {
                // E4M3 widens exactly to BF16, so the tuned BF16 tensor-core
                // GEMM computes the same products as the FP8 M64 kernel.
                let count = n as usize * k as usize;
                let wide =
                    ctx.gpu
                        .op_cache()
                        .scratch(ctx.gpu, "shared_fp8_bf16_weight", count * 2)?;
                ops::fp8_e4m3_to_bf16(ctx.gpu, weight, wide, count, stream)?;
                return ops::bf16_gemm(input, wide.0, output, rows, n, k, stream);
            }
            if let Some(budget) = super::shared_fp8_cache::prefill_slab_budget()? {
                // Keep the previously qualified M64 row envelope. Only the
                // explicit larger-prefill profile replaces the legacy T fallback.
                let slabs = super::shared_fp8_cache::prefill_rows::plan(
                    rows,
                    ctx.buffers.max_batch_tokens(),
                    n,
                    k,
                    input.0,
                    output.0,
                    budget,
                )?;
                for slab in slabs {
                    launch_cached_m64(
                        ctx.gpu,
                        self.fp8_gemm_k,
                        DevicePtr(slab.input),
                        weight,
                        DevicePtr(slab.output),
                        slab.rows,
                        n,
                        k,
                        stream,
                    )?;
                }
                return Ok(());
            }
            return reference();
        }
        anyhow::ensure!(
            self.fp8_gemm_k.0 != 0,
            "shared FP8 cache M64 handle missing before output/oracle work"
        );
        let launch = || {
            launch_cached_m64(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                weight,
                output,
                rows,
                n,
                k,
                stream,
            )
        };
        let bit = 1u8 << projection;
        if !state.verify || rows != 5 || state.checked.load(Ordering::Relaxed) & bit != 0 {
            return launch();
        }
        verify_output(ctx.gpu, output, bytes, stream, reference, launch)?;
        state.checked.fetch_or(bit, Ordering::Relaxed);
        tracing::info!(
            layer = state.layer,
            projection,
            rows,
            bytes,
            "GLM shared FP8 resident output oracle passed"
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_cached_m64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        kernel.0 != 0
            && (1..=1024).contains(&rows)
            && matches!((n, k), (2048, 4096) | (4096, 2048)),
        "shared FP8 cache M64 handle/geometry contract"
    );
    // Deliberately bypass generic fp8_gemm_n128's default-on LDMAB rewrite:
    // only this exact existing M64 kernel was validated for the cache.
    spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), rows.div_ceil(64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

fn verify_output(
    gpu: &dyn GpuBackend,
    output: DevicePtr,
    bytes: usize,
    stream: u64,
    reference_launch: impl FnOnce() -> Result<()>,
    candidate_launch: impl FnOnce() -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(
        !gpu.stream_is_capturing(stream) && bytes > 0 && bytes <= 40960 && bytes.is_multiple_of(2),
        "shared FP8 output oracle eager/bounded contract"
    );
    super::m5_projections::span(output, bytes, 2)?;
    gpu.memset_async(output, 0xff, bytes, stream)?;
    reference_launch()?;
    let mut reference = vec![0; bytes];
    gpu.copy_d2h_on_stream(output, &mut reference, stream)?;
    gpu.memset_async(output, 0xff, bytes, stream)?;
    candidate_launch()?;
    let mut actual = vec![0; bytes];
    gpu.copy_d2h_on_stream(output, &mut actual, stream)?;
    compare_outputs(&reference, &actual)
}

fn compare_outputs(reference: &[u8], actual: &[u8]) -> Result<()> {
    anyhow::ensure!(
        !reference.is_empty()
            && reference.len().is_multiple_of(2)
            && reference.len() == actual.len(),
        "shared FP8 oracle output length"
    );
    for (i, (r, a)) in reference
        .chunks_exact(2)
        .zip(actual.chunks_exact(2))
        .enumerate()
    {
        let r = u16::from_le_bytes([r[0], r[1]]);
        let a = u16::from_le_bytes([a[0], a[1]]);
        anyhow::ensure!(
            r & 0x7f80 != 0x7f80 && a & 0x7f80 != 0x7f80 && r == a,
            "shared FP8 resident output mismatch/nonfinite at {i}: {r:04x} vs {a:04x}"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "shared_fp8_cache_output_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "shared_fp8_cache_dispatch_tests.rs"]
mod dispatch_tests;
