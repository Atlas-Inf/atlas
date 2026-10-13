// SPDX-License-Identifier: AGPL-3.0-only

//! Quantized drafter twins on the tensor-core GEMV tiers.
//!
//! main's drafter NVFP4 twins serve only blocks of up to four rows through
//! the scalar `w4a16_gemv_batch4`, so a γ=8 DFlash2 drafter (GLM-5.3) read
//! its BF16 projections every proposal. These levers serve every row block
//! from smaller twins instead. All of them change drafts only — the target
//! verifies every token — so they can move acceptance, never output:
//!
//! - `ATLAS_DFLASH_NVFP4_TC=1`: the layer projections' NVFP4 twins on the
//!   tensor-core `w4a16_gemv_tc{8,16,32}` tiers (`ATLAS_W4A16_TC=1` resolves
//!   them), in 32-row pieces for batched proposals.
//! - `ATLAS_DFLASH_MXFP8=1`: MXFP8 twins (E4M3 + one E8M0 scale per 32
//!   values, lossless into BF16 MMA operands) of q/o/gate/up/down on
//!   `mxfp8_gemv_tc{8,16,32}`. With `ATLAS_DFLASH_NVFP4_TC=1` the NVFP4 tiers
//!   serve the layers instead and no MXFP8 layer twin is built.
//! - `ATLAS_DFLASH_NVFP4_HEAD=1` / `ATLAS_DFLASH_MXFP8_HEAD=1`: a drafter-owned
//!   NVFP4 (else MXFP8) twin of the shared BF16 lm_head — the drafter's
//!   largest read (154880 x 4096 on GLM-5.3). The verifier keeps its head.
//! - `ATLAS_DFLASH_CTX_NVFP4=1`: NVFP4 twins of `fc` and the fused K/V
//!   weight for context precomputes of up to 32 rows (decode steps); wider
//!   precomputes (prompt catch-up) keep BF16.
//!
//! One context-precompute lever changes no value at all:
//! `ATLAS_DFLASH_CTX_ASYNC_POS=1` uploads the precompute's RoPE positions in
//! stream order instead of after a drain of the stream (`precompute_ctx_kv`
//! step 4). Without it the host waits for the `fc` and K/V projections and
//! then launches the ~17 small ops of the append's tail one by one while the
//! GPU idles (a ~0.4 ms gap a C1 propose in the 2026-10-03 nsys profiles).
//!
//! Row blocks the tiers cannot take fall back to the BF16 weights.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Mxfp8Weight, QuantizedWeight};

/// Startup-resolved twin state beyond main's per-layer NVFP4/FP8 fields.
#[derive(Default)]
pub struct DflashTwins {
    /// `ATLAS_DFLASH_NVFP4_TC=1`: layer projections take the NVFP4 twins on
    /// the tensor-core tiers at any row count (main: only up to 4 rows).
    pub nvfp4_tc: bool,
    /// NVFP4 twin of the shared BF16 lm_head (`ATLAS_DFLASH_NVFP4_HEAD=1`),
    /// half the MXFP8 bytes, on the tensor-core tiers in 32-row pieces.
    /// Unlike a target NVFP4 head it is the drafter's own copy.
    pub lm_head_q4: Option<QuantizedWeight>,
    /// MXFP8 twin of the shared BF16 lm_head (`ATLAS_DFLASH_MXFP8_HEAD=1`,
    /// superseded by the NVFP4 head), up to 32 rows.
    pub lm_head_mx: Option<Mxfp8Weight>,
    /// NVFP4 twins of `[fc, fused_kv_weight]` (`ATLAS_DFLASH_CTX_NVFP4=1`).
    pub ctx_q4: Option<[QuantizedWeight; 2]>,
    /// `ATLAS_DFLASH_CTX_ASYNC_POS=1`: the context precompute's position
    /// upload does not drain the stream.
    pub ctx_async_positions: bool,
}

/// MXFP8 twins of a layer's five large projections (`ATLAS_DFLASH_MXFP8=1`);
/// k/v stay BF16 (the context precompute reads them through the fused K/V).
pub struct LayerMxfp8 {
    pub q_proj: Mxfp8Weight,
    pub o_proj: Mxfp8Weight,
    pub gate_proj: Mxfp8Weight,
    pub up_proj: Mxfp8Weight,
    pub down_proj: Mxfp8Weight,
}

fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

impl BlockDiffusionDraftHead {
    /// Install the twins the `ATLAS_DFLASH_*` levers request, after the
    /// Phase G FP8 install. main's NVFP4 install is skipped when MXFP8 serves
    /// the layers (its twins would never be read at γ > 4).
    pub(super) fn install_twins(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let mxfp8 = env_on("ATLAS_DFLASH_MXFP8");
        let nvfp4_tc = env_on("ATLAS_DFLASH_NVFP4_TC");
        if mxfp8 && !nvfp4_tc {
            self.install_mxfp8_layers(gpu)?;
        } else {
            self.try_install_nvfp4(gpu)?;
        }
        self.twins.nvfp4_tc =
            nvfp4_tc && matches!(self.quant, super::DflashQuantization::Nvfp4Weights);
        if nvfp4_tc && !self.twins.nvfp4_tc {
            tracing::warn!(
                "ATLAS_DFLASH_NVFP4_TC=1 but the drafter NVFP4 twins were not installed \
                 (ATLAS_NO_DFLASH_DRAFTER_NVFP4, drafter FP8, or missing quantizer)"
            );
        }
        self.install_head_twin(gpu)?;
        if env_on("ATLAS_DFLASH_CTX_NVFP4") {
            self.install_ctx_twins(gpu)?;
        }
        self.twins.ctx_async_positions = env_on("ATLAS_DFLASH_CTX_ASYNC_POS");
        tracing::info!(
            "DFlash twins: NVFP4 tensor-core layers {}, MXFP8 layers {}, head NVFP4 {} / MXFP8 {}, context NVFP4 {}, async context positions {}",
            self.twins.nvfp4_tc,
            self.layers.iter().filter(|l| l.mx.is_some()).count(),
            self.twins.lm_head_q4.is_some(),
            self.twins.lm_head_mx.is_some(),
            self.twins.ctx_q4.is_some(),
            self.twins.ctx_async_positions,
        );
        Ok(())
    }

    fn install_ctx_twins(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let Some(fused_kv) = self.fused_kv_weight else {
            return Ok(());
        };
        let h = self.hidden_size;
        let fc_in = self.target_layer_ids.len() * self.target_hidden_size;
        let fused_n = self.num_layers * 2 * self.num_kv_heads * self.head_dim;
        let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quant = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let q4 = |w: &DenseWeight, n, k| {
            crate::weight_map::quantize_to_nvfp4(w, n, k, gpu, absmax, quant, stream)
        };
        self.twins.ctx_q4 = Some([
            q4(&self.fc, h, fc_in)?,
            q4(&DenseWeight { weight: fused_kv }, fused_n, h)?,
        ]);
        gpu.synchronize(stream)
    }

    /// Context projection `out[n, n_out] = input[n, k] · Wᵀ` for the twin at
    /// `which` (0 = `fc`, 1 = fused K/V) of up to 32 rows on the tensor-core
    /// tiers, else the BF16 `w`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ctx_projection(
        &self,
        gpu: &dyn GpuBackend,
        which: usize,
        w: &DenseWeight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if m <= 32
            && let Some(q4) = self.twins.ctx_q4.as_ref()
            && self.nvfp4_tc_rows(gpu, &q4[which], input, out, m, n, k, stream)?
        {
            return Ok(());
        }
        self.drafter_dense_gemm(gpu, input, w, out, m, n, k, stream)
    }

    /// Drafter-owned twin of a BF16 shared lm_head. A target NVFP4 head
    /// (`lm_head_nvfp4`) already drafts from packed weights, so it gets none.
    fn install_head_twin(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.lm_head_nvfp4.is_some() || self.lm_head_shared.0 == 0 {
            return Ok(());
        }
        let head = DenseWeight {
            weight: self.lm_head_shared,
        };
        let (vocab, h) = (self.vocab_size, self.hidden_size);
        if env_on("ATLAS_DFLASH_NVFP4_HEAD") {
            let stream = gpu.default_stream();
            self.twins.lm_head_q4 = Some(crate::weight_map::quantize_to_nvfp4(
                &head,
                vocab,
                h,
                gpu,
                gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
                gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
                stream,
            )?);
            gpu.synchronize(stream)?;
        } else if env_on("ATLAS_DFLASH_MXFP8_HEAD") {
            self.twins.lm_head_mx = Some(self.quantize_mxfp8(gpu, &head, vocab, h)?);
            gpu.synchronize(gpu.default_stream())?;
        }
        Ok(())
    }

    /// Drafter logits `out[rows, vocab] = input[rows, H] · lm_headᵀ` for a
    /// BF16 shared head: the NVFP4 twin on the tensor-core tiers (32-row
    /// pieces: the NVFP4 head read twice still beats one BF16 read), else the
    /// MXFP8 twin (≤32 rows), else the BF16 head.
    pub(super) fn project_head(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        out: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let (vocab, h) = (self.vocab_size as u32, self.hidden_size as u32);
        if let Some(q4) = self.twins.lm_head_q4.as_ref()
            && self.nvfp4_tc_rows(gpu, q4, input, out, rows, vocab, h, stream)?
        {
            return Ok(());
        }
        if let Some(mx) = self.twins.lm_head_mx.as_ref()
            && self.mxfp8_rows(gpu, mx, input, out, rows, vocab, h, stream)?
        {
            return Ok(());
        }
        let head = DenseWeight {
            weight: self.lm_head_shared,
        };
        self.drafter_dense_gemm(gpu, input, &head, out, rows, vocab, h, stream)
    }

    fn install_mxfp8_layers(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let h = self.hidden_size;
        let q_dim = self.num_q_heads * self.head_dim;
        let inter = self.intermediate_size;
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let mx = LayerMxfp8 {
                q_proj: self.quantize_mxfp8(gpu, &layer.q_proj, q_dim, h)?,
                o_proj: self.quantize_mxfp8(gpu, &layer.o_proj, h, q_dim)?,
                gate_proj: self.quantize_mxfp8(gpu, &layer.gate_proj, inter, h)?,
                up_proj: self.quantize_mxfp8(gpu, &layer.up_proj, inter, h)?,
                down_proj: self.quantize_mxfp8(gpu, &layer.down_proj, h, inter)?,
            };
            self.layers[i].mx = Some(mx);
        }
        gpu.synchronize(gpu.default_stream())
    }

    /// E4M3 `[n, k]` + E8M0 `[n, k/32]` twin of a BF16 `[n, k]` weight.
    pub(super) fn quantize_mxfp8(
        &self,
        gpu: &dyn GpuBackend,
        w: &DenseWeight,
        n: usize,
        k: usize,
    ) -> Result<Mxfp8Weight> {
        let quant = self.kernels.mxfp8_quantize;
        anyhow::ensure!(
            quant.0 != 0 && self.kernels.mxfp8_gemv.iter().all(|k| k.0 != 0),
            "ATLAS_DFLASH_MXFP8 twins need the mxfp8_gemv kernels (GLM-5.3 kernel target)"
        );
        anyhow::ensure!(
            w.weight.0 != 0,
            "DFlash MXFP8 twin of a dropped BF16 weight"
        );
        let data = gpu.alloc(n * k)?;
        let scales = gpu.alloc(n * k / ops::MXFP8_BLOCK)?;
        ops::mxfp8_quantize(
            gpu,
            quant,
            w.weight,
            data,
            scales,
            n,
            k,
            gpu.default_stream(),
        )?;
        Ok(Mxfp8Weight { data, scales })
    }

    /// `out[m, n] = input[m, k] · Wᵀ` from an NVFP4 twin on the tensor-core
    /// tiers, in 32-row pieces. `Ok(false)` (nothing launched) when a tier
    /// covering a piece is not resolved (`ATLAS_W4A16_TC` unset, or a target
    /// without the tiers).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn nvfp4_tc_rows(
        &self,
        gpu: &dyn GpuBackend,
        w: &QuantizedWeight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let tier = crate::layers::w4a16_gemv_tiers::tc_kernel;
        let pieces = || (0..m).step_by(32).map(move |r0| (r0, (m - r0).min(32)));
        if m == 0 || !pieces().all(|(_, rows)| tier(rows).0 != 0) {
            return Ok(false);
        }
        for (r0, rows) in pieces() {
            ops::w4a16_gemv_batchm(
                gpu,
                tier(rows),
                input.offset(r0 as usize * k as usize * 2),
                w,
                out.offset(r0 as usize * n as usize * 2),
                rows,
                n,
                k,
                stream,
            )?;
        }
        Ok(true)
    }

    /// `out[m, n] = input[m, k] · Wᵀ` from an MXFP8 twin (up to 32 rows).
    /// `Ok(false)` when the row count or the kernels rule it out.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mxfp8_rows(
        &self,
        gpu: &dyn GpuBackend,
        w: &Mxfp8Weight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let kernel = match m {
            1..=8 => self.kernels.mxfp8_gemv[0],
            9..=16 => self.kernels.mxfp8_gemv[1],
            17..=32 => self.kernels.mxfp8_gemv[2],
            _ => return Ok(false),
        };
        if kernel.0 == 0 {
            return Ok(false);
        }
        ops::mxfp8_gemv(
            gpu, kernel, input, w.data, w.scales, out, m, n, k, n, stream,
        )?;
        Ok(true)
    }
}
