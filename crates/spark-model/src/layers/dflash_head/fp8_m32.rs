// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 mirrors of the drafter's dense weights for the small-M FP8 GEMV
//! (`fp8_gemv_m32`): `ATLAS_DFLASH_DRAFTER_FP8_M32=1` on a target that ships
//! the kernel (Ricursive). The drafter only proposes tokens and verification
//! is exact, so this moves acceptance, never output; it halves the drafter's
//! weight stream per propose. The BF16 weights stay for every other path.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::BlockDiffusionDraftHead;
use crate::weight_map::{DenseWeight, Fp8DenseWeight};

fn requested() -> bool {
    std::env::var("ATLAS_DFLASH_DRAFTER_FP8_M32").as_deref() == Ok("1")
}

/// Quantize the seven per-layer projections, `fc` and the shared lm_head.
pub(super) fn build(head: &mut BlockDiffusionDraftHead, gpu: &dyn GpuBackend) -> Result<()> {
    if !requested() {
        return Ok(());
    }
    anyhow::ensure!(
        head.kernels.fp8_gemv_m32.0 != 0,
        "ATLAS_DFLASH_DRAFTER_FP8_M32=1 but this kernel target has no fp8_gemv_m32"
    );
    let (h, inter) = (head.hidden_size, head.intermediate_size);
    let q_dim = head.num_q_heads * head.head_dim;
    let kv_dim = head.num_kv_heads * head.head_dim;
    let quant_k = head.kernels.quantize_bf16_to_fp8;
    let stream = gpu.default_stream();
    let mut mirrors = Vec::new();
    let mut add = |w: &DenseWeight, n: usize, k: usize| -> Result<()> {
        mirrors.push((w.weight.0, w.quantize_to_fp8(gpu, quant_k, n, k, stream)?));
        Ok(())
    };
    for layer in &head.layers {
        add(&layer.q_proj, q_dim, h)?;
        add(&layer.k_proj, kv_dim, h)?;
        add(&layer.v_proj, kv_dim, h)?;
        add(&layer.o_proj, h, q_dim)?;
        add(&layer.gate_proj, inter, h)?;
        add(&layer.up_proj, inter, h)?;
        add(&layer.down_proj, h, inter)?;
    }
    let fc_in = head.target_layer_ids.len() * head.target_hidden_size;
    add(&head.fc, h, fc_in)?;
    // The shared lm_head is BF16 unless the target packed it (NVFP4 twin).
    if head.lm_head_nvfp4.is_none() {
        let lm = DenseWeight {
            weight: head.lm_head_shared,
        };
        add(&lm, head.vocab_size, h)?;
    }
    gpu.synchronize(stream)?;
    tracing::info!(
        "DFlash drafter: {} FP8 GEMV mirrors (ATLAS_DFLASH_DRAFTER_FP8_M32)",
        mirrors.len()
    );
    head.fp8_m32 = mirrors;
    Ok(())
}

impl BlockDiffusionDraftHead {
    /// The FP8 mirror of `w` when the small-M FP8 GEMV can serve `m` x `k`.
    pub(super) fn fp8_m32_for(&self, w: &DenseWeight, m: u32, k: u32) -> Option<&Fp8DenseWeight> {
        if m > crate::layers::ops::BF16_GEMV_MAX_M || !k.is_multiple_of(512) {
            return None;
        }
        self.fp8_m32
            .iter()
            .find(|(ptr, _)| *ptr == w.weight.0)
            .map(|(_, w8)| w8)
    }
}
