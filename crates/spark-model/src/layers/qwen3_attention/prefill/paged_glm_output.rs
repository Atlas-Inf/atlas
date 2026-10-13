// SPDX-License-Identifier: AGPL-3.0-only

//! GLM paged prefill row math: W_uk query absorb and the W_uv + o_proj
//! output.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Mxfp8Weight;

/// The MXFP8 twin `glm_head_gemm` reads for `rows` of the `g`-head weight
/// `weight` (`[g, n, k]`): only a twin quantized as exactly `[g * n, k]`, for
/// at most 16 rows, with both grouped tiers resolved. Else BF16.
fn glm_head_twin<'a>(
    twins: &'a [(DevicePtr, [usize; 2], Mxfp8Weight)],
    tiers: &[KernelHandle; 2],
    weight: DevicePtr,
    [rows, g, k, n]: [u32; 4],
) -> Option<&'a Mxfp8Weight> {
    let shape = [g as usize * n as usize, k as usize];
    let servable = rows <= ops::MXFP8_GROUPED_MAX_M && tiers.iter().all(|t| t.0 != 0);
    twins
        .iter()
        .find(|&&(w, s, _)| servable && w == weight && s == shape)
        .map(|(.., mx)| mx)
}

impl Qwen3AttentionLayer {
    /// W_uk absorb of `rows` q_b outputs into `nq` per-head latent queries.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_absorb_queries(
        &self,
        q_full: DevicePtr,
        q_absorbed: DevicePtr,
        rows: u32,
        nq: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mla = self.mla.as_ref().expect("GLM absorb without MLA");
        let (hd, kv_lora) = (mla.nope as u32, mla.kv_lora_rank as u32);
        self.glm_head_gemm(
            q_full,
            mla.w_uk_t.weight,
            q_absorbed,
            [rows, nq, hd, kv_lora, nq * hd, nq * kv_lora],
            ctx,
            stream,
        )
    }

    /// Per-head MLA GEMM `c[:, h*n..] = a[:, h*k..] · weight_hᵀ` over `g`
    /// heads: its MXFP8 twin (`ATLAS_GLM_MLA_KVB_MXFP8=1`, see
    /// `glm_head_twin`), else the BF16 grouped GEMM.
    fn glm_head_gemm(
        &self,
        a: DevicePtr,
        weight: DevicePtr,
        c: DevicePtr,
        [rows, g, k, n, a_stride, c_stride]: [u32; 6],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let tiers = &self.mxfp8_gemv_grouped_k;
        if let Some(mx) = glm_head_twin(&self.mla_mx, tiers, weight, [rows, g, k, n]) {
            return ops::mxfp8_gemv_grouped(
                ctx.gpu, tiers, a, mx.data, mx.scales, c, rows, g, k, n, a_stride, c_stride, stream,
            );
        }
        ops::glm_paged_grouped_gemm_mla(
            ctx.gpu,
            self.grouped_gemm_mla_k,
            &ctx.config.model_type,
            a,
            weight,
            c,
            rows,
            g,
            k,
            n,
            a_stride,
            c_stride,
            stream,
        )
    }

    /// W_uv then the row-parallel o_proj for `rows` latent attention rows
    /// of `nq` local heads into `h` hidden columns.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn paged_glm_output(
        &self,
        mla: &MlaWeights,
        latent: DevicePtr,
        out: DevicePtr,
        [rows, nq, h]: [u32; 3],
        ctx: &ForwardContext,
        stream: u64,
        accelerated: bool,
    ) -> Result<()> {
        let (kv_lora, v_dim) = (mla.kv_lora_rank as u32, mla.v_dim as u32);
        let v_extracted = ctx.buffers.qkv_output();
        self.glm_head_gemm(
            latent,
            mla.w_uv.weight,
            v_extracted,
            [rows, nq, kv_lora, v_dim, nq * kv_lora, nq * v_dim],
            ctx,
            stream,
        )?;
        self.paged_glm_projection(
            v_extracted,
            &mla.wo,
            out,
            rows,
            h,
            nq * v_dim,
            ctx,
            stream,
            accelerated,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIERS: [KernelHandle; 2] = [KernelHandle(8), KernelHandle(16)];
    const W_UK: DevicePtr = DevicePtr(0x100);
    const W_UV: DevicePtr = DevicePtr(0x200);
    /// `[g, k, n]` of the absorb (W_uk: k = nope, n = kv_lora) and of W_uv.
    const UK: [u32; 3] = [32, 256, 512];
    const UV: [u32; 3] = [32, 512, 256];

    /// Twins as `install_mla_mxfp8` records them: `[heads * n, k]`.
    fn twins() -> Vec<(DevicePtr, [usize; 2], Mxfp8Weight)> {
        [
            (W_UK, [32 * 512, 256], 0x1000),
            (W_UV, [32 * 256, 512], 0x2000),
        ]
        .map(|(w, shape, p)| {
            let (data, scales) = (DevicePtr(p), DevicePtr(p + 0x800));
            (w, shape, Mxfp8Weight { data, scales })
        })
        .to_vec()
    }

    fn route(
        twins: &[(DevicePtr, [usize; 2], Mxfp8Weight)],
        tiers: [KernelHandle; 2],
        weight: DevicePtr,
        rows: u32,
        [g, k, n]: [u32; 3],
    ) -> Option<u64> {
        glm_head_twin(twins, &tiers, weight, [rows, g, k, n]).map(|mx| mx.data.0)
    }

    #[test]
    fn head_gemm_takes_the_twin_up_to_16_rows() {
        for rows in [1, 8, 9, 16] {
            assert_eq!(route(&twins(), TIERS, W_UK, rows, UK), Some(0x1000));
            assert_eq!(route(&twins(), TIERS, W_UV, rows, UV), Some(0x2000));
        }
    }

    #[test]
    fn head_gemm_keeps_bf16_past_16_rows_or_without_a_twin() {
        assert_eq!(route(&twins(), TIERS, W_UK, 17, UK), None);
        assert_eq!(route(&twins(), TIERS, W_UV, 32, UV), None);
        assert_eq!(route(&[], TIERS, W_UK, 8, UK), None);
        assert_eq!(route(&twins(), TIERS, DevicePtr(0x300), 8, UK), None);
    }

    #[test]
    fn head_gemm_keeps_bf16_for_a_twin_of_another_shape() {
        // k and n swapped, or a head count other than the quantized one.
        assert_eq!(route(&twins(), TIERS, W_UK, 8, UV), None);
        assert_eq!(route(&twins(), TIERS, W_UV, 8, UK), None);
        assert_eq!(route(&twins(), TIERS, W_UK, 8, [16, 256, 512]), None);
    }

    #[test]
    fn head_gemm_keeps_bf16_without_both_grouped_tiers() {
        for tiers in [
            [KernelHandle(0), KernelHandle(16)],
            [KernelHandle(8), KernelHandle(0)],
        ] {
            assert_eq!(route(&twins(), tiers, W_UK, 4, UK), None);
        }
    }
}
