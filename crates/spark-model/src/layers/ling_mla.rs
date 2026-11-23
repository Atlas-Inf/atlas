// SPDX-License-Identifier: AGPL-3.0-only

//! Native one-token MLA for Ling 3.2 air.
//!
//! Ling's full-attention layers are the direct (q_lora-free) Bailing MLA:
//! q_proj, kv_a_proj_with_mqa + RMSNorm, kv_b_proj, interleaved RoPE,
//! paged K/V, a head-wise sigmoid gate, and the checkpoint's dense output
//! projection. The stock paged-attention ABI has one head width for K and V;
//! Ling's K width is 192 while V is 128. This adapter stores V in a zero-padded
//! 192-wide row, then narrows attention output back to 128 before gating and
//! dense. The cache layer therefore must be configured as (32, 192).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::AttnMetadataDev;
use crate::layers::ling_projection::{LingProjection, LingProjectionScratch};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Current Ling 3.2 air MLA geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LingMlaConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl Default for LingMlaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2560,
            num_heads: 32,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            rope_theta: 6_000_000.0,
            rms_eps: 1e-6,
        }
    }
}

impl LingMlaConfig {
    pub fn qk_head_dim(self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn q_proj_rows(self) -> usize {
        self.num_heads * self.qk_head_dim()
    }

    pub fn kv_b_rows(self) -> usize {
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
    }

    pub fn validate(self) -> Result<()> {
        ensure!(self.hidden_size > 0, "Ling MLA hidden size is zero");
        ensure!(self.num_heads > 0, "Ling MLA head count is zero");
        ensure!(self.kv_lora_rank > 0, "Ling MLA KV rank is zero");
        ensure!(self.qk_nope_head_dim > 0, "Ling MLA NoPE width is zero");
        ensure!(self.qk_rope_head_dim > 0, "Ling MLA RoPE width is zero");
        ensure!(
            self.qk_rope_head_dim.is_multiple_of(2),
            "Ling MLA RoPE width must be even"
        );
        ensure!(self.v_head_dim > 0, "Ling MLA value width is zero");
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "invalid Ling MLA RoPE theta"
        );
        ensure!(
            self.rms_eps.is_finite() && self.rms_eps > 0.0,
            "invalid Ling MLA RMS epsilon"
        );
        Ok(())
    }
}

/// Tensors required by one Ling MLA layer. Prefix is normally
/// model.layers.{layer}.attention.
pub struct LingMlaWeights {
    pub q_proj: LingProjection,
    pub kv_a_proj_with_mqa: LingProjection,
    pub kv_a_layernorm: DenseWeight,
    pub kv_b_proj: LingProjection,
    pub g_proj: LingProjection,
    /// The checkpoint calls this projection dense, not o_proj.
    pub dense: LingProjection,
    /// Device table [qk_rope_head_dim / 2] for the interleaved RoPE kernel.
    pub rope_inv_freq: DevicePtr,
}

impl LingMlaWeights {
    pub fn load(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        prefix: &str,
        config: LingMlaConfig,
        rope_inv_freq: DevicePtr,
    ) -> Result<Self> {
        config.validate()?;
        let h = config.hidden_size;
        let q = config.q_proj_rows();
        let kv_a = config.kv_lora_rank + config.qk_rope_head_dim;
        let kv_b = config.kv_b_rows();
        let norm = store.get(&format!("{prefix}.kv_a_layernorm.weight"))?;
        ensure!(
            norm.shape == [config.kv_lora_rank],
            "{prefix}.kv_a_layernorm.weight has shape {:?}; expected [{}]",
            norm.shape,
            config.kv_lora_rank
        );
        ensure!(
            norm.dtype == WeightDtype::BF16,
            "{prefix}.kv_a_layernorm.weight must be BF16"
        );
        Ok(Self {
            q_proj: LingProjection::load(gpu, store, &format!("{prefix}.q_proj"), q, h)?,
            kv_a_proj_with_mqa: LingProjection::load(
                gpu,
                store,
                &format!("{prefix}.kv_a_proj_with_mqa"),
                kv_a,
                h,
            )?,
            kv_a_layernorm: DenseWeight { weight: norm.ptr },
            kv_b_proj: LingProjection::load(
                gpu,
                store,
                &format!("{prefix}.kv_b_proj"),
                kv_b,
                config.kv_lora_rank,
            )?,
            g_proj: LingProjection::load(
                gpu,
                store,
                &format!("{prefix}.g_proj"),
                config.num_heads,
                h,
            )?,
            dense: LingProjection::load(
                gpu,
                store,
                &format!("{prefix}.dense"),
                h,
                config.num_heads * config.v_head_dim,
            )?,
            rope_inv_freq,
        })
    }
}

/// Scratch layout for a single decode token. All pointers are BF16 except
/// projection activation/scales, whose sizes are described by the nested
/// LingProjectionScratch. The caller owns and reuses this storage.
#[derive(Clone, Copy)]
pub struct LingMlaScratch {
    pub projection: LingProjectionScratch,
    pub q_full: DevicePtr,
    pub kv_a: DevicePtr,
    pub kv_norm: DevicePtr,
    pub q_rope: DevicePtr,
    pub k_rope: DevicePtr,
    pub kv_b: DevicePtr,
    pub k_token: DevicePtr,
    pub v_compact: DevicePtr,
    pub v_padded: DevicePtr,
    pub attn_padded: DevicePtr,
    pub attn_compact: DevicePtr,
    pub attn_gated: DevicePtr,
    pub gate: DevicePtr,
}

/// Kernel family used by the native path. The MLA assembly kernels are shared
/// with DeepSeek's absorbed implementation; only their generic K/V assembly
/// ABI is used here.
#[derive(Clone, Copy)]
pub struct LingMlaKernels {
    pub rms_norm: KernelHandle,
    /// Ling's body-diagonal MRoPE; unlike Qwen3.6 it maps H/W/T by
    /// contiguous frequency sections after the adjacent-pair reshape.
    pub ling_mrope: KernelHandle,
    pub q_rope_extract: KernelHandle,
    pub q_rope_writeback: KernelHandle,
    pub kv_assemble: KernelHandle,
    pub reshape_cache: KernelHandle,
    pub paged_attention: KernelHandle,
    pub sigmoid_gate: KernelHandle,
}

impl LingMlaKernels {
    /// Resolve the BF16 one-token family used by the pinned Ling checkpoint.
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rms_norm: gpu.kernel("ling_norm", "ling_rms_norm")?,
            ling_mrope: gpu.kernel("rope_mrope_interleaved", "ling_mrope_rope_bf16")?,
            q_rope_extract: gpu.kernel("mla_absorbed", "mla_q_rope_extract_batched")?,
            q_rope_writeback: gpu.kernel("mla_absorbed", "mla_q_rope_writeback_batched")?,
            kv_assemble: gpu.kernel("mla_absorbed", "mla_kv_assemble_batched")?,
            reshape_cache: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
            // Ling's absorbed K/V rows are 192-wide (qk_nope 128 + RoPE 64),
            // so resolve the target-local HDIM=192 module rather than the
            // borrowed generic HDIM=256 paged decoder.
            paged_attention: gpu.kernel("paged_decode_ling", "paged_decode_attn_ling")?,
            sigmoid_gate: gpu.kernel("residual_add", "sigmoid_gate_mul_head_broadcast")?,
        })
    }
}

/// Upload the pinned text RoPE frequency table. The returned allocation must
/// stay live for the lifetime of the layer weights.
pub fn alloc_rope_inv_freq(gpu: &dyn GpuBackend, config: LingMlaConfig) -> Result<DevicePtr> {
    config.validate()?;
    let pairs = config.qk_rope_head_dim / 2;
    let mut bytes = Vec::with_capacity(pairs * std::mem::size_of::<f32>());
    for i in 0..pairs {
        let exponent = (2 * i) as f64 / config.qk_rope_head_dim as f64;
        let freq = (config.rope_theta as f64).powf(-exponent) as f32;
        bytes.extend_from_slice(&freq.to_ne_bytes());
    }
    let table = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, table)?;
    Ok(table)
}

/// Apply the pinned Ling body-diagonal MRoPE to one compact MLA token.
/// `positions`, `positions_h` and `positions_w` each point at one u32 metadata
/// word; the kernel reinterprets H/W as signed i32 to preserve centered
/// coordinates from the reference implementation.
fn rope_ling_mrope(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    positions_h: DevicePtr,
    positions_w: DevicePtr,
    q_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    inv_freq: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([q_heads + 1, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_ptr(positions_h)
        .arg_ptr(positions_w)
        .arg_u32(q_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_ptr(inv_freq)
        .arg_u32(8)
        .arg_u32(12)
        .arg_u32(12)
        .launch(stream)
}

impl LingMlaWeights {
    /// Run one causal decode token and optionally append its K/V to the shared
    /// paged cache. `write_kv=false` is used when replaying a prefix row whose
    /// K/V already exists in the cache: the query still attends through the
    /// caller-provided causal metadata, but the replay must not overwrite the
    /// cached slot. The cache layer must be BF16 with dimensions
    /// (num_heads, qk_head_dim).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode(
        &self,
        config: LingMlaConfig,
        kernels: LingMlaKernels,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        layer_idx: usize,
        metadata: AttnMetadataDev,
        input: DevicePtr,
        output: DevicePtr,
        scratch: LingMlaScratch,
        write_kv: bool,
        stream: u64,
    ) -> Result<()> {
        config.validate()?;
        ensure!(
            metadata.num_seqs == 1,
            "Ling MLA decode expects one sequence"
        );
        ensure!(
            kv_cache.dtype_for_layer(layer_idx) == KvCacheDtype::Bf16,
            "Ling MLA native decode currently requires BF16 KV cache"
        );
        ensure!(
            kv_cache.config().dims_for_layer(layer_idx) == (config.num_heads, config.qk_head_dim()),
            "Ling MLA cache geometry must be ({} heads, {}-wide K/V), got {:?}",
            config.num_heads,
            config.qk_head_dim(),
            kv_cache.config().dims_for_layer(layer_idx)
        );
        ensure!(
            !self.rope_inv_freq.is_null(),
            "Ling MLA RoPE frequency table is missing"
        );

        let heads = config.num_heads as u32;
        let nope = config.qk_nope_head_dim as u32;
        let rope = config.qk_rope_head_dim as u32;
        let qk = config.qk_head_dim() as u32;
        let vdim = config.v_head_dim as u32;
        let qdim = config.q_proj_rows() as u32;

        self.q_proj
            .forward(gpu, input, scratch.q_full, 1, scratch.projection, stream)?;
        self.kv_a_proj_with_mqa
            .forward(gpu, input, scratch.kv_a, 1, scratch.projection, stream)?;
        ops::rms_norm(
            gpu,
            kernels.rms_norm,
            scratch.kv_a,
            &self.kv_a_layernorm,
            scratch.kv_norm,
            1,
            config.kv_lora_rank as u32,
            config.rms_eps,
            stream,
        )?;

        // Extract q RoPE channels, rotate q and the shared k RoPE row together,
        // then write q's rotated channels back into q_full.
        ops::mla_q_rope_extract_batched(
            gpu,
            kernels.q_rope_extract,
            scratch.q_full,
            scratch.q_rope,
            1,
            heads,
            qk,
            nope,
            rope,
            qdim,
            stream,
        )?;
        gpu.copy_d2d_async(
            scratch.kv_a.offset(config.kv_lora_rank * 2),
            scratch.k_rope,
            config.qk_rope_head_dim * 2,
            stream,
        )?;
        rope_ling_mrope(
            gpu,
            kernels.ling_mrope,
            scratch.q_rope,
            scratch.k_rope,
            metadata.positions,
            metadata.positions_h,
            metadata.positions_w,
            heads,
            rope,
            rope,
            self.rope_inv_freq,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            gpu,
            kernels.q_rope_writeback,
            scratch.q_rope,
            scratch.q_full,
            1,
            heads,
            qk,
            nope,
            rope,
            qdim,
            stream,
        )?;

        self.kv_b_proj.forward(
            gpu,
            scratch.kv_norm,
            scratch.kv_b,
            1,
            scratch.projection,
            stream,
        )?;
        ops::mla_kv_assemble_batched(
            gpu,
            kernels.kv_assemble,
            scratch.kv_b,
            scratch.k_rope,
            scratch.k_token,
            scratch.v_compact,
            1,
            heads,
            nope,
            vdim,
            rope,
            qk,
            (config.num_heads * (config.qk_nope_head_dim + config.v_head_dim)) as u32,
            stream,
        )?;

        // The stock paged-attention ABI uses one width for K and V. Keep V's
        // true 128 channels and zero-pad each head to the 192-wide cache row.
        let k_row_bytes = config.qk_head_dim() * 2;
        let v_row_bytes = config.v_head_dim * 2;
        gpu.memset_async(scratch.v_padded, 0, config.num_heads * k_row_bytes, stream)?;
        gpu.copy_d2d_2d_async(
            scratch.v_compact,
            v_row_bytes,
            scratch.v_padded,
            k_row_bytes,
            v_row_bytes,
            config.num_heads,
            stream,
        )?;
        if write_kv {
            ops::reshape_and_cache(
                gpu,
                kernels.reshape_cache,
                scratch.k_token,
                scratch.v_padded,
                kv_cache.k_pool_ptr(layer_idx),
                kv_cache.v_pool_ptr(layer_idx),
                metadata.slot,
                1,
                heads,
                qk,
                kv_cache.block_size() as u32,
                heads * qk,
                heads * qk,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
        }

        ops::paged_decode_attn_bf16(
            gpu,
            kernels.paged_attention,
            scratch.q_full,
            kv_cache.k_pool_ptr(layer_idx),
            kv_cache.v_pool_ptr(layer_idx),
            scratch.attn_padded,
            metadata.block_table,
            metadata.seq_len,
            metadata.max_blocks_per_seq,
            1,
            heads,
            heads,
            qk,
            kv_cache.block_size() as u32,
            1.0 / (config.qk_head_dim() as f32).sqrt(),
            qdim,
            0,
            stream,
        )?;

        // Narrow [heads, 192] attention output to the true [heads, 128] V
        // width before applying the head-wise gate and dense projection.
        gpu.copy_d2d_2d_async(
            scratch.attn_padded,
            k_row_bytes,
            scratch.attn_compact,
            v_row_bytes,
            v_row_bytes,
            config.num_heads,
            stream,
        )?;
        self.g_proj
            .forward(gpu, input, scratch.gate, 1, scratch.projection, stream)?;
        ops::sigmoid_gate_mul_head_broadcast(
            gpu,
            kernels.sigmoid_gate,
            scratch.attn_compact,
            scratch.gate,
            scratch.attn_gated,
            heads,
            vdim,
            1,
            stream,
        )?;
        self.dense.forward(
            gpu,
            scratch.attn_gated,
            output,
            1,
            scratch.projection,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::LingMlaConfig;

    #[test]
    fn pinned_geometry_is_direct_headwise_mla() {
        let c = LingMlaConfig::default();
        assert_eq!(c.qk_head_dim(), 192);
        assert_eq!(c.q_proj_rows(), 32 * 192);
        assert_eq!(c.kv_b_rows(), 32 * (128 + 128));
        assert!(c.validate().is_ok());
    }
}
