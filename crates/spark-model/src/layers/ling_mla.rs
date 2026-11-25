// SPDX-License-Identifier: AGPL-3.0-only

//! Native causal MLA for Ling 3.2 air.
//!
//! Ling's full-attention layers are the direct (q_lora-free) Bailing MLA:
//! q_proj, kv_a_proj_with_mqa + RMSNorm, kv_b_proj, interleaved RoPE,
//! paged K/V, a head-wise sigmoid gate, and the checkpoint's dense output
//! projection. Each BF16 cache row stores 128 NoPE channels per head and
//! one shared 64-channel RoPE tail. Values share an all-zero padding tail.

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
        // The latent kernels (ling_mla_latent.cu) are built for this geometry.
        ensure!(
            (
                self.num_heads,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim
            ) == (32, 512, 128, 64, 128),
            "Ling MLA latent kernels need 32 heads, rank 512, NoPE 128, RoPE 64 and V 128; got {self:?}"
        );
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
    /// kv_b_proj as BF16 `[heads * 256, 512]`: the absorbed query and value
    /// sides of the latent cache.
    pub kv_b_bf16: DevicePtr,
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
        let kv_b_proj = LingProjection::load(
            gpu,
            store,
            &format!("{prefix}.kv_b_proj"),
            kv_b,
            config.kv_lora_rank,
        )?;
        let kv_b_bf16 = kv_b_proj.bf16_weight(gpu, store, &format!("{prefix}.kv_b_proj"))?;
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
            kv_b_proj,
            kv_b_bf16,
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

/// Maximum rows per MLA projection tile. Decode uses the first row.
pub(super) const MLA_TILE_ROWS: usize = 128;
const MLA_PROMPT_TILE_ROWS: usize = 2048;

/// Width of one cached token: the normalized latent then the rotated RoPE key
/// shared by every head (`ling_mla_latent.cu`). V aliases K in the pool.
pub(crate) const LATENT_ROW: usize = 512 + 64;
/// Rows one absorbed decode launch attends (its split buffers' height).
const DECODE_ROWS: usize = 32;
/// Key splits per decode row (`DEC_SPLITS` in the kernel).
const DECODE_SPLITS: usize = 64;
/// Keys a prompt tile rebuilds from the latent before attending them. The
/// last tile also holds the prompt's own rows, up to MLA_PROMPT_TILE_ROWS.
const PROMPT_KEY_TILE: usize = 8192;
/// Expanded per-token row width the paged prefill kernel reads
/// (`ling_cache_layout.cuh`): 32 x 128 NoPE channels and the RoPE tail.
const EXPANDED_ROW: usize = 32 * 128 + 64;

/// Scratch layout for a projection tile. All pointers are BF16 except
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
    pub attn_padded: DevicePtr,
    pub attn_compact: DevicePtr,
    pub attn_gated: DevicePtr,
    pub gate: DevicePtr,
    pub latent: LingLatentScratch,
}

impl LingMlaScratch {
    /// Borrow the routed-expert output arena while attention is running.
    /// FFN starts after all MLA tiles on the same stream have finished.
    pub(super) fn prompt_tile(
        self,
        config: LingMlaConfig,
        arena: DevicePtr,
        bytes: usize,
        tokens: usize,
    ) -> (Self, usize) {
        let padded = config.num_heads * config.qk_head_dim();
        let compact = config.num_heads * config.v_head_dim;
        let widths = [
            config.q_proj_rows(),
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.kv_lora_rank,
            config.num_heads * config.qk_rope_head_dim,
            config.qk_rope_head_dim,
            padded,
            compact,
            compact,
            config.num_heads,
        ];
        let row_bytes = 2 * widths.iter().sum::<usize>();
        let rows = (bytes / row_bytes).min(MLA_PROMPT_TILE_ROWS).min(tokens) / MLA_TILE_ROWS
            * MLA_TILE_ROWS;
        if rows <= MLA_TILE_ROWS || arena.is_null() {
            return (self, MLA_TILE_ROWS);
        }
        let mut offset = 0;
        let [
            q_full,
            kv_a,
            kv_norm,
            q_rope,
            k_rope,
            attn_padded,
            attn_compact,
            attn_gated,
            gate,
        ] = widths.map(|width| {
            let ptr = arena.offset(offset);
            offset += 2 * rows * width;
            ptr
        });
        (
            Self {
                q_full,
                kv_a,
                kv_norm,
                q_rope,
                k_rope,
                attn_padded,
                attn_compact,
                attn_gated,
                gate,
                ..self
            },
            rows,
        )
    }
}

/// Buffers every Ling MLA layer (and the native predictor's) shares on the
/// one model stream: absorbed decode and prompt-tile expansion.
#[derive(Clone, Copy)]
pub struct LingLatentScratch {
    q_abs: DevicePtr,
    latent_out: DevicePtr,
    part_o: DevicePtr,
    part_ml: DevicePtr,
    gather_latent: DevicePtr,
    gather_rope: DevicePtr,
    expand_kv_b: DevicePtr,
    k_rows: DevicePtr,
    v_rows: DevicePtr,
    identity: DevicePtr,
    tile_o: DevicePtr,
    tile_ml: DevicePtr,
    acc_o: DevicePtr,
    acc_ml: DevicePtr,
}

static LATENT_SCRATCH: std::sync::OnceLock<LingLatentScratch> = std::sync::OnceLock::new();

impl LingLatentScratch {
    /// Allocate once per process; every later caller shares it.
    pub fn shared(gpu: &dyn GpuBackend) -> Result<Self> {
        if let Some(s) = LATENT_SCRATCH.get() {
            return Ok(*s);
        }
        let heads = 32;
        let keys = PROMPT_KEY_TILE + MLA_PROMPT_TILE_ROWS;
        let prompt = MLA_PROMPT_TILE_ROWS;
        let v_rows = gpu.alloc(keys * EXPANDED_ROW * 2)?;
        // The prefill kernel reads V's padding tail; it stays zero.
        gpu.memset(v_rows, 0, keys * EXPANDED_ROW * 2)?;
        let blocks = keys / 16;
        let identity = gpu.alloc(blocks * 4)?;
        let table: Vec<u8> = (0..blocks as i32).flat_map(i32::to_le_bytes).collect();
        gpu.copy_h2d(&table, identity)?;
        let s = Self {
            q_abs: gpu.alloc(MLA_TILE_ROWS * heads * LATENT_ROW * 2)?,
            latent_out: gpu.alloc(MLA_TILE_ROWS * heads * 512 * 2)?,
            part_o: gpu.alloc(DECODE_ROWS * DECODE_SPLITS * heads * 512 * 4)?,
            part_ml: gpu.alloc(DECODE_ROWS * DECODE_SPLITS * heads * 2 * 4)?,
            gather_latent: gpu.alloc(prompt * 512 * 2)?,
            gather_rope: gpu.alloc(prompt * 64 * 2)?,
            expand_kv_b: gpu.alloc(prompt * heads * 256 * 2)?,
            k_rows: gpu.alloc(keys * EXPANDED_ROW * 2)?,
            v_rows,
            identity,
            tile_o: gpu.alloc(prompt * heads * 192 * 4)?,
            tile_ml: gpu.alloc(prompt * heads * 2 * 4)?,
            acc_o: gpu.alloc(prompt * heads * 192 * 4)?,
            acc_ml: gpu.alloc(prompt * heads * 2 * 4)?,
        };
        Ok(*LATENT_SCRATCH.get_or_init(|| s))
    }
}

/// Kernel family used by the native path.
#[derive(Clone, Copy)]
pub struct LingMlaKernels {
    pub rms_norm: KernelHandle,
    /// Ling's body-diagonal MRoPE; unlike Qwen3.6 it maps H/W/T by
    /// contiguous frequency sections after the adjacent-pair reshape.
    pub ling_mrope: KernelHandle,
    pub q_rope_extract: KernelHandle,
    pub q_rope_writeback: KernelHandle,
    pub cache_write: KernelHandle,
    pub q_absorb: KernelHandle,
    pub decode: KernelHandle,
    pub decode_merge: KernelHandle,
    pub v_up: KernelHandle,
    pub gather: KernelHandle,
    pub expand: KernelHandle,
    pub prefill_attention: KernelHandle,
    pub prefill_partial: KernelHandle,
    pub tile_merge: KernelHandle,
    pub sigmoid_gate: KernelHandle,
}

impl LingMlaKernels {
    /// Resolve the BF16 family used by the pinned Ling checkpoint.
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let latent = |name| gpu.kernel("ling_mla_latent", name);
        Ok(Self {
            rms_norm: gpu.kernel("ling_norm", "ling_rms_norm")?,
            ling_mrope: gpu.kernel("ling_mrope", "ling_mrope_rope_bf16")?,
            q_rope_extract: gpu.kernel("mla_absorbed", "mla_q_rope_extract_batched")?,
            q_rope_writeback: gpu.kernel("mla_absorbed", "mla_q_rope_writeback_batched")?,
            cache_write: latent("ling_latent_cache_write")?,
            q_absorb: latent("ling_q_absorb")?,
            decode: latent("ling_latent_decode")?,
            decode_merge: latent("ling_latent_decode_merge")?,
            v_up: latent("ling_v_up")?,
            gather: latent("ling_latent_gather")?,
            expand: latent("ling_latent_expand")?,
            prefill_attention: gpu.kernel("ling_mla_prefill", "ling_mla_prefill_64")?,
            prefill_partial: gpu.kernel("ling_mla_prefill_part", "ling_mla_prefill_part_64")?,
            tile_merge: latent("ling_prompt_tile_merge")?,
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
    rows: u32,
    q_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    inv_freq: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([q_heads + 1, rows, 1])
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
    /// Process a tile of causal queries from one sequence (a prompt tile) or
    /// one query per row (decode, batched decode, verify). Prefix hits
    /// precede the write floor and already occupy their cache slots.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_rows(
        &self,
        config: LingMlaConfig,
        kernels: LingMlaKernels,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        layer_idx: usize,
        metadata: AttnMetadataDev,
        input: DevicePtr,
        output: Option<DevicePtr>,
        scratch: LingMlaScratch,
        rows: u32,
        write_floor: u32,
        prompt_offset: Option<u32>,
        stream: u64,
    ) -> Result<()> {
        config.validate()?;
        ensure!(
            (metadata.num_seqs == 1 || metadata.num_seqs == rows)
                && (1..=MLA_PROMPT_TILE_ROWS as u32).contains(&rows)
                && (prompt_offset.is_some() || rows as usize <= MLA_TILE_ROWS)
                && write_floor <= rows,
            "Ling MLA expects one sequence and a valid causal tile"
        );
        ensure!(
            (output.is_none()
                || (!metadata.block_table.is_null()
                    && metadata.max_blocks_per_seq > 0
                    && (prompt_offset.is_some() || !metadata.seq_len.is_null())))
                && (write_floor == rows || !metadata.slot.is_null()),
            "Ling MLA requires live paged KV metadata from the first prompt token"
        );
        ensure!(
            kv_cache.dtype_for_layer(layer_idx) == KvCacheDtype::Bf16
                && kv_cache.config().dims_for_layer(layer_idx) == (1, LATENT_ROW)
                && kv_cache.k_pool_ptr(layer_idx) == kv_cache.v_pool_ptr(layer_idx),
            "Ling MLA needs a BF16 latent cache of one {LATENT_ROW}-wide row per token, got {:?}",
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

        if output.is_some() {
            self.q_proj
                .forward(gpu, input, scratch.q_full, rows, scratch.projection, stream)?;
        }
        self.kv_a_proj_with_mqa.forward(
            gpu,
            input,
            scratch.kv_a,
            rows,
            scratch.projection,
            stream,
        )?;
        let rank_bytes = config.kv_lora_rank * 2;
        let rope_bytes = config.qk_rope_head_dim * 2;
        let latent = if rows == 1 {
            scratch.kv_a
        } else {
            // kv_a interleaves the latent and RoPE channels in each row.
            gpu.copy_d2d_2d_async(
                scratch.kv_a,
                rank_bytes + rope_bytes,
                scratch.q_rope,
                rank_bytes,
                rank_bytes,
                rows as usize,
                stream,
            )?;
            scratch.q_rope
        };
        ops::rms_norm(
            gpu,
            kernels.rms_norm,
            latent,
            &self.kv_a_layernorm,
            scratch.kv_norm,
            rows,
            config.kv_lora_rank as u32,
            config.rms_eps,
            stream,
        )?;

        // Extract q RoPE channels, rotate q and the shared k RoPE row together,
        // then write q's rotated channels back into q_full.
        if output.is_some() {
            ops::mla_q_rope_extract_batched(
                gpu,
                kernels.q_rope_extract,
                scratch.q_full,
                scratch.q_rope,
                rows,
                heads,
                qk,
                nope,
                rope,
                qdim,
                stream,
            )?;
        }
        gpu.copy_d2d_2d_async(
            scratch.kv_a.offset(rank_bytes),
            rank_bytes + rope_bytes,
            scratch.k_rope,
            rope_bytes,
            rope_bytes,
            rows as usize,
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
            rows,
            if output.is_some() { heads } else { 0 },
            rope,
            rope,
            self.rope_inv_freq,
            stream,
        )?;
        if output.is_some() {
            ops::mla_q_rope_writeback_batched(
                gpu,
                kernels.q_rope_writeback,
                scratch.q_rope,
                scratch.q_full,
                rows,
                heads,
                qk,
                nope,
                rope,
                qdim,
                stream,
            )?;
        }

        let pool = kv_cache.k_pool_ptr(layer_idx);
        if write_floor < rows {
            let skip = write_floor as usize;
            KernelLaunch::new(gpu, kernels.cache_write)
                .grid([rows - write_floor, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(scratch.kv_norm.offset(skip * rank_bytes))
                .arg_ptr(scratch.k_rope.offset(skip * rope_bytes))
                .arg_ptr(pool)
                .arg_ptr(metadata.slot.offset(skip * 8))
                .launch(stream)?;
        }
        let Some(output) = output else {
            return Ok(());
        };
        let scale = 1.0 / (qk as f32).sqrt();
        let block_size = kv_cache.block_size() as u32;
        if let Some(offset) = prompt_offset {
            self.prompt_attention(
                kernels,
                gpu,
                pool,
                metadata.block_table,
                block_size,
                scratch,
                rows,
                offset,
                scale,
                stream,
            )?;
            // Narrow [heads, 192] attention output to the true [heads, 128] V
            // width before applying the head-wise gate and dense projection.
            gpu.copy_d2d_2d_async(
                scratch.attn_padded,
                qk as usize * 2,
                scratch.attn_compact,
                vdim as usize * 2,
                vdim as usize * 2,
                rows as usize * config.num_heads,
                stream,
            )?;
        } else {
            self.absorbed_decode(
                kernels, gpu, pool, metadata, block_size, scratch, rows, scale, stream,
            )?;
        }
        self.g_proj
            .forward(gpu, input, scratch.gate, rows, scratch.projection, stream)?;
        ops::sigmoid_gate_mul_head_broadcast(
            gpu,
            kernels.sigmoid_gate,
            scratch.attn_compact,
            scratch.gate,
            scratch.attn_gated,
            heads,
            vdim,
            rows,
            stream,
        )?;
        self.dense.forward(
            gpu,
            scratch.attn_gated,
            output,
            rows,
            scratch.projection,
            stream,
        )
    }

    /// One query per row against its own sequence: absorb the query into
    /// latent space, attend the latent cache, project the result to values.
    /// Every launch keeps a row's arithmetic independent of the others.
    #[allow(clippy::too_many_arguments)]
    fn absorbed_decode(
        &self,
        kernels: LingMlaKernels,
        gpu: &dyn GpuBackend,
        pool: DevicePtr,
        metadata: AttnMetadataDev,
        block_size: u32,
        scratch: LingMlaScratch,
        rows: u32,
        scale: f32,
        stream: u64,
    ) -> Result<()> {
        let l = scratch.latent;
        KernelLaunch::new(gpu, kernels.q_absorb)
            .grid([32, rows.div_ceil(8), 1])
            .block([512, 1, 1])
            .arg_ptr(scratch.q_full)
            .arg_ptr(self.kv_b_bf16)
            .arg_ptr(l.q_abs)
            .arg_u32(rows)
            .launch(stream)?;
        let stride = if metadata.num_seqs == 1 {
            0
        } else {
            metadata.max_blocks_per_seq
        };
        for first in (0..rows).step_by(DECODE_ROWS) {
            let n = (rows - first).min(DECODE_ROWS as u32);
            let at = first as usize;
            let seq_len = metadata.seq_len.offset(at * 4);
            KernelLaunch::new(gpu, kernels.decode)
                .grid([DECODE_SPLITS as u32, n, 1])
                .block([256, 1, 1])
                .arg_ptr(l.q_abs.offset(at * 32 * LATENT_ROW * 2))
                .arg_ptr(pool)
                .arg_ptr(metadata.block_table.offset(at * stride as usize * 4))
                .arg_ptr(seq_len)
                .arg_u32(stride)
                .arg_u32(block_size)
                .arg_f32(scale)
                .arg_ptr(l.part_o)
                .arg_ptr(l.part_ml)
                .launch(stream)?;
            KernelLaunch::new(gpu, kernels.decode_merge)
                .grid([32, n, 1])
                .block([128, 1, 1])
                .arg_ptr(l.part_o)
                .arg_ptr(l.part_ml)
                .arg_ptr(seq_len)
                .arg_ptr(l.latent_out.offset(at * 32 * 512 * 2))
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, kernels.v_up)
            .grid([32, rows.div_ceil(8), 1])
            .block([256, 1, 1])
            .arg_ptr(l.latent_out)
            .arg_ptr(self.kv_b_bf16)
            .arg_ptr(scratch.attn_compact)
            .arg_u32(rows)
            .launch(stream)
    }

    /// A prompt tile at `offset` attends keys [0, offset + rows). Keys are
    /// rebuilt from the latent cache through kv_b_proj, PROMPT_KEY_TILE at a
    /// time; the last tile starts at or before the prompt tile and so holds
    /// every row's own key. Each earlier tile is entirely visible to every row.
    #[allow(clippy::too_many_arguments)]
    fn prompt_attention(
        &self,
        kernels: LingMlaKernels,
        gpu: &dyn GpuBackend,
        pool: DevicePtr,
        block_table: DevicePtr,
        block_size: u32,
        scratch: LingMlaScratch,
        rows: u32,
        offset: u32,
        scale: f32,
        stream: u64,
    ) -> Result<()> {
        let l = scratch.latent;
        let total = offset as usize + rows as usize;
        let last = offset as usize / PROMPT_KEY_TILE * PROMPT_KEY_TILE;
        let tiles = last / PROMPT_KEY_TILE + 1;
        for tile in 0..tiles {
            let begin = tile * PROMPT_KEY_TILE;
            let end = if tile + 1 == tiles {
                total
            } else {
                begin + PROMPT_KEY_TILE
            };
            for first in (begin..end).step_by(MLA_PROMPT_TILE_ROWS) {
                let n = (end - first).min(MLA_PROMPT_TILE_ROWS) as u32;
                KernelLaunch::new(gpu, kernels.gather)
                    .grid([n, 1, 1])
                    .block([64, 1, 1])
                    .arg_ptr(pool)
                    .arg_ptr(block_table)
                    .arg_u32(block_size)
                    .arg_u32(first as u32)
                    .arg_ptr(l.gather_latent)
                    .arg_ptr(l.gather_rope)
                    .launch(stream)?;
                self.kv_b_proj.forward(
                    gpu,
                    l.gather_latent,
                    l.expand_kv_b,
                    n,
                    scratch.projection,
                    stream,
                )?;
                KernelLaunch::new(gpu, kernels.expand)
                    .grid([n, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(l.expand_kv_b)
                    .arg_ptr(l.gather_rope)
                    .arg_ptr(l.k_rows)
                    .arg_ptr(l.v_rows)
                    .arg_u32((first - begin) as u32)
                    .launch(stream)?;
            }
            let single = tiles == 1;
            let kernel = if single {
                kernels.prefill_attention
            } else {
                kernels.prefill_partial
            };
            let launch = KernelLaunch::new(gpu, kernel)
                .grid([32, rows.div_ceil(64), 1])
                .block([256, 1, 1])
                .arg_ptr(scratch.q_full)
                .arg_ptr(l.k_rows)
                .arg_ptr(l.v_rows)
                .arg_ptr(if single {
                    scratch.attn_padded
                } else {
                    l.tile_o
                })
                .arg_ptr(l.identity)
                .arg_u32(rows)
                .arg_u32((end - begin) as u32)
                .arg_u32(offset - begin as u32)
                .arg_u32(32)
                .arg_u32(32)
                .arg_u32(192)
                .arg_u32(16)
                .arg_u32(0)
                .arg_u32(1)
                .arg_f32(scale);
            if single {
                return launch.launch(stream);
            }
            launch.arg_ptr(l.tile_ml).launch(stream)?;
            KernelLaunch::new(gpu, kernels.tile_merge)
                .grid([32, rows, 1])
                .block([64, 1, 1])
                .arg_ptr(l.tile_o)
                .arg_ptr(l.tile_ml)
                .arg_ptr(l.acc_o)
                .arg_ptr(l.acc_ml)
                .arg_ptr(scratch.attn_padded)
                .arg_u32(u32::from(tile == 0))
                .arg_u32(u32::from(tile + 1 == tiles))
                .launch(stream)?;
        }
        Ok(())
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
