// SPDX-License-Identifier: AGPL-3.0-only
//! Ling's single appended predictor uses the next token embedding, the
//! normalized target hidden, and an MLA + MoE body with its own attention KV.

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, TransformerLayer};
use crate::layers::dflash_head::SequenceGeneration;
use crate::layers::mtp_meta::pack_mtp_attn_meta;
use crate::layers::{ling_decoder, ops};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::DenseWeight;
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::{WeightDtype, WeightStore};
use std::any::Any;

struct State {
    blocks: Vec<u32>,
    rows: usize,
    capacity_exhausted: bool,
    pair: Option<usize>,
    body: Box<dyn LayerState>,
}
impl ProposerState for State {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
fn state(value: &mut dyn ProposerState) -> Result<&mut State> {
    value
        .as_any_mut()
        .downcast_mut::<State>()
        .context("invalid Ling MTP state")
}

pub(crate) struct LingMtp {
    body: ling_decoder::LingDecoderLayer,
    embed: DenseWeight,
    head: DenseWeight,
    draft_head: Option<(crate::weight_map::QuantizedWeight, KernelHandle)>,
    enorm: DenseWeight,
    hnorm: DenseWeight,
    merge: DenseWeight,
    norm: DenseWeight,
    cache: Mutex<PagedKvCache>,
    meta: DevicePtr,
    max_meta_bytes: usize,
    pair: DevicePtr,
    hidden: DevicePtr,
    residual: DevicePtr,
    logits: DevicePtr,
    inputs: DevicePtr,
    embed_k: KernelHandle,
    norm_k: KernelHandle,
    gemv: KernelHandle,
    argmax: KernelHandle,
}

impl LingMtp {
    pub(crate) fn load(
        store: &WeightStore,
        c: &ModelConfig,
        gpu: &dyn GpuBackend,
        embed: DenseWeight,
        head: DenseWeight,
        max_seq_len: usize,
    ) -> Result<Self> {
        ensure!(
            c.mtp_num_hidden_layers == 1,
            "LingSante Air-VL does not contain a native MTP head"
        );
        let width = c.hidden_size;
        let weight = |name: &str, shape: &[usize]| -> Result<DenseWeight> {
            let t = store.get(&format!(
                "model.layers.{}.{name}.weight",
                c.num_hidden_layers
            ))?;
            ensure!(
                t.dtype == WeightDtype::BF16 && t.shape == shape,
                "unexpected Ling MTP {name} format"
            );
            Ok(DenseWeight { weight: t.ptr })
        };
        let body = ling_decoder::load_predictor_body(store, c, gpu)?;
        let blocks = max_seq_len.div_ceil(16) + 1;
        let max_meta_bytes = 256 + 4 * blocks;
        // The predictor's one MLA layer caches the same latent rows as the
        // target's (`ling_mla::LATENT_ROW`), V aliasing K.
        let cache = PagedKvCache::new_with_v_alias(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: super::ling_mla::LATENT_ROW,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            blocks,
            gpu,
            true,
        )?;
        let tile = super::ling_mla::MLA_TILE_ROWS;
        Ok(Self {
            body,
            embed,
            head,
            draft_head: None,
            enorm: weight("enorm", &[width])?,
            hnorm: weight("hnorm", &[width])?,
            merge: weight("eh_proj", &[width, 2 * width])?,
            norm: weight("final_layernorm", &[width])?,
            cache: Mutex::new(cache),
            meta: gpu.alloc(max_meta_bytes)?,
            max_meta_bytes,
            pair: gpu.alloc(tile * 4 * width)?,
            hidden: gpu.alloc(tile * 2 * width)?,
            residual: gpu.alloc(tile * 2 * width)?,
            logits: gpu.alloc(2 * c.vocab_size)?,
            inputs: gpu.alloc(tile * 16)?,
            embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            norm_k: gpu.kernel("ling_norm", "ling_rms_norm")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    pub(crate) fn set_draft_head(
        &mut self,
        weight: crate::weight_map::QuantizedWeight,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.draft_head = Some((weight, gpu.kernel("w4a16_gemv", "w4a16_gemv")?));
        Ok(())
    }

    /// Prime committed pairs in tiles. The next prediction needs only this layer's
    /// K/V, so historical pairs skip the query projection, attention, MoE and head.
    fn prime(
        &self,
        tokens: &[u32],
        target: DevicePtr,
        first_position: usize,
        s: &mut State,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let tile = super::ling_mla::MLA_TILE_ROWS;
        let mut cache = self.cache.lock();
        let bs = cache.block_size();
        let needed = (s.rows + tokens.len())
            .div_ceil(bs)
            .saturating_sub(s.blocks.len());
        if s.capacity_exhausted || needed > cache.num_free_blocks() {
            s.capacity_exhausted = true;
            return Ok(0);
        }
        for _ in 0..needed {
            s.blocks.push(cache.alloc_block()?);
        }
        for (i, ids) in tokens.chunks(tile).enumerate() {
            let start = i * tile;
            let rows = ids.len();
            ensure!(
                ids.iter().all(|&id| (id as usize) < ctx.config.vocab_size),
                "invalid Ling predictor prompt token"
            );
            let mut data = Vec::with_capacity(16 * rows);
            for id in ids {
                data.extend(id.to_le_bytes());
            }
            for j in 0..rows {
                data.extend(u32::try_from(first_position + start + j)?.to_le_bytes());
            }
            for j in 0..rows {
                let row = s.rows + j;
                data.extend(((s.blocks[row / bs] as usize * bs + row % bs) as u64).to_le_bytes());
            }
            gpu.copy_h2d_async(&data, self.inputs, stream)?;
            ops::batched_embed(
                gpu,
                self.embed_k,
                self.inputs,
                self.embed.weight,
                self.hidden,
                rows as u32,
                h as u32,
                stream,
            )?;
            for (input, weight, offset) in [
                (self.hidden, &self.enorm, 0),
                (target.offset(start * 2 * h), &self.hnorm, 2 * h),
            ] {
                ops::rms_norm(
                    gpu,
                    self.norm_k,
                    input,
                    weight,
                    self.residual,
                    rows as u32,
                    h as u32,
                    ctx.config.rms_norm_eps as f32,
                    stream,
                )?;
                gpu.copy_d2d_2d_async(
                    self.residual,
                    2 * h,
                    self.pair.offset(offset),
                    4 * h,
                    2 * h,
                    rows,
                    stream,
                )?;
            }
            ops::bf16_gemm(
                self.pair,
                self.merge.weight.0,
                self.hidden,
                rows as u32,
                h as u32,
                (2 * h) as u32,
                stream,
            )?;
            let positions = self.inputs.offset(4 * rows);
            let meta = AttnMetadataDev {
                positions,
                positions_h: positions,
                positions_w: positions,
                slot: self.inputs.offset(8 * rows),
                seq_len: DevicePtr::NULL,
                block_table: DevicePtr::NULL,
                max_blocks_per_seq: 0,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            };
            self.body.prime_predictor_kv(
                self.hidden,
                self.residual,
                rows as u32,
                meta,
                &cache,
                ctx,
                stream,
            )?;
            s.rows += rows;
            s.pair = Some(first_position + start + rows - 1);
        }
        Ok(tokens.len())
    }

    fn append(
        &self,
        token: u32,
        target: DevicePtr,
        position: usize,
        s: &mut State,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        ensure!(
            (token as usize) < ctx.config.vocab_size,
            "invalid Ling MTP token"
        );
        let normalize = |x, w, out| {
            ops::rms_norm(
                gpu,
                self.norm_k,
                x,
                w,
                out,
                1,
                h as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )
        };
        normalize(
            self.embed.weight.offset(token as usize * h * 2),
            &self.enorm,
            self.pair,
        )?;
        normalize(target, &self.hnorm, self.pair.offset(h * 2))?;
        ops::dense_gemv(
            gpu,
            self.gemv,
            self.pair,
            &self.merge,
            self.hidden,
            h as u32,
            (2 * h) as u32,
            stream,
        )?;

        let mut cache = self.cache.lock();
        let bs = cache.block_size();
        if s.rows / bs == s.blocks.len() {
            s.blocks.push(cache.alloc_block()?);
        }
        let slot = s.blocks[s.rows / bs] as usize * bs + s.rows % bs;
        let data = pack_mtp_attn_meta(
            position.try_into()?,
            slot.try_into()?,
            (s.rows + 1).try_into()?,
            &s.blocks,
            self.max_meta_bytes,
        )?;
        gpu.copy_h2d_async(&data, self.meta, stream)?;
        let meta = AttnMetadataDev {
            positions: self.meta,
            positions_h: self.meta,
            positions_w: self.meta,
            slot: self.meta.offset(8),
            seq_len: self.meta.offset(16),
            block_table: self.meta.offset(256),
            max_blocks_per_seq: s.blocks.len() as u32,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let context = ForwardContext {
            attn_metadata: Some(meta),
            ssm_batch: None,
            graph_capture: false,
            comm: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            ..*ctx
        };
        self.body.decode(
            self.hidden,
            self.residual,
            s.body.as_mut(),
            &mut cache,
            s.rows,
            &mut s.blocks,
            &mut Vec::new(),
            &mut vec![0],
            &context,
            stream,
        )?;
        s.rows += 1;
        s.pair = Some(position);
        Ok(())
    }
}

impl DraftProposer for LingMtp {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(State {
            blocks: vec![],
            rows: 0,
            capacity_exhausted: false,
            pair: None,
            body: self.body.alloc_state(gpu)?,
        }))
    }
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        value: &mut dyn ProposerState,
        _owner: Option<SequenceGeneration>,
        ctx: &ForwardContext,
        stream: u64,
        _embed: Option<DevicePtr>,
        mask: Option<&[i32]>,
        _stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        ensure!(num_drafts == 1, "Ling has one native draft stage");
        let s = state(value)?;
        {
            let cache = self.cache.lock();
            if s.capacity_exhausted
                || (s.rows == s.blocks.len() * cache.block_size() && cache.num_free_blocks() == 0)
            {
                s.capacity_exhausted = true;
                return Ok(vec![]);
            }
        }
        self.append(
            last_token,
            target_hidden,
            position.saturating_sub(1),
            s,
            ctx,
            stream,
        )?;
        let h = ctx.config.hidden_size as u32;
        let vocab = ctx.config.vocab_size as u32;
        ops::rms_norm(
            ctx.gpu,
            self.norm_k,
            self.hidden,
            &self.norm,
            self.residual,
            1,
            h,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        if let Some((weight, kernel)) = &self.draft_head {
            ops::w4a16_gemv(
                ctx.gpu,
                *kernel,
                self.residual,
                weight,
                self.logits,
                vocab,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.gemv,
                self.residual,
                &self.head,
                self.logits,
                vocab,
                h,
                stream,
            )?;
        }
        let id = if let Some(mask) = mask {
            super::deepseek_v4_mtp::argmax_grammar_masked(
                ctx.gpu,
                self.logits,
                vocab as usize,
                mask,
                position,
            )?
        } else {
            ops::argmax_bf16(
                ctx.gpu,
                self.argmax,
                self.logits,
                self.residual,
                vocab,
                stream,
            )?;
            let mut raw = [0; 4];
            ctx.gpu
                .copy_d2h_on_stream(self.residual, &mut raw, stream)?;
            u32::from_le_bytes(raw)
        };
        Ok(vec![id])
    }
    fn prefill_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        value: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let s = state(value)?;
        if s.rows > 0 {
            return Ok(0);
        }
        self.prime(tokens.get(1..).unwrap_or(&[]), hiddens, 0, s, ctx, stream)
    }
    fn drafter_rows(&self, value: &mut dyn ProposerState) -> usize {
        state(value).map_or(0, |s| s.rows)
    }
    fn last_pair_key(&self, value: &mut dyn ProposerState) -> Option<usize> {
        state(value).ok().and_then(|s| s.pair)
    }
    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        value: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let s = state(value)?;
        ensure!(row_base == s.rows, "Ling MTP catch-up boundary mismatch");
        self.prime(
            tokens.get(1..).unwrap_or(&[]),
            hiddens,
            pos_base.saturating_sub(1),
            s,
            ctx,
            stream,
        )
    }
    fn after_verify(
        &self,
        _accepted: usize,
        _owner: Option<SequenceGeneration>,
        _state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        // One proposal consumed the committed anchor token. Its private KV row
        // remains valid whether the proposed next token was accepted or rejected.
        Ok(())
    }
    fn free_state(
        &self,
        _gpu: &dyn GpuBackend,
        _owner: Option<SequenceGeneration>,
        value: &mut dyn ProposerState,
    ) -> Result<()> {
        let s = state(value)?;
        self.cache.lock().free_blocks(&s.blocks);
        s.blocks.clear();
        s.rows = 0;
        s.capacity_exhausted = false;
        s.pair = None;
        Ok(())
    }
}
