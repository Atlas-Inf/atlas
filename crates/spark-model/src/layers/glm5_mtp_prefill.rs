// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt and accepted-pair KV construction for the native GLM MTP head.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::ops;
use super::{Glm5MtpHead, Glm5MtpProposerState};
use crate::layer::ForwardContext;
use crate::layers::glm5_mtp_cursor::required_blocks;
use crate::layers::mtp_meta::MTP_META_OFFSET;

const PREFILL_ROWS: usize = 128;

/// Opt-in: zero the MTP position-0 embedding row before `enorm`, matching the
/// reference fused EH norm. Strictly `0`/`1`; unset defaults off.
pub(super) const ENV_ZERO_FIRST_EMBED: &str = "ATLAS_GLM_NATIVE_MTP_ZERO_FIRST_EMBED";

/// Parse [`ENV_ZERO_FIRST_EMBED`]. Unset or `0` disables, `1` enables; any
/// other value (including non-UTF-8) is a hard error.
pub(super) fn parse_zero_first_embedding_env() -> Result<bool> {
    parse_zero_first_embedding(std::env::var_os(ENV_ZERO_FIRST_EMBED).as_deref())
}

fn parse_zero_first_embedding(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => anyhow::bail!("{ENV_ZERO_FIRST_EMBED} must be unset, 0, or 1"),
    }
}

pub(super) fn zero_first_embedding_at(enabled: bool, base: usize) -> bool {
    enabled && base == 0
}

impl Glm5MtpHead {
    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Glm5MtpProposerState> {
        let body_state = self.module.body.alloc_state(gpu)?;
        Ok(Glm5MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state,
            confidence: Default::default(),
            target_hidden: gpu.alloc(self.draft_config.hidden_size * 2)?,
            target_end: 0,
        })
    }

    pub(super) fn draft_context<'a>(&'a self, ctx: &'a ForwardContext) -> ForwardContext<'a> {
        ForwardContext {
            ssm_batch: None,
            buffers: &self.arena,
            gpu: ctx.gpu,
            config: &self.draft_config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        }
    }

    pub(super) fn project_eh_rows(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        zero_first_embedding: bool,
        stream: u64,
        ctx: &ForwardContext,
    ) -> Result<usize> {
        let rows = tokens.len();
        anyhow::ensure!(rows > 0 && rows <= PREFILL_ROWS);
        let h = self.draft_config.hidden_size;
        let row_bytes = h
            .checked_mul(2)
            .context("GLM-5 MTP: hidden row byte size overflows")?;
        let h_u32 = u32::try_from(h).context("GLM-5 MTP: hidden size exceeds u32")?;
        let eps = self.draft_config.rms_norm_eps as f32;
        let embed = self.arena.ssm_deinterleaved();
        for (row, &token) in tokens.iter().enumerate() {
            let token = usize::try_from(token).context("GLM-5 MTP: token id conversion")?;
            anyhow::ensure!(
                token < self.draft_config.vocab_size,
                "GLM-5 MTP: token {token} exceeds full embedding vocabulary {}",
                self.draft_config.vocab_size
            );
            ctx.gpu.copy_d2d_async(
                self.embed_tokens.weight.offset(token * row_bytes),
                embed.offset(row * row_bytes),
                row_bytes,
                stream,
            )?;
        }
        if zero_first_embedding {
            ctx.gpu.memset_async(embed, 0, row_bytes, stream)?;
        }
        let norm_embed = self.arena.attn_output();
        let norm_hidden = self.arena.residual();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed,
            &self.module.enorm,
            norm_embed,
            rows as u32,
            h_u32,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hiddens,
            &self.module.hnorm,
            norm_hidden,
            rows as u32,
            h_u32,
            eps,
            stream,
        )?;
        let concat = self.arena.ssm_qkvz();
        for row in 0..rows {
            ops::bf16_concat(
                ctx.gpu,
                self.bf16_concat_k,
                norm_embed.offset(row * row_bytes),
                norm_hidden.offset(row * row_bytes),
                concat.offset(row * 2 * row_bytes),
                h_u32,
                stream,
            )?;
        }
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            concat,
            &self.module.eh_proj,
            self.arena.hidden_states(),
            rows as u32,
            h_u32,
            (2 * h)
                .try_into()
                .context("GLM-5 MTP: combined hidden size exceeds u32")?,
            stream,
        )?;
        Ok(rows)
    }

    pub(super) fn write_kv_rows(
        &self,
        rows: usize,
        row_base: usize,
        _pos_base: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut cache = self.cache()?;
        let block_size = cache.block_size();
        let final_row = row_base
            .checked_add(rows)
            .context("GLM-5 MTP: prefill row count overflows")?;
        anyhow::ensure!(
            final_row <= self.max_seq_len,
            "native MTP private append exceeds context limit"
        );
        let needed = required_blocks(final_row.saturating_sub(1), block_size)?;
        while state.block_table.len() < needed {
            state.block_table.push(cache.alloc_block()?);
        }
        let mut slots = Vec::with_capacity(rows);
        for row in 0..rows {
            let logical = row_base + row;
            let block = state.block_table[logical / block_size];
            slots.push((block as i64) * (block_size as i64) + (logical % block_size) as i64);
        }
        let slot_bytes =
            unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, slots.len() * 8) };
        anyhow::ensure!(
            MTP_META_OFFSET + slot_bytes.len() <= self.arena.scratch_bytes(),
            "GLM-5 MTP: primer slot metadata exceeds private scratch"
        );
        let slots_dev = self.arena.scratch().offset(MTP_META_OFFSET);
        ctx.gpu.copy_h2d_async(slot_bytes, slots_dev, stream)?;
        let mtp_ctx = self.draft_context(ctx);
        anyhow::ensure!(
            self.module.body.prefill_mla_kv_only(
                self.arena.hidden_states(),
                rows,
                &mut cache,
                slots_dev,
                &mtp_ctx,
                stream,
            )?,
            "GLM-5 MTP: body does not support zero-RoPE MLA KV-only writes"
        );
        // `slots` backs an asynchronous H2D copy and must remain alive until
        // the writer has consumed it. The eager sync also makes a failed
        // primer fail closed before its temporary vectors are dropped.
        ctx.gpu.synchronize(stream)?;
        state.seq_len = final_row;
        Ok(())
    }

    pub(super) fn prefill_rows(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        anyhow::ensure!(
            state.seq_len == 0,
            "GLM-5 MTP: prompt primer requires a fresh state"
        );
        anyhow::ensure!(prompt_tokens.len() >= 2);
        let rows_total = prompt_tokens.len() - 1;
        let mut done = 0;
        let result = (|| -> Result<()> {
            while done < rows_total {
                let rows = (rows_total - done).min(PREFILL_ROWS);
                // Pair t[k+1] with hidden[k] at private KV row k (NoPE).
                let token_slice = &prompt_tokens[done + 1..done + rows + 1];
                self.project_eh_rows(
                    token_slice,
                    hiddens.offset(done * self.draft_config.hidden_size * 2),
                    zero_first_embedding_at(self.zero_first_embed, done),
                    stream,
                    ctx,
                )?;
                self.write_kv_rows(rows, done, done + 1, state, ctx, stream)?;
                done += rows;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = ctx.gpu.synchronize(stream);
            let mut cache = self.cache()?;
            if !state.block_table.is_empty() {
                cache.free_blocks(&state.block_table);
            }
            state.block_table.clear();
            state.seq_len = 0;
        }
        result?;
        Ok(rows_total)
    }

    /// Native C1 catch-up: append committed rows `row_base .. row_base + len - 1`
    /// to the private MTP KV cache from a `(tokens, hiddens)` window whose
    /// FIRST token is a dummy (as in `prefill_rows`). Row `k` pairs token
    /// `tokens[k+1]` with hidden row `k`; RoPE position is `pos_base + k`.
    ///
    /// Transactional: on any error, synchronize, free ONLY the blocks this
    /// call freshly allocated, restore `state.seq_len` (rows written beyond
    /// the old length stay invisible because the slot mapping is rebuilt from
    /// `seq_len` on the next call), and return the error.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn catchup_rows(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        anyhow::ensure!(
            tokens.len() >= 2,
            "GLM-5 MTP catch-up needs at least two tokens"
        );
        anyhow::ensure!(
            row_base == state.seq_len,
            "GLM-5 MTP catch-up row_base {} must equal seq_len {}",
            row_base,
            state.seq_len,
        );
        anyhow::ensure!(
            row_base.checked_add(1) == Some(pos_base),
            "GLM-5 MTP catch-up pos_base {} must equal row_base + 1 ({})",
            pos_base,
            row_base.saturating_add(1),
        );
        let rows_total = tokens.len() - 1;
        anyhow::ensure!(
            row_base
                .checked_add(rows_total)
                .is_some_and(|end| end <= self.max_seq_len),
            "GLM-5 MTP catch-up exceeds max_seq_len"
        );
        let row_bytes = self
            .draft_config
            .hidden_size
            .checked_mul(2)
            .context("GLM-5 MTP catch-up row size overflow")?;
        rows_total
            .checked_mul(row_bytes)
            .context("GLM-5 MTP catch-up byte span overflow")?;
        let old_len = state.seq_len;
        let old_blocks = state.block_table.len();
        let mut done = 0;
        let result = (|| -> Result<()> {
            while done < rows_total {
                let rows = (rows_total - done).min(PREFILL_ROWS);
                let token_slice = &tokens[done + 1..done + rows + 1];
                self.project_eh_rows(
                    token_slice,
                    hiddens.offset(done * row_bytes),
                    zero_first_embedding_at(self.zero_first_embed, row_base + done),
                    stream,
                    ctx,
                )?;
                self.write_kv_rows(rows, row_base + done, pos_base + done, state, ctx, stream)?;
                done += rows;
            }
            Ok(())
        })();
        if result.is_err() {
            // Fail closed: quiesce the writer, then drop only newly allocated
            // blocks. `write_kv_rows` sets `state.seq_len` only on success.
            let _ = ctx.gpu.synchronize(stream);
            let mut cache = self.cache()?;
            if state.block_table.len() > old_blocks {
                let extra: Vec<u32> = state.block_table.split_off(old_blocks);
                cache.free_blocks(&extra);
            }
            state.seq_len = old_len;
        }
        result?;
        Ok(rows_total)
    }

    pub(super) fn repair_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(state.last_num_drafted == 0);
        anyhow::ensure!(
            state.seq_len.checked_add(1) == Some(position),
            "GLM-5 MTP repair position must follow its committed cache"
        );
        // Repair rows are always positions > 0: never masked.
        self.project_eh_rows(
            std::slice::from_ref(&token),
            target_hidden,
            false,
            stream,
            ctx,
        )?;
        self.write_kv_rows(1, state.seq_len, position, state, ctx, stream)
    }
}

#[cfg(test)]
mod zero_first_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn zero_first_selector_is_strict() {
        assert!(!parse_zero_first_embedding(None).unwrap());
        assert!(!parse_zero_first_embedding(Some(OsStr::new("0"))).unwrap());
        assert!(parse_zero_first_embedding(Some(OsStr::new("1"))).unwrap());
        for value in ["", "true", "01", " 1", "2"] {
            assert!(parse_zero_first_embedding(Some(OsStr::new(value))).is_err());
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let value = std::ffi::OsString::from_vec(vec![255]);
            assert!(parse_zero_first_embedding(Some(&value)).is_err());
        }
    }

    #[test]
    fn zero_first_mask_does_not_repeat_at_chunk_boundaries() {
        assert!(zero_first_embedding_at(true, 0));
        for base in [1, 127, 128, 129, 256, usize::MAX] {
            assert!(!zero_first_embedding_at(true, base));
        }
        assert!(!zero_first_embedding_at(false, 0));
    }
}
