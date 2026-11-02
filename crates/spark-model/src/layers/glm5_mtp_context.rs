// SPDX-License-Identifier: AGPL-3.0-only
//! Incremental canonical-target ingestion with one retained hidden row per request.
use super::*;

fn append_end(
    start: usize,
    count: usize,
    retained_end: usize,
    private_rows: usize,
    capacity: usize,
) -> Result<usize> {
    anyhow::ensure!(count > 0, "native MTP cannot append an empty target chunk");
    let end = start
        .checked_add(count)
        .context("native MTP target chunk end overflows")?;
    anyhow::ensure!(
        end <= capacity,
        "native MTP target chunk exceeds context capacity"
    );
    anyhow::ensure!(
        retained_end == start,
        "native MTP target chunk has a gap or overlap"
    );
    anyhow::ensure!(
        private_rows == start.saturating_sub(1),
        "native MTP private context does not match canonical target boundary"
    );
    Ok(end)
}
impl Glm5MtpHead {
    pub(super) fn append_target_context(
        &self,
        tokens: &[u32],
        start: usize,
        norm_hiddens: DevicePtr,
        state: &mut Glm5MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let committed = crate::layers::glm5_mtp_cursor::abandon_draft_tail(
            state.seq_len,
            state.last_num_drafted,
        )?;
        let end = append_end(
            start,
            tokens.len(),
            state.target_end,
            committed,
            self.max_seq_len,
        )?;
        anyhow::ensure!(
            !state.target_hidden.is_null() && !norm_hiddens.is_null(),
            "native MTP target hidden storage is unavailable"
        );
        state.seq_len = committed;
        state.last_num_drafted = 0;
        state.confidence.clear();
        let old_blocks = state.block_table.len();
        let row_bytes = self
            .draft_config
            .hidden_size
            .checked_mul(2)
            .context("native MTP target hidden row size overflows")?;
        let result = (|| -> Result<()> {
            if start > 0 {
                self.project_eh_rows(
                    &tokens[..1],
                    state.target_hidden,
                    prefill::zero_first_embedding_at(self.zero_first_embed, committed),
                    stream,
                    ctx,
                )?;
                self.write_kv_rows(1, committed, start, state, ctx, stream)?;
            }
            if tokens.len() > 1 {
                self.catchup_rows(tokens, norm_hiddens, start, start + 1, state, ctx, stream)?;
            }
            anyhow::ensure!(
                state.seq_len == end - 1,
                "native MTP incremental append wrote wrong row count"
            );
            let offset = (tokens.len() - 1)
                .checked_mul(row_bytes)
                .context("native MTP target hidden offset overflows")?;
            ctx.gpu.copy_d2d_async(
                norm_hiddens.offset(offset),
                state.target_hidden,
                row_bytes,
                stream,
            )?;
            state.target_end = end;
            Ok(())
        })();
        if result.is_err() {
            let _ = ctx.gpu.synchronize(stream);
            let extra = state.block_table.split_off(old_blocks);
            self.cache()?.free_blocks(&extra);
            state.seq_len = committed;
        }
        result
    }
    pub(super) fn retain_target_hidden(
        &self,
        hidden: DevicePtr,
        end: usize,
        state: &mut Glm5MtpProposerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            end > 0 && end <= self.max_seq_len && end >= state.target_end,
            "native MTP retained target position is invalid"
        );
        anyhow::ensure!(
            !hidden.is_null() && !state.target_hidden.is_null(),
            "native MTP retained target hidden is unavailable"
        );
        gpu.copy_d2d_async(
            hidden,
            state.target_hidden,
            self.draft_config.hidden_size * 2,
            stream,
        )?;
        state.target_end = end;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_chunks_keep_only_the_boundary_row() {
        assert_eq!(append_end(0, 128, 0, 0, 1_048_576).unwrap(), 128);
        assert_eq!(append_end(128, 128, 128, 127, 1_048_576).unwrap(), 256);
        assert_eq!(append_end(256, 1, 256, 255, 1_048_576).unwrap(), 257);
        assert_eq!(append_end(0, 1, 0, 0, 1_048_576).unwrap(), 1);
        assert_eq!(append_end(1, 1, 1, 0, 1_048_576).unwrap(), 2);
        assert_eq!(
            append_end(1_048_575, 1, 1_048_575, 1_048_574, 1_048_576).unwrap(),
            1_048_576
        );
    }
    #[test]
    fn append_refuses_gaps_overlaps_stale_private_rows_and_overflow() {
        for args in [
            (128, 1, 127, 127, 1024),
            (128, 1, 129, 127, 1024),
            (128, 1, 128, 128, 1024),
            (128, 0, 128, 127, 1024),
            (128, 1, 128, 127, 128),
            (usize::MAX, 1, usize::MAX, usize::MAX - 1, usize::MAX),
        ] {
            assert!(append_end(args.0, args.1, args.2, args.3, args.4).is_err());
        }
    }
}
