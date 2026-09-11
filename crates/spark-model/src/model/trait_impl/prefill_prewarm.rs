// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use spark_runtime::graph_runtime::{GraphPayload, GraphSegment};

use super::super::types::TransformerModel;

impl TransformerModel {
    pub(super) fn prewarm_graphs_dispatch(&self) -> Result<usize> {
        let requests = self.graph_runtime.take_prewarm_requests();
        let mut prewarmed = 0;
        for identity in requests {
            let (request_count, token_count, chunk_start) = match &identity.key.payload {
                GraphPayload::Prefill {
                    request_count,
                    token_count,
                    chunk_start,
                    ..
                } => (*request_count, *token_count as usize, *chunk_start as usize),
                _ => {
                    tracing::warn!(
                        phase = %identity.key.phase,
                        "graph prewarm key is not safely recreatable at startup"
                    );
                    continue;
                }
            };
            if request_count != 1 || token_count < 2 || chunk_start != 0 {
                tracing::warn!(
                    request_count,
                    token_count,
                    chunk_start,
                    "graph prewarm key requires live request state; skipping"
                );
                continue;
            }
            let mut seq = self.alloc_sequence_dispatch()?;
            let tokens = vec![self.config.bos_token_id; token_count];
            let stream = self.gpu.default_stream();
            let result = match identity.key.segment {
                GraphSegment::PrefillCompute => self.prefill_dispatch(&tokens, &mut seq, stream),
                GraphSegment::PrefillLayers => {
                    self.prefill_chunk_dispatch(&tokens, &mut seq, 0, token_count, true, stream)
                }
                _ => {
                    tracing::warn!(
                        segment = ?identity.key.segment,
                        "graph prewarm segment is not safely recreatable at startup"
                    );
                    self.free_sequence_dispatch(&mut seq)?;
                    continue;
                }
            };
            let cleanup = self.free_sequence_dispatch(&mut seq);
            result.context("execute CUDA graph prewarm request")?;
            cleanup.context("release CUDA graph prewarm sequence")?;
            if self.graph_runtime.lookup(&identity)?.is_none() {
                anyhow::bail!(
                    "CUDA graph prewarm completed but key {} was not recreated",
                    spark_runtime::graph_runtime::graph_key_hash(&identity.key)
                );
            }
            prewarmed += 1;
        }
        Ok(prewarmed)
    }
}
