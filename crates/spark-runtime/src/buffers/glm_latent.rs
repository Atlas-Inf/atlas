// SPDX-License-Identifier: AGPL-3.0-only

//! `BufferArena` GLM `fp8_g128` latent scratch attach. Split from `buffers.rs`
//! (500-LoC cap).

use super::BufferArena;
use crate::gpu::GpuBackend;
use anyhow::Result;

impl BufferArena {
    /// Allocate the BF16 latent view for an `fp8_g128` GLM cache: `tokens`
    /// latents (a multiple of 16) plus their identity block table.
    pub fn attach_glm_latent_scratch(&mut self, tokens: usize, gpu: &dyn GpuBackend) -> Result<()> {
        anyhow::ensure!(
            self.glm_latent_bf16.is_null() && tokens > 0 && tokens.is_multiple_of(16),
            "GLM latent scratch needs one attach of a positive multiple of 16 tokens"
        );
        let blocks = tokens / 16;
        let table: Vec<u8> = (0..blocks as u32).flat_map(u32::to_le_bytes).collect();
        let latent = gpu.alloc(tokens * 512 * 2)?;
        let identity = match gpu
            .alloc(table.len())
            .and_then(|t| gpu.copy_h2d(&table, t).map(|()| t))
        {
            Ok(t) => t,
            Err(error) => {
                let _ = gpu.free(latent);
                return Err(error);
            }
        };
        self.glm_latent_bf16 = latent;
        self.glm_identity_table = identity;
        self.glm_latent_tokens = tokens;
        Ok(())
    }
}
