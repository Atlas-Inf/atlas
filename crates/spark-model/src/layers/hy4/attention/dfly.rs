// SPDX-License-Identifier: AGPL-3.0-only

use super::AttentionCache;
use crate::layers::hy4::dfly::TargetCache;
use anyhow::{Result, ensure};

impl AttentionCache<'_, '_> {
    /// Borrow the accepted query's indices and writable future latent slots.
    /// The caller must checkpoint before drafting and hold the request lock
    /// until target verification has overwritten or restored these slots.
    pub fn dfly_target(&mut self, stream: u64) -> Result<TargetCache> {
        ensure!(
            !self.poisoned && self.len > 0 && self.stream == Some(stream),
            "DFly target cache poisoned, empty or on another stream"
        );
        ensure!(
            self.layer.index.is_some()
                && !self.layout.circular
                && self.layout.table.iter().enumerate().all(|(i, p)| i == *p),
            "DFly requires contiguous DSA target cache storage"
        );
        Ok(TargetCache {
            latent: self.latent,
            indices: self.scratch.indices,
            len: self.len,
            capacity: self.max_tokens,
        })
    }
}
