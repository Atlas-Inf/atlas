// SPDX-License-Identifier: AGPL-3.0-only
//! Narrow factory access to the actual target FFN after all weights load.
use super::*;
impl Glm5KdaLayer {
    pub(crate) fn glm_shared_cache_ffn(&mut self) -> (usize, &mut FfnComponent) {
        (self.layer_idx, &mut self.ffn)
    }
}
