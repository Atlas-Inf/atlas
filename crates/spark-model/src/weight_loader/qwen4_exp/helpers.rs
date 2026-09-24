// SPDX-License-Identifier: AGPL-3.0-only

//! The loader's small private helpers.
//!
//! Split out of `qwen4_exp.rs` for the file-size cap. The seam is a real one:
//! none of these touches the loader's state or its trait impl — they are the
//! three free functions the impl calls, and they are the only free functions
//! in the file.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use crate::weight_map::DenseWeight;

/// A ones-filled `[n]` BF16 norm scale.
///
/// BF16 1.0 is `0x3F80`, so the buffer cannot be produced with `memset`.
pub(super) fn ones_norm(n: usize, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let host: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3Fu8], n).flatten().collect();
    let ptr = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// `model.language_model` for the multimodal layout, `model` otherwise.
pub(super) fn embed_prefix(config: &ModelConfig) -> String {
    if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    }
}

/// The model-level hyper-connection mixer that collapses the residual streams.
pub(super) fn mixer_prefix(config: &ModelConfig) -> String {
    format!("{}.hyper_connection_mixer", embed_prefix(config))
}
