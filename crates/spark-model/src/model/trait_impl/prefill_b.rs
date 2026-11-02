// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_chunk_dispatch` orchestrator.
//!
//! Refactor wave-4e split a 1000-LoC monolith into Pattern-B phase fns
//! (siblings under `prefill_b/`). The MutexGuard on `kv_cache` is
//! acquired here once and threaded through each phase as `&mut`.
//!
//! Phases (by section comment in original):
//!   1+1b → embed_chunk     (token embed + vision-pad overlay)
//!   2    → prefix_lookup   (prefix-cache hit + EP-sync + Marconi)
//!   2b   → proc_range      (recompute proc_start/count after skip; may early-return)
//!   3    → upload_meta     (positions + MRoPE + slots staging upload)
//!   3b   → upload_paged    (paged-prefill block_table + seq_len upload)
//!   4    → forward_layers  (per-layer prefill/decode + diagnostics)
//!   5-8  → finalize_last   (final norm + lm_head + snapshot save) — last chunk
//!   9    → save_intermediate_checkpoint — non-last chunk
#![allow(unused_imports, dead_code, clippy::too_many_arguments)]
use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
pub(in crate::model) mod embed_chunk;
mod exact_direct;
mod finalize_last;
mod forward_layers;
mod h_state_ptrs;
mod midchunk_capture;
mod multi;
mod multi_ep;
mod multi_head;
pub use multi_ep::{multi_cached, multi_requested};
#[path = "prefill_b/dispatch.rs"]
mod dispatch;
pub(in crate::model) mod pc_inflight;
pub(in crate::model) mod pc_policy;
mod prefix_lookup;
mod proc_range;
mod prompt_logprobs;
mod qwen4exp_ckpt;
pub(in crate::model) mod qwen4exp_points;
mod save_checkpoint;
mod stage_batched;
mod upload_meta;
mod upload_paged;
mod warm;

#[path = "prefill_b/dispatch_with.rs"]
mod dispatch_with;
pub(in crate::model) use dispatch_with::*;
