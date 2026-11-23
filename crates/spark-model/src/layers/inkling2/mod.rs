// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking Machines Inkling 2 Mini: the per-visit `Inkling2Layer` (decode,
//! prefill and decode batches), its forward, kernel launchers, visit
//! schedule, weight handles and shared buffers.

mod attn_ops;
mod batch;
mod calls;
mod forward;
mod gate_ops;
pub(crate) mod glue;
pub(crate) mod layer;
mod moe_ops;
mod moe_prefill_hip;
pub(crate) mod ops;
mod owner_runs;
mod proj;
pub(crate) mod schedule;
pub(crate) mod shared;
pub(crate) mod weights;
