// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking Machines Inkling 2 Mini: the per-visit `Inkling2Layer` (decode),
//! its kernel launchers, visit schedule, weight handles and shared buffers.

pub(crate) mod glue;
pub(crate) mod layer;
pub(crate) mod ops;
pub(crate) mod schedule;
pub(crate) mod shared;
pub(crate) mod weights;
