// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking Machines Inkling 2 Mini layer pieces (work in progress): kernel
//! kernel launchers and the visit schedule now, the per-visit `Inkling2Layer` next.

// These land before the layer that uses them.
#[allow(dead_code)]
pub(crate) mod ops;
#[allow(dead_code)]
pub(crate) mod schedule;
