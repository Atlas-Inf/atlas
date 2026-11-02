// SPDX-License-Identifier: AGPL-3.0-only
//! Short-lived launch authority shared by construction tests and resident reads.
//! Private checked launches share the production mathematical/ABI implementation.
use super::{kernels::KernelFamily, native_source::Span};
use crate::weight_map::QuantizedWeight;

/// Private to gate_up_repack's children. No source/store borrow or raw getter
/// escapes to a serving caller. `stream` is the live caller stream, not load time.
pub(super) struct LaunchLease<'a> {
    pub(super) family: KernelFamily<'a>,
    pub(super) tables: [Span; 6],
    pub(super) shared: Option<([QuantizedWeight; 2], [Span; 4])>,
    pub(super) stream: u64,
}
