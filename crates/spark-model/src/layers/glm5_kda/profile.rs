// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use crate::layer::ForwardContext;

pub(super) use crate::layers::prefill_profile::start;

pub(super) fn step(
    ctx: &ForwardContext,
    stream: u64,
    timer: &mut Option<std::time::Instant>,
    label: &str,
) -> Result<()> {
    crate::layers::prefill_profile::step(ctx, stream, timer, "KDA", label)
}
