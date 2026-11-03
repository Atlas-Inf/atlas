// SPDX-License-Identifier: AGPL-3.0-only

//! Per-step GLM prefill timing (`ForwardContext::profile`): each step syncs
//! the stream and logs the time since the previous one as
//! `GLM {layer} prefill [{label}]`. A no-op unless profiling.

use anyhow::Result;

use crate::layer::ForwardContext;

pub(crate) fn start(ctx: &ForwardContext, stream: u64) -> Result<Option<std::time::Instant>> {
    if !ctx.profile {
        return Ok(None);
    }
    ctx.gpu.synchronize(stream)?;
    Ok(Some(std::time::Instant::now()))
}

pub(crate) fn step(
    ctx: &ForwardContext,
    stream: u64,
    timer: &mut Option<std::time::Instant>,
    layer: &str,
    label: &str,
) -> Result<()> {
    let Some(started) = timer.take() else {
        return Ok(());
    };
    ctx.gpu.synchronize(stream)?;
    tracing::info!(
        "  GLM {layer} prefill [{label}]: {}us",
        started.elapsed().as_micros()
    );
    *timer = Some(std::time::Instant::now());
    Ok(())
}
