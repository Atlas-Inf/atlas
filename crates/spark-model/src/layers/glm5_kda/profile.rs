// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use crate::layer::ForwardContext;

pub(super) fn start(ctx: &ForwardContext, stream: u64) -> Result<Option<std::time::Instant>> {
    if !ctx.profile {
        return Ok(None);
    }
    ctx.gpu.synchronize(stream)?;
    Ok(Some(std::time::Instant::now()))
}

pub(super) fn step(
    ctx: &ForwardContext,
    stream: u64,
    timer: &mut Option<std::time::Instant>,
    label: &str,
) -> Result<()> {
    let Some(started) = timer.take() else {
        return Ok(());
    };
    ctx.gpu.synchronize(stream)?;
    tracing::info!(
        "  GLM KDA prefill [{label}]: {}us",
        started.elapsed().as_micros()
    );
    *timer = Some(std::time::Instant::now());
    Ok(())
}
