// SPDX-License-Identifier: AGPL-3.0-only
//! Side-stream helpers of `qwen4exp_sp_pipe` (split for the line cap): pair
//! swaps issued piece by piece ([`SideExchanges`], the QSA list split) and
//! reduce-scatter offers between a caller and the site computing its partial
//! ([`RsOffer`], `ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE`), and the SP scratch
//! preallocation size ([`prealloc_rows`]).

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use std::cell::Cell;

use super::{Side, side};
use crate::layer::ForwardContext;

/// Pair exchanges on the side stream, each issued after everything the main
/// stream enqueued before it ([`SideExchanges::send`]); [`SideExchanges::join`]
/// makes the main stream wait for them. Both ranks must send the same sizes
/// in the same order.
pub struct SideExchanges<'a> {
    ctx: &'a ForwardContext<'a>,
    side: Side,
}

impl<'a> SideExchanges<'a> {
    /// `None` without a communicator.
    pub fn new(ctx: &'a ForwardContext<'a>) -> Result<Option<Self>> {
        if ctx.comm.is_none() {
            return Ok(None);
        }
        Ok(Some(Self {
            ctx,
            side: side(ctx.gpu)?,
        }))
    }

    /// Swap `bytes` at `send` for the peer's, landing at `dst` (a copy).
    pub fn send(&self, send: DevicePtr, dst: DevicePtr, bytes: usize, stream: u64) -> Result<()> {
        let gpu = self.ctx.gpu;
        gpu.record_event(self.side.to_side, stream)?;
        gpu.stream_wait_event(self.side.stream, self.side.to_side)?;
        let comm = self.ctx.comm.expect("checked in new");
        ensure!(
            comm.exchange_async(send.0, dst.0, bytes, false, self.side.stream)?,
            "qwen4exp side exchange refused ({bytes} bytes)"
        );
        Ok(())
    }

    /// Make `stream` wait for every exchange sent.
    pub fn join(self, stream: u64) -> Result<()> {
        let gpu = self.ctx.gpu;
        gpu.record_event(self.side.to_main, self.side.stream)?;
        gpu.stream_wait_event(stream, self.side.to_main)
    }
}

thread_local! {
    /// 1: a caller offered its reduce-scatter to the site producing the
    /// partial ([`RsOffer`]); 2: the site took it.
    static OFFER: Cell<u8> = const { Cell::new(0) };
}

/// A caller about to reduce-scatter a partial lets the site that computes
/// it run `compute_and_reduce_scatter` itself; [`RsOffer::taken`] says
/// whether it did (then the caller must not reduce again). Withdrawn on drop,
/// so a site reached from any other caller never takes one.
pub struct RsOffer(());

impl RsOffer {
    /// Offer the caller's reduce-scatter until the offer drops.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        OFFER.with(|c| c.set(1));
        Self(())
    }
    /// The site ran the reduce-scatter.
    pub fn taken(&self) -> bool {
        OFFER.with(Cell::get) == 2
    }
}

impl Drop for RsOffer {
    fn drop(&mut self) {
        OFFER.with(|c| c.set(0));
    }
}

/// Whether the current caller offered its reduce-scatter.
pub fn rs_offered() -> bool {
    OFFER.with(Cell::get) == 1
}

/// The offered reduce-scatter ran in the site.
pub fn rs_took() {
    OFFER.with(|c| c.set(2));
}

/// `ATLAS_QWEN4EXP_SP_PREALLOC=1`: the SP scratch (this stage, the QSA list
/// slots, the route buffer) is sized for the largest chunk the arena takes on
/// its first use, so a later, longer prompt never grows it mid-pass (a stream
/// drain plus `cuMemFree` / `cuMemAlloc` inside a measured prefill).
pub fn prealloc_rows(ctx: &ForwardContext<'_>) -> usize {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_SP_PREALLOC").as_deref(),
            Ok("1") | Ok("true")
        )
    });
    let row = ctx.config.hidden_size * 2;
    if on && row > 0 {
        ctx.buffers.sizes().hidden_states / row
    } else {
        0
    }
}
