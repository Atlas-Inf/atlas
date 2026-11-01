// SPDX-License-Identifier: AGPL-3.0-only

//! Exit when an EP peer process dies (see `spark_comm::peer_lifeline`).
//!
//! Without this, a rank whose peer exited blocks forever in its next NCCL
//! collective while `/health` still reports ready. Each rank watches its
//! bootstrap connections and terminates (exit 74) the moment one closes, so
//! the pair dies together and a supervisor can restart both.
//! `ATLAS_EP_PEER_LIFELINE=0` opts out; the connections stay open either way,
//! so the peer's own lifeline is unaffected.

use std::sync::atomic::{AtomicBool, Ordering};

static PEER_EXIT_EXPECTED: AtomicBool = AtomicBool::new(false);

/// Exit status when an EP rank stops because its pair's state is uncertain.
const EXIT_EP_PAIR_UNCERTAIN: i32 = 74;

/// Stop this rank immediately. The peer may be blocked in a collective, so no
/// destructor, `atexit` handler or device call may run first.
pub(crate) fn terminate() -> ! {
    // SAFETY: `_exit` never returns and touches no Rust state.
    unsafe { libc::_exit(EXIT_EP_PAIR_UNCERTAIN) }
}

#[cfg(feature = "nccl")]
pub(crate) fn watch(backend: &spark_comm::NcclBackend) -> anyhow::Result<()> {
    if std::env::var("ATLAS_EP_PEER_LIFELINE").as_deref() == Ok("0") {
        tracing::warn!("ATLAS_EP_PEER_LIFELINE=0: EP peer death will not stop this rank");
        return Ok(());
    }
    backend.peer_lifeline().watch(on_peer_lost)?;
    tracing::info!("EP peer lifeline armed");
    Ok(())
}

/// Call on both ranks once the worker shutdown command is sent or received:
/// from then on the peer's exit is the shutdown handshake, not a failure.
pub(crate) fn expect_peer_exit() {
    PEER_EXIT_EXPECTED.store(true, Ordering::Release);
}

#[cfg(feature = "nccl")]
fn on_peer_lost(how: String) {
    if PEER_EXIT_EXPECTED.load(Ordering::Acquire) {
        tracing::info!("EP peer exited during shutdown ({how})");
        return;
    }
    let reason = format!("EP peer lost: {how}");
    tracing::error!("{reason}; terminating rank");
    atlas_core::fault::global().latch(reason);
    terminate();
}
