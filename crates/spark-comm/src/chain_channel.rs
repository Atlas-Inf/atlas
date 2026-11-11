// SPDX-License-Identifier: AGPL-3.0-only

//! Which of the RDMA chain's one-shot channels the calling thread's
//! collectives take (`ATLAS_RDMA_ONESHOT_CHANNELS=2`). Each channel is an
//! independent sequence, so two streams may each run collectives at once,
//! each on its own channel, as long as every rank issues each channel's ops
//! in the same order. Channel 0 unless a [`Scope`] says otherwise; a chain
//! with one channel serves every channel's ops on it.

use std::cell::Cell;

thread_local! {
    static CURRENT: Cell<usize> = const { Cell::new(0) };
}

/// The calling thread's channel.
pub fn current() -> usize {
    CURRENT.with(Cell::get)
}

/// Restores the previous channel on drop.
pub struct Scope(usize);

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.0));
    }
}

/// Run this thread's collectives on `channel` until the scope drops.
pub fn enter(channel: usize) -> Scope {
    Scope(CURRENT.with(|c| c.replace(channel)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_nest_and_restore() {
        assert_eq!(current(), 0);
        {
            let _a = enter(1);
            assert_eq!(current(), 1);
            {
                let _b = enter(0);
                assert_eq!(current(), 0);
            }
            assert_eq!(current(), 1);
        }
        assert_eq!(current(), 0);
    }
}
