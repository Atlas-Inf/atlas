// SPDX-License-Identifier: AGPL-3.0-only

//! The pair proxy's side of the one-shot channel: it sends each staged op to
//! the peer and clears `stage` once the WRITEs completed.

use super::{
    Config, PAIR_FLAG_SRC, STAGE, ctrl_off, fail_at, flag_line, flag_word, recv_off, stripes,
};
use anyhow::{Result, ensure};
use atlas_rdma::Verbs;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

/// Proxy side: sends each staged op.
pub(in crate::nccl_backend::rdma_pair) struct Channel {
    cfg: Config,
    host: usize,
    /// The peer's address of its one-shot region.
    peer: u64,
    served: u64,
}

impl Channel {
    pub(in crate::nccl_backend::rdma_pair) fn new(cfg: Config, host: usize, peer: u64) -> Self {
        Self {
            cfg,
            host,
            peer,
            served: 0,
        }
    }

    /// The proxy is going away and nothing will clear `stage` again: poison
    /// the channel, then release a stream blocked in the stage wait so it
    /// reaches a kernel, which traps on the poison.
    pub(in crate::nccl_backend::rdma_pair) fn fail(&self) {
        fail_at(self.host + ctrl_off(self.cfg.max, 1), self.served);
    }

    /// Send the staged op, if any: data then flag per rail, chained in one
    /// post when `chain` (same-QP WRITEs land in order without relaxed
    /// ordering), else each flag after the data completions. `false` when
    /// nothing is staged.
    pub(in crate::nccl_backend::rdma_pair) fn serve(
        &mut self,
        rails: &mut [Verbs],
        lkeys: &[u32],
        rkeys: &[u32],
        chain: bool,
    ) -> Result<bool> {
        let ctrl = self.host + ctrl_off(self.cfg.max, 1);
        // SAFETY: the control page lives in the pinned region for the pair's
        // lifetime; `stage` is written by the GPU (stream memop) and here.
        let stage = unsafe { &*((ctrl + STAGE) as *const AtomicU32) };
        let bytes = stage.load(Ordering::Acquire) as usize;
        if bytes == 0 {
            return Ok(false);
        }
        // Only our own stream writes `stage`, with an eligible size; refuse
        // anything else rather than WRITE past the peer's receive slot.
        ensure!(
            self.cfg.eligible(bytes),
            "RDMA one-shot: staged size {bytes} is not eligible"
        );
        let seq = self.served + 1;
        let parts = stripes(bytes, rails.len(), self.cfg.stripe_min);
        let recv = self.peer + recv_off(self.cfg.max, 0, (seq & 1) as usize) as u64;
        let flag = |r: usize| {
            (
                ctrl + PAIR_FLAG_SRC + 64 * r,
                self.peer + (ctrl_off(self.cfg.max, 1) + flag_line(0, r)) as u64,
            )
        };
        for (r, &(off, len)) in parts.iter().enumerate() {
            let (src_flag, dst_flag) = flag(r);
            // SAFETY: our flag source word; its previous WRITE was reaped
            // before the previous serve returned.
            unsafe { (src_flag as *mut u64).write_volatile(flag_word(seq, bytes)) };
            let data = (self.host + off) as *mut c_void;
            let (dst, len) = (recv + off as u64, u32::try_from(len)?);
            // SAFETY: `S` and the flag word lie in the registered region and
            // stay unmodified until these WRITEs are reaped below (the GPU
            // only restages after `stage` is cleared).
            unsafe {
                if chain {
                    rails[r].post_write_flag(
                        data,
                        lkeys[r],
                        dst,
                        rkeys[r],
                        len,
                        src_flag as *mut c_void,
                        dst_flag,
                        seq,
                    )
                } else {
                    rails[r].post_write(data, lkeys[r], dst, rkeys[r], len, seq)
                }
            }?;
        }
        if !chain {
            for rail in rails.iter_mut().take(parts.len()) {
                rail.poll()?;
            }
            for (r, rail) in rails.iter_mut().enumerate().take(parts.len()) {
                let (src_flag, dst_flag) = flag(r);
                // SAFETY: as above.
                unsafe {
                    rail.post_write(
                        src_flag as *mut c_void,
                        lkeys[r],
                        dst_flag,
                        rkeys[r],
                        8,
                        seq,
                    )
                }?;
            }
        }
        for rail in rails.iter_mut().take(parts.len()) {
            rail.poll()?;
        }
        self.served = seq;
        stage.store(0, Ordering::Release);
        Ok(true)
    }
}
