// SPDX-License-Identifier: AGPL-3.0-only

//! A chain rank's links to its neighbours, with completions reaped as they
//! come. The pair proxy waits out each WRITE's completion before it moves on;
//! the chain proxy keeps its own op's sends on both links and the forwards of
//! other ranks' payloads in flight together, so a payload that arrives while
//! another is still being acknowledged goes on at once. An RC send queue
//! completes in posting order and every request here is signaled (a chained
//! data + flag post signals its flag only), so a request is done once the
//! completions reaped reach the count it was posted at: its ticket.

use super::super::cmd_ring::{self, Wire};
use anyhow::{Result, bail};
use atlas_rdma::Verbs;
use std::ffi::c_void;

/// Requests a QP may have unreaped (its completion queue holds 256).
const IN_FLIGHT: u64 = 64;

/// One rail of a link: an RC QP to the neighbour's matching rail.
pub(in crate::nccl_backend::rdma_pair) struct Qp {
    verbs: Verbs,
    lkey: u32,
    rkey: u32,
    posted: u64,
    done: u64,
}

impl Qp {
    pub(in crate::nccl_backend::rdma_pair) fn new(verbs: Verbs, lkey: u32, rkey: u32) -> Self {
        Self {
            verbs,
            lkey,
            rkey,
            posted: 0,
            done: 0,
        }
    }

    /// Reap every completion that is ready.
    pub(in crate::nccl_backend::rdma_pair) fn reap(&mut self) -> Result<()> {
        while self.done < self.posted && self.verbs.try_poll()?.is_some() {
            self.done += 1;
        }
        Ok(())
    }

    /// Whether the request with `ticket` has completed (as of the last reap).
    pub(in crate::nccl_backend::rdma_pair) fn done(&self, ticket: u64) -> bool {
        self.done >= ticket
    }

    /// Reap until the request with `ticket` has completed.
    pub(in crate::nccl_backend::rdma_pair) fn wait(&mut self, ticket: u64) -> Result<()> {
        while self.done < ticket {
            self.verbs.poll()?;
            self.done += 1;
        }
        Ok(())
    }

    /// Room for one more request.
    fn admit(&mut self) -> Result<u64> {
        if self.posted - self.done >= IN_FLIGHT {
            self.wait(self.posted + 1 - IN_FLIGHT)?;
        }
        self.posted += 1;
        Ok(self.posted)
    }

    /// WRITE `len` bytes at host address `local` to the neighbour's `remote`,
    /// then the word at `flag` to `flag_remote`, in one post (same-QP WRITEs
    /// land in order; see [`Verbs::post_write_flag`]). Returns the ticket.
    ///
    /// # Safety
    /// Both local spans lie in this QP's registered region and stay
    /// unmodified until the ticket is done; both remote ones in the
    /// neighbour's.
    pub(in crate::nccl_backend::rdma_pair) unsafe fn write_flag(
        &mut self,
        local: usize,
        remote: u64,
        len: usize,
        flag: usize,
        flag_remote: u64,
    ) -> Result<u64> {
        let ticket = self.admit()?;
        let len = u32::try_from(len)?;
        // SAFETY: the caller's contract.
        unsafe {
            self.verbs.post_write_flag(
                local as *mut c_void,
                self.lkey,
                remote,
                self.rkey,
                len,
                flag as *mut c_void,
                flag_remote,
                ticket,
            )
        }?;
        Ok(ticket)
    }

    /// WRITE `len` bytes at `local` to the neighbour's `remote`; the ticket.
    ///
    /// # Safety
    /// As [`Self::write_flag`].
    pub(in crate::nccl_backend::rdma_pair) unsafe fn write(
        &mut self,
        local: usize,
        remote: u64,
        len: usize,
    ) -> Result<u64> {
        let ticket = self.admit()?;
        let len = u32::try_from(len)?;
        // SAFETY: the caller's contract.
        unsafe {
            self.verbs.post_write(
                local as *mut c_void,
                self.lkey,
                remote,
                self.rkey,
                len,
                ticket,
            )
        }?;
        Ok(ticket)
    }
}

/// The link to one neighbour: its rails and where its region starts.
pub(in crate::nccl_backend::rdma_pair) struct Link {
    pub(in crate::nccl_backend::rdma_pair) qps: Vec<Qp>,
    pub(in crate::nccl_backend::rdma_pair) base: u64,
}

/// A rank's links: to the rank below, and to the one above.
pub(in crate::nccl_backend::rdma_pair) type Links = [Option<Link>; 2];

/// Reap every rail of every link.
pub(in crate::nccl_backend::rdma_pair) fn reap(links: &mut Links) -> Result<()> {
    for link in links.iter_mut().flatten() {
        for qp in &mut link.qps {
            qp.reap()?;
        }
    }
    Ok(())
}

/// The command ring's WRITEs on a chain: the stream goes on toward the ranks
/// above, an acknowledgement back to the one below ([`cmd_ring::upstream`]),
/// each on rail 0 and complete before `write` returns.
pub(in crate::nccl_backend::rdma_pair) struct CmdWire<'a> {
    pub(in crate::nccl_backend::rdma_pair) links: &'a mut Links,
    /// Host address of this rank's region, and the command region's offset
    /// in every rank's.
    pub(in crate::nccl_backend::rdma_pair) host: usize,
    pub(in crate::nccl_backend::rdma_pair) at: usize,
}

impl Wire for CmdWire<'_> {
    fn write(&mut self, src: usize, dst: usize, len: usize) -> Result<()> {
        let side = usize::from(!cmd_ring::upstream(dst));
        let Some(link) = self.links[side].as_mut() else {
            bail!(
                "RDMA chain command ring: no rank {} this one",
                ["below", "above"][side]
            );
        };
        let remote = link.base + (self.at + dst) as u64;
        let qp = &mut link.qps[0];
        // SAFETY: both spans lie in the command regions, inside the
        // registered regions; the ring leaves the local span unmodified
        // until this WRITE completes, just below.
        let ticket = unsafe { qp.write(self.host + self.at + src, remote, len) }?;
        qp.wait(ticket)
    }
}
