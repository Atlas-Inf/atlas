// SPDX-License-Identifier: AGPL-3.0-only

//! Switchless RDMA: `ATLAS_RDMA_ALLREDUCE=1` with three or four ranks wired
//! in a line, each Spark cabled to its neighbours only (no switch, and a
//! middle host cannot forward RoCE). Every rank connects RC QPs to the rank
//! below on `ATLAS_RDMA_LEFT_RAILS` and to the one above on
//! `ATLAS_RDMA_RIGHT_RAILS` (comma lists of RDMA devices, as
//! `ATLAS_RDMA_RAILS`; each end of a link names its own), and the proxies
//! relay what is not for them, so ranks that share no cable never need a
//! link. `ATLAS_RDMA_TOPOLOGY=ring` closes a four-rank ring: rank 0
//! also connects left to rank 3, and rank 3 right to rank 0. Flood
//! collectives use this edge to reach every peer in at most two hops;
//! specialized reductions keep their exact linear arithmetic schedule.
//! The links bootstrap through rank 0 (`bootstrap::chain`), and all ranks
//! must agree on topology. Omission preserves the legacy chain setting.
//!
//! It carries the pair's graph-capturable one-shot channel, generalized to
//! the chain ([`oneshot`], [`super::oneshot::relay`]): every all-reduce lands
//! the same pairwise sum of all ranks' payloads on every rank, a broadcast
//! lands the root's, and a gather each rank's part in its place. Payloads above the one-shot maximum go as
//! consecutive one-shot ops. With `ATLAS_GLM_CMD_RDMA=1` the command ring
//! runs along the chain: the head writes to rank 1, and each rank with one
//! above passes on what it receives. There is no legacy staged channel.
//!
//! Both require explicit settings: `ATLAS_RDMA_ONESHOT=1`, and
//! `ATLAS_RDMA_PAIR_CHAIN=1` because every hop chains its data and flag on
//! one QP (same-QP WRITEs land in order: a NIC property, opt-in as on the
//! pair).

mod collectives;
pub(in crate::nccl_backend) mod link;

use super::bootstrap::chain::{Head, Rendezvous, rendezvous};
use super::oneshot::{
    self, MAX_RANKS, OneShot, Shape,
    relay::{Relay, RelayFused},
};
use super::proxy::{Idle, chain_requested};
use super::rails::{Rails, connect_rails, pinned, rail_names};
use super::{cmd_ring, cuMemFreeHost, topology::Topology};
use anyhow::{Context, Result, ensure};
use cmd_ring::{CmdProxy, CmdRing};
use link::{CmdWire, Link, Links, Qp};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(in crate::nccl_backend) struct RdmaChain {
    /// Pinned, device-mapped region: the one-shot channel, then the command
    /// ring when on.
    host: *mut u8,
    stop: Arc<AtomicBool>,
    proxy: Option<std::thread::JoinHandle<()>>,
    /// The one-shot channels (`ATLAS_RDMA_ONESHOT_CHANNELS`, 1 or 2): each
    /// its own staging, receive slots, control page and sequence, all over
    /// the same links. The calling thread's [`crate::chain_channel`] picks
    /// the one an op takes.
    oneshot: Vec<OneShot>,
    cmd: Option<CmdRing>,
    rank: usize,
    /// A rank above to pass received command words on to.
    relay_cmd: bool,
}

// SAFETY: as for `RdmaPair`: `host` is a pinned allocation owned here; the
// proxy thread works on it through its own copy of the address.
unsafe impl Send for RdmaChain {}
unsafe impl Sync for RdmaChain {}

/// The largest one-shot maximum two channels run at (`connect`).
const TWO_CHANNEL_MAX: usize = 1 << 20;

/// `bytes` as consecutive `(offset, len)` ops of at most `max` (a multiple
/// of `unit`): as few as fit, of equal length in whole `unit`s but the last,
/// so a payload just over the maximum is not followed by a sliver op.
fn chunks(bytes: usize, max: usize, unit: usize) -> impl Iterator<Item = (usize, usize)> {
    let n = bytes.div_ceil(max.max(1)).max(1);
    let len = bytes.div_ceil(n).next_multiple_of(unit).min(max.max(unit));
    (0..bytes)
        .step_by(len.max(1))
        .map(move |at| (at, len.min(bytes - at)))
}

/// All-reduces of at least this many bytes run as a reduce-scatter and an
/// all-gather on four ranks. Every rank must use the same value (the chain
/// head carries it).
const AR_SPLIT_MIN: usize = 4 << 20;

/// The rail lists of `rank`'s links, below and above (empty where there is
/// no rank), from `var`. Each is required where the link exists.
fn chain_rails(
    rank: usize,
    ranks: usize,
    topology: Topology,
    var: impl Fn(&str) -> Option<String>,
) -> Result<[Vec<String>; 2]> {
    let side = |key: &str, peer: Option<usize>| -> Result<Vec<String>> {
        let Some(peer) = peer else {
            return Ok(Vec::new());
        };
        let names = rail_names(var(key), None);
        ensure!(
            var(key).is_some_and(|v| !v.trim().is_empty()) && !names.is_empty(),
            "RDMA chain: rank {rank} of {ranks} needs {key} (the RDMA devices of its link to rank {})",
            peer
        );
        Ok(names)
    };
    Ok([
        side("ATLAS_RDMA_LEFT_RAILS", topology.neighbor(rank, ranks, 0))?,
        side("ATLAS_RDMA_RIGHT_RAILS", topology.neighbor(rank, ranks, 1))?,
    ])
}

/// `ATLAS_RDMA_ONESHOT_CHANNELS`: one-shot channels the chain opens (1, the
/// default, or 2: a second, independent sequence for a second stream's
/// collectives, `crate::chain_channel`).
fn channels_requested(var: impl Fn(&str) -> Option<String>) -> Result<usize> {
    match var("ATLAS_RDMA_ONESHOT_CHANNELS").as_deref() {
        None | Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some(v) => anyhow::bail!("ATLAS_RDMA_ONESHOT_CHANNELS must be 1 or 2, got {v:?}"),
    }
}

impl RdmaChain {
    /// Bring the chain up for `rank` of `ranks`, bootstrapping at `at`.
    pub(in crate::nccl_backend) fn connect(
        rank: usize,
        ranks: usize,
        at: &Rendezvous,
    ) -> Result<Self> {
        ensure!(
            (3..=MAX_RANKS).contains(&ranks),
            "RDMA chain: {ranks} ranks (supported: 3..={MAX_RANKS})"
        );
        let cfg = oneshot::Config::from_env()
            .context(
                "RDMA chain: collectives ride the one-shot channel only; set ATLAS_RDMA_ONESHOT=1",
            )?
            .for_chain();
        ensure!(
            chain_requested(),
            "RDMA chain: every hop chains data and flag on one QP; set ATLAS_RDMA_PAIR_CHAIN=1"
        );
        let topology_value = std::env::var("ATLAS_RDMA_TOPOLOGY").ok();
        let topology = Topology::parse(topology_value.as_deref(), ranks)?;
        let names = chain_rails(rank, ranks, topology, |k| std::env::var(k).ok())?;
        let gid_override: Option<u32> = std::env::var("ATLAS_RDMA_GID")
            .ok()
            .and_then(|v| v.parse().ok());
        let shape = Shape {
            ranks,
            rank,
            rails: names.each_ref().map(Vec::len),
        };
        // Before the rendezvous: a shape the channel refuses fails this rank's
        // boot, and with it every other rank's, before any QP is up.
        shape.check()?;
        let cmd_on = cmd_ring::requested();
        let channels = channels_requested(|k| std::env::var(k).ok())?;
        // Two channels are proven only at ops of up to eight pieces (a 1 MiB
        // one-shot maximum): above it the two-channel NIC loopback hangs.
        // Refuse before the rendezvous, as every rank does alike.
        ensure!(
            channels == 1 || cfg.max <= TWO_CHANNEL_MAX,
            "ATLAS_RDMA_ONESHOT_CHANNELS=2 needs ATLAS_RDMA_ONESHOT_MAX <= {TWO_CHANNEL_MAX} \
             (ops of at most eight pieces); got {}",
            cfg.max
        );
        let chan_bytes = oneshot::region_bytes(cfg.max, shape.origins()).next_multiple_of(4096);
        let cmd_at = channels * chan_bytes;
        // The fused reduce-scatter's region, after the command ring, at the
        // same offset on every rank (`oneshot::fused`).
        let fused_at =
            (cmd_at + if cmd_on { cmd_ring::REGION_BYTES } else { 0 }).next_multiple_of(4096);
        let fused_cap = if ranks == MAX_RANKS { cfg.fused_seg } else { 0 };
        let bytes = fused_at + oneshot::fused::entries(rank) * fused_cap;
        let (host, dev) = pinned(bytes, "RDMA chain")?;
        let mut sides = [None, None];
        for (side, names) in sides.iter_mut().zip(&names) {
            if !names.is_empty() {
                // No address of this rank on the link is known here (the
                // bootstrap runs on rank 0's route): the lowest GID.
                *side = Some(Rails::open(names, gid_override, None, rank, host, bytes)?);
            }
        }
        let head = Head {
            rank,
            rails: shape.rails,
            ranks,
            // The channel count rides the staging word's second byte: every
            // rank must open the same channels.
            oneshot: {
                let mut w = oneshot::Config::wire(Some(cfg));
                w[2] |= ((channels - 1) as u64) << 8;
                w
            },
            cmd: cmd_ring::wire(cmd_on),
            ar_split: AR_SPLIT_MIN as u64,
            rs_fused: fused_cap as u64,
            topology,
        };
        let idents = sides.each_ref().map(|s| {
            s.as_ref()
                .map_or_else(Vec::new, |r: &Rails| r.ident.clone())
        });
        let mut links: Links = [None, None];
        rendezvous(&head, idents, at, |theirs| {
            for ((side, rails), remote) in links.iter_mut().zip(&mut sides).zip(theirs) {
                let Some(mut rails) = rails.take() else {
                    continue;
                };
                let (base, rkeys) = connect_rails(&mut rails.verbs, &remote)?;
                let qps = rails.verbs.into_iter().zip(rails.lkeys).zip(rkeys);
                *side = Some(Link {
                    qps: qps
                        .map(|((v, lkey), rkey)| Qp::new(v, lkey, rkey))
                        .collect(),
                    base,
                });
            }
            Ok(())
        })?;

        let oneshot = (0..channels)
            .map(|c| {
                let os = OneShot::new(
                    cfg,
                    shape,
                    host as usize + c * chan_bytes,
                    dev + (c * chan_bytes) as u64,
                )?;
                if c == 0 && fused_cap > 0 {
                    os.with_fused(dev + fused_at as u64, fused_cap)
                } else {
                    Ok(os)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let relays: Vec<Relay> = (0..channels)
            .map(|c| {
                let relay = Relay::new(cfg, shape, host as usize + c * chan_bytes, c * chan_bytes)
                    .with_topology(topology);
                if c == 0 && fused_cap > 0 {
                    relay.with_fused(RelayFused {
                        host: host as usize + fused_at,
                        at: fused_at,
                        cap: fused_cap,
                    })
                } else {
                    relay
                }
            })
            .collect();
        let (cmd, cmd_proxy) = if cmd_on {
            let (ring, proxy) = cmd_ring::channel(host as usize + cmd_at);
            (Some(ring), Some(proxy))
        } else {
            (None, None)
        };
        let stop = Arc::new(AtomicBool::new(false));
        let proxy = {
            let stop = stop.clone();
            let host = host as usize;
            std::thread::Builder::new()
                .name("atlas-rdma-chain".into())
                .spawn(move || {
                    super::pin::pin_proxy();
                    if let Err(e) = proxy_loop(links, relays, cmd_proxy, host, cmd_at, &stop) {
                        tracing::error!("RDMA chain proxy failed: {e:#}");
                    }
                })?
        };
        tracing::info!(
            "RDMA {} ready: rank {rank} of {ranks}, rails below {:?} above {:?}, one-shot {cfg:?} x{channels}, command ring {}",
            topology.name(),
            names[0],
            names[1],
            if cmd_on { "on" } else { "off" }
        );
        Ok(Self {
            host: host as *mut u8,
            stop,
            proxy: Some(proxy),
            oneshot,
            cmd,
            rank,
            relay_cmd: rank + 1 < ranks,
        })
    }

    /// The graph-capturable channel the calling thread selects
    /// ([`crate::chain_channel`]; channel 0 unless the chain opened more).
    pub(in crate::nccl_backend) fn oneshot(&self) -> &OneShot {
        &self.oneshot[crate::chain_channel::current().min(self.oneshot.len() - 1)]
    }

    /// The one-shot channels the chain opened.
    pub(in crate::nccl_backend) fn channels(&self) -> &[OneShot] {
        &self.oneshot
    }

    /// Send the head's command words down the chain (rank 0 only: the
    /// stream runs one way).
    pub(in crate::nccl_backend) fn send_command_words(&self, words: &[u32]) -> Result<()> {
        ensure!(
            self.rank == 0,
            "RDMA chain: only rank 0 sends command words (rank {})",
            self.rank
        );
        self.ring()?.send(words)
    }

    /// Receive the head's next command words, passing them on to the rank
    /// above, if any.
    pub(in crate::nccl_backend) fn recv_command_words(&self, words: &mut [u32]) -> Result<()> {
        self.ring()?.recv_relay(words, self.relay_cmd)
    }

    /// The command ring, when `ATLAS_GLM_CMD_RDMA=1`.
    pub(in crate::nccl_backend) fn cmd(&self) -> Option<&CmdRing> {
        self.cmd.as_ref()
    }

    fn ring(&self) -> Result<&CmdRing> {
        self.cmd
            .as_ref()
            .context("RDMA chain: no command ring (ATLAS_GLM_CMD_RDMA)")
    }
}

impl Drop for RdmaChain {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(proxy) = self.proxy.take() {
            let _ = proxy.join();
        }
        unsafe { cuMemFreeHost(self.host as *mut c_void) };
    }
}

/// Run the chain proxy until `stop` (see `super::proxy::proxy_loop`, whose
/// failure handling this shares).
fn proxy_loop(
    mut links: Links,
    mut relays: Vec<Relay>,
    mut cmd: Option<CmdProxy>,
    host: usize,
    cmd_at: usize,
    stop: &AtomicBool,
) -> Result<()> {
    // A panic takes the error path too: every relay is failed, so no rank
    // spins on a broadcast that this proxy will never forward.
    let end = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        serve(&mut links, &mut relays, &mut cmd, host, cmd_at, stop)
    }))
    .unwrap_or_else(|_| Err(anyhow::anyhow!("RDMA chain proxy panicked")));
    if end.is_err() {
        for relay in &relays {
            relay.fail();
        }
    }
    if let Some(cmd) = &cmd {
        cmd.close();
    }
    end
}

fn serve(
    links: &mut Links,
    relays: &mut [Relay],
    cmd: &mut Option<CmdProxy>,
    host: usize,
    cmd_at: usize,
    stop: &AtomicBool,
) -> Result<()> {
    let mut idle = Idle::from_env(true, cmd.is_some());
    loop {
        link::reap(links)?;
        let mut worked = false;
        for relay in relays.iter_mut() {
            worked |= relay.serve(links)?;
        }
        if let Some(cmd) = cmd.as_mut() {
            worked |= cmd.serve(&mut CmdWire {
                links,
                host,
                at: cmd_at,
            })?;
        }
        if worked || relays.iter().any(Relay::busy) {
            idle.reset();
            continue;
        }
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        idle.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(set: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let set: Vec<(String, String)> = set
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| set.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
    }

    #[test]
    fn each_rank_needs_the_rails_of_the_links_it_has() {
        let both = vars(&[
            ("ATLAS_RDMA_LEFT_RAILS", "rocep1s0f0,roceP2p1s0f0"),
            ("ATLAS_RDMA_RIGHT_RAILS", "rocep1s0f1"),
        ]);
        assert_eq!(
            chain_rails(1, 4, Topology::Chain, &both).unwrap(),
            [vec!["rocep1s0f0", "roceP2p1s0f0"], vec!["rocep1s0f1"]]
        );
        // The ends have one link each and ignore the other list.
        assert_eq!(
            chain_rails(0, 4, Topology::Chain, &both).unwrap(),
            [Vec::<String>::new(), vec!["rocep1s0f1".to_owned()]]
        );
        assert_eq!(
            chain_rails(3, 4, Topology::Chain, &both).unwrap()[1],
            Vec::<String>::new()
        );
        let left = vars(&[("ATLAS_RDMA_LEFT_RAILS", "rocep1s0f1")]);
        let said = format!(
            "{:#}",
            chain_rails(2, 4, Topology::Chain, &left).unwrap_err()
        );
        assert!(
            said.contains("rank 2 of 4 needs ATLAS_RDMA_RIGHT_RAILS") && said.contains("rank 3"),
            "{said}"
        );
        assert!(chain_rails(3, 4, Topology::Chain, &left).is_ok());
        let empty = vars(&[("ATLAS_RDMA_RIGHT_RAILS", " ")]);
        assert!(chain_rails(0, 4, Topology::Chain, &empty).is_err());

        // A closed ring requires both sides on every rank, including the
        // new endpoint-to-endpoint link; an absent cable is never ignored.
        assert_eq!(
            chain_rails(0, 4, Topology::Ring, &both).unwrap()[0].len(),
            2
        );
        assert_eq!(
            chain_rails(3, 4, Topology::Ring, &both).unwrap()[1].len(),
            1
        );
        assert!(chain_rails(3, 4, Topology::Ring, &left).is_err());
        assert!(chain_rails(0, 4, Topology::Ring, &empty).is_err());
    }
}
