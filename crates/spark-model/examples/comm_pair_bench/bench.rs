// SPDX-License-Identifier: AGPL-3.0-only
//! The checks and timings of `comm_pair_bench`, per rank.

use anyhow::{Result, ensure};
use spark_comm::{CommBackend, NcclBackend};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};
use std::time::{Duration, Instant};

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

/// Small integers: every rank's value and every sum over ranks is exact in
/// BF16.
fn value(rank: usize, seq: u64, i: usize) -> f32 {
    ((rank * 7 + seq as usize * 13 + i * 3) % 61) as f32 - 30.0
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 31)).wrapping_mul(0x7fb5_d329_728e_a185);
    x = (x ^ (x >> 27)).wrapping_mul(0x81da_def4_bc2d_d44d);
    x ^ (x >> 33)
}

/// Rank `rank`'s payload `seq`: exact small integers, or (`rounded`) finite
/// pseudo-random BF16 values whose sums round.
pub fn payload(rank: usize, seq: u64, n: usize, rounded: bool) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            if !rounded {
                return bf16(value(rank, seq, i));
            }
            let x = mix((seq << 34) ^ ((rank as u64) << 32) ^ i as u64);
            let bits = (x as u16 & 0x807f) | ((0x60 + (x >> 16) % 0x40) as u16) << 7;
            bits.to_le_bytes()
        })
        .collect()
}

/// The exact sum of every rank's small-integer payload `seq`.
fn exact_sum(world: usize, seq: u64, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| bf16((0..world).map(|r| value(r, seq, i)).sum()))
        .collect()
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

fn wrong_bytes(got: &[u8], want: &[u8]) -> usize {
    got.iter().zip(want).filter(|(x, y)| x != y).count()
}

pub struct Bench<'a> {
    pub g: &'a dyn GpuBackend,
    pub comm: &'a NcclBackend,
    pub stream: u64,
    pub rank: usize,
    pub world: usize,
    pub seq: u64,
}

impl Bench<'_> {
    fn fetch(&self, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        self.g.synchronize(self.stream)?;
        let mut v = vec![0u8; bytes];
        self.g.copy_d2h(p, &mut v)?;
        Ok(v)
    }

    /// Load the next payload into each buffer; their sequence numbers.
    fn prepare(&mut self, bufs: &[DevicePtr], bytes: usize, rounded: bool) -> Result<Vec<u64>> {
        bufs.iter()
            .map(|&b| {
                self.seq += 1;
                let v = payload(self.rank, self.seq, bytes / 2, rounded);
                self.g.copy_h2d(&v, b)?;
                Ok(self.seq)
            })
            .collect()
    }

    /// Wrong bytes across `bufs` after one exact all-reduce each.
    fn wrong(&self, bufs: &[DevicePtr], seqs: &[u64], bytes: usize) -> Result<usize> {
        let mut bad = 0;
        for (&b, &s) in bufs.iter().zip(seqs) {
            bad += wrong_bytes(&self.fetch(b, bytes)?, &exact_sum(self.world, s, bytes / 2));
        }
        Ok(bad)
    }

    /// Wrong bytes in `dst` against the other rank's payload `seq` (a pair).
    fn wrong_copy(&self, dst: DevicePtr, seq: u64, bytes: usize) -> Result<usize> {
        let want = payload(1 - self.rank, seq, bytes / 2, false);
        Ok(wrong_bytes(&self.fetch(dst, bytes)?, &want))
    }

    /// One capturable all-reduce per buffer in a graph; `None` when
    /// unavailable.
    pub fn capture(&self, bufs: &[DevicePtr], bytes: usize) -> Result<Option<GraphHandle>> {
        if self.comm.capturable_all_reduce_max_bytes() < bytes {
            return Ok(None);
        }
        self.g.begin_capture(self.stream)?;
        for b in bufs {
            if !self.comm.all_reduce_capturable(b.0, bytes, self.stream)? {
                self.g.abort_capture_if_active(self.stream);
                return Ok(None);
            }
        }
        Ok(Some(self.g.end_capture(self.stream)?))
    }

    /// Exact sums: eager all-reduces, then graph replays between eager calls.
    pub fn check(&mut self, bytes: usize, rounds: usize) -> Result<usize> {
        let one = [self.g.alloc(bytes)?];
        let bufs: Vec<DevicePtr> = (0..8).map(|_| self.g.alloc(bytes)).collect::<Result<_>>()?;
        let mut bad = 0;
        let graph = self.capture(&bufs, bytes)?;
        for _ in 0..rounds {
            let s = self.prepare(&one, bytes, false)?;
            self.comm.all_reduce_async(one[0].0, bytes, self.stream)?;
            bad += self.wrong(&one, &s, bytes)?;
            if let Some(h) = graph {
                let s = self.prepare(&bufs, bytes, false)?;
                self.g.launch_graph(h, self.stream)?;
                bad += self.wrong(&bufs, &s, bytes)?;
            }
        }
        if let Some(h) = graph {
            self.g.destroy_graph(h)?;
        }
        Ok(bad)
    }

    /// Rounded sums: ranks whose result hash differs from rank 0's, summed
    /// over `rounds` (every rank learns every hash through an all-gather).
    pub fn identity(&mut self, bytes: usize, rounds: usize) -> Result<usize> {
        let buf = self.g.alloc(bytes)?;
        let (mine, all) = (self.g.alloc(8)?, self.g.alloc(8 * self.world)?);
        let mut differ = 0;
        for _ in 0..rounds {
            self.prepare(&[buf], bytes, true)?;
            self.comm.all_reduce_async(buf.0, bytes, self.stream)?;
            let hash = fnv(&self.fetch(buf, bytes)?);
            self.g.copy_h2d(&hash.to_le_bytes(), mine)?;
            self.comm.all_gather(mine.0, all.0, 8)?;
            let hashes = self.fetch(all, 8 * self.world)?;
            differ += hashes
                .chunks_exact(8)
                .filter(|h| h != &&hashes[..8])
                .count();
        }
        Ok(differ)
    }

    /// A broadcast of `bytes` from every root: wrong bytes.
    pub fn broadcast_check(&mut self, bytes: usize) -> Result<usize> {
        let buf = self.g.alloc(bytes)?;
        let mut bad = 0;
        for root in 0..self.world {
            let s = self.prepare(&[buf], bytes, false)?[0];
            self.comm.broadcast(buf.0, bytes, root)?;
            bad += wrong_bytes(
                &self.fetch(buf, bytes)?,
                &payload(root, s, bytes / 2, false),
            );
        }
        Ok(bad)
    }

    /// A chain's sequence-parallel gather of `part` bytes per rank (rank
    /// `r`'s at part `world - 1 - r`, as `layers::glm_sp`), from every rank's
    /// payload filling its whole buffer: wrong bytes.
    pub fn gather_check(&mut self, part: usize, rounds: usize) -> Result<usize> {
        let (world, bytes) = (self.world, part * self.world);
        let buf = self.g.alloc(bytes)?;
        let mut bad = 0;
        for _ in 0..rounds {
            let s = self.prepare(&[buf], bytes, false)?[0];
            let top = buf.offset((world - 1) * part);
            ensure!(
                self.comm
                    .gather_async(top.0, -(part as i64), part, self.stream)?,
                "gather of {part} B refused"
            );
            let want: Vec<u8> = (0..world)
                .flat_map(|at| {
                    payload(world - 1 - at, s, bytes / 2, false)[at * part..][..part].to_vec()
                })
                .collect();
            bad += wrong_bytes(&self.fetch(buf, bytes)?, &want);
        }
        self.g.free(buf)?;
        Ok(bad)
    }

    /// A pair's every entry point, back to back with one sync per round, per
    /// size: buffers 0-2 are all-reduced, 3 is sent and lands in 4 and 5.
    pub fn routing(&mut self, sizes: [usize; 2], rounds: usize) -> Result<usize> {
        let (comm, st, mut bad) = (self.comm, self.stream, 0);
        let mut sets = Vec::new();
        for bytes in sizes {
            let b: Vec<DevicePtr> = (0..6).map(|_| self.g.alloc(bytes)).collect::<Result<_>>()?;
            sets.push((bytes, b));
        }
        let graph = if comm.capturable_all_reduce_max_bytes() >= sizes[0] {
            let (bytes, b) = &sets[0];
            self.g.begin_capture(st)?;
            ensure!(comm.all_reduce_capturable(b[0].0, *bytes, st)?);
            ensure!(comm.peer_exchange_capturable(b[3].0, b[5].0, *bytes, st)?);
            Some(self.g.end_capture(st)?)
        } else {
            None
        };
        for _ in 0..rounds {
            let mut seqs = Vec::new();
            for (bytes, b) in &sets {
                seqs.push(self.prepare(&b[..4], *bytes, false)?);
            }
            for (bytes, b) in &sets {
                let bytes = *bytes;
                comm.all_reduce(b[0].0, bytes)?;
                comm.all_reduce_async(b[1].0, bytes, st)?;
                if !comm.exchange_async(b[2].0, b[2].0, bytes, true, st)? {
                    comm.all_reduce_async(b[2].0, bytes, st)?;
                }
                if !comm.exchange_async(b[3].0, b[4].0, bytes, false, st)? {
                    comm.peer_exchange_async(b[3].0, b[4].0, bytes, st)?;
                }
                comm.peer_exchange_async(b[3].0, b[5].0, bytes, st)?;
            }
            for ((bytes, b), s) in sets.iter().zip(&seqs) {
                bad += self.wrong(&b[..3], &s[..3], *bytes)?;
                bad += self.wrong_copy(b[4], s[3], *bytes)?;
                bad += self.wrong_copy(b[5], s[3], *bytes)?;
            }
            if let Some(h) = graph {
                let (bytes, b) = &sets[0];
                let s = self.prepare(&[b[0], b[3]], *bytes, false)?;
                self.g.launch_graph(h, st)?;
                bad += self.wrong(&b[..1], &s[..1], *bytes)?;
                bad += self.wrong_copy(b[5], s[1], *bytes)?;
            }
        }
        Ok(bad)
    }

    /// A pair's peer exchange of `bytes`: wrong bytes.
    pub fn peer_exchange_check(&mut self, bytes: usize) -> Result<usize> {
        let (send, recv) = (self.g.alloc(bytes)?, self.g.alloc(bytes)?);
        let mut bad = 0;
        for _ in 0..16 {
            let s = self.prepare(&[send], bytes, false)?[0];
            self.comm
                .peer_exchange_async(send.0, recv.0, bytes, self.stream)?;
            bad += self.wrong_copy(recv, s, bytes)?;
        }
        Ok(bad)
    }

    pub fn barrier(&self) -> Result<()> {
        self.comm.barrier()?;
        self.g.synchronize(self.stream)
    }

    /// Synced per-call latency percentiles (us) of `op`; the last rank first
    /// sleeps `skew_us`, so the others report early ranks and it a late one.
    pub fn latency(
        &self,
        iters: usize,
        skew_us: u64,
        op: &dyn Fn() -> Result<()>,
    ) -> Result<[f64; 3]> {
        self.barrier()?;
        let mut t = Vec::with_capacity(iters);
        for _ in 0..iters {
            if self.rank + 1 == self.world && skew_us > 0 {
                std::thread::sleep(Duration::from_micros(skew_us));
            }
            let t0 = Instant::now();
            op()?;
            self.g.synchronize(self.stream)?;
            t.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        t.sort_by(f64::total_cmp);
        Ok([t[iters / 10], t[iters / 2], t[iters * 9 / 10]])
    }

    /// Mean us per call of `calls` back-to-back calls of `op`.
    pub fn throughput(
        &self,
        iters: usize,
        calls: usize,
        op: &dyn Fn() -> Result<()>,
    ) -> Result<f64> {
        self.barrier()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            op()?;
        }
        self.g.synchronize(self.stream)?;
        Ok(t0.elapsed().as_secs_f64() * 1e6 / (iters * calls) as f64)
    }

    /// One command message the way the engine moves it (`ep_broadcast_u32`,
    /// `ep_broadcast_tokens`, the worker's idle first word): rank 0 sends
    /// `words`, every other rank receives as many. Returns the message.
    fn command(&self, buf: DevicePtr, words: &[u32], idle: bool) -> Result<Vec<u32>> {
        let (g, comm, n) = (self.g, self.comm, words.len());
        if n <= comm.command_words_max() {
            let mut got = words.to_vec();
            if self.rank == 0 {
                g.synchronize(self.stream)?;
                comm.send_command_words(words)?;
            } else {
                comm.recv_command_words(&mut got)?;
                g.synchronize(self.stream)?;
            }
            return Ok(got);
        }
        let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        if self.rank == 0 {
            g.copy_h2d(&bytes, buf)?;
            comm.broadcast(buf.0, 4 * n, 0)?;
            return Ok(words.to_vec());
        }
        if idle {
            comm.receive_idle_command_word(buf.0)?;
        } else {
            comm.broadcast(buf.0, 4 * n, 0)?;
        }
        g.synchronize(self.stream)?;
        g.copy_d2h(buf, &mut bytes)?;
        let word = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        Ok(bytes.chunks_exact(4).map(word).collect())
    }

    /// `steps` verify-step starts: us from rank 0's first word to its first
    /// collective landing (p10/p50/p90), and the wrong messages read.
    pub fn command_steps(
        &mut self,
        steps: usize,
        pause_us: u64,
        bytes: usize,
    ) -> Result<([f64; 3], usize)> {
        let (cmd, ar) = (self.g.alloc(64)?, self.g.alloc(bytes)?);
        let (mut t, mut bad) = (Vec::with_capacity(steps), 0);
        self.barrier()?;
        for step in 0..steps as u32 {
            let k = 2 + step % 7;
            let tokens: Vec<u32> = (0..k).map(|i| step * 31 + i).collect();
            if self.rank == 0 {
                let until = Instant::now() + Duration::from_micros(pause_us);
                while Instant::now() < until {
                    std::hint::spin_loop();
                }
            }
            let t0 = Instant::now();
            let mut got = Vec::new();
            for (i, word) in [step % 4, 0xffff_fff5, k].into_iter().enumerate() {
                got.extend(self.command(cmd, &[word], i == 0)?);
            }
            got.extend(self.command(cmd, &tokens, false)?);
            self.comm.all_reduce_async(ar.0, bytes, self.stream)?;
            self.g.synchronize(self.stream)?;
            t.push(t0.elapsed().as_secs_f64() * 1e6);
            bad += usize::from(got[..3] != [step % 4, 0xffff_fff5, k] || got[3..] != tokens[..]);
        }
        t.sort_by(f64::total_cmp);
        Ok(([t[steps / 10], t[steps / 2], t[steps * 9 / 10]], bad))
    }
}
