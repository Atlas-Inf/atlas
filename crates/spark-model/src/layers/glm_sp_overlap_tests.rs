// SPDX-License-Identifier: AGPL-3.0-only
//! The overlapped seam on both ranks at once (mock GPUs joined by the
//! uneven tests' mailbox pair, which lands each exchange immediately): block
//! by block it must leave every byte the whole reduce-scatter, site and
//! all-gather leave.

use std::sync::Arc;

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::layers::glm_sp_uneven::tests::{LoopPair, Wire, bf16, with_ctx};

const W: usize = 4;

#[test]
fn overlap_switch_parses_strictly() {
    assert_eq!(parse_blocks(None).unwrap(), 0);
    assert_eq!(parse_blocks(Some("0")).unwrap(), 0);
    assert_eq!(parse_blocks(Some("1")).unwrap(), 4);
    assert_eq!(parse_blocks(Some("3")).unwrap(), 3);
    for bad in ["9", "true", "-1"] {
        assert!(parse_blocks(Some(bad)).is_err(), "{bad}");
    }
}

#[test]
fn blocks_tile_the_rows_and_keep_the_seam_form() {
    assert_eq!(
        cut(4096, 4).unwrap(),
        [(0, 1024), (1024, 1024), (2048, 1024), (3072, 1024)]
    );
    assert_eq!(
        cut(2694, 4).unwrap(),
        [(0, 674), (674, 674), (1348, 674), (2022, 672)]
    );
    // A block under the seam's 512 rows refuses the split.
    assert!(cut(2048, 5).is_none());
    assert!(cut(4096, 1).is_none());
}

/// The site: each row block's values replaced by a function of the reduced
/// rows (what the mHC seam does to `normed`), on the host.
fn site_on_host(gpu: &MockGpuBackend, buf: DevicePtr, first: usize, (r0, rows): (usize, usize)) {
    let at = buf.offset((first + r0) * W * 2);
    let mut b = vec![0u8; rows * W * 2];
    gpu.copy_d2h(at, &mut b).unwrap();
    for v in b.chunks_exact_mut(2) {
        let x = f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16);
        v.copy_from_slice(&bf16(x * 0.5 + 3.0).to_le_bytes());
    }
    gpu.copy_h2d(&b, at).unwrap();
}

fn run_pair(total: usize, overlapped: bool) -> [Vec<u8>; 2] {
    let wire = Arc::new(Wire::default());
    let run = |rank: usize| {
        let gpu = MockGpuBackend::new();
        let pair = LoopPair {
            gpu: &gpu,
            rank,
            wire: wire.clone(),
        };
        let sp = SpRows::for_rank(total, rank, 2);
        let start: Vec<u8> = (0..total * W)
            .flat_map(|i| bf16(i as f32 * 0.11 + rank as f32 * 7.0 - 50.0).to_le_bytes())
            .collect();
        let buf = gpu.alloc(start.len()).unwrap();
        let g = [(buf, W)]; // the one tensor gathered
        gpu.copy_h2d(&start, buf).unwrap();
        with_ctx(&gpu, &pair, |ctx| {
            if overlapped {
                let blocks = cut(sp.rows, 4).unwrap();
                pipelined(sp, &blocks, Some(buf), Some((&g, 1)), W, ctx, 0, |b| {
                    site_on_host(&gpu, buf, sp.row0, b);
                    Ok(())
                })
                .unwrap();
            } else {
                sp.reduce_scatter(buf, W, ctx, 0).unwrap();
                site_on_host(&gpu, buf, sp.row0, (0, sp.rows));
                sp.all_gather(buf, W, ctx, 0).unwrap();
            }
        });
        let mut out = vec![0u8; start.len()];
        gpu.copy_d2h(buf, &mut out).unwrap();
        out
    };
    std::thread::scope(|s| {
        let a = s.spawn(|| run(0));
        let b = s.spawn(|| run(1));
        [a.join().unwrap(), b.join().unwrap()]
    })
}

#[test]
fn overlapped_seam_leaves_the_whole_seams_bytes() {
    for total in [4096usize, 8192, 5388] {
        let whole = run_pair(total, false);
        let blocked = run_pair(total, true);
        assert!(whole[0] == whole[1], "{total}: ranks agree");
        assert!(blocked == whole, "{total}: overlapped seam differs");
    }
}

/// Four ranks through a host mailbox, landing at once: the chain's
/// all-reduce ((x0 + x1) + (x2 + x3)), reduce-scatter (that sum of each
/// rank's own segment) and gather. `rs` says whether the reduce-scatter is
/// offered (else SP takes the all-reduce).
struct Chain4<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    rs: bool,
    slots: &'a std::sync::Mutex<Vec<Vec<u8>>>,
    barrier: &'a std::sync::Barrier,
}

impl Chain4<'_> {
    fn read(&self, at: u64, bytes: usize) -> Vec<u8> {
        let mut b = vec![0u8; bytes];
        self.gpu.copy_d2h(DevicePtr(at), &mut b).unwrap();
        b
    }

    /// Post `mine`; every rank's posts, in rank order.
    fn post(&self, mine: Vec<u8>) -> Vec<Vec<u8>> {
        self.slots.lock().unwrap()[self.rank] = mine;
        self.barrier.wait();
        let all = self.slots.lock().unwrap().clone();
        self.barrier.wait();
        all
    }

    fn tree(all: &[Vec<u8>]) -> Vec<u8> {
        let (mut lo, mut hi) = (all[0].clone(), all[2].clone());
        crate::layers::glm_sp_uneven::tests::add_into(&mut lo, &all[1]);
        crate::layers::glm_sp_uneven::tests::add_into(&mut hi, &all[3]);
        crate::layers::glm_sp_uneven::tests::add_into(&mut lo, &hi);
        lo
    }
}

impl spark_comm::CommBackend for Chain4<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        4
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, _: u64) -> anyhow::Result<()> {
        let all = self.post(self.read(ptr, bytes));
        self.gpu.copy_h2d(&Self::tree(&all), DevicePtr(ptr))
    }
    fn supports_reduce_scatter_async(&self, _: usize) -> bool {
        self.rs
    }
    fn reduce_scatter_async(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        _: u64,
    ) -> anyhow::Result<bool> {
        let at = |d: usize| seg0.wrapping_add_signed(stride * d as i64);
        let mine: Vec<u8> = (0..4).flat_map(|d| self.read(at(d), bytes)).collect();
        let all = self.post(mine);
        let own: Vec<Vec<u8>> = all
            .iter()
            .map(|p| p[self.rank * bytes..][..bytes].to_vec())
            .collect();
        self.gpu
            .copy_h2d(&Self::tree(&own), DevicePtr(at(self.rank)))?;
        Ok(true)
    }
    fn supports_gather_async(&self, _: usize) -> bool {
        true
    }
    fn gather_async(&self, base: u64, stride: i64, bytes: usize, _: u64) -> anyhow::Result<bool> {
        let at = |r: usize| base.wrapping_add_signed(stride * r as i64);
        let all = self.post(self.read(at(self.rank), bytes));
        for (r, part) in all.iter().enumerate() {
            self.gpu.copy_h2d(part, DevicePtr(at(r)))?;
        }
        Ok(true)
    }
    fn all_reduce(&self, _: u64, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("unexpected all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("unexpected reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("unexpected broadcast")
    }
    fn barrier(&self) -> anyhow::Result<()> {
        anyhow::bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        anyhow::bail!("unexpected receive")
    }
}

/// How [`run_chain`] runs its seam.
#[derive(Clone, Copy, PartialEq)]
enum Seam {
    Whole,
    Overlapped,
    /// Overlapped, its gathers two blocks at a time.
    Grouped,
    /// Reduced ahead ([`reduce_ahead`], each block's partials written by its
    /// `produce`), then the overlapped rest.
    Ahead,
    /// Reduced ahead, joined ([`join_ahead`]), then the whole site and gather.
    AheadJoined,
}

/// One seam on four ranks: over the all-reduce (`rs` false) or the
/// reduce-scatter, whole or block by block.
fn run_chain(total: usize, rs: bool, seam: Seam) -> Vec<Vec<u8>> {
    let slots = std::sync::Mutex::new(vec![Vec::new(); 4]);
    let barrier = std::sync::Barrier::new(4);
    let run = |rank: usize| {
        let gpu = MockGpuBackend::new();
        let comm = Chain4 {
            gpu: &gpu,
            rank,
            rs,
            slots: &slots,
            barrier: &barrier,
        };
        let sp = SpRows::for_rank(total, rank, 4);
        let start: Vec<u8> = (0..total * W)
            .flat_map(|i| bf16(i as f32 * 0.13 + rank as f32 * 5.0 - 40.0).to_le_bytes())
            .collect();
        let buf = gpu.alloc(start.len()).unwrap();
        let g = [(buf, W)]; // the one tensor gathered
        let ahead = matches!(seam, Seam::Ahead | Seam::AheadJoined);
        if !ahead {
            gpu.copy_h2d(&start, buf).unwrap();
        }
        let row = W * 2;
        // Each block of every part's partials, written just before its reduce.
        let produce = |(r0, rows): (usize, usize)| {
            for part in 0..4 {
                let at = (part * sp.rows + r0) * row;
                gpu.copy_h2d(&start[at..at + rows * row], buf.offset(at))?;
            }
            Ok(())
        };
        crate::layers::glm_sp_uneven::tests::with_comm_ctx(&gpu, &comm, |ctx| {
            let blocks = cut(sp.rows, 4).unwrap();
            let site = |b| {
                site_on_host(&gpu, buf, sp.row0, b);
                Ok(())
            };
            match seam {
                Seam::Overlapped => {
                    pipelined(sp, &blocks, Some(buf), Some((&g, 1)), W, ctx, 0, site).unwrap()
                }
                Seam::Grouped => {
                    pipelined(sp, &blocks, Some(buf), Some((&g, 2)), W, ctx, 0, site).unwrap()
                }
                Seam::Ahead => {
                    reduce_ahead(sp, &blocks, buf, W, ctx, 0, produce).unwrap();
                    assert!(ahead_matches(buf, &blocks));
                    pipelined(sp, &blocks, Some(buf), Some((&g, 1)), W, ctx, 0, site).unwrap();
                    assert!(!ahead_matches(buf, &blocks));
                }
                Seam::AheadJoined | Seam::Whole => {
                    if seam == Seam::AheadJoined {
                        reduce_ahead(sp, &blocks, buf, W, ctx, 0, produce).unwrap();
                        join_ahead(ctx, 0).unwrap();
                        assert!(!ahead_matches(buf, &blocks));
                    } else {
                        sp.reduce_scatter(buf, W, ctx, 0).unwrap();
                    }
                    site_on_host(&gpu, buf, sp.row0, (0, sp.rows));
                    sp.all_gather(buf, W, ctx, 0).unwrap();
                }
            }
        });
        let mut out = vec![0u8; start.len()];
        gpu.copy_d2h(buf, &mut out).unwrap();
        out
    };
    std::thread::scope(|s| {
        let ranks: Vec<_> = (0..4).map(|r| s.spawn(move || run(r))).collect();
        ranks.into_iter().map(|r| r.join().unwrap()).collect()
    })
}

#[test]
fn chain_seams_leave_the_all_reduce_seams_bytes() {
    for total in [8192usize, 10240, 16384] {
        let reduced = run_chain(total, false, Seam::Whole);
        assert!(
            reduced.iter().all(|r| r == &reduced[0]),
            "{total}: ranks agree"
        );
        for seam in [
            Seam::Whole,
            Seam::Overlapped,
            Seam::Grouped,
            Seam::Ahead,
            Seam::AheadJoined,
        ] {
            assert!(
                run_chain(total, true, seam) == reduced,
                "{total}: reduce-scatter seam differs"
            );
        }
    }
}

#[test]
fn a_block_waits_for_every_piece_it_touches() {
    assert_eq!(pieces_of((0, 512), 256), 0..2);
    assert_eq!(pieces_of((512, 512), 256), 2..4);
    // Blocks need not align to pieces.
    assert_eq!(pieces_of((500, 524), 144), 3..8);
    assert_eq!(pieces_of((0, 1152), 144), 0..8);
}
