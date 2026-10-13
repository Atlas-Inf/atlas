// SPDX-License-Identifier: AGPL-3.0-only
//! The uneven exchanges, both ranks at once: two threads, each with its own
//! mock GPU, joined by a mailbox pair. The mock does not run kernels, so the
//! staging-side BF16 add is replayed on the host from the staging buffer --
//! what is checked is that the right rows land in the right place.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_comm::CommBackend;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

/// One rank's outgoing payloads, in order.
#[derive(Default)]
struct Wire {
    queues: Mutex<[VecDeque<Vec<u8>>; 2]>,
    cv: Condvar,
}

struct LoopPair<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    wire: Arc<Wire>,
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn f32_of(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// `dst + src` per BF16 element, as `__hadd` rounds it.
fn add_into(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
        let a = f32_of(u16::from_le_bytes([d[0], d[1]]));
        let b = f32_of(u16::from_le_bytes([s[0], s[1]]));
        d.copy_from_slice(&bf16(a + b).to_le_bytes());
    }
}

impl CommBackend for LoopPair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn exchange_async(&self, send: u64, dst: u64, bytes: usize, add: bool, _: u64) -> Result<bool> {
        let mut out = vec![0u8; bytes];
        self.gpu.copy_d2h(DevicePtr(send), &mut out)?;
        let mut q = self.wire.queues.lock().unwrap();
        q[self.rank].push_back(out);
        self.wire.cv.notify_all();
        let got = loop {
            if let Some(p) = q[1 - self.rank].pop_front() {
                break p;
            }
            q = self.wire.cv.wait(q).unwrap();
        };
        drop(q);
        assert_eq!(got.len(), bytes, "both ranks move the same bytes");
        let mut land = got;
        if add {
            let mut cur = vec![0u8; bytes];
            self.gpu.copy_d2h(DevicePtr(dst), &mut cur)?;
            add_into(&mut cur, &land);
            land = cur;
        }
        self.gpu.copy_h2d(&land, DevicePtr(dst))?;
        Ok(true)
    }
    fn supports_exchange_async(&self, _: usize) -> bool {
        true
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected reduce-scatter")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected receive")
    }
}

fn with_ctx<R>(gpu: &MockGpuBackend, comm: &LoopPair, f: impl FnOnce(&ForwardContext) -> R) -> R {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.tp_world_size = 2;
    let buffers = BufferArena::new(&config, 1, 256, 16, 1, gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    f(&ForwardContext {
        ssm_batch: None,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: Some(comm as &dyn spark_comm::CommBackend),
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    })
}

const W: usize = 4;

/// Distinct BF16 bytes per (tensor, row, column).
fn tensor(seed: u32, rows: usize) -> Vec<u8> {
    (0..rows * W)
        .flat_map(|i| bf16((seed as f32) + (i as f32) * 0.37 - 900.0).to_le_bytes())
        .collect()
}

/// Run `op` on both ranks of `total` rows split at `split`; each rank starts
/// from `init(rank)` and returns its buffer afterwards (with any staged add
/// replayed).
fn run_pair(
    total: usize,
    split: usize,
    add: bool,
    init: impl Fn(usize) -> Vec<u8> + Sync,
) -> [Vec<u8>; 2] {
    let wire = Arc::new(Wire::default());
    let run = |rank: usize| {
        let gpu = MockGpuBackend::new();
        let pair = LoopPair {
            gpu: &gpu,
            rank,
            wire: wire.clone(),
        };
        let sp = SpRows::split_at(total, split, rank);
        let start = init(rank);
        let buf = gpu.alloc(start.len()).unwrap();
        gpu.copy_h2d(&start, buf).unwrap();
        with_ctx(&gpu, &pair, |ctx| {
            if add {
                sp.reduce_scatter(buf, W, ctx, 0).unwrap();
            } else {
                sp.all_gather(buf, W, ctx, 0).unwrap();
            }
        });
        let mut out = vec![0u8; start.len()];
        gpu.copy_d2h(buf, &mut out).unwrap();
        let p = plan(sp, add);
        if add && p.recv_n != p.m {
            // The staged peer rows; the mock skipped the add kernel.
            let (stage, _) = STAGING.with(Cell::get);
            let mut staged = vec![0u8; p.recv_n * W * 2];
            gpu.copy_d2h(DevicePtr(stage).offset(p.skip * W * 2), &mut staged)
                .unwrap();
            let r = p.recv0 * W * 2;
            add_into(&mut out[r..r + staged.len()], &staged);
        }
        out
    };
    std::thread::scope(|s| {
        let a = s.spawn(|| run(0));
        let b = s.spawn(|| run(1));
        [a.join().unwrap(), b.join().unwrap()]
    })
}

const SPLITS: [(usize, usize); 4] = [(5000, 2048), (12288, 6144), (13000, 6144), (16046, 8192)];

#[test]
fn uneven_all_gather_rebuilds_the_chunk() {
    for (total, split) in SPLITS {
        let x = tensor(1, total);
        let out = run_pair(total, split, false, |rank| {
            let sp = SpRows::split_at(total, split, rank);
            let mut b = vec![0xABu8; total * W * 2];
            let r = sp.row0 * W * 2..(sp.row0 + sp.rows) * W * 2;
            b[r.clone()].copy_from_slice(&x[r]);
            b
        });
        assert!(out[0] == x && out[1] == x, "all-gather {total}/{split}");
    }
}

#[test]
fn uneven_reduce_scatter_sums_like_the_all_reduce() {
    for (total, split) in SPLITS {
        let p = [tensor(3, total), tensor(77, total)];
        let mut sum = p[0].clone();
        add_into(&mut sum, &p[1]);
        let out = run_pair(total, split, true, |rank| p[rank].clone());
        for (rank, got) in out.iter().enumerate() {
            let sp = SpRows::split_at(total, split, rank);
            let r = sp.row0 * W * 2..(sp.row0 + sp.rows) * W * 2;
            assert!(
                got[r.clone()] == sum[r],
                "reduce-scatter {total}/{split} rank {rank}"
            );
        }
    }
}

/// Both ranks' windows: what each receives is exactly the region it expects.
#[test]
fn windows_carry_the_regions() {
    for (total, split) in SPLITS {
        for add in [true, false] {
            let (a, b) = (
                SpRows::split_at(total, split, 0),
                SpRows::split_at(total, split, 1),
            );
            for (me, peer) in [(a, b), (b, a)] {
                let (pm, pp) = (plan(me, add), plan(peer, add));
                assert_eq!(pm.m, pp.m);
                assert!(pp.send0 + pp.m <= total && pm.send0 + pm.m <= total);
                assert_eq!(pp.send0 + pm.skip, pm.recv0);
                assert!(pm.skip + pm.recv_n <= pm.m);
                if pm.recv_n == pm.m {
                    assert_eq!(pm.skip, 0);
                }
            }
        }
    }
}
