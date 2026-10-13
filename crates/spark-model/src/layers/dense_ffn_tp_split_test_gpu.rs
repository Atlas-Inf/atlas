// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build, `ATLAS_W4A16_TC=1`): `ATLAS_GLM_DENSE_TP_SPLIT`'s two ranks,
//! as two threads on one GPU over a host-mailbox pair exchange, against the
//! unsplit canonical FFN (`glm_verify_rows_tc`'s launches) at GLM's dense
//! shape (12288 x 4096): both ranks' joined `[rows, 4096]` output must be
//! the unsplit output word for word, rows 1..=32.
//!
//!   ATLAS_W4A16_TC=1 cargo test --release -p spark-model --lib dense_tp_split -- --ignored --nocapture

use super::*;
use anyhow::{bail, ensure};
use spark_comm::CommBackend;
use std::sync::{Barrier, Mutex};

const H: u32 = 4096;
const I: u32 = 12288;

/// Two ranks' exchange through host memory: each side syncs its stream,
/// posts its bytes, and lands the peer's in `dst`.
struct Pair<'a> {
    gpu: &'a dyn GpuBackend,
    rank: usize,
    slots: &'a Mutex<[Vec<u8>; 2]>,
    barrier: &'a Barrier,
}

impl CommBackend for Pair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn supports_exchange_async(&self, _: usize) -> bool {
        true
    }
    fn exchange_async(
        &self,
        send: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        ensure!(!add, "the split only copies");
        self.gpu.synchronize(stream)?;
        let mut own = vec![0u8; bytes];
        self.gpu.copy_d2h(DevicePtr(send), &mut own)?;
        self.slots.lock().unwrap()[self.rank] = own;
        self.barrier.wait();
        let peer = self.slots.lock().unwrap()[1 - self.rank].clone();
        ensure!(peer.len() == bytes, "ranks exchange different sizes");
        self.gpu.copy_h2d(&peer, DevicePtr(dst))?;
        self.barrier.wait();
        Ok(true)
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
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn weight(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<QuantizedWeight> {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|_| 0x20 + (rng.next() % 24) as u8)
        .collect();
    Ok(QuantizedWeight {
        weight: upload(gpu, &packed)?,
        weight_scale: upload(gpu, &scales)?,
        weight_scale_2: 1.0 / 64.0,
        ..QuantizedWeight::null()
    })
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize, stream: u64) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    gpu.copy_d2h_on_stream(p, &mut b, stream)?;
    crate::layers::glm_gpu_test_backend::finite_bf16("dense TP split output", &b)?;
    Ok(b)
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn dense_tp_split_matches_the_unsplit_ffn() -> Result<()> {
    let gpu = crate::layers::glm_gpu_test_backend::backend()?;
    let g: &dyn GpuBackend = gpu;
    crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(g);
    let tc = g.kernel("w4a16_gemv", "w4a16_gemv_tc8")?;
    let act_mul = g.kernel("moe_silu_mul", "moe_silu_mul")?;
    let mut rng = Rng(0xd15c_0de5);
    let weights = DenseFfnWeights {
        gate_proj: weight(g, &mut rng, I, H)?,
        up_proj: weight(g, &mut rng, I, H)?,
        down_proj: weight(g, &mut rng, H, I)?,
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    };
    let scratch = |g: &dyn GpuBackend| -> Result<[DevicePtr; 3]> {
        Ok([
            g.alloc(32 * I as usize * 2)?,
            g.alloc(32 * I as usize * 2)?,
            g.alloc(32 * H as usize * 2)?,
        ])
    };
    for rows in 1..=32u32 {
        let x: Vec<u8> = (0..rows * H)
            .map(|_| {
                let v = ((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 2.0;
                ((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16
            })
            .flat_map(u16::to_le_bytes)
            .collect();
        let input = upload(g, &x)?;
        // The unsplit canonical FFN.
        let stream = g.default_stream();
        let [gate, up, out] = scratch(g)?;
        let w = &weights;
        ops::w4a16_gemv_batchm(g, tc, input, &w.gate_proj, gate, rows, I, H, stream)?;
        ops::w4a16_gemv_batchm(g, tc, input, &w.up_proj, up, rows, I, H, stream)?;
        ops::silu_mul(g, act_mul, gate, up, gate, rows * I, stream)?;
        ops::w4a16_gemv_batchm(g, tc, gate, &w.down_proj, out, rows, H, I, stream)?;
        let want = read(g, out, (rows * H) as usize * 2, stream)?;
        // The two ranks, each on its own thread and stream.
        let (slots, barrier) = (Mutex::new([Vec::new(), Vec::new()]), Barrier::new(2));
        let got: Vec<Vec<u8>> = std::thread::scope(|s| {
            let ranks: Vec<_> = (0..2)
                .map(|rank| {
                    let (gpu, weights, slots, barrier) = (gpu, &weights, &slots, &barrier);
                    s.spawn(move || -> Result<Vec<u8>> {
                        gpu.bind_to_thread()?;
                        let g: &dyn GpuBackend = gpu;
                        let stream = g.create_stream()?;
                        let comm = Pair {
                            gpu: g,
                            rank,
                            slots,
                            barrier,
                        };
                        let split = SplitFfn {
                            gpu: g,
                            comm: &comm,
                            weights,
                            act_mul,
                            tc,
                            h: H,
                            inter: I,
                        };
                        let bufs = scratch(g)?;
                        split.run(input, rows, bufs, stream)?;
                        read(g, bufs[2], (rows * H) as usize * 2, stream)
                    })
                })
                .collect();
            ranks
                .into_iter()
                .map(|r| r.join().expect("rank thread"))
                .collect::<Result<_>>()
        })?;
        for (rank, out) in got.iter().enumerate() {
            let bad = out
                .chunks(2)
                .zip(want.chunks(2))
                .filter(|(a, b)| a != b)
                .count();
            ensure!(
                bad == 0,
                "rows={rows} rank {rank}: {bad} of {} outputs differ",
                rows * H
            );
        }
    }
    println!("dense TP split == unsplit FFN on both ranks, rows 1..=32");
    Ok(())
}
