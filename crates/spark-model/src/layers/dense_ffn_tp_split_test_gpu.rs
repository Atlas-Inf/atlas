// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build, `ATLAS_W4A16_TC=1`): `ATLAS_GLM_DENSE_TP_SPLIT`'s ranks
//! (the pair and a four-rank chain), as threads on one GPU over a
//! host-mailbox pair exchange or gather, against the unsplit canonical FFN
//! (`glm_verify_rows_tc`'s launches, and the native MXFP8 ones of
//! `try_native`) at GLM's dense shape (12288 x 4096):
//! every rank's joined `[rows, 4096]` output must be the unsplit output word
//! for word, rows 1..=32.
//!
//!   ATLAS_W4A16_TC=1 cargo test --release -p spark-model --lib dense_tp_split -- --ignored --nocapture

use super::*;
use crate::weight_map::Mxfp8Weight;
use anyhow::{bail, ensure};
use spark_comm::CommBackend;
use std::sync::{Barrier, Mutex};

const H: u32 = 4096;
const I: u32 = 12288;

/// The ranks' exchange (a pair) or gather (a chain) through host memory:
/// each side syncs its stream, posts its bytes, and lands the others'.
struct Pair<'a> {
    gpu: &'a dyn GpuBackend,
    rank: usize,
    world: usize,
    slots: &'a Mutex<Vec<Vec<u8>>>,
    barrier: &'a Barrier,
}

impl Pair<'_> {
    /// Post `bytes` from `send`; every rank's posted bytes, in rank order.
    fn post(&self, send: u64, bytes: usize, stream: u64) -> Result<Vec<Vec<u8>>> {
        self.gpu.synchronize(stream)?;
        let mut own = vec![0u8; bytes];
        self.gpu.copy_d2h(DevicePtr(send), &mut own)?;
        self.slots.lock().unwrap()[self.rank] = own;
        self.barrier.wait();
        let all = self.slots.lock().unwrap().clone();
        self.barrier.wait();
        ensure!(
            all.iter().all(|p| p.len() == bytes),
            "ranks exchange different sizes"
        );
        Ok(all)
    }
}

impl CommBackend for Pair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
    fn supports_exchange_async(&self, _: usize) -> bool {
        self.world == 2
    }
    fn supports_gather_async(&self, _: usize) -> bool {
        self.world > 2
    }
    fn gather_async(&self, base: u64, stride: i64, bytes: usize, stream: u64) -> Result<bool> {
        ensure!(self.world > 2, "the pair swaps");
        let at = |r: usize| base.wrapping_add_signed(stride * r as i64);
        let all = self.post(at(self.rank), bytes, stream)?;
        for (r, part) in all.iter().enumerate() {
            self.gpu.copy_h2d(part, DevicePtr(at(r)))?;
        }
        Ok(true)
    }
    fn exchange_async(
        &self,
        send: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        ensure!(
            !add && self.world == 2,
            "the split only copies, a pair swaps"
        );
        let all = self.post(send, bytes, stream)?;
        self.gpu.copy_h2d(&all[1 - self.rank], DevicePtr(dst))?;
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

/// A random row-major MXFP8 `[n, k]` weight: E4M3 codes without NaNs and
/// E8M0 scales near 1.
fn mx_weight(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<Mxfp8Weight> {
    let codes: Vec<u8> = (0..n * k).map(|_| rng.next() as u8 & 0xF7).collect();
    let scales: Vec<u8> = (0..n * k / 32)
        .map(|_| 120 + (rng.next() % 6) as u8)
        .collect();
    Ok(Mxfp8Weight {
        data: upload(gpu, &codes)?,
        scales: upload(gpu, &scales)?,
        tiled: false,
    })
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn dense_tp_split_matches_the_unsplit_ffn() -> Result<()> {
    let gpu = crate::layers::glm_gpu_test_backend::backend()?;
    let g: &dyn GpuBackend = gpu;
    crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(g, true);
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
    let native = DenseNative {
        gate: mx_weight(g, &mut rng, I, H)?,
        up: mx_weight(g, &mut rng, I, H)?,
        down: mx_weight(g, &mut rng, H, I)?,
        w8a8: None,
    };
    let scratch = |g: &dyn GpuBackend| -> Result<[DevicePtr; 3]> {
        Ok([
            g.alloc(32 * I as usize * 2)?,
            g.alloc(32 * I as usize * 2)?,
            g.alloc(32 * H as usize * 2)?,
        ])
    };
    let kinds = [
        (
            "NVFP4",
            SplitProj::Nvfp4 {
                weights: &weights,
                tc,
            },
        ),
        ("MXFP8", SplitProj::Mxfp8(&native)),
    ];
    for ((name, proj), (world, rows)) in kinds.into_iter().flat_map(|k| {
        [2usize, 4]
            .into_iter()
            .flat_map(|w| (1..=32u32).map(move |r| (w, r)))
            .map(move |wr| (k, wr))
    }) {
        let x: Vec<u8> = (0..rows * H)
            .map(|_| {
                let v = ((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 2.0;
                ((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16
            })
            .flat_map(u16::to_le_bytes)
            .collect();
        let input = upload(g, &x)?;
        // The unsplit FFN.
        let stream = g.default_stream();
        let [gate, up, out] = scratch(g)?;
        match proj {
            SplitProj::Nvfp4 { weights: w, tc } => {
                ops::w4a16_gemv_batchm(g, tc, input, &w.gate_proj, gate, rows, I, H, stream)?;
                ops::w4a16_gemv_batchm(g, tc, input, &w.up_proj, up, rows, I, H, stream)?;
                ops::silu_mul(g, act_mul, gate, up, gate, rows * I, stream)?;
                ops::w4a16_gemv_batchm(g, tc, gate, &w.down_proj, out, rows, H, I, stream)?;
            }
            SplitProj::Mxfp8(n) => {
                let k = ops::native_kernels(g);
                let (i, h, r) = (I as usize, H as usize, rows as usize);
                let parts = [
                    ops::MxPart::mx(&n.gate, gate, i),
                    ops::MxPart::mx(&n.up, up, i),
                ];
                ops::mx_multi_rows(g, &k.mx_multi, false, input, &parts, r, h, stream)?;
                ops::silu_mul(g, act_mul, gate, up, gate, rows * I, stream)?;
                ops::mxfp8_gemv_rows_touched(
                    g,
                    &k.mx.row_major,
                    &n.down,
                    gate,
                    out,
                    r,
                    h,
                    i,
                    h,
                    stream,
                )?;
            }
        }
        let want = read(g, out, (rows * H) as usize * 2, stream)?;
        // The ranks, each on its own thread and stream.
        let (slots, barrier) = (Mutex::new(vec![Vec::new(); world]), Barrier::new(world));
        let got: Vec<Vec<u8>> = std::thread::scope(|s| {
            let ranks: Vec<_> = (0..world)
                .map(|rank| {
                    let (gpu, slots, barrier) = (gpu, &slots, &barrier);
                    s.spawn(move || -> Result<Vec<u8>> {
                        gpu.bind_to_thread()?;
                        let g: &dyn GpuBackend = gpu;
                        let stream = g.create_stream()?;
                        let comm = Pair {
                            gpu: g,
                            rank,
                            world,
                            slots,
                            barrier,
                        };
                        let split = SplitFfn {
                            gpu: g,
                            comm: &comm,
                            proj,
                            act_mul,
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
                "{name} world {world} rows={rows} rank {rank}: {bad} of {} outputs differ",
                rows * H
            );
        }
    }
    println!(
        "dense TP split == unsplit FFN (NVFP4 and MXFP8) on every rank of the pair and the chain, rows 1..=32"
    );
    // Timing at a three-row verify: the unsplit native MLP against one rank's
    // projections of the four-way split (the joins are the chain's).
    let (stream, rows) = (g.default_stream(), 3u32);
    let input = g.alloc(32 * H as usize * 2)?;
    let [gate, up, out] = scratch(g)?;
    let k = ops::native_kernels(g);
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        g.synchronize(stream)?;
        let t = std::time::Instant::now();
        for _ in 0..20 {
            f()?;
        }
        g.synchronize(stream)?;
        Ok(t.elapsed().as_secs_f64() / 20.0 * 1e6)
    };
    let unsplit = time(&|| {
        let (i, h) = (I as usize, H as usize);
        let parts = [
            ops::MxPart::mx(&native.gate, gate, i),
            ops::MxPart::mx(&native.up, up, i),
        ];
        ops::mx_multi_rows(g, &k.mx_multi, false, input, &parts, 3, h, stream)?;
        ops::silu_mul(g, act_mul, gate, up, gate, rows * I, stream)?;
        ops::mxfp8_gemv_rows_touched(
            g,
            &k.mx.row_major,
            &native.down,
            gate,
            out,
            3,
            h,
            i,
            h,
            stream,
        )
    })?;
    let comm = Pair {
        gpu: g,
        rank: 1,
        world: 4,
        slots: &Mutex::new(Vec::new()),
        barrier: &Barrier::new(1),
    };
    let split = SplitFfn {
        gpu: g,
        comm: &comm,
        proj: SplitProj::Mxfp8(&native),
        act_mul,
        h: H,
        inter: I,
    };
    let quarter = time(&|| {
        split.gate_up(input, rows, I / 4, I / 4, [gate, up], stream)?;
        ops::silu_mul(g, act_mul, gate, up, gate, rows * I / 4, stream)?;
        split.down(up, rows, H / 4, H / 4, out, stream)
    })?;
    println!("MXFP8 dense MLP, 3 rows: unsplit {unsplit:.1} us, one rank of four {quarter:.1} us");
    Ok(())
}
