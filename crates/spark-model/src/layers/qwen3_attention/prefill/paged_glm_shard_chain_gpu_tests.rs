// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): a four-rank token shard on one GPU (four threads with their
//! own streams, the chain's all-gathers copied in device memory). The merge
//! form ([`ChainMerge::run`]) must give every rank's 16 heads the
//! double-precision softmax attention over the selected tokens of all ranks,
//! including rows whose tokens one rank stores none of; the view form
//! ([`ViewParts::gather`]) must assemble the history bit for bit.
//!
//!   cargo test --release -p spark-model --lib chain_shard -- --ignored --nocapture

use super::*;
use anyhow::{Context, bail};
use spark_runtime::kv_cache::LatentShardSpec;
use std::sync::{Barrier, Mutex};

const WORLD: usize = 4;
const BS: usize = 16;
const BB: usize = BS * 512 * 2; // BF16 latent block

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// The chain's in-place all-gather on one GPU: every rank posts its base,
/// all meet, each copies the others' parts into place, and all meet again.
struct MemChain<'a> {
    gpu: &'a dyn GpuBackend,
    rank: usize,
    posts: &'a Mutex<[(u64, i64, usize); WORLD]>,
    meet: &'a Barrier,
}

impl CommBackend for MemChain<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        WORLD
    }
    fn gather_async(&self, base: u64, stride: i64, bytes: usize, s: u64) -> Result<bool> {
        self.gpu.synchronize(s)?;
        self.posts.lock().unwrap()[self.rank] = (base, stride, bytes);
        self.meet.wait();
        let posts = *self.posts.lock().unwrap();
        for (o, &(peer, ps, pb)) in posts.iter().enumerate().filter(|(o, _)| *o != self.rank) {
            ensure!(
                ps == stride && pb == bytes,
                "the ranks gather different shapes"
            );
            let at = |b: u64| DevicePtr(b.wrapping_add_signed(stride * o as i64));
            self.gpu.copy_d2d(at(peer), at(base), bytes)?;
        }
        self.meet.wait();
        Ok(true)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected NCCL gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected scatter")
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

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c.tp_world_size = WORLD;
    c
}

/// One rank's table over its own ids: logical block `l` gets an id of
/// residue `l % 4` (the sharded allocator's rule), shuffled per rank.
fn table(blocks: usize, slots: usize, rng: &mut Rng) -> Vec<u32> {
    let mut lists: Vec<Vec<u32>> = (0..WORLD)
        .map(|r| {
            let mut ids: Vec<u32> = (0..slots as u32)
                .map(|s| s * WORLD as u32 + r as u32)
                .collect();
            rng.shuffle(&mut ids);
            ids
        })
        .collect();
    (0..blocks)
        .map(|l| lists[l % WORLD].pop().unwrap())
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    /// Distinct random tokens, about one in twenty -1; `skip`: none stored
    /// by that rank.
    Sparse { skip: Option<usize> },
    /// Row `r` attends to tokens `[0, start + r + 1)`.
    Causal(u32),
}

fn rows_ids(
    rows: usize,
    ctx: usize,
    sel: Selection,
    rng: &mut Rng,
) -> (Option<Vec<i32>>, Vec<Vec<i32>>) {
    let w = WIDTH as usize;
    match sel {
        Selection::Causal(start) => (
            None,
            (0..rows)
                .map(|r| (0..=(start as i32 + r as i32)).collect())
                .collect(),
        ),
        Selection::Sparse { skip } => {
            let mut flat = vec![];
            let mut kept = vec![];
            for _ in 0..rows {
                let mut pool: Vec<i32> = (0..ctx as i32)
                    .filter(|t| skip != Some(*t as usize / BS % WORLD))
                    .collect();
                rng.shuffle(&mut pool);
                let mut row: Vec<i32> = pool.into_iter().take(w).collect();
                row.resize(w, -1);
                row.iter_mut()
                    .filter(|_| rng.below(20) == 0)
                    .for_each(|t| *t = -1);
                kept.push(row.iter().copied().filter(|&t| t >= 0).collect());
                flat.extend(row);
            }
            (Some(flat), kept)
        }
    }
}

/// Double-precision attention of `heads` heads of `query` (`[rows, 64,
/// 512]`) from `h0` over each row's tokens.
fn reference(
    logical: &[Vec<u8>],
    query: &[u16],
    ids: &[Vec<i32>],
    h0: usize,
    heads: usize,
) -> Vec<f64> {
    let latent = |t: usize, d: usize| {
        let b = &logical[t / BS];
        let i = (t % BS * 512 + d) * 2;
        from_bf16(u16::from_le_bytes([b[i], b[i + 1]])) as f64
    };
    let mut out = vec![0f64; ids.len() * heads * 512];
    for (r, tokens) in ids.iter().enumerate() {
        for h in 0..heads {
            let q = &query[(r * 64 + h0 + h) * 512..][..512];
            let logits: Vec<f64> = tokens
                .iter()
                .map(|&t| {
                    (0..512)
                        .map(|d| from_bf16(q[d]) as f64 * latent(t as usize, d))
                        .sum::<f64>()
                        * 0.0625
                })
                .collect();
            let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
            let w: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            for d in 0..512 {
                let v: f64 = tokens
                    .iter()
                    .zip(&w)
                    .map(|(&t, w)| w * latent(t as usize, d))
                    .sum();
                out[(r * heads + h) * 512 + d] = v / z;
            }
        }
    }
    out
}

fn upload(gpu: &dyn GpuBackend, live: &Mutex<Vec<DevicePtr>>, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(256))?;
    live.lock().unwrap().push(p);
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn alloc(
    gpu: &dyn GpuBackend,
    live: &Mutex<Vec<DevicePtr>>,
    bytes: usize,
    fill: u8,
) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.max(256))?;
    live.lock().unwrap().push(p);
    gpu.memset(p, fill, bytes.max(256))?;
    Ok(p)
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[path = "paged_glm_shard_chain_gpu_cases.rs"]
mod cases;
