// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): the merge form of a token-sharded pair ([`ShardMerge::run`]
//! on both ranks, two threads with their own streams, the pair's exchanges
//! copied in device memory) against the canonical form an unsharded pair runs
//! (`ops::glm_sparse_canonical`, the whole pool behind a block table whose ids
//! keep no residue), for both ranks' heads: BF16 outputs and merged LSEs must
//! be bitwise equal, owner after owner on the same scratch. BF16 cases are
//! also held to a double-precision softmax attention on the CPU.
//! `canonical_and_shard_are_history_free` holds both forms to the same bits
//! wherever the latents sit (other tables, production-sized pools) and
//! whatever the pools' unmapped blocks, the scratch and the output held before
//! (poison, or an earlier owner's scratch of another shape).
//!
//!   cargo test --release -p spark-model --lib canonical_matches_the_shard -- --ignored --nocapture
//! `a_row_does_not_depend_on_the_rows_beside_it` holds one row to the same
//! bits in owners of 1 to 64 rows (a DFlash verify's width follows the
//! drafter, so it must not reach the target's arithmetic).
//!
//!   cargo test --release -p spark-model --lib canonical_and_shard_are_history_free -- --ignored --nocapture
//!   cargo test --release -p spark-model --lib a_row_does_not_depend_on_the_rows_beside_it -- --ignored --nocapture

use super::*;
use anyhow::{Context, bail};
use spark_runtime::kv_cache::LatentShardSpec;
use std::sync::{Barrier, Mutex};

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
    /// Uniform in [-1, 1).
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

/// The pair's exchange on one GPU: each rank posts its send buffer, both
/// meet, copy the peer's into their receive buffer, and meet again.
struct MemPair<'a> {
    gpu: &'a dyn GpuBackend,
    rank: usize,
    posts: &'a Mutex<[(u64, usize); 2]>,
    meet: &'a Barrier,
}

impl CommBackend for MemPair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn exchange_async(&self, send: u64, recv: u64, n: usize, add: bool, s: u64) -> Result<bool> {
        ensure!(!add, "shard exchanges copy, never add");
        self.gpu.synchronize(s)?;
        self.posts.lock().unwrap()[self.rank] = (send, n);
        self.meet.wait();
        let (peer, bytes) = self.posts.lock().unwrap()[1 - self.rank];
        ensure!(bytes == n, "the ranks exchange {n} and {bytes} bytes");
        self.gpu.copy_d2d(DevicePtr(peer), DevicePtr(recv), n)?;
        self.meet.wait();
        Ok(true)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected gather")
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

#[derive(Clone, Copy, Debug)]
enum Selection {
    /// Distinct tokens, `skew` percent of them in even blocks (stored by rank
    /// 0); `holes`: about one in twenty IDs is -1.
    Sparse { skew: usize, holes: bool },
    /// No selection: row `r` attends to tokens `[0, start + r + 1)`.
    Causal(u32),
    /// What the indexer selects for row `r` at position `ctx - rows + r`: up
    /// to 512 random 4-token pools in ascending order, -1 to 2048, then the
    /// partial pool's tokens and -1 to the width.
    Indexer,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    dtype: KvCacheDtype,
    rows: u32,
    ctx: usize,
    selection: Selection,
    owners: usize,
    /// Each owner's row 0 (queries and IDs) is drawn on its own, so it is
    /// the same row whatever `rows` is.
    pinned_row: bool,
}

const BS: usize = 16;

fn block_bytes(dtype: KvCacheDtype) -> usize {
    match dtype {
        KvCacheDtype::Fp8G128 => BS * 528,
        _ => BS * 512 * 2,
    }
}

/// One logical block of latents: random E4M3 codes (no NaN) and positive
/// per-128 scales, or BF16 values in [-1, 1).
fn latent_block(dtype: KvCacheDtype, rng: &mut Rng) -> Vec<u8> {
    match dtype {
        KvCacheDtype::Fp8G128 => {
            let mut b: Vec<u8> = (0..BS * 512)
                .map(|_| match rng.next() as u8 {
                    c if c & 0x7f == 0x7f => c ^ 1,
                    c => c,
                })
                .collect();
            for _ in 0..BS * 4 {
                let scale = (1.0 + rng.below(1000) as f32 / 250.0) / 448.0;
                b.extend(scale.to_le_bytes());
            }
            b
        }
        _ => (0..BS * 512)
            .flat_map(|_| bf16(rng.unit()).to_le_bytes())
            .collect(),
    }
}

/// `n` table entries over physical ids `[0, 2 * half)`: any ids, or with
/// `residues` each on its logical index's residue (a sharded pool's rule).
fn table(n: usize, half: usize, residues: bool, rng: &mut Rng) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..2 * half as u32).collect();
    rng.shuffle(&mut ids);
    if !residues {
        return ids[..n].to_vec();
    }
    let (even, odd): (Vec<u32>, Vec<u32>) = ids.iter().partition(|&&b| b % 2 == 0);
    (0..n)
        .map(|l| if l % 2 == 0 { even[l / 2] } else { odd[l / 2] })
        .collect()
}

fn indexer_row(len: usize, rng: &mut Rng) -> Vec<i32> {
    let pools = len / 4;
    let mut order: Vec<usize> = (0..pools).collect();
    rng.shuffle(&mut order);
    let mut chosen = order[..pools.min(512)].to_vec();
    chosen.sort_unstable();
    let mut row: Vec<i32> = chosen
        .iter()
        .flat_map(|p| (0..4).map(move |i| (p * 4 + i) as i32))
        .collect();
    row.resize(2048, -1);
    row.extend((pools * 4..len).map(|t| t as i32));
    row.resize(WIDTH as usize, -1);
    row
}

fn selection(case: &Case, rng: &mut Rng) -> Option<Vec<i32>> {
    let (skew, holes) = match case.selection {
        Selection::Sparse { skew, holes } => (skew, holes),
        Selection::Causal(_) => return None,
        Selection::Indexer => {
            let first = case.ctx - case.rows as usize;
            let rows = (0..case.rows as usize).flat_map(|r| indexer_row(first + r + 1, rng));
            return Some(rows.collect());
        }
    };
    let width = WIDTH as usize;
    let mut out = Vec::with_capacity(case.rows as usize * width);
    for _ in 0..case.rows {
        let mut by_residue: [Vec<i32>; 2] = [vec![], vec![]];
        for t in 0..case.ctx {
            by_residue[t / BS % 2].push(t as i32);
        }
        by_residue.iter_mut().for_each(|v| rng.shuffle(v));
        let even = (width * skew / 100).min(by_residue[0].len());
        let mut row: Vec<i32> = by_residue[0][..even].to_vec();
        row.extend(&by_residue[1][..width - even]);
        rng.shuffle(&mut row);
        if holes {
            row.iter_mut()
                .filter(|_| rng.below(20) == 0)
                .for_each(|t| *t = -1);
        }
        out.extend(row);
    }
    Some(out)
}

/// One case's device allocations, freed together when it ends.
struct Allocs<'a> {
    gpu: &'a dyn GpuBackend,
    live: Mutex<Vec<DevicePtr>>,
}

impl<'a> Allocs<'a> {
    fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self {
            gpu,
            live: Mutex::new(vec![]),
        }
    }

    /// `bytes` of device memory, every byte `fill`.
    fn alloc(&self, bytes: usize, fill: u8) -> Result<DevicePtr> {
        let bytes = bytes.max(256);
        let p = self.gpu.alloc(bytes)?;
        self.live.lock().unwrap().push(p);
        self.gpu.memset(p, fill, bytes)?;
        Ok(p)
    }

    fn upload<T: Copy>(&self, v: &[T]) -> Result<DevicePtr> {
        let bytes = std::mem::size_of_val(v);
        // SAFETY: plain-old-data slices (u8, u16, u32, i32) viewed as bytes.
        let raw = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), bytes) };
        let p = self.alloc(bytes, 0)?;
        self.gpu.copy_h2d(raw, p)?;
        Ok(p)
    }
}

impl Drop for Allocs<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.synchronize(self.gpu.default_stream());
        for p in self.live.lock().unwrap().drain(..) {
            let _ = self.gpu.free(p);
        }
    }
}

fn download(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c.tp_world_size = 2;
    c
}

/// Double-precision softmax attention of `query` (`[rows, 32, 512]` BF16)
/// over the BF16 latents of each row's tokens.
fn reference(case: &Case, logical: &[Vec<u8>], query: &[u16], ids: &[Vec<i32>]) -> Vec<f64> {
    let latent = |t: usize, d: usize| {
        let b = &logical[t / BS];
        let i = (t % BS * 512 + d) * 2;
        from_bf16(u16::from_le_bytes([b[i], b[i + 1]])) as f64
    };
    let mut out = vec![0f64; query.len()];
    for (r, tokens) in ids.iter().enumerate().take(case.rows as usize) {
        for h in 0..32 {
            let q = &query[(r * 32 + h) * 512..][..512];
            let logits: Vec<f64> = tokens
                .iter()
                .map(|&t| {
                    let dot: f64 = (0..512)
                        .map(|d| from_bf16(q[d]) as f64 * latent(t as usize, d))
                        .sum();
                    dot * 0.0625
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
                out[(r * 32 + h) * 512 + d] = v / z;
            }
        }
    }
    out
}

/// Where a case's latents sit and what the memory it reads held before; no
/// placement may change a bit. `seed` draws the block tables, `pool_blocks`
/// sizes the unsharded pool (the shard's halves follow; at least the
/// history), and `poison` fills every pool block no table maps, every scratch
/// and the outputs first. `scratch`: the canonical form's region, kept across
/// cases, instead of a fresh one.
#[derive(Clone, Copy, Debug)]
struct Placement {
    seed: u64,
    pool_blocks: usize,
    poison: u8,
    scratch: Option<(DevicePtr, usize)>,
}

impl Placement {
    /// Tight pools, zeroed memory, fresh scratch.
    fn fresh() -> Self {
        Self {
            seed: 1,
            pool_blocks: 0,
            poison: 0,
            scratch: None,
        }
    }
}

/// One case's results: the canonical form's outputs and LSEs per owner and
/// rank, then the shard's, and how many of the forms' bytes differ.
struct CaseOut {
    differing: usize,
    bits: Vec<Vec<u8>>,
}

#[path = "paged_glm_shard_merge_gpu_cases.rs"]
mod cases;
#[path = "paged_glm_shard_merge_gpu_run.rs"]
mod runner;
