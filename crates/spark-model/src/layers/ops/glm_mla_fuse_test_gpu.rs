// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): `ATLAS_GLM_MLA_FUSE`'s two launches against the twelve they
//! replace (MXFP8 q_a / kv_a, `rms_norm_vanilla_regs` on each,
//! `mla_cache_assemble_batched`, `glm_kv_shard_map_slots`,
//! `glm_latent_cache_write_fp8g128`, the canonical `dense_gemv_bf16_tc*` key
//! and gate projections around `glm_index_layernorm_bf16`,
//! `glm_index_tail_write_bf16`, `glm_index_kpool_finalize_bf16` and
//! `glm_kv_shard_localize_compact`) at GLM-5.3's shape, for every block of
//! 1..=32 rows, starting mid-pool and mid-block, unsharded and on each rank
//! of a sharded pair: the same q and kv latents, index keys and gates, fp8
//! latent pool, raw tails, pooled keys and localized IDs, byte for byte.
//!
//!   cargo test --release -p spark-model --lib glm_mla_fuse -- --ignored --nocapture

use super::*;
use anyhow::Context;
use spark_runtime::cuda_backend::AtlasCudaBackend;

const H: u32 = 4096;
const QL: u32 = 1536;
const KVL: u32 = 512;
const ID: u32 = 128;
const BS: u32 = 16;
const WIDTH: u32 = 2051;
/// Logical blocks of the test sequence ([`physical`] places them): a
/// whole dense-selection context (2048 tokens).
const BLOCKS: u32 = 128;
const PHYS: u32 = 3 * BLOCKS + 2;
const TAIL_STRIDE: u64 = (2 * BS * ID * 2) as u64;
const POOL_STRIDE: u64 = (BS / 4 * ID * 2) as u64;
const LATENT_BLOCK: usize = (BS * 528) as usize;

/// Logical block `l`'s physical block: distinct, and with the residue of `l`
/// (the shard allocator's invariant).
fn physical(l: u32) -> u32 {
    3 * l + 2 * (l % 2)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn bf16_bytes(n: usize, rng: &mut Rng, scale: f32, offset: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| bf16(offset + scale * rng.unit()).to_le_bytes())
        .collect()
}

fn random_bytes(n: usize, rng: &mut Rng) -> Vec<u8> {
    (0..n).map(|_| rng.next() as u8).collect()
}

fn backend() -> Result<&'static AtlasCudaBackend> {
    crate::layers::glm_gpu_test_backend::backend()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn raw(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// One MXFP8 weight `[n, H]`: E4M3 codes below 2 in magnitude (no NaN),
/// scales 2^-9 .. 2^-6.
fn mx(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32) -> Result<(DevicePtr, DevicePtr)> {
    let values: Vec<u8> = (0..n * H).map(|_| rng.next() as u8 & 0xbf).collect();
    let scales: Vec<u8> = (0..n * H / 32)
        .map(|_| 118 + (rng.next() % 4) as u8)
        .collect();
    Ok((upload(gpu, &values)?, upload(gpu, &scales)?))
}

/// The layer's weights, shared by both runs.
struct Weights {
    q_a: (DevicePtr, DevicePtr),
    kv_a: (DevicePtr, DevicePtr),
    wk: DevicePtr,
    gate: DevicePtr,
    q_norm: DevicePtr,
    kv_norm: DevicePtr,
    k_norm_w: DevicePtr,
    k_norm_b: DevicePtr,
    ape: DevicePtr,
}

impl Weights {
    fn new(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        Ok(Self {
            q_a: mx(gpu, rng, QL)?,
            kv_a: mx(gpu, rng, KVL)?,
            wk: upload(gpu, &bf16_bytes((ID * H) as usize, rng, 0.02, 0.0))?,
            gate: upload(gpu, &bf16_bytes((ID * H) as usize, rng, 0.02, 0.0))?,
            q_norm: upload(gpu, &bf16_bytes(QL as usize, rng, 0.25, 1.0))?,
            kv_norm: upload(gpu, &bf16_bytes(KVL as usize, rng, 0.25, 1.0))?,
            k_norm_w: upload(gpu, &bf16_bytes(ID as usize, rng, 0.25, 1.0))?,
            k_norm_b: upload(gpu, &bf16_bytes(ID as usize, rng, 0.1, 0.0))?,
            ape: upload(gpu, &bf16_bytes((4 * ID) as usize, rng, 0.5, 0.0))?,
        })
    }
}

/// The buffers one run writes, initialized alike for both runs.
struct Run {
    q: DevicePtr,
    kv: DevicePtr,
    entries: DevicePtr,
    keys: DevicePtr,
    gates: DevicePtr,
    latent: DevicePtr,
    tail: DevicePtr,
    pool: DevicePtr,
    local: DevicePtr,
    ids: DevicePtr,
    counts: DevicePtr,
}

/// What one run leaves, read back.
#[derive(PartialEq, Debug)]
struct Out {
    q: Vec<u8>,
    kv: Vec<u8>,
    keys: Vec<u8>,
    gates: Vec<u8>,
    latent: Vec<u8>,
    tail: Vec<u8>,
    pool: Vec<u8>,
    ids: Vec<u8>,
    counts: Vec<u8>,
}

impl Run {
    /// Buffers sized for `rows` rows, uninitialized ([`Run::fill`]).
    fn new(gpu: &dyn GpuBackend, rows: u32) -> Result<Self> {
        let sizes = Self::sizes(rows);
        let mut b = sizes.iter().map(|&n| gpu.alloc(n));
        let mut next = || b.next().expect("one size per buffer");
        Ok(Self {
            q: next()?,
            kv: next()?,
            entries: next()?,
            keys: next()?,
            gates: next()?,
            latent: next()?,
            tail: next()?,
            pool: next()?,
            local: next()?,
            ids: next()?,
            counts: next()?,
        })
    }

    fn buffers(&self) -> [DevicePtr; 11] {
        [
            self.q,
            self.kv,
            self.entries,
            self.keys,
            self.gates,
            self.latent,
            self.tail,
            self.pool,
            self.local,
            self.ids,
            self.counts,
        ]
    }

    /// Bytes of each buffer, in [`Run::buffers`] order.
    fn sizes(rows: u32) -> [usize; 11] {
        let r = rows as usize;
        [
            r * QL as usize * 2,
            r * KVL as usize * 2,
            2 * r * KVL as usize * 2,
            r * ID as usize * 2,
            r * ID as usize * 2,
            PHYS as usize * LATENT_BLOCK,
            PHYS as usize * TAIL_STRIDE as usize,
            PHYS as usize * POOL_STRIDE as usize,
            r * 8,
            r * WIDTH as usize * 4,
            r * 4,
        ]
    }

    /// Every buffer to the random bytes of `seed` (alike for both runs).
    fn fill(&self, gpu: &dyn GpuBackend, rows: u32, seed: u64) -> Result<()> {
        let mut rng = Rng(seed);
        for (i, (p, n)) in self
            .buffers()
            .into_iter()
            .zip(Self::sizes(rows))
            .enumerate()
        {
            // Existing tail/pool entries are legitimate model inputs. Random
            // bit patterns could contain NaNs and make matching NaNs pass.
            let bytes = if matches!(i, 6 | 7) {
                bf16_bytes(n / 2, &mut rng, 1.0, 0.0)
            } else {
                random_bytes(n, &mut rng)
            };
            gpu.copy_h2d(&bytes, p)?;
        }
        Ok(())
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.buffers().into_iter().try_for_each(|p| gpu.free(p))
    }

    fn read(&self, gpu: &dyn GpuBackend, rows: u32) -> Result<Out> {
        let r = rows as usize;
        Ok(Out {
            q: raw(gpu, self.q, r * QL as usize * 2)?,
            kv: raw(gpu, self.kv, r * KVL as usize * 2)?,
            keys: raw(gpu, self.keys, r * ID as usize * 2)?,
            gates: raw(gpu, self.gates, r * ID as usize * 2)?,
            latent: raw(gpu, self.latent, PHYS as usize * LATENT_BLOCK)?,
            tail: raw(gpu, self.tail, PHYS as usize * TAIL_STRIDE as usize)?,
            pool: raw(gpu, self.pool, PHYS as usize * POOL_STRIDE as usize)?,
            ids: raw(gpu, self.ids, r * WIDTH as usize * 4)?,
            counts: raw(gpu, self.counts, r * 4)?,
        })
    }
}

/// The sequence's metadata: `rows` rows from token `start`.
struct Meta {
    slots: DevicePtr,
    table: DevicePtr,
    start: u32,
}

impl Meta {
    fn new(gpu: &dyn GpuBackend, rows: u32, start: u32) -> Result<Self> {
        let slots: Vec<u8> = (start..start + rows)
            .flat_map(|t| i64::from(physical(t / BS) * BS + t % BS).to_le_bytes())
            .collect();
        let table: Vec<u8> = (0..BLOCKS)
            .flat_map(|l| physical(l).to_le_bytes())
            .collect();
        Ok(Self {
            slots: upload(gpu, &slots)?,
            table: upload(gpu, &table)?,
            start,
        })
    }
}

fn kernel(gpu: &dyn GpuBackend, module: &str, name: &str) -> Result<KernelHandle> {
    let k = gpu.kernel(module, name)?;
    anyhow::ensure!(k.0 != 0, "{module}::{name} unavailable");
    Ok(k)
}

#[path = "glm_mla_fuse_oracle_test_gpu.rs"]
mod oracle;
use oracle::unfused;
#[path = "glm_mla_fuse_finite_test_gpu.rs"]
mod finite;

/// The two launches.
fn fused(
    gpu: &dyn GpuBackend,
    f: &MlaFuse,
    w: &Weights,
    run: &Run,
    meta: &Meta,
    input: DevicePtr,
    rows: u32,
    owner: ShardRank,
    s: u64,
) -> Result<()> {
    let proj = |(data, scales), out, n| MxProj {
        data,
        scales,
        out,
        n,
    };
    let down = DownArgs {
        input,
        rows,
        k: H,
        wk: w.wk,
        gate: w.gate,
        keys: run.keys,
        gates: run.gates,
        ni: ID,
        q_a: proj(w.q_a, run.q, QL),
        kv_a: proj(w.kv_a, run.kv, KVL),
        // Touching is loads only; without PDL it runs before the body.
        touch: Some((12 << 20, 32)),
    };
    f.launch_down(gpu, &down, s)?;
    let mid = MidArgs {
        rows,
        q_latent: run.q,
        q_norm: w.q_norm,
        q_dim: QL,
        kv_latent: run.kv,
        kv_norm: w.kv_norm,
        kv_dim: KVL,
        eps: 1e-6,
        latent_cache: run.latent,
        slots: meta.slots,
        block_size: BS,
        owner,
        index: IndexCache {
            keys: run.keys,
            gates: run.gates,
            norm_weight: w.k_norm_w,
            norm_bias: w.k_norm_b,
            eps: 1e-6,
            dim: ID,
            tail: run.tail,
            tail_map: DevicePtr::NULL,
            tail_stride: TAIL_STRIDE,
            ape: w.ape,
            pool: run.pool,
            pool_stride: POOL_STRIDE,
            tokens_per_pool: 4,
        },
        localize: (owner.world > 1).then_some((meta.table, run.ids, run.counts, WIDTH, meta.start)),
    };
    f.launch_mid(gpu, &mid, s)
}

#[test]
#[ignore = "needs a GB10 and the glm-5.3-flash kernels"]
fn fused_front_matches_the_twelve_launches() -> Result<()> {
    let gpu = backend()?;
    let f = MlaFuse::load(gpu).context("glm_mla_fuse kernels")?;
    let s = gpu.default_stream();
    let mut rng = Rng(0x6d6c_6166_7573_6501);
    let w = Weights::new(gpu, &mut rng)?;
    let owners = [
        ShardRank { rank: 0, world: 1 },
        ShardRank { rank: 0, world: 2 },
        ShardRank { rank: 1, world: 2 },
    ];
    for rows in (1..=32).filter(|r| *r <= 8 || r % 4 == 0 || [9, 17, 31].contains(r)) {
        let (a, b) = (Run::new(gpu, rows)?, Run::new(gpu, rows)?);
        // Mid-pool and mid-block starts, and a block ending at 2048.
        for start in [29u32, 64, 126, 2016 - rows.min(16)] {
            let input = upload(gpu, &bf16_bytes((rows * H) as usize, &mut rng, 1.0, 0.0))?;
            let meta = Meta::new(gpu, rows, start)?;
            for owner in owners {
                let seed = rng.next();
                a.fill(gpu, rows, seed)?;
                b.fill(gpu, rows, seed)?;
                let before = raw(gpu, a.latent, PHYS as usize * LATENT_BLOCK)?;
                unfused(gpu, &w, &a, &meta, input, rows, owner, s)?;
                fused(gpu, &f, &w, &b, &meta, input, rows, owner, s)?;
                gpu.synchronize(s)?;
                let (want, got) = (a.read(gpu, rows)?, b.read(gpu, rows)?);
                for output in [&want, &got] {
                    output.check_finite(start, rows, owner)?;
                }
                let what = format!("rows {rows} from {start}, {owner:?}");
                for (name, x, y) in [
                    ("q latent", &want.q, &got.q),
                    ("kv latent", &want.kv, &got.kv),
                    ("index keys", &want.keys, &got.keys),
                    ("index gates", &want.gates, &got.gates),
                    ("fp8 latent pool", &want.latent, &got.latent),
                    ("raw tails", &want.tail, &got.tail),
                    ("pooled keys", &want.pool, &got.pool),
                    ("localized ids", &want.ids, &got.ids),
                    ("localized counts", &want.counts, &got.counts),
                ] {
                    let first = x.iter().zip(y.iter()).position(|(p, q)| p != q);
                    anyhow::ensure!(first.is_none(), "{what}: {name} differ at byte {first:?}");
                }
                // Not a vacuous comparison: an unsharded run writes every
                // row's latent (a rank of a pair may own none of them).
                anyhow::ensure!(
                    owner.world > 1 || want.latent != before,
                    "{what}: the latent write wrote nothing"
                );
            }
            [input, meta.slots, meta.table]
                .into_iter()
                .try_for_each(|p| gpu.free(p))?;
        }
        a.free(gpu)?;
        b.free(gpu)?;
    }
    eprintln!(
        "fused MLA front == twelve launches: rows 1..=32 x 4 starts x unsharded/rank 0/rank 1"
    );
    Ok(())
}
