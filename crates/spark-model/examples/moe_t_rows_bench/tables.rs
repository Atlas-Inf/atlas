// SPDX-License-Identifier: AGPL-3.0-only
//! Layer tables, RNG and constants for `moe_t_rows_bench`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(crate) const N_EXP: usize = 40; // routed experts (FP4-Local)
pub(crate) const N_OUT: usize = 640; // gate/up output N
pub(crate) const K_RED: usize = 2560; // reduction K (gate/up)
pub(crate) const N_DOWN: usize = 2560; // down output N
pub(crate) const K_DOWN: usize = 640; // down reduction K
pub(crate) const TOP_K: usize = 10;

pub(crate) struct Lcg(pub(crate) u64);
impl Lcg {
    pub(crate) fn u(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    pub(crate) fn f(&mut self) -> f32 {
        (self.u() as f64 / (1u64 << 53) as f64) as f32
    }
    pub(crate) fn b(&mut self) -> u8 {
        (self.u() & 0xFF) as u8
    }
    pub(crate) fn sc(&mut self) -> u8 {
        0x28 + (self.u() % 0x21) as u8
    } // E4M3 ~2^-6..2^2
    pub(crate) fn s2(&mut self) -> f32 {
        0.001 + self.f() * 0.009
    } // ~1e-3..1e-2
    pub(crate) fn act(&mut self) -> f32 {
        (self.f() * 4.0 - 2.0).clamp(-2.0, 2.0)
    }
}

pub(crate) fn up(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
pub(crate) fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// One layer's transposed ptr tables: each expert separately allocated.
pub(crate) struct Layer {
    gpk: Vec<DevicePtr>,
    gsc: Vec<DevicePtr>,
    upk: Vec<DevicePtr>,
    usc: Vec<DevicePtr>,
    dpk: Vec<DevicePtr>,
    dsc: Vec<DevicePtr>,
    s2: Vec<f32>,
}
impl Layer {
    pub(crate) fn new(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Self> {
        let mut s2: Vec<f32> = Vec::new();
        let mk_packed = |bytes: usize, rng: &mut Lcg| -> Result<Vec<DevicePtr>> {
            (0..N_EXP)
                .map(|_| {
                    let d: Vec<u8> = (0..bytes).map(|_| rng.b()).collect();
                    up(g, &d)
                })
                .collect()
        };
        let mk_scale = |bytes: usize, rng: &mut Lcg| -> Result<Vec<DevicePtr>> {
            (0..N_EXP)
                .map(|_| {
                    let d: Vec<u8> = (0..bytes).map(|_| rng.sc()).collect();
                    up(g, &d)
                })
                .collect()
        };
        for _ in 0..N_EXP {
            s2.push(rng.s2());
        }
        Ok(Self {
            gpk: mk_packed(K_RED / 2 * N_OUT, rng)?,
            gsc: mk_scale(K_RED / 16 * N_OUT, rng)?,
            upk: mk_packed(K_RED / 2 * N_OUT, rng)?,
            usc: mk_scale(K_RED / 16 * N_OUT, rng)?,
            dpk: mk_packed(K_DOWN / 2 * N_DOWN, rng)?,
            dsc: mk_scale(K_DOWN / 16 * N_DOWN, rng)?,
            s2,
        })
    }
    pub(crate) fn ptr_tables(&self, g: &dyn GpuBackend) -> Result<[DevicePtr; 7]> {
        let v = |ps: &[DevicePtr]| -> Result<DevicePtr> {
            let d: Vec<u8> = ps.iter().flat_map(|p| p.0.to_le_bytes()).collect();
            up(g, &d)
        };
        let s2d: Vec<u8> = self.s2.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok([
            v(&self.gpk)?,
            v(&self.gsc)?,
            up(g, &s2d)?,
            v(&self.upk)?,
            v(&self.usc)?,
            v(&self.dpk)?,
            v(&self.dsc)?,
        ])
    }
}

pub(crate) const LAYER_SET: usize = 48; // ~5 GB of tables: cold-streams L2 like a real step.
/// Routing with an exact routed-expert union of `union` experts across
/// `rows` x TOP_K distinct-per-row ids. Requires TOP_K <= union <=
/// min(N_EXP, rows*TOP_K). pool[0..union] ends up exactly covering rows.
pub(crate) fn union_ids(rng: &mut Lcg, rows: usize, union: usize) -> Vec<u32> {
    let mut pool: Vec<u32> = (0..N_EXP as u32).collect();
    for i in (1..pool.len()).rev() {
        let j = (rng.u() % (i as u64 + 1)) as usize;
        pool.swap(i, j);
    }
    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); rows];
    ids[0] = pool[..TOP_K].to_vec();
    let extra = union - TOP_K;
    let per = if rows > 1 { extra / (rows - 1) } else { 0 };
    let mut rem = extra - per * (rows - 1);
    let mut used = TOP_K;
    for r in 1..rows {
        let f = per
            + if rem > 0 {
                rem -= 1;
                1
            } else {
                0
            };
        for _ in 0..f {
            ids[r].push(pool[used]);
            used += 1;
        }
        while ids[r].len() < TOP_K {
            let id = pool[(rng.u() % used as u64) as usize];
            if !ids[r].contains(&id) {
                ids[r].push(id);
            }
        }
    }
    debug_assert_eq!(used, union);
    ids.concat()
}

/// Weight bytes one expert contributes per kernel pair (packed + scale for
/// gate & up + down).
pub(crate) fn expert_bytes() -> usize {
    (K_RED / 2 * N_OUT + K_RED / 16 * N_OUT) * 2 + (K_DOWN / 2 * N_DOWN + K_DOWN / 16 * N_DOWN)
}

pub(crate) fn mem_kb() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("MemAvailable"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
    })
}
