// SPDX-License-Identifier: AGPL-3.0-only
//! Synthetic weights and top-8 routings for `moe_verify_bench`.

use super::*;

pub(super) struct Rng(pub(super) u64);
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
    pub(super) fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
    /// Random E2M1 bytes.
    pub(super) fn packed(&mut self, n: usize) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n.div_ceil(8))
            .flat_map(|_| self.next().to_le_bytes())
            .collect();
        v.truncate(n);
        v
    }
    /// UE4M3 block scales in a sane exponent range (0x30..=0x3F).
    pub(super) fn scales(&mut self, n: usize) -> Vec<u8> {
        self.packed(n)
            .into_iter()
            .map(|b| 0x30 | (b & 0x0F))
            .collect()
    }
}

pub(super) fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

pub(super) fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

/// One expert projection: transposed `[K/2, N]` packed + `[K/16, N]` scale
/// pointer tables over all `EXPERTS` (every expert local) and scale2 values.
pub(super) struct Proj {
    pub(super) packed: DevicePtr,
    pub(super) scales: DevicePtr,
    pub(super) scale2: DevicePtr,
    pub(super) owned: Vec<DevicePtr>,
}

impl Proj {
    pub(super) fn new(
        g: &dyn GpuBackend,
        rng: &mut Rng,
        n: usize,
        k: usize,
        scale2: f32,
    ) -> Result<Self> {
        let (mut pt, mut st, mut owned) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..EXPERTS {
            let (w, s) = (
                up(g, &rng.packed(n * k / 2))?,
                up(g, &rng.scales(n * k / 16))?,
            );
            pt.push(w.0);
            st.push(s.0);
            owned.extend([w, s]);
        }
        let (packed, scales) = (
            up(g, &le(&pt, u64::to_le_bytes))?,
            up(g, &le(&st, u64::to_le_bytes))?,
        );
        let scale2 = up(g, &le(&vec![scale2; EXPERTS], f32::to_le_bytes))?;
        owned.extend([packed, scales, scale2]);
        Ok(Self {
            packed,
            scales,
            scale2,
            owned,
        })
    }
    pub(super) fn table(&self) -> [DevicePtr; 3] {
        [self.packed, self.scales, self.scale2]
    }
}

/// Top-8 routes of `t` tokens over exactly the `active` experts (each token
/// picks 8 distinct experts; the experts are dealt from reshuffled decks, so
/// every active expert gets a row once `t * 8 >= active + 8`). Returns
/// (`expert_offsets[E + 1]`, `sorted_token_ids`) as `moe_sort_by_expert`.
pub(super) fn routing(rng: &mut Rng, t: usize, active: &[usize]) -> (Vec<i32>, Vec<i32>) {
    let mut deck: Vec<usize> = Vec::new();
    let mut routes = Vec::with_capacity(t * TOP_K);
    for _ in 0..t {
        let (mut chosen, mut skipped) = (Vec::with_capacity(TOP_K), Vec::new());
        while chosen.len() < TOP_K {
            if deck.is_empty() {
                deck = active.to_vec();
                rng.shuffle(&mut deck);
            }
            let e = deck.pop().unwrap();
            if chosen.contains(&e) {
                skipped.push(e)
            } else {
                chosen.push(e)
            }
        }
        deck.extend(skipped);
        routes.extend(chosen);
    }
    let mut offsets = vec![0i32; EXPERTS + 1];
    let mut sorted = Vec::with_capacity(routes.len());
    for e in 0..EXPERTS {
        for (i, _) in routes.iter().enumerate().filter(|(_, r)| **r == e) {
            sorted.push((i / TOP_K) as i32);
        }
        offsets[e + 1] = sorted.len() as i32;
    }
    (offsets, sorted)
}
