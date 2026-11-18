// SPDX-License-Identifier: AGPL-3.0-only

/// NumPy SeedSequence(pool_size=4) -> PCG64 XSL-RR 128/64. This is distinct
/// from the SplitMix draws used by Qwen PLE.
pub(super) struct Pcg64 {
    state: u128,
    increment: u128,
    half: Option<u32>,
}
fn mix(x: u32, y: u32) -> u32 {
    let v = 0xca01f9ddu32
        .wrapping_mul(x)
        .wrapping_sub(0x4973f715u32.wrapping_mul(y));
    v ^ (v >> 16)
}
fn hash(v: u32, c: &mut u32) -> u32 {
    let v = v ^ *c;
    *c = c.wrapping_mul(0x931e8875);
    let v = v.wrapping_mul(*c);
    v ^ (v >> 16)
}
pub(super) fn seed_sequence(seed: u64) -> [u64; 4] {
    let entropy = [seed as u32, (seed >> 32) as u32];
    let n = if entropy[1] == 0 { 1 } else { 2 };
    let mut c = 0x43b0d7e5;
    let mut pool = [0; 4];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hash(if i < n { entropy[i] } else { 0 }, &mut c);
    }
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                pool[j] = mix(pool[j], hash(pool[i], &mut c));
            }
        }
    }
    let mut out = [0u64; 4];
    let mut c = 0x8b51f9ddu32;
    for i in 0..8 {
        let v = pool[i % 4] ^ c;
        c = c.wrapping_mul(0x58f38ded);
        let v = v.wrapping_mul(c);
        let v = v ^ (v >> 16);
        out[i / 2] |= (v as u64) << ((i % 2) * 32);
    }
    out
}
impl Pcg64 {
    pub fn new(seed: u64) -> Self {
        let s = seed_sequence(seed);
        let mut g = Self {
            state: 0,
            increment: (((s[2] as u128) << 64 | s[3] as u128) << 1) | 1,
            half: None,
        };
        g.step();
        g.state = g.state.wrapping_add((s[0] as u128) << 64 | s[1] as u128);
        g.step();
        g
    }
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(0x2360ed051fc65da44385df649fccf645)
            .wrapping_add(self.increment);
    }
    pub fn raw(&mut self) -> u64 {
        self.step();
        ((self.state as u64) ^ ((self.state >> 64) as u64)).rotate_right((self.state >> 122) as u32)
    }
    fn raw32(&mut self) -> u32 {
        if let Some(v) = self.half.take() {
            v
        } else {
            let x = self.raw();
            self.half = Some((x >> 32) as u32);
            x as u32
        }
    }
    pub fn bounded(&mut self, b: u64) -> u64 {
        if b == 1 {
            return 0;
        }
        if b <= u32::MAX as u64 {
            let b = b as u32;
            let threshold = b.wrapping_neg() % b;
            loop {
                let p = (self.raw32() as u64) * (b as u64);
                if p as u32 >= threshold {
                    return p >> 32;
                }
            }
        }
        let threshold = b.wrapping_neg() % b;
        loop {
            let p = (self.raw() as u128) * (b as u128);
            if p as u64 >= threshold {
                return (p >> 64) as u64;
            }
        }
    }
}
