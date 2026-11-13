// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side helpers of `skinny_mma_microtest`: the RNG, FP8/FP4 decoders,
//! H2D/D2H shims and the error metrics.

use anyhow::Result;
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(super) struct Lcg(pub(super) u64);
impl Lcg {
    pub(super) fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    pub(super) fn byte(&mut self) -> u8 {
        (self.f() * 256.0) as u8
    }
    /// Roughly normal, with a heavy tail (one value in 64 up to ~+-200).
    pub(super) fn act(&mut self) -> f32 {
        let n = (0..4).map(|_| self.f()).sum::<f64>() - 2.0;
        let big = self.f() < 1.0 / 64.0;
        (n * if big { 120.0 } else { 1.2 }) as f32
    }
}

pub(super) fn e4m3(b: u8) -> f64 {
    let (s, e, m) = (b >> 7, (b >> 3) & 0xF, b & 7);
    let v = if e == 0 {
        m as f64 / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}
pub(super) const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
pub(super) fn e2m1(code: u8) -> f64 {
    let v = E2M1[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

pub(super) fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
pub(super) fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
pub(super) fn a_bytes(a: &[u16]) -> Vec<u8> {
    a.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// (RMS error / output RMS, max error / output RMS, share not correctly rounded).
pub(super) fn errors(got: &[u8], want: &[f64]) -> (f64, f64, f64) {
    let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
    let (mut se, mut mx, mut off) = (0f64, 0f64, 0usize);
    for (b, &w) in got.chunks_exact(2).zip(want) {
        let v = bf16::from_bits(u16::from_le_bytes([b[0], b[1]]));
        let e = (v.to_f64() - w).abs();
        se += e * e;
        mx = mx.max(e);
        off += usize::from(v != bf16::from_f64(w));
    }
    (
        (se / want.len() as f64).sqrt() / rms,
        mx / rms,
        off as f64 / want.len() as f64,
    )
}
