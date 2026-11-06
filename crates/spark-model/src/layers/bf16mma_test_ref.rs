// SPDX-License-Identifier: AGPL-3.0-only

//! Test support for the BF16-MMA prefill GEMM GPU tests (MoE and dense):
//! a deterministic RNG, BF16 / e2m1 / e4m3 decode in FP64, device upload
//! and readback, and the tolerance-scaled error measure.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    /// An e4m3 block scale: mostly 2^-4..2^3, some zero, some subnormal.
    pub(crate) fn scale(&mut self) -> u8 {
        match self.next() % 50 {
            0 => 0,
            1 => 1 + (self.next() % 7) as u8,
            _ => (((3 + self.next() % 8) << 3) | (self.next() % 8)) as u8,
        }
    }
}

pub(crate) fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

pub(crate) fn f32_of(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub(crate) fn e2m1(code: u8) -> f64 {
    const MAG: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

pub(crate) fn e4m3(b: u8) -> f64 {
    let (e, m) = (((b >> 3) & 15) as i32, (b & 7) as f64);
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

pub(crate) fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    // SAFETY: plain integer/float slices, read as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
        .to_vec()
}

pub(crate) fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

pub(crate) fn read_u16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Max of |got - want| / (rel * |want| + abs * rms(want row)) over a row.
pub(crate) fn worst(got: &[u16], want: &[f64], rel: f64, abs: f64) -> f64 {
    let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f32_of(g) as f64 - w).abs() / (rel * w.abs() + abs * rms + 1e-30))
        .fold(0.0, f64::max)
}
