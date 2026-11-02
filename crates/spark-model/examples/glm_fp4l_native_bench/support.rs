// SPDX-License-Identifier: AGPL-3.0-only
//! Weights, timing and print helpers of `glm_fp4l_native_bench`.

use super::*;

pub(super) struct Rng(pub(super) u64);
impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub(super) fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// An E4M3 byte without the NaN codes.
    pub(super) fn e4m3(&mut self) -> u8 {
        let b = (self.next() >> 32) as u8;
        if b & 0x7F == 0x7F { b ^ 1 } else { b }
    }
}

pub(super) fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

pub(super) fn fetch(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

pub(super) fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

pub(super) fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// One weight in every format a group uses, `copies` times (cold cycling).
pub(super) struct Weight {
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) mx: Vec<Mxfp8Weight>,
    pub(super) fp8b: Vec<Fp8BlockWeight>,
    pub(super) nv: Vec<(DevicePtr, DevicePtr)>,
    /// Host bytes of copy 0 (E4M3 values, f32 grid) for the reference.
    pub(super) host: (Vec<u8>, Vec<f32>),
}

pub(super) const NV_SCALE2: f32 = 1.0 / 64.0;

impl Weight {
    pub(super) fn new(g: &dyn GpuBackend, rng: &mut Rng, n: usize, k: usize) -> Result<Self> {
        let values: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
        let grid: Vec<f32> = (0..(n / 128) * (k / 128))
            .map(|_| 0.01 + 0.01 * rng.unit().abs())
            .collect();
        let grid_bytes: Vec<u8> = grid.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mx_scales: Vec<u8> = (0..n * k / 32)
            .map(|_| 118 + (rng.next() % 4) as u8)
            .collect();
        let nv_packed: Vec<u8> = (0..n * k / 2).map(|_| (rng.next() >> 24) as u8).collect();
        let nv_scales: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x30 + (rng.next() % 16) as u8)
            .collect();
        let copies = (COLD_BYTES / (n * k)).clamp(2, 64);
        let mut w = Self {
            n,
            k,
            mx: Vec::new(),
            fp8b: Vec::new(),
            nv: Vec::new(),
            host: (values.clone(), grid),
        };
        for _ in 0..copies {
            let data = up(g, &values)?;
            w.mx.push(Mxfp8Weight {
                data,
                scales: up(g, &mx_scales)?,
                tiled: false,
            });
            w.fp8b.push(Fp8BlockWeight {
                data,
                scales: up(g, &grid_bytes)?,
                tiled: false,
            });
            w.nv.push((up(g, &nv_packed)?, up(g, &nv_scales)?));
        }
        Ok(w)
    }

    pub(super) fn mx_bytes(&self) -> usize {
        self.n * self.k * 33 / 32
    }
    pub(super) fn fp8b_bytes(&self) -> usize {
        self.n * self.k + (self.n / 128) * (self.k / 128) * 4
    }
    pub(super) fn nv_bytes(&self) -> usize {
        self.n * self.k * 9 / 16
    }
}

/// The canonical NVFP4 tensor-core tier (`w4a16_gemv_tc8`, 1..32 rows).
pub(super) fn nv_tc(
    g: &dyn GpuBackend,
    k8: KernelHandle,
    (w, s): (DevicePtr, DevicePtr),
    x: DevicePtr,
    y: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, k8)
        .grid([div_ceil(n as u32, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_f32(NV_SCALE2)
        .arg_ptr(y)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)
}

/// `w4a16_gemv_tc8_ld` on columns `[col0, col0 + k)` of a `[n, ld]` NVFP4 weight.
#[allow(clippy::too_many_arguments)]
pub(super) fn nv_tc_ld(
    g: &dyn GpuBackend,
    k8ld: KernelHandle,
    (w, s): (DevicePtr, DevicePtr),
    x: DevicePtr,
    y: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ld: usize,
    col0: usize,
) -> Result<()> {
    KernelLaunch::new(g, k8ld)
        .grid([div_ceil(n as u32, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w.offset(col0 / 2))
        .arg_ptr(s.offset(col0 / 16))
        .arg_f32(NV_SCALE2)
        .arg_ptr(y)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32((ld / 2) as u32)
        .arg_u32((ld / 16) as u32)
        .launch(0)
}

/// Best-of-`reps` microseconds per call of `f(copy)` over `copies` copies.
pub(super) fn time(
    g: &dyn GpuBackend,
    reps: usize,
    copies: usize,
    f: &dyn Fn(usize) -> Result<()>,
) -> Result<f64> {
    for c in 0..copies.min(4) {
        f(c)?;
    }
    g.synchronize(0)?;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        for i in 0..ITERS {
            f(i % copies)?;
        }
        g.synchronize(0)?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
    }
    Ok(best)
}

pub(super) fn line(label: &str, m: usize, us: f64, bytes: usize) {
    println!(
        "  {label:<34} M={m:<2} {us:>8.1} us  {:>7.1} MB  {:>6.1} GB/s",
        bytes as f64 / 1e6,
        bytes as f64 / us / 1e3
    );
}

/// E4M3 byte to f64 (no NaN codes are generated).
pub(super) fn e4m3(b: u8) -> f64 {
    let (s, e, m) = ((b >> 7) & 1, ((b >> 3) & 0xF) as i32, (b & 7) as f64);
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if s == 1 { -v } else { v }
}
