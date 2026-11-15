// SPDX-License-Identifier: AGPL-3.0-only

//! Shared by the V4.1 GPU tests: the backend on this build's kernel target,
//! host <-> device copies, and the reference recording.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::weight_map::test_checkpoint::{read_checkpoint_tensor, read_safetensors_tensor};
use crate::weight_map::{Mxfp8Weight, fp8_block_scales_to_rows};

pub(crate) fn backend() -> Result<&'static AtlasCudaBackend> {
    static GPU: std::sync::OnceLock<AtlasCudaBackend> = std::sync::OnceLock::new();
    if GPU.get().is_none() {
        let gpu =
            AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules()).context("CUDA backend")?;
        let _ = GPU.set(gpu);
    }
    let gpu = GPU.get().expect("set above");
    gpu.bind_to_thread()?;
    Ok(gpu)
}

pub(crate) fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

pub(crate) fn bf16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

pub(crate) fn le_bytes<T: Copy, const N: usize>(
    values: &[T],
    le: impl Fn(T) -> [u8; N],
) -> Vec<u8> {
    values.iter().flat_map(|&v| le(v)).collect()
}

/// `values` BF16 values at `p`, once everything queued has run.
pub(crate) fn download(gpu: &dyn GpuBackend, p: DevicePtr, values: usize) -> Result<Vec<u16>> {
    gpu.synchronize(gpu.default_stream())?;
    let mut b = vec![0u8; values * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(bf16s(&b))
}

pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// BF16 values spread over many binades, as activations are.
pub(crate) fn activations(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let r = rng.next();
            let mantissa = (r & 0x7f) as u16;
            let exponent = 112 + ((r >> 8) % 20) as u16; // 2^-15 .. 2^4
            let sign = ((r >> 16) & 1) as u16;
            sign << 15 | exponent << 7 | mantissa
        })
        .collect()
}

/// The reference recording (`sparkseek/results/2026-11-15-reference-layers`),
/// at `ATLAS_DSV41_REC`.
pub(crate) struct Recording {
    dir: PathBuf,
    pub(crate) prefill: usize,
    pub(crate) decode: usize,
}

impl Recording {
    pub(crate) fn open() -> Result<Self> {
        let dir: PathBuf = std::env::var("ATLAS_DSV41_REC")
            .context("ATLAS_DSV41_REC=<recording dir>")?
            .into();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        let count =
            |key: &str| Ok::<_, anyhow::Error>(meta[key].as_u64().context("meta")? as usize);
        Ok(Self {
            prefill: count("prefill")?,
            decode: count("decode")?,
            dir,
        })
    }

    /// `(shape, bytes)` of a tensor of run A's prefill.
    pub(crate) fn prefill(&self, name: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        self.tensor("A_prefill.safetensors", name)
    }

    /// One of run A's decode steps: steps of one shape were saved stacked,
    /// others one by one.
    pub(crate) fn step(&self, name: &str, step: usize) -> Result<(Vec<usize>, Vec<u8>)> {
        if let Ok((shape, bytes)) = self.tensor("A_decode.safetensors", name) {
            let each = bytes.len() / self.decode;
            let bytes = bytes[step * each..(step + 1) * each].to_vec();
            return Ok((shape[1..].to_vec(), bytes));
        }
        self.tensor("A_decode.safetensors", &format!("{name}.{step}"))
    }

    fn tensor(&self, file: &str, name: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        let (_, shape, bytes) = read_safetensors_tensor(&self.dir.join(file), name)?;
        Ok((shape, bytes))
    }
}

/// Rows `picks` of a recorded `[.., rows, width]` tensor.
pub(crate) fn rows_of(all: &[u8], width: usize, picks: &[usize]) -> Vec<u8> {
    picks
        .iter()
        .flat_map(|&row| all[row * width * 2..(row + 1) * width * 2].iter().copied())
        .collect()
}

/// How close `got` is to the recorded `want`: the share of BF16 values equal,
/// and the relative L2 distance.
pub(crate) fn closeness(got: &[u16], want: &[u16]) -> (f64, f64) {
    let value = |bits: u16| f64::from(crate::layers::deepseek_v41::cache_recipe::bf16_to_f32(bits));
    let same = got.iter().zip(want).filter(|(a, b)| a == b).count();
    let (mut diff, mut norm) = (0.0, 0.0);
    for (&a, &b) in got.iter().zip(want) {
        diff += (value(a) - value(b)).powi(2);
        norm += value(b).powi(2);
    }
    (same as f64 / got.len() as f64, (diff / norm).sqrt())
}

/// Prefill rows spread over the prompt.
pub(crate) fn picks(rec: &Recording, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| i * (rec.prefill - 1) / (count - 1))
        .collect()
}

pub(crate) fn checkpoint() -> Result<String> {
    std::env::var("ATLAS_DSV41_CKPT").context("ATLAS_DSV41_CKPT=<checkpoint dir>")
}

pub(crate) fn tiers(gpu: &dyn GpuBackend) -> Result<[KernelHandle; 3]> {
    Ok([
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc8")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc16")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc32")?,
    ])
}

/// An FP8 projection of the release, with its 32 x 32 block scales repeated
/// into the per-row layout the MXFP8 tiers read. `(weight, n, k)`.
pub(crate) fn load_fp8(gpu: &dyn GpuBackend, prefix: &str) -> Result<(Mxfp8Weight, usize, usize)> {
    let ckpt = checkpoint()?;
    let (dtype, shape, values) =
        read_checkpoint_tensor(Path::new(&ckpt), &format!("{prefix}.weight"))?;
    let (_, _, grid) = read_checkpoint_tensor(Path::new(&ckpt), &format!("{prefix}.scale"))?;
    anyhow::ensure!(dtype == "F8_E4M3", "{prefix}: {dtype}");
    let (n, k) = (shape[0], shape[1]);
    let w = Mxfp8Weight {
        data: upload(gpu, &values)?,
        scales: upload(gpu, &fp8_block_scales_to_rows(&grid, n, k)?)?,
        tiled: false,
    };
    Ok((w, n, k))
}
