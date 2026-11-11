// SPDX-License-Identifier: AGPL-3.0-only
//! Vision-tower native weights: `mxfp8_proj_dequant` and
//! `dequant_nvfp4_to_bf16` must reproduce the host expansion bit-for-bit.
//! Run:
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example vit_dq_microtest /path/to/FP4-Local/snapshot

use anyhow::Result;
use atlas_core::modelopt_codec::{mxfp8_to_bf16, nvfp4_rows_to_bf16};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::collections::HashMap;

struct Store {
    headers: HashMap<String, (std::path::PathBuf, u64)>, // name -> (file, data_off)
    shapes: HashMap<String, Vec<u64>>,
}

impl Store {
    fn open(dir: &str) -> Result<Self> {
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
            "{dir}/model.safetensors.index.json"
        ))?)?;
        let mut headers = HashMap::new();
        let mut shapes = HashMap::new();
        for e in std::fs::read_dir(format!("{dir}/tensors"))? {
            let p = e?.path();
            if p.extension().map(|x| x == "safetensors").unwrap_or(false) {
                let b = std::fs::read(&p)?;
                let nlen = u64::from_le_bytes(b[..8].try_into()?) as usize;
                let hdr: serde_json::Value = serde_json::from_slice(&b[8..8 + nlen])?;
                for (name, t) in hdr.as_object().unwrap() {
                    if name == "__metadata__" {
                        continue;
                    }
                    let off: Vec<u64> = serde_json::from_value(t["data_offsets"].clone())?;
                    let shape: Vec<u64> = serde_json::from_value(t["shape"].clone())?;
                    headers.insert(name.clone(), (p.clone(), 8 + nlen as u64 + off[0]));
                    shapes.insert(name.clone(), shape);
                }
            }
        }
        let _ = idx;
        Ok(Self { headers, shapes })
    }
    fn shape(&self, name: &str) -> Vec<u64> {
        self.shapes[name].clone()
    }
    fn bytes(&self, name: &str, len: usize) -> Result<Vec<u8>> {
        let (p, off) = &self.headers[name];
        let mut f = std::fs::File::open(p)?;
        use std::io::{Read, Seek};
        f.seek(std::io::SeekFrom::Start(*off))?;
        let mut b = vec![0u8; len];
        f.read_exact(&mut b)?;
        Ok(b)
    }
}

fn up(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn main() -> Result<()> {
    let snap = std::env::args().nth(1).expect("snapshot dir");
    let st = Store::open(&snap)?;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k8 = gpu.kernel("mxfp8_proj", "mxfp8_proj_dequant")?;
    let k4 = gpu.kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16")?;

    let mut checked = 0usize;
    for i in 0..27usize {
        for proj in ["attn.qkv", "attn.proj"] {
            let base = format!("model.visual.blocks.{i}.{proj}");
            let shp = st.shape(&format!("{base}.weight"));
            let (n, k) = (shp[0] as u32, shp[1] as u32);
            let wb = st.bytes(&format!("{base}.weight"), (n * k) as usize)?;
            let sb = st.bytes(&format!("{base}.weight_scale"), (n * k / 32) as usize)?;
            let out = gpu.alloc((n * k * 2) as usize)?;
            KernelLaunch::new(&gpu, k8)
                .grid([((n as u64 * (k as u64 / 16)).div_ceil(256)) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(up(&gpu, &wb)?)
                .arg_ptr(up(&gpu, &sb)?)
                .arg_ptr(out)
                .arg_u32(n)
                .arg_u32(k)
                .launch(0)?;
            gpu.synchronize(0)?;
            let got = down(&gpu, out, (n * k * 2) as usize)?;
            let want = mxfp8_to_bf16(&wb, &sb, n as usize, k as usize)?;
            anyhow::ensure!(got == want, "MXFP8 {base} differs");
            checked += 1;
        }
    }
    println!("{checked} MXFP8 projections (all 27 blocks): IDENTICAL");

    let mut nv = 0;
    for i in 0..27usize {
        for proj in ["mlp.linear_fc1", "mlp.linear_fc2"] {
            let base = format!("model.visual.blocks.{i}.{proj}");
            let shp = st.shape(&format!("{base}.weight"));
            let (n, k) = (shp[0] as u32, (shp[1] * 2) as u32);
            let wb = st.bytes(&format!("{base}.weight"), (n * k / 2) as usize)?;
            let sb = st.bytes(&format!("{base}.weight_scale"), (n * k / 16) as usize)?;
            let g = st.bytes(&format!("{base}.weight_scale_2"), 4)?;
            let global = f32::from_le_bytes(g[..4].try_into()?);
            let out = gpu.alloc((n * k * 2) as usize)?;
            KernelLaunch::new(&gpu, k4)
                .grid([n, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(up(&gpu, &wb)?)
                .arg_ptr(up(&gpu, &sb)?)
                .arg_ptr(out)
                .arg_f32(global)
                .arg_u32(n)
                .arg_u32(k)
                .launch(0)?;
            gpu.synchronize(0)?;
            let got = down(&gpu, out, (n * k * 2) as usize)?;
            let mut want = vec![0u8; (n * k * 2) as usize];
            nvfp4_rows_to_bf16(&wb, &sb, global, k as usize, 16, &mut want)?;
            anyhow::ensure!(got == want, "NVFP4 {base} differs");
            nv += 1;
        }
    }
    for proj in ["linear_fc1", "linear_fc2"] {
        let base = format!("model.visual.merger.{proj}");
        let shp = st.shape(&format!("{base}.weight"));
        let (n, k) = (shp[0] as u32, (shp[1] * 2) as u32);
        let wb = st.bytes(&format!("{base}.weight"), (n * k / 2) as usize)?;
        let sb = st.bytes(&format!("{base}.weight_scale"), (n * k / 16) as usize)?;
        let g = st.bytes(&format!("{base}.weight_scale_2"), 4)?;
        let global = f32::from_le_bytes(g[..4].try_into()?);
        let out = gpu.alloc((n * k * 2) as usize)?;
        KernelLaunch::new(&gpu, k4)
            .grid([n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(up(&gpu, &wb)?)
            .arg_ptr(up(&gpu, &sb)?)
            .arg_ptr(out)
            .arg_f32(global)
            .arg_u32(n)
            .arg_u32(k)
            .launch(0)?;
        gpu.synchronize(0)?;
        let got = down(&gpu, out, (n * k * 2) as usize)?;
        let mut want = vec![0u8; (n * k * 2) as usize];
        nvfp4_rows_to_bf16(&wb, &sb, global, k as usize, 16, &mut want)?;
        anyhow::ensure!(got == want, "NVFP4 {base} differs");
        nv += 1;
    }
    println!("{nv} NVFP4 projections (all 27 blocks + merger): IDENTICAL");
    println!("PASS");
    Ok(())
}
