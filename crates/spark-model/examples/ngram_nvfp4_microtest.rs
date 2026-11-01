// SPDX-License-Identifier: AGPL-3.0-only
//! `batched_embed_nvfp4` vs the host oracle `nvfp4_rows_to_bf16`, bit-identical.
//! (a) all 256 scale codes x 16 nibble pairs x real global; (b) real FP4-Local
//! shard rows through the segmented scale path. Run:
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example ngram_nvfp4_microtest [-- /path/to/snapshot]

use anyhow::Result;
use atlas_core::modelopt_codec::nvfp4_rows_to_bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const DIM: usize = 160;
const GROUP: usize = 16;
const GLOBAL: f32 = 3.324236e-5;

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

fn gather(
    g: &dyn GpuBackend,
    k: spark_runtime::gpu::KernelHandle,
    slots: &DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    n: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(*slots)
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_f32(GLOBAL)
        .arg_ptr(out)
        .arg_u32(DIM as u32)
        .arg_u32(GROUP as u32)
        .arg_u32((DIM / 2) as u32)
        .arg_u32((DIM / GROUP) as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k = gpu.kernel("embed_from_argmax", "batched_embed_nvfp4")?;

    // (a) exhaustive scale-code coverage: 256 scale codes x 256 packed bytes.
    let n = 256;
    let mut packed = vec![0u8; n * DIM / 2];
    let mut scales = vec![0u8; n * DIM / GROUP];
    for r in 0..n {
        for i in 0..DIM / 2 {
            packed[r * DIM / 2 + i] = ((i + r) & 0xFF) as u8;
        }
        for i in 0..DIM / GROUP {
            scales[r * DIM / GROUP + i] = r as u8;
        }
    }
    let slots_host: Vec<u8> = (0..n as u32).flat_map(|s| s.to_le_bytes()).collect();
    let (pd, sd, out) = (
        up(&gpu, &packed)?,
        up(&gpu, &scales)?,
        gpu.alloc(n * DIM * 2)?,
    );
    let slots = up(&gpu, &slots_host)?;
    gather(&gpu, k, &slots, pd, sd, out, n)?;
    gpu.synchronize(0)?;
    let got = down(&gpu, out, n * DIM * 2)?;
    let mut want = vec![0u8; n * DIM * 2];
    for r in 0..n {
        nvfp4_rows_to_bf16(
            &packed[r * DIM / 2..(r + 1) * DIM / 2],
            &scales[r * DIM / GROUP..(r + 1) * DIM / GROUP],
            GLOBAL,
            DIM,
            GROUP,
            &mut want[r * DIM * 2..(r + 1) * DIM * 2],
        )?;
    }
    anyhow::ensure!(got == want, "synthetic sweep: outputs differ");
    println!("synthetic sweep (256 scale codes x all bytes): IDENTICAL");

    // (b) real rows: read FP4-Local shard_0 weight+scale through the cache.
    if let Some(snap) = std::env::args().nth(1) {
        use std::io::Read;
        let find = |name: &str| -> Result<(std::path::PathBuf, u64, usize)> {
            for e in std::fs::read_dir(std::path::Path::new(&snap).join("tensors"))? {
                let p = e?.path();
                if p.extension().map(|x| x == "safetensors").unwrap_or(false) {
                    let b = std::fs::read(&p)?;
                    let nlen = u64::from_le_bytes(b[..8].try_into()?) as usize;
                    let hdr: serde_json::Value = serde_json::from_slice(&b[8..8 + nlen])?;
                    if let Some(t) = hdr.get(name) {
                        let off: Vec<u64> = serde_json::from_value(t["data_offsets"].clone())?;
                        return Ok((p, 8 + nlen as u64 + off[0], (off[1] - off[0]) as usize));
                    }
                }
            }
            anyhow::bail!("{name} not found")
        };
        let name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
        let sname = format!("{name}_scale");
        let (wp, woff, wlen) = find(name)?;
        let (sp, soff, slen) = find(&sname)?;
        let rows = wlen / (DIM / 2);
        anyhow::ensure!(slen == rows * DIM / GROUP, "scale rows mismatch");
        let n2 = rows.min(4096);
        let mut cache = spark_storage::NgramRowCache::open_segmented(
            &vec![(wp.clone(), woff)],
            rows as u64,
            None,
            DIM / 2,
            n2 + 16,
        )?;
        cache.attach_segmented_block_scales(&vec![(sp.clone(), soff)], DIM / GROUP)?;
        let _ = &mut cache;
        let ids: Vec<u64> = (0..n2 as u64).map(|i| i * 7919 % rows as u64).collect();
        let mut slots_v = Vec::new();
        cache.resolve(&ids, &mut slots_v)?;
        let sbytes: Vec<u8> = slots_v.iter().flat_map(|s| s.to_le_bytes()).collect();
        let slots = up(&gpu, &sbytes)?;
        let out2 = gpu.alloc(n2 * DIM * 2)?;
        gather(
            &gpu,
            k,
            &slots,
            DevicePtr(cache.table_dev_va()?),
            DevicePtr(cache.block_scale_dev_va()?.unwrap()),
            out2,
            n2,
        )?;
        gpu.synchronize(0)?;
        let got2 = down(&gpu, out2, n2 * DIM * 2)?;
        let mut want2 = vec![0u8; n2 * DIM * 2];
        for (i, &id) in ids.iter().enumerate() {
            let mut f = std::fs::File::open(&wp)?;
            let mut pb = vec![0u8; DIM / 2];
            use std::io::Seek;
            f.seek(std::io::SeekFrom::Start(woff + id * (DIM / 2) as u64))?;
            f.read_exact(&mut pb)?;
            let mut sf = std::fs::File::open(&sp)?;
            let mut sb = vec![0u8; DIM / GROUP];
            sf.seek(std::io::SeekFrom::Start(soff + id * (DIM / GROUP) as u64))?;
            sf.read_exact(&mut sb)?;
            nvfp4_rows_to_bf16(
                &pb,
                &sb,
                GLOBAL,
                DIM,
                GROUP,
                &mut want2[i * DIM * 2..(i + 1) * DIM * 2],
            )?;
        }
        anyhow::ensure!(got2 == want2, "real rows: outputs differ");
        println!("{n2} real FP4-Local rows through the segmented cache: IDENTICAL");
    }
    println!("PASS");
    Ok(())
}
