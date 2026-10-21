// SPDX-License-Identifier: AGPL-3.0-only

//! Decode tensors of a ModelOpt FP4-Local checkpoint with
//! `atlas_core::modelopt_codec` and write the raw result, so an independent
//! reference can check it byte for byte.
//!
//! ```text
//! fp4local_decode <snapshot> csf   <scale-name>       <out.bin>   # E4M3 bytes
//! fp4local_decode <snapshot> mxfp8 <module>           <out.bin>   # BF16 bytes
//! fp4local_decode <snapshot> ple   <shard-base> <row> <out.bin>   # BF16 row
//! ```

use anyhow::{Context, Result, bail};
use atlas_core::modelopt_codec::{decode_csf_scales, mxfp8_to_bf16, nvfp4_rows_to_bf16};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

struct Tensor {
    file: std::path::PathBuf,
    offset: u64,
    len: usize,
    shape: Vec<usize>,
}

fn locate(snap: &Path, name: &str) -> Result<Tensor> {
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(snap.join("model.safetensors.index.json"))?)?;
    let shard = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name} is not in the index"))?;
    let file = snap.join(shard);
    let mut f = std::fs::File::open(&file)?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let hlen = u64::from_le_bytes(n);
    let mut header = vec![0u8; hlen as usize];
    f.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    let t = &header[name];
    let offs = t["data_offsets"].as_array().context("data_offsets")?;
    let (a, b) = (offs[0].as_u64().unwrap(), offs[1].as_u64().unwrap());
    let shape = t["shape"]
        .as_array()
        .context("shape")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    Ok(Tensor {
        file,
        offset: 8 + hlen + a,
        len: (b - a) as usize,
        shape,
    })
}

fn read(t: &Tensor, start: usize, len: usize) -> Result<Vec<u8>> {
    let mut f = std::fs::File::open(&t.file)?;
    f.seek(SeekFrom::Start(t.offset + start as u64))?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snap = Path::new(args.get(1).context("snapshot dir")?);
    let out = match args.get(2).map(String::as_str) {
        Some("csf") => {
            let name = &args[3];
            let fixed = locate(snap, &format!("{name}.nvfp4_csf_fixed"))?;
            let exc = locate(snap, &format!("{name}.nvfp4_csf_exceptions"))?;
            decode_csf_scales(
                &read(&fixed, 0, fixed.len)?,
                fixed.shape[0],
                fixed.shape[1],
                &read(&exc, 0, exc.len)?,
            )?
        }
        Some("mxfp8") => {
            let m = &args[3];
            let w = locate(snap, &format!("{m}.weight"))?;
            let s = locate(snap, &format!("{m}.weight_scale"))?;
            mxfp8_to_bf16(
                &read(&w, 0, w.len)?,
                &read(&s, 0, s.len)?,
                w.shape[0],
                w.shape[1],
            )?
        }
        Some("ple") => {
            let (base, row): (&str, usize) = (&args[3], args[4].parse()?);
            let w = locate(snap, &format!("{base}.weight"))?;
            let s = locate(snap, &format!("{base}.weight_scale"))?;
            let table = base.rsplit_once(".shard_").context("shard base")?.0;
            let g = locate(snap, &format!("{table}.weight_scale_2"))?;
            let g = read(&g, 0, 4)?;
            let global = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
            let (pw, sw) = (w.shape[1], s.shape[1]);
            let dim = pw * 2;
            let mut dst = vec![0u8; dim * 2];
            nvfp4_rows_to_bf16(
                &read(&w, row * pw, pw)?,
                &read(&s, row * sw, sw)?,
                global,
                dim,
                dim / sw,
                &mut dst,
            )?;
            dst
        }
        _ => bail!("mode must be csf | mxfp8 | ple"),
    };
    std::fs::write(args.last().unwrap(), &out)?;
    println!("wrote {} bytes", out.len());
    Ok(())
}
