// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`; a GB10 and a glm-5.3-flash kernel build):
//! `ATLAS_GLM_KDA_CONV_PLANES`'s planar prefill convolution against
//! `kda_pack_qkv` + the packed convolution, at the TP2 width (p = 32 x 128)
//! and prefill row counts: the convolved rows and the outgoing conv state must
//! be the same bytes.
//!
//!   cargo test --release -p spark-model --lib kda_conv_planes -- --ignored --nocapture

use super::*;
use spark_runtime::cuda_backend::AtlasCudaBackend;

fn random(count: usize, seed: &mut u64, f32_words: bool) -> Vec<u8> {
    (0..count)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((*seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 2.0;
            if f32_words {
                v.to_le_bytes().to_vec()
            } else {
                (((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16)
                    .to_le_bytes()
                    .to_vec()
            }
        })
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(g.default_stream())?;
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn kda_conv_planes_match_pack_then_conv() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let kernels = [
        g.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
        g.kernel("causal_conv1d", "causal_conv1d_update_prefill_tp")?,
    ];
    let pack = g.kernel("kda", "kda_pack_qkv")?;
    let stream = g.default_stream();
    let (p, d_conv) = (32u32 * 128, 4u32);
    let width = 3 * p as usize;
    let mut seed = 0xc0de_c0afu64;
    let weight = DenseWeight {
        weight: upload(g, &random(width * d_conv as usize, &mut seed, false))?,
    };
    let state0 = random(width * d_conv as usize, &mut seed, true);
    for m in [1u32, 31, 2048, 5387, 8192, 16384] {
        let planes = upload(g, &random(m as usize * width, &mut seed, false))?;
        let rows = m as usize * width * 2;
        let (packed, want, got) = (g.alloc(rows)?, g.alloc(rows)?, g.alloc(rows)?);
        let (state_want, state_got) = (upload(g, &state0)?, upload(g, &state0)?);
        let geometry = [m, p, d_conv];
        kda_pack_qkv(g, pack, planes, packed, m, p, stream)?;
        kda_conv_prefill(
            g,
            kernels,
            state_want,
            (packed, false),
            &weight,
            want,
            geometry,
            stream,
        )?;
        kda_conv_prefill(
            g,
            kernels,
            state_got,
            (planes, true),
            &weight,
            got,
            geometry,
            stream,
        )?;
        assert_eq!(
            read(g, want, rows)?,
            read(g, got, rows)?,
            "m={m}: convolved"
        );
        let state_bytes = state0.len();
        assert_eq!(
            read(g, state_want, state_bytes)?,
            read(g, state_got, state_bytes)?,
            "m={m}: conv state"
        );
        println!("m={m}: convolved rows and conv state identical");
        for ptr in [planes, packed, want, got, state_want, state_got] {
            g.free(ptr)?;
        }
    }
    g.free(weight.weight)?;
    Ok(())
}
