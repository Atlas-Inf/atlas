// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`; a GB10 and a glm-5.3-flash kernel build):
//! `ATLAS_GLM_HC_PREFILL_FUSE2`'s `glm_hc_pre_finalize_ss_norm{,_bf16}`
//! against `glm_hc_pre_finalize_ss_vec{,_bf16}` followed by the block input
//! norm (`rms_norm_vanilla` and its `_regs` twin, 1024 threads): `hidden`,
//! post, comb and the normed rows must be the same bytes, for both highway
//! storages, at prefill row counts. Also prints both paths' time.
//!
//!   cargo test --release -p spark-model --lib hc_finalize_norm -- --ignored --nocapture

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const H: usize = 4096;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32 - 0.5
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// One finalizer's outputs: hidden, post, comb, normed.
struct Outs([DevicePtr; 4]);

impl Outs {
    fn new(g: &dyn GpuBackend, tokens: usize) -> Result<Self> {
        Ok(Self([
            g.alloc(tokens * H * 2)?,
            g.alloc(tokens * 4 * 4)?,
            g.alloc(tokens * 16 * 4)?,
            g.alloc(tokens * H * 2)?,
        ]))
    }

    fn read(&self, g: &dyn GpuBackend, tokens: usize) -> Result<Vec<Vec<u8>>> {
        let sizes = [tokens * H * 2, tokens * 16, tokens * 64, tokens * H * 2];
        self.0
            .iter()
            .zip(sizes)
            .map(|(&p, n)| down(g, p, n))
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    inputs: [DevicePtr; 5],
    outs: &Outs,
    norm_weight: Option<DevicePtr>,
    tokens: u32,
    eps: f32,
) -> Result<()> {
    let [streams, raw_mix, ss, scale, base] = inputs;
    let [y, post, comb, normed] = outs.0;
    let mut l = KernelLaunch::new(g, kernel)
        .grid([tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(raw_mix)
        .arg_ptr(ss)
        .arg_ptr(scale)
        .arg_ptr(base)
        .arg_ptr(y)
        .arg_ptr(post)
        .arg_ptr(comb);
    if let Some(w) = norm_weight {
        l = l.arg_ptr(w).arg_ptr(normed);
    }
    l.arg_u32(20).arg_f32(eps).arg_f32(1e-6).launch(0)
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn hc_finalize_norm_matches_finalize_then_rms_norm() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let m = "glm_hc_prefill_vec";
    let norms = [
        g.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
        g.kernel("glm_rms_norm_regs", "rms_norm_vanilla_regs")?,
    ];
    let eps = 1e-5f32;
    for bf16_highway in [true, false] {
        let suffix = if bf16_highway { "_bf16" } else { "" };
        let reference = g.kernel(m, &format!("glm_hc_pre_finalize_ss_vec{suffix}"))?;
        let fused = g.kernel(m, &format!("glm_hc_pre_finalize_ss_norm{suffix}"))?;
        for tokens in [1usize, 31, 2048, 5387, 8192, 16384] {
            let mut r = Lcg(tokens as u64 * 2 + bf16_highway as u64);
            let highway: Vec<f32> = (0..tokens * 4 * H).map(|_| r.f() * 4.0).collect();
            let raw_mix: Vec<f32> = (0..tokens * 24).map(|_| r.f() * 40.0).collect();
            let ss: Vec<f32> = (0..tokens).map(|_| 300.0 + r.f() * 200.0).collect();
            let weight: Vec<f32> = (0..H).map(|_| 1.0 + r.f()).collect();
            let streams = if bf16_highway {
                up(g, &bf16_bytes(&highway))?
            } else {
                up(g, &f32_bytes(&highway))?
            };
            let inputs = [
                streams,
                up(g, &f32_bytes(&raw_mix))?,
                up(g, &f32_bytes(&ss))?,
                up(g, &f32_bytes(&[0.7, 0.9, 1.1]))?,
                up(g, &f32_bytes(&(0..24).map(|_| r.f()).collect::<Vec<_>>()))?,
            ];
            let w = up(g, &bf16_bytes(&weight))?;
            let t = tokens as u32;
            let (want, got) = (Outs::new(g, tokens)?, Outs::new(g, tokens)?);
            let run_got = || finalize(g, fused, inputs, &got, Some(w), t, eps);
            run_got()?;
            let got_bytes = got.read(g, tokens)?;
            for norm in norms {
                let run_want = || -> Result<()> {
                    finalize(g, reference, inputs, &want, None, t, eps)?;
                    let [y, _, _, normed] = want.0;
                    KernelLaunch::new(g, norm)
                        .grid([t, 1, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(y)
                        .arg_ptr(w)
                        .arg_ptr(normed)
                        .arg_u32(H as u32)
                        .arg_f32(eps)
                        .launch(0)
                };
                run_want()?;
                let want_bytes = want.read(g, tokens)?;
                for (i, name) in ["hidden", "post", "comb", "normed"].iter().enumerate() {
                    ensure!(
                        want_bytes[i] == got_bytes[i],
                        "T={tokens} bf16={bf16_highway}: {name} differs"
                    );
                }
                let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                    f()?;
                    g.synchronize(0)?;
                    let t0 = std::time::Instant::now();
                    for _ in 0..20 {
                        f()?;
                    }
                    g.synchronize(0)?;
                    Ok(t0.elapsed().as_secs_f64() / 20.0 * 1e6)
                };
                println!(
                    "T={tokens} bf16={bf16_highway}: bitwise; finalize+norm {:7.1}us fused {:7.1}us",
                    time(&run_want)?,
                    time(&run_got)?
                );
            }
            for p in inputs.into_iter().chain([w]).chain(want.0).chain(got.0) {
                g.free(p)?;
            }
        }
    }
    Ok(())
}

/// `ATLAS_GLM_SP_GATHER_OVERLAP` runs `glm_hc_mix_ss_bf16` and
/// `glm_hc_post_mix_ss_bf16` on row blocks: each block's raw mix, sum of
/// squares and (post) updated highway must be the whole call's bytes.
#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn hc_mix_row_blocks_match_the_whole_call() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let m = "glm_hc_prefill_vec";
    let (tokens, cuts) = (4096usize, [0usize, 1000, 2731, 4096]);
    let mut r = Lcg(0x5eed);
    let highway: Vec<f32> = (0..tokens * 4 * H).map(|_| r.f() * 4.0).collect();
    let block: Vec<f32> = (0..tokens * H).map(|_| r.f() * 2.0).collect();
    let post: Vec<f32> = (0..tokens * 4).map(|_| r.f() + 1.0).collect();
    let comb: Vec<f32> = (0..tokens * 16).map(|_| r.f() * 0.5 + 0.25).collect();
    let hc_fn: Vec<f32> = (0..24 * 4 * H).map(|_| r.f() * 0.02).collect();
    let (blk, pst, cmb, hfn) = (
        up(g, &bf16_bytes(&block))?,
        up(g, &f32_bytes(&post))?,
        up(g, &f32_bytes(&comb))?,
        up(g, &f32_bytes(&hc_fn))?,
    );
    let hw_row = 4 * H * 2;
    for with_post in [false, true] {
        let k = g.kernel(
            m,
            if with_post {
                "glm_hc_post_mix_ss_bf16"
            } else {
                "glm_hc_mix_ss_bf16"
            },
        )?;
        let launch = |hw: DevicePtr, scratch: DevicePtr, row0: usize, rows: usize| {
            let l = KernelLaunch::new(g, k)
                .grid([(rows as u32).div_ceil(32), 1, 1])
                .block([256, 1, 1]);
            let l = if with_post {
                l.arg_ptr(blk.offset(row0 * H * 2))
                    .arg_ptr(hw.offset(row0 * hw_row))
                    .arg_ptr(pst.offset(row0 * 16))
                    .arg_ptr(cmb.offset(row0 * 64))
            } else {
                l.arg_ptr(hw.offset(row0 * hw_row))
            };
            l.arg_ptr(hfn)
                .arg_ptr(scratch)
                .arg_ptr(scratch.offset(rows * 24 * 4))
                .arg_u32(rows as u32)
                .launch(0)
        };
        let (hw_whole, hw_blocks) = (up(g, &bf16_bytes(&highway))?, up(g, &bf16_bytes(&highway))?);
        let whole = g.alloc(tokens * 25 * 4)?;
        launch(hw_whole, whole, 0, tokens)?;
        let mix_whole = down(g, whole, tokens * 25 * 4)?;
        let scratch = g.alloc(tokens * 25 * 4)?;
        for w in cuts.windows(2) {
            let (row0, rows) = (w[0], w[1] - w[0]);
            launch(hw_blocks, scratch, row0, rows)?;
            let got = down(g, scratch, rows * 25 * 4)?;
            ensure!(
                got[..rows * 96] == mix_whole[row0 * 96..(row0 + rows) * 96],
                "post={with_post}: raw mix of rows {row0}..+{rows} differs"
            );
            let ss = tokens * 96;
            ensure!(
                got[rows * 96..] == mix_whole[ss + row0 * 4..ss + (row0 + rows) * 4],
                "post={with_post}: sum of squares of rows {row0}..+{rows} differs"
            );
        }
        ensure!(
            down(g, hw_whole, tokens * hw_row)? == down(g, hw_blocks, tokens * hw_row)?,
            "post={with_post}: updated highway differs"
        );
        println!("post={with_post}: row blocks bitwise");
        for p in [hw_whole, hw_blocks, whole, scratch] {
            g.free(p)?;
        }
    }
    Ok(())
}
