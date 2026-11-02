// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`; a GB10 with cuBLASLt and the CUTLASS build) of
//! the packed-prefill pins (`glm_packed_scope`): under the scope a BF16
//! projection (`ops::bf16_gemm`) and the MLA head GEMM (CUTLASS grouped)
//! give each prompt's rows the same bytes whether the prompt runs alone or
//! packed with others, at the prompt lengths and pass heights a packed
//! prefill sees.
//!
//!   cargo test --release -p spark-model --lib packed_pin -- --ignored --nocapture
//!
//! End to end, `ATLAS_GLM_PACKED_PREFILL_CHECK=1` logs per-prompt hashes of
//! every layer's highway rows, the final hidden rows and the logits
//! (`PACKED_CHECK`), which `scripts/dev/packed_check_compare.py` compares
//! between a C1 and a C4/C8 run.

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

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

fn bf16(n: usize, r: &mut Lcg, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| half::bf16::from_f32(r.f() * scale).to_bits().to_le_bytes())
        .collect()
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

/// Packs of prompt lengths, each run back to back and then prompt by
/// prompt: the 98-row pack holding a 30-row prompt (the C1-vs-C4 case seen
/// on hardware), and a wide mix up to 512 rows.
const PACKS: [&[usize]; 2] = [&[30, 40, 28], &[2, 7, 33, 50, 64, 129, 300, 512]];

/// Each prompt's first row in the packed pass.
fn offsets(lens: &[usize]) -> Vec<usize> {
    lens.iter()
        .scan(0usize, |next, &len| {
            let at = *next;
            *next += len;
            Some(at)
        })
        .collect()
}

#[test]
#[ignore = "needs a GB10 with cuBLASLt and the CUTLASS build"]
fn packed_pin_bf16_gemm_rows_match_alone() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let _pin = crate::layers::glm_packed_scope::enter();
    for lens in PACKS {
        let total: usize = lens.iter().sum();
        // (N, K) of the GLM prefill BF16 projections the pin covers: q_a, kv_a,
        // index keys/gates, router, q_b, o_proj, dense gate/up and down halves.
        for (n, k) in [
            (1536, 4096),
            (512, 4096),
            (128, 4096),
            (288, 4096),
            (8192, 1536),
            (4096, 8192),
            (6144, 4096),
            (4096, 6144),
        ] {
            let mut r = Lcg((n * 31 + k) as u64);
            let act = up(g, &bf16(total * k, &mut r, 2.0))?;
            let weight = up(g, &bf16(n * k, &mut r, 0.1))?;
            let packed = g.alloc(total * n * 2)?;
            let alone = g.alloc(lens.iter().max().unwrap() * n * 2)?;
            crate::layers::ops::bf16_gemm(
                act,
                weight.0,
                packed,
                total as u32,
                n as u32,
                k as u32,
                0,
            )?;
            let packed_bytes = down(g, packed, total * n * 2)?;
            for (&row0, &len) in offsets(lens).iter().zip(lens) {
                crate::layers::ops::bf16_gemm(
                    act.offset(row0 * k * 2),
                    weight.0,
                    alone,
                    len as u32,
                    n as u32,
                    k as u32,
                    0,
                )?;
                let alone_bytes = down(g, alone, len * n * 2)?;
                ensure!(
                    alone_bytes[..] == packed_bytes[row0 * n * 2..(row0 + len) * n * 2],
                    "{n}x{k}: a {len}-row prompt at row {row0} differs packed vs alone"
                );
            }
            println!("{n}x{k}: every prompt's rows identical packed ({total} rows) and alone");
            for p in [act, weight, packed, alone] {
                g.free(p)?;
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 with the CUTLASS build"]
fn packed_pin_mla_head_gemm_rows_match_alone() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    for lens in PACKS {
        let total: usize = lens.iter().sum();
        // W_uk absorb and W_uv per head: 32 heads x [512, 256] / [256, 512].
        for (heads, n, k) in [(32u32, 512u32, 256u32), (32, 256, 512)] {
            let (a_stride, c_stride) = (heads * k, heads * n);
            let mut r = Lcg((heads * n + k) as u64);
            let act = up(g, &bf16(total * a_stride as usize, &mut r, 2.0))?;
            let weight = up(g, &bf16((heads * n * k) as usize, &mut r, 0.1))?;
            let row = c_stride as usize * 2;
            let packed = g.alloc(total * row)?;
            let alone = g.alloc(lens.iter().max().unwrap() * row)?;
            let gemm = |a: DevicePtr, c: DevicePtr, m: usize| {
                spark_runtime::cutlass::bf16_grouped_gemm_act_weight_t(
                    a.0, weight.0, c.0, m as u32, heads, n, k, a_stride, c_stride, 0,
                )
            };
            gemm(act, packed, total)?;
            let packed_bytes = down(g, packed, total * row)?;
            for (&row0, &len) in offsets(lens).iter().zip(lens) {
                gemm(act.offset(row0 * a_stride as usize * 2), alone, len)?;
                let alone_bytes = down(g, alone, len * row)?;
                ensure!(
                    alone_bytes[..] == packed_bytes[row0 * row..(row0 + len) * row],
                    "heads {heads} [{n}, {k}]: a {len}-row prompt at row {row0} differs"
                );
            }
            println!("heads {heads} [{n}, {k}]: identical packed and alone");
            for p in [act, weight, packed, alone] {
                g.free(p)?;
            }
        }
    }
    Ok(())
}
