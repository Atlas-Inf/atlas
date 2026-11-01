// SPDX-License-Identifier: AGPL-3.0-only
//! #158 regression microtest: the GDN FLA prefill pipeline must not depend on
//! gdn_fla_scratch's prior contents.
//!
//! Runs the production atlas_scale launch sequence — recompute_wu →
//! chunk_delta_h_vfused → chunk_fwd_o — via `ops::gdn_prefill_fla` (the same
//! function `fla_dispatch.rs` calls), at the Qwen3.8-27B GDN dims
//! (NK=16, NV=48, kd=vd=128). For each T with a partial last chunk, the whole
//! scratch region is pre-filled with several patterns — zeros (reference),
//! bf16 0x7F7F (max finite), 0x3F80 (1.0), 0xFFFF/0x7F80 (NaN/Inf patterns),
//! fixed-seed random u16 (full range, includes NaN/Inf), fixed-seed random
//! finite bf16 in [-8, 8], bf16 denormals, -0.0, and tiny normals — plus
//! `stale1024`/`stale1024decay` cases that leave a real prior pipeline run's
//! bytes in place (the decay variant uses gates in [1e-12, 1e-3]). `output` and final `h_state` are compared
//! BYTES against the zeros reference. On any diff the failing fill is rerun
//! applied to ONE scratch region at a time to localize the reader.
//!
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example gdn_fla_scratch_garbage
//! Exit 0 = sanity passed AND every fill bit-identical for every T.

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops::gdn_prefill_fla;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 48;
const C: usize = 64;
// Packed qkvz row pitch: [Q | K | V | Z] of bf16 elements per token.
const CONV_DIM: usize = NK * KD + NK * KD + NV * VD + NV * VD;
const GB: usize = NV;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
    fn u16(&mut self) -> u16 {
        (self.f() * 65536.0) as u16
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Scratch region byte sizes for a given chunk count (fla_dispatch.rs layout).
fn regions(nt: usize) -> [(&'static str, usize); 5] {
    [
        ("w", nt * NV * C * KD * 2),
        ("u", nt * NV * C * VD * 2),
        ("s", nt * NV * KD * VD * 2),
        ("uc", nt * NV * C * VD * 2),
        ("gc", nt * NV * C * 4),
    ]
}
fn region_base(scratch: DevicePtr, nt: usize, name: &str) -> DevicePtr {
    let [w_b, u_b, s_b, uc_b, _] = {
        let r = regions(nt);
        [r[0].1, r[1].1, r[2].1, r[3].1, r[4].1]
    };
    match name {
        "w" => scratch,
        "u" => scratch.offset(w_b),
        "s" => scratch.offset(w_b + u_b),
        "uc" => scratch.offset(w_b + u_b + s_b),
        "gc" => scratch.offset(w_b + u_b + s_b + uc_b),
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy)]
enum Fill {
    Zero,
    Bf16(u16),    // repeated bf16 bit pattern
    RandU16(u64), // fixed-seed random u16 (includes NaN/Inf)
    RandFin(u64), // fixed-seed random finite bf16 in [-8, 8]
    Denorm(u64),  // bf16 denormals: exp==0, mant!=0, random sign
    Tiny(u64),    // normal bf16 |x| in [1e-38, 1e-30], random sign
}

fn fill_region(g: &dyn GpuBackend, ptr: DevicePtr, bytes: usize, f: Fill) -> Result<()> {
    let mut rng = match f {
        Fill::RandU16(s) | Fill::RandFin(s) | Fill::Denorm(s) | Fill::Tiny(s) => Lcg(s),
        _ => Lcg(0),
    };
    let mut hb = vec![0u8; bytes];
    for c in hb.chunks_exact_mut(2) {
        let v: u16 = match f {
            Fill::Zero => 0,
            Fill::Bf16(p) => p,
            Fill::RandU16(_) => rng.u16(),
            Fill::RandFin(_) => bf16::from_f32(rng.r(-8.0, 8.0) as f32).to_bits(),
            Fill::Denorm(_) => {
                let mant = 1 + (rng.f() * 127.0) as u16; // 1..=127
                let sign = if rng.f() < 0.5 { 0x8000 } else { 0 };
                sign | mant
            }
            Fill::Tiny(_) => {
                let mag = rng.r(1e-38, 1e-30) as f32;
                let v = if rng.f() < 0.5 { -mag } else { mag };
                bf16::from_f32(v).to_bits()
            }
        };
        c.copy_from_slice(&v.to_le_bytes());
    }
    g.copy_h2d(&hb, ptr)?;
    Ok(())
}

/// One pipeline run on `scratch` (region layout recomputed for `nt`).
#[allow(clippy::too_many_arguments)]
fn run_case(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_cdh_fused: KernelHandle,
    k_fo: KernelHandle,
    q_ptr: DevicePtr,
    k_ptr: DevicePtr,
    v_ptr: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    h0: &[f32],
    t: usize,
    scratch: DevicePtr,
    stream: u64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let nt = t.div_ceil(C);
    let w_out = region_base(scratch, nt, "w");
    let u_out = region_base(scratch, nt, "u");
    let s_out = region_base(scratch, nt, "s");
    let uc_out = region_base(scratch, nt, "uc");
    let gc_out = region_base(scratch, nt, "gc");
    let hp = up_f32(g, h0)?; // mutated → final h_state
    let out = g.alloc(t * NV * VD * 2)?;
    gdn_prefill_fla(
        g,
        k_wu,
        KernelHandle(0), // ksplit fallback (not atlas_scale)
        KernelHandle(0), // tc_vblock
        k_cdh_fused,
        KernelHandle(0), // tma
        k_fo,
        hp,
        q_ptr,
        k_ptr,
        v_ptr,
        gate,
        beta,
        out,
        w_out,
        u_out,
        s_out,
        uc_out,
        gc_out,
        1,
        t as u32,
        nt as u32,
        NK as u32,
        NV as u32,
        KD as u32,
        VD as u32,
        CONV_DIM as u32,
        CONV_DIM as u32,
        GB as u32,
        false,
        DevicePtr::NULL,
        DevicePtr::NULL,
        false,
        false,
        stream,
    )?;
    g.synchronize(stream)?;
    let ob = dn(g, out, t * NV * VD * 2)?;
    let hb = dn(g, hp, NV * KD * VD * 4)?;
    for p in [hp, out] {
        let _ = g.free(p);
    }
    Ok((ob, hb))
}

fn diff_stat_bf16(a: &[u8], b: &[u8]) -> (usize, f64, Option<usize>) {
    let mut n = 0;
    let mut mx = 0f64;
    let mut first = None;
    for (i, (x, y)) in a.chunks_exact(2).zip(b.chunks_exact(2)).enumerate() {
        if x != y {
            n += 1;
            if first.is_none() {
                first = Some(i);
            }
            let d = (bf16::from_le_bytes([x[0], x[1]]).to_f64()
                - bf16::from_le_bytes([y[0], y[1]]).to_f64())
            .abs();
            if d > mx {
                mx = d;
            }
        }
    }
    (n, mx, first)
}
fn diff_stat_f32(a: &[u8], b: &[u8]) -> (usize, f64, Option<usize>) {
    let mut n = 0;
    let mut mx = 0f64;
    let mut first = None;
    for (i, (x, y)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        if x != y {
            n += 1;
            if first.is_none() {
                first = Some(i);
            }
            let d = (f32::from_le_bytes([x[0], x[1], x[2], x[3]]) as f64
                - f32::from_le_bytes([y[0], y[1], y[2], y[3]]) as f64)
                .abs();
            if d > mx {
                mx = d;
            }
        }
    }
    (n, mx, first)
}

struct Inputs {
    t: usize,
    qkvz_p: DevicePtr,
    q_ptr: DevicePtr,
    k_ptr: DevicePtr,
    v_ptr: DevicePtr,
    gate_p: DevicePtr,
    beta_p: DevicePtr,
    h0: Vec<f32>,
}
fn gen_inputs(g: &dyn GpuBackend, t: usize, seed: u64) -> Result<Inputs> {
    gen_inputs_gates(g, t, seed, 0.80, 0.999)
}
fn gen_inputs_gates(
    g: &dyn GpuBackend,
    t: usize,
    seed: u64,
    gate_lo: f64,
    gate_hi: f64,
) -> Result<Inputs> {
    let mut r = Lcg(seed);
    let mut qkvz = vec![bf16::ZERO; t * CONV_DIM];
    for x in qkvz.iter_mut() {
        *x = bf16::from_f64(r.r(-0.5, 0.5));
    }
    let mut gate = vec![];
    let mut beta = vec![];
    let mut h0 = vec![];
    for _ in 0..t * GB {
        gate.push(r.r(gate_lo, gate_hi) as f32);
        beta.push(r.r(0.0, 1.0) as f32);
    }
    for _ in 0..NV * KD * VD {
        h0.push(r.r(-0.1, 0.1) as f32);
    }
    let qkvz_p = up_bf16(g, &qkvz)?;
    Ok(Inputs {
        t,
        qkvz_p,
        q_ptr: qkvz_p,
        k_ptr: qkvz_p.offset(NK * KD * 2),
        v_ptr: qkvz_p.offset((NK * KD + NK * KD) * 2),
        gate_p: up_f32(g, &gate)?,
        beta_p: up_f32(g, &beta)?,
        h0,
    })
}
impl Inputs {
    fn run(
        &self,
        g: &dyn GpuBackend,
        ks: &(KernelHandle, KernelHandle, KernelHandle),
        scratch: DevicePtr,
        stream: u64,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        run_case(
            g,
            ks.0,
            ks.1,
            ks.2,
            self.q_ptr,
            self.k_ptr,
            self.v_ptr,
            self.gate_p,
            self.beta_p,
            &self.h0,
            self.t,
            scratch,
            stream,
        )
    }
}

/// Fill `fill` into only `only` region of the T-layout (or all when None).
fn prep_scratch(
    g: &dyn GpuBackend,
    scratch: DevicePtr,
    nt: usize,
    fill: Fill,
    only: Option<&str>,
) -> Result<()> {
    for (name, bytes) in regions(nt) {
        let f = match only {
            None => fill,
            Some(o) if o == name => fill,
            _ => Fill::Zero,
        };
        fill_region(g, region_base(scratch, nt, name), bytes, f)?;
    }
    g.synchronize(0)?;
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let ks = (
        g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?,
        g.kernel(
            "gated_delta_rule_fla",
            "gated_delta_rule_chunk_delta_h_vfused",
        )?,
        g.kernel("gated_delta_rule_fla", "gated_delta_rule_chunk_fwd_o")?,
    );
    let stream = g.create_stream()?;
    let fills: &[(&str, Fill)] = &[
        ("max", Fill::Bf16(0x7F7F)),
        ("one", Fill::Bf16(0x3F80)),
        ("nan", Fill::Bf16(0xFFFF)),
        ("inf", Fill::Bf16(0x7F80)),
        ("rand", Fill::RandU16(0xBAD5EED)),
        ("randfin", Fill::RandFin(0xF1A17E)),
        ("denorm", Fill::Denorm(0xDE0AA1)),
        ("negzero", Fill::Bf16(0x8000)),
        ("tiny", Fill::Tiny(0x71B1)),
    ];
    // One scratch allocation big enough for the largest T (nt=17); region
    // offsets are recomputed per T exactly as fla_dispatch.rs does.
    let nt_max = 1075usize.div_ceil(C);
    let scratch = g.alloc(regions(nt_max).iter().map(|r| r.1).sum())?;

    let mut all_ok = true;
    for &t in &[107usize, 179, 1075] {
        let inp = gen_inputs(g, t, 0x158 ^ (t as u64))?;
        let nt = t.div_ceil(C);

        // Reference: zeros fill.
        prep_scratch(g, scratch, nt, Fill::Zero, None)?;
        let (o0, h0b) = inp.run(g, &ks, scratch, stream)?;

        // Sanity: the pipeline actually ran.
        let mut nonfinite = 0usize;
        let mut omax = 0f64;
        for c in o0.chunks_exact(2) {
            let v = bf16::from_le_bytes([c[0], c[1]]).to_f64();
            if !v.is_finite() {
                nonfinite += 1;
            } else if v.abs() > omax {
                omax = v.abs();
            }
        }
        let hdiff = h0b
            .chunks_exact(4)
            .zip(inp.h0.iter())
            .filter(|&(c, &x)| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != x)
            .count();
        let sane = nonfinite == 0 && omax > 1e-3 && hdiff > 0;
        all_ok &= sane;
        println!(
            "T={t:5} sanity: nonfinite_out={nonfinite} max|out|={omax:.4} h_state_diff_elems={hdiff} -> {}",
            if sane { "ok" } else { "FAIL" }
        );

        for &(name, fill) in fills {
            prep_scratch(g, scratch, nt, fill, None)?;
            let (ob, hb) = inp.run(g, &ks, scratch, stream)?;
            let (on, omx, of) = diff_stat_bf16(&o0, &ob);
            let (hn, hmx, hf) = diff_stat_f32(&h0b, &hb);
            let ok = on == 0 && hn == 0;
            all_ok &= ok;
            println!(
                "T={t:5} fill={name:8}: identical={}  out ndiff={on} maxdiff={omx:.6} first={of:?}  h ndiff={hn} maxdiff={hmx:.6} first={hf:?}",
                if ok { "yes" } else { "no" },
            );
            if !ok {
                // Localize: same fill into ONE region at a time, rest zeroed.
                for (rname, _) in regions(nt) {
                    prep_scratch(g, scratch, nt, fill, Some(rname))?;
                    let (o1, h1) = inp.run(g, &ks, scratch, stream)?;
                    let (rn, _, _) = diff_stat_bf16(&o0, &o1);
                    let (rhn, _, _) = diff_stat_f32(&h0b, &h1);
                    if rn > 0 || rhn > 0 {
                        println!(
                            "    localize T={t} fill={name}: region {rname} diffs (out {rn}, h {rhn})"
                        );
                    }
                }
            }
        }

        // stale1024: run a DIFFERENT input set at T=1024 on the same scratch
        // base, then this case's T with no refill — real stale bytes.
        prep_scratch(g, scratch, 16, Fill::Zero, None)?;
        let prev = gen_inputs(g, 1024, 0xDEADBEEF)?;
        let _ = prev.run(g, &ks, scratch, stream)?;
        let (ob, hb) = inp.run(g, &ks, scratch, stream)?;
        let (on, omx, of) = diff_stat_bf16(&o0, &ob);
        let (hn, hmx, hf) = diff_stat_f32(&h0b, &hb);
        let ok = on == 0 && hn == 0;
        all_ok &= ok;
        println!(
            "T={t:5} fill=stale1024: identical={}  out ndiff={on} maxdiff={omx:.6} first={of:?}  h ndiff={hn} maxdiff={hmx:.6} first={hf:?}",
            if ok { "yes" } else { "no" },
        );
        // stale1024decay: same shape, but the prior T=1024 run uses gates in
        // [1e-12, 1e-3] (strong decay) so W/U/S hold realistic tiny/denormal
        // bf16 values.
        prep_scratch(g, scratch, 16, Fill::Zero, None)?;
        let decay = gen_inputs_gates(g, 1024, 0xDECA1, 1e-12, 1e-3)?;
        let _ = decay.run(g, &ks, scratch, stream)?;
        let wb = dn(g, region_base(scratch, 16, "w"), regions(16)[0].1)?;
        let mut denorm = 0usize;
        for c in wb.chunks_exact(2) {
            if u16::from_le_bytes([c[0], c[1]]) & 0x7F80 == 0 {
                denorm += 1;
            }
        }
        let total_w = regions(16)[0].1 / 2;
        println!(
            "T={t:5} stale1024decay: prior W denorm-or-zero = {denorm}/{total_w} ({:.2}%)",
            100.0 * denorm as f64 / total_w as f64,
        );
        let (ob, hb) = inp.run(g, &ks, scratch, stream)?;
        let (on, omx, of) = diff_stat_bf16(&o0, &ob);
        let (hn, hmx, hf) = diff_stat_f32(&h0b, &hb);
        let ok = on == 0 && hn == 0;
        all_ok &= ok;
        println!(
            "T={t:5} fill=stale1024decay: identical={}  out ndiff={on} maxdiff={omx:.6} first={of:?}  h ndiff={hn} maxdiff={hmx:.6} first={hf:?}",
            if ok { "yes" } else { "no" },
        );
        for p in [
            inp.qkvz_p,
            inp.gate_p,
            inp.beta_p,
            prev.qkvz_p,
            prev.gate_p,
            prev.beta_p,
            decay.qkvz_p,
            decay.gate_p,
            decay.beta_p,
        ] {
            let _ = g.free(p);
        }
    }
    let _ = g.free(scratch);
    println!(
        "{}",
        if all_ok {
            "PASS: sanity ok and output/h_state bit-identical across all scratch fills"
        } else {
            "FAIL: pipeline output depends on scratch contents (or sanity failed)"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
