// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// E4M3 byte as f64.
fn e4m3(b: u8) -> f64 {
    let (ex, m) = (((b >> 3) & 15) as i32, (b & 7) as f64);
    let v = if ex == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(ex - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Expert `e`'s `[k/2, n]` packed and `[k/16, n]` scale tables and scale2.
fn host_table(
    gpu: &dyn GpuBackend,
    t: &[DevicePtr; 3],
    e: usize,
    n: u32,
    k: u32,
) -> Result<(Vec<u8>, Vec<u8>, f32)> {
    let mut ptr = [0u8; 8];
    let mut get = |table: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
        gpu.copy_d2h(table.offset(e * 8), &mut ptr)?;
        let mut v = vec![0u8; bytes];
        gpu.copy_d2h(DevicePtr(u64::from_le_bytes(ptr)), &mut v)?;
        Ok(v)
    };
    let w = get(t[0], (k / 2 * n) as usize)?;
    let s = get(t[1], (k / 16 * n) as usize)?;
    let mut s2 = [0u8; 4];
    gpu.copy_d2h(t[2].offset(e * 4), &mut s2)?;
    Ok((w, s, f32::from_le_bytes(s2)))
}

/// `x · W` in f64 over exact dequantized weights, times scale2, BF16-rounded.
fn host_gemv(t: &(Vec<u8>, Vec<u8>, f32), n: u32, k: u32, x: &[f32]) -> Vec<f32> {
    let mut o = vec![0f64; n as usize];
    for (kk, &xv) in x.iter().enumerate().take(k as usize) {
        if xv == 0.0 {
            continue;
        }
        for (c, acc) in o.iter_mut().enumerate() {
            let b = t.0[kk / 2 * n as usize + c];
            let q = if kk & 1 == 1 { b >> 4 } else { b & 15 };
            *acc += xv as f64 * E2M1[q as usize] * e4m3(t.1[kk / 16 * n as usize + c]);
        }
    }
    o.iter()
        .map(|&v| bf16_val(bf16_bits((v * t.2 as f64) as f32)))
        .collect()
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn a16_twins_match_a_host_w4a16_reference() -> Result<()> {
    let gpu: &dyn GpuBackend = backend()?;
    let mut rng = Rng(0xa16_5eed);
    let rig = Rig::new(gpu, &mut rng)?;
    // BF16 activation rows (gate/up input) and the BF16 SiLU·mul rows.
    let x: Vec<u16> = (0..32 * H as usize)
        .map(|_| bf16_bits(((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0))
        .collect();
    let xd = upload(gpu, words(&x))?;
    let mid = gpu.alloc(MAX_ROUTED * I as usize * 2)?;
    let k = |name: String| gpu.kernel("moe_w4a16", &name).with_context(|| name.clone());
    let twins = |family: &str| -> Result<[KernelHandle; 2]> {
        Ok([
            k(format!(
                "glm_moe_decode_{family}_a16_gate_up_silu_k128w_l2pf_p"
            ))?,
            k(format!("glm_moe_decode_{family}_a16_k128w_l2pf_p"))?,
        ])
    };
    let families = [twins("m16s")?, twins("m32s")?, twins("m64s")?];
    // Both projections of `rows` rows (from row `first`) on `pair` over `ctas`
    // CTAs: the SiLU·mul and down BF16 words.
    let run = |pair: [KernelHandle; 2], rows: u32, ctas: u32| -> Result<(Vec<u16>, Vec<u16>)> {
        let routed = rows as usize * TOPK;
        gpu.memset(mid, POISON, routed * I as usize * 2)?;
        gpu.memset(rig.c, POISON, routed * H as usize * 2)?;
        let schedule = ops::K128wSchedule::Stride { ctas };
        let kernel = |grid| ops::K128wKernel {
            grid,
            persist: KernelHandle(0),
        };
        ops::moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
            gpu,
            kernel(pair[0]),
            xd,
            DevicePtr::NULL,
            rig.gate,
            rig.up,
            [mid, DevicePtr::NULL, DevicePtr::NULL],
            rig.offsets,
            rig.sorted,
            E,
            I,
            H,
            rig.scratch,
            schedule,
            0,
        )?;
        ops::moe_w4a4_grouped_gemm_prequant_k128w(
            gpu,
            kernel(pair[1]),
            mid,
            DevicePtr::NULL,
            rig.down[0],
            rig.down[1],
            rig.down[2],
            rig.c,
            rig.offsets,
            DevicePtr::NULL,
            E,
            H,
            I,
            rig.scratch,
            schedule,
            0,
        )?;
        gpu.synchronize(0)?;
        let mut m = vec![0u16; routed * I as usize];
        let mut c = vec![0u16; routed * H as usize];
        // SAFETY: u16 buffers read as bytes.
        gpu.copy_d2h(mid, unsafe {
            std::slice::from_raw_parts_mut(m.as_mut_ptr().cast(), m.len() * 2)
        })?;
        gpu.copy_d2h(rig.c, unsafe {
            std::slice::from_raw_parts_mut(c.as_mut_ptr().cast(), c.len() * 2)
        })?;
        Ok((m, c))
    };
    let (mut worst, mut failed) = ([0f64; 2], vec![]);
    for (rows, kind) in [(1u32, 0u32), (3, 1), (8, 2), (16, 1), (32, 2)] {
        let ids = picks(&mut rng, rows as usize, kind);
        let r = route(&ids);
        rig.set_route(&r, 0)?;
        let family = families[if rows <= 16 { 0 } else { 1 }];
        let (m, c) = run(family, rows, 48)?;
        // Other CTA counts and the wider slab families: the same words.
        for (pair, ctas) in [(family, 7), (families[2], 48)] {
            if run(pair, rows, ctas)? != (m.clone(), c.clone()) {
                failed.push(format!(
                    "rows {rows} routing {kind} ctas {ctas}: words differ"
                ));
            }
        }
        // Host W4A16 reference, the down on the kernel's own SiLU·mul rows.
        for e in 0..E as usize {
            if r.offsets[e + 1] == r.offsets[e] {
                continue;
            }
            let (g, u) = (
                host_table(gpu, &rig.gate, e, I, H)?,
                host_table(gpu, &rig.up, e, I, H)?,
            );
            let d = host_table(gpu, &rig.down, e, H, I)?;
            for p in r.offsets[e] as usize..r.offsets[e + 1] as usize {
                let tok = r.sorted[p] as usize;
                let xin: Vec<f32> = x[tok * H as usize..(tok + 1) * H as usize]
                    .iter()
                    .map(|&b| bf16_val(b))
                    .collect();
                let (gv, uv) = (host_gemv(&g, I, H, &xin), host_gemv(&u, I, H, &xin));
                let want: Vec<f32> = gv
                    .iter()
                    .zip(&uv)
                    .map(|(&g, &u)| {
                        let (g, u) = (g.min(10.0), u.clamp(-10.0, 10.0));
                        bf16_val(bf16_bits(g / (1.0 + (-g).exp()) * u))
                    })
                    .collect();
                let got_m: Vec<f32> = m[p * I as usize..(p + 1) * I as usize]
                    .iter()
                    .map(|&b| bf16_val(b))
                    .collect();
                let want_c = host_gemv(&d, H, I, &got_m);
                for (i, (got, want)) in [
                    (&got_m, &want),
                    (
                        &c[p * H as usize..(p + 1) * H as usize]
                            .iter()
                            .map(|&b| bf16_val(b))
                            .collect(),
                        &want_c,
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    let peak = want.iter().fold(0f32, |a, &v| a.max(v.abs())) as f64 + 1e-30;
                    let err = got
                        .iter()
                        .zip(want.iter())
                        .fold(0f64, |a, (&g, &w)| a.max((g - w).abs() as f64))
                        / peak;
                    worst[i] = worst[i].max(err);
                }
            }
        }
        // Each row alone: its own bits.
        for (t, row_ids) in ids.iter().enumerate().take(3) {
            let alone = route(std::slice::from_ref(row_ids));
            rig.set_route(&alone, t)?;
            let (m1, c1) = run(families[0], 1, 48)?;
            for kk in 0..TOPK {
                let (p, p1) = (r.pos[t][kk], alone.pos[0][kk]);
                if m[p * I as usize..(p + 1) * I as usize]
                    != m1[p1 * I as usize..(p1 + 1) * I as usize]
                    || c[p * H as usize..(p + 1) * H as usize]
                        != c1[p1 * H as usize..(p1 + 1) * H as usize]
                {
                    failed.push(format!(
                        "rows {rows} routing {kind} row {t}: depends on the batch"
                    ));
                }
            }
        }
    }
    println!(
        "max |got - host| / row peak: SiLU·mul {:.2e}, down {:.2e}",
        worst[0], worst[1]
    );
    // FP32 accumulation in MMA order against f64: BF16 roundings one ulp apart.
    ensure!(
        worst[0] < 1.0 / 64.0 && worst[1] < 1.0 / 64.0,
        "W4A16 twins off the host reference: {worst:?}"
    );
    ensure!(failed.is_empty(), "W4A16 twins: {failed:#?}");
    Ok(())
}
