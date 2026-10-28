// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark cases: union routing, memory guard and the RESULT printer.

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::gpu::GpuBackend;

use crate::tables::*;

/// Weight bytes one expert contributes per kernel pair (packed + scale for
/// gate & up + down).
fn expert_bytes() -> usize {
    (K_RED / 2 * N_OUT + K_RED / 16 * N_OUT) * 2 + (K_DOWN / 2 * N_DOWN + K_DOWN / 16 * N_DOWN)
}

pub(crate) fn mem_kb() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("MemAvailable"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
    })
}

pub(crate) fn bench_case(
    g: &dyn GpuBackend,
    layers: &[Layer],
    rows: usize,
    union: usize,
    shared: bool,
    vname: &str,
    vsuffix: &str,
    vbs: u32,
) -> Result<()> {
    if let Some(kb) = mem_kb()
        && kb < 20 * 1024 * 1024
    {
        println!("ABORT: MemAvailable {kb} kB < 20 GB");
        std::process::exit(2);
    }
    let gpu = g;
    let mut rng = Lcg(0xC0FFEE ^ (rows as u64) << 8 | union as u64);
    let tabs: Vec<_> = layers
        .iter()
        .map(|l| l.ptr_tables(gpu))
        .collect::<Result<_>>()?;

    let input: Vec<u8> = (0..rows * K_RED)
        .map(|_| bf16::from_f32(rng.act()))
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    let input = up(gpu, &input)?;
    // expert ids: K rows x TOP_K, distinct per row, exact union `union`.
    let ids = union_ids(&mut rng, rows, union);
    let ids_d = up(
        gpu,
        &ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )?;

    let gate_out = gpu.alloc(rows * TOP_K * N_OUT * 2)?;
    let up_out = gpu.alloc(rows * TOP_K * N_OUT * 2)?;
    let down_out = gpu.alloc(rows * TOP_K * N_DOWN * 2)?;
    let _sh_g = QuantizedWeight::null();
    let _sh_u = QuantizedWeight::null();
    let _sh_d = QuantizedWeight::null();
    let sh_go = gpu.alloc(rows * N_OUT * 2)?;
    let sh_uo = gpu.alloc(rows * N_OUT * 2)?;
    let sh_gi = up(
        gpu,
        &(0..rows * K_DOWN)
            .map(|_| bf16::from_f32(rng.act()))
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let sh_ui = up(
        gpu,
        &(0..rows * K_DOWN)
            .map(|_| bf16::from_f32(rng.act()))
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let sh_do = gpu.alloc(rows * N_DOWN * 2)?;

    let (sh_gw, sh_uw, sh_dw) = if shared {
        let mut mkq = |pk: usize, sc: usize| -> Result<QuantizedWeight> {
            let pw = up(gpu, &(0..pk).map(|_| rng.b()).collect::<Vec<_>>())?;
            let sw = up(gpu, &(0..sc).map(|_| rng.sc()).collect::<Vec<_>>())?;
            Ok(QuantizedWeight {
                weight: pw,
                weight_scale: sw,
                weight_scale_2: rng.s2(),
                ..QuantizedWeight::null()
            })
        };
        (
            mkq(K_RED / 2 * N_OUT, K_RED / 16 * N_OUT)?,
            mkq(K_RED / 2 * N_OUT, K_RED / 16 * N_OUT)?,
            mkq(K_DOWN / 2 * N_DOWN, K_DOWN / 16 * N_DOWN)?,
        )
    } else {
        (
            QuantizedWeight::null(),
            QuantizedWeight::null(),
            QuantizedWeight::null(),
        )
    };

    let uk = gpu.kernel("moe_shared_expert_fused_t", "moe_expert_union_t")?;
    let gk = gpu.kernel("moe_shared_expert_fused_t", "moe_expert_gate_up_shared_t")?;
    let gkr = gpu.kernel(
        "moe_shared_expert_fused_t",
        &format!("moe_expert_gate_up_shared_t_rows{vsuffix}"),
    )?;
    let dk = gpu.kernel("moe_shared_expert_fused_t", "moe_expert_silu_down_shared_t")?;
    let dkr = gpu.kernel(
        "moe_shared_expert_fused_t",
        &format!("moe_expert_silu_down_shared_t_rows{vsuffix}"),
    )?;

    let u_experts = gpu.alloc(rows * TOP_K * 4)?;
    let u_rowmask = gpu.alloc(rows * TOP_K * 4)?;
    let u_rowslot = gpu.alloc(rows * TOP_K * 8)?;
    let u_count = gpu.alloc(4)?;

    let gate_ref = gpu.alloc(rows * TOP_K * N_OUT * 2)?;
    let up_ref = gpu.alloc(rows * TOP_K * N_OUT * 2)?;
    let down_ref = gpu.alloc(rows * TOP_K * N_DOWN * 2)?;
    let sh_go_r = gpu.alloc(rows * N_OUT * 2)?;
    let sh_uo_r = gpu.alloc(rows * N_OUT * 2)?;
    let sh_do_r = gpu.alloc(rows * N_DOWN * 2)?;

    // ── byte-identity check (fixed ids) ──
    ops::moe_expert_union_t(
        gpu,
        uk,
        ids_d,
        u_experts,
        u_rowmask,
        u_rowslot,
        u_count,
        rows as u32,
        TOP_K as u32,
        0,
    )?;
    {
        gpu.synchronize(0)?;
        let t = &tabs[0];
        let (gpk, gsc, gs2, upk, usc, dpk, dsc) = (t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
        for r in 0..rows {
            let ids_r = ids_d.offset(r * TOP_K * 4);
            ops::moe_expert_gate_up_shared_t(
                gpu,
                gk,
                input.offset(r * K_RED * 2),
                gpk,
                gsc,
                gs2,
                gate_ref.offset(r * TOP_K * N_OUT * 2),
                upk,
                usc,
                gs2,
                up_ref.offset(r * TOP_K * N_OUT * 2),
                ids_r,
                &sh_gw,
                sh_go_r.offset(r * N_OUT * 2),
                &sh_uw,
                sh_uo_r.offset(r * N_OUT * 2),
                N_OUT as u32,
                K_RED as u32,
                TOP_K as u32,
                0,
            )?;
            ops::moe_expert_silu_down_shared_t(
                gpu,
                dk,
                gate_ref.offset(r * TOP_K * N_OUT * 2),
                up_ref.offset(r * TOP_K * N_OUT * 2),
                dpk,
                dsc,
                gs2,
                down_ref.offset(r * TOP_K * N_DOWN * 2),
                ids_r,
                sh_gi.offset(r * K_DOWN * 2),
                sh_ui.offset(r * K_DOWN * 2),
                &sh_dw,
                sh_do_r.offset(r * N_DOWN * 2),
                N_DOWN as u32,
                K_DOWN as u32,
                TOP_K as u32,
                0,
            )?;
        }
        ops::moe_expert_gate_up_shared_t_rows(
            gpu,
            gkr,
            input,
            gpk,
            gsc,
            gs2,
            gate_out,
            upk,
            usc,
            gs2,
            up_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            (rows * TOP_K) as u32,
            &sh_gw,
            sh_go,
            &sh_uw,
            sh_uo,
            N_OUT as u32,
            K_RED as u32,
            TOP_K as u32,
            K_RED as u32,
            rows as u32,
            vbs,
            0,
        )?;
        ops::moe_expert_silu_down_shared_t_rows(
            gpu,
            dkr,
            gate_out,
            up_out,
            dpk,
            dsc,
            gs2,
            down_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            (rows * TOP_K) as u32,
            sh_gi,
            sh_ui,
            &sh_dw,
            sh_do,
            N_DOWN as u32,
            K_DOWN as u32,
            TOP_K as u32,
            rows as u32,
            vbs,
            0,
        )?;
    }
    gpu.synchronize(0)?;
    // Capture the check outputs NOW — the timed loops reuse these buffers.
    let mut ref_host: Vec<Vec<u8>> = Vec::new();
    let mut rows_host: Vec<Vec<u8>> = Vec::new();
    for (a, nb) in [
        (gate_ref, rows * TOP_K * N_OUT * 2),
        (up_ref, rows * TOP_K * N_OUT * 2),
        (down_ref, rows * TOP_K * N_DOWN * 2),
        (sh_go_r, rows * N_OUT * 2),
        (sh_uo_r, rows * N_OUT * 2),
        (sh_do_r, rows * N_DOWN * 2),
    ] {
        ref_host.push(dn(gpu, a, nb)?);
    }
    for (b, nb) in [
        (gate_out, rows * TOP_K * N_OUT * 2),
        (up_out, rows * TOP_K * N_OUT * 2),
        (down_out, rows * TOP_K * N_DOWN * 2),
        (sh_go, rows * N_OUT * 2),
        (sh_uo, rows * N_OUT * 2),
        (sh_do, rows * N_DOWN * 2),
    ] {
        rows_host.push(dn(gpu, b, nb)?);
    }

    // ── single-row reference (timed, cold) ──
    // Fixed ids across iterations: the union stays at the target while the
    // 48-layer table rotation streams weights cold, like a real decode step.
    const ITERS: usize = 24;
    let t0 = std::time::Instant::now();
    for it in 0..ITERS {
        let (gpk, gsc, gs2, upk, usc, dpk, dsc) = {
            let t = &tabs[it % LAYER_SET];
            (t[0], t[1], t[2], t[3], t[4], t[5], t[6])
        };
        for r in 0..rows {
            let ids_r = ids_d.offset(r * TOP_K * 4);
            ops::moe_expert_gate_up_shared_t(
                gpu,
                gk,
                input.offset(r * K_RED * 2),
                gpk,
                gsc,
                gs2,
                gate_ref.offset(r * TOP_K * N_OUT * 2),
                upk,
                usc,
                gs2,
                up_ref.offset(r * TOP_K * N_OUT * 2),
                ids_r,
                &sh_gw,
                sh_go_r.offset(r * N_OUT * 2),
                &sh_uw,
                sh_uo_r.offset(r * N_OUT * 2),
                N_OUT as u32,
                K_RED as u32,
                TOP_K as u32,
                0,
            )?;
            ops::moe_expert_silu_down_shared_t(
                gpu,
                dk,
                gate_ref.offset(r * TOP_K * N_OUT * 2),
                up_ref.offset(r * TOP_K * N_OUT * 2),
                dpk,
                dsc,
                gs2,
                down_ref.offset(r * TOP_K * N_DOWN * 2),
                ids_r,
                sh_gi.offset(r * K_DOWN * 2),
                sh_ui.offset(r * K_DOWN * 2),
                &sh_dw,
                sh_do_r.offset(r * N_DOWN * 2),
                N_DOWN as u32,
                K_DOWN as u32,
                TOP_K as u32,
                0,
            )?;
        }
    }
    gpu.synchronize(0)?;
    let t_single = t0.elapsed() / ITERS as u32;

    // ── multi-row dedup ──
    let t1 = std::time::Instant::now();
    for it in 0..ITERS {
        let (gpk, gsc, gs2, upk, usc, dpk, dsc) = {
            let t = &tabs[it % LAYER_SET];
            (t[0], t[1], t[2], t[3], t[4], t[5], t[6])
        };
        ops::moe_expert_union_t(
            gpu,
            uk,
            ids_d,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            rows as u32,
            TOP_K as u32,
            0,
        )?;
        ops::moe_expert_gate_up_shared_t_rows(
            gpu,
            gkr,
            input,
            gpk,
            gsc,
            gs2,
            gate_out,
            upk,
            usc,
            gs2,
            up_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            (rows * TOP_K) as u32,
            &sh_gw,
            sh_go,
            &sh_uw,
            sh_uo,
            N_OUT as u32,
            K_RED as u32,
            TOP_K as u32,
            K_RED as u32,
            rows as u32,
            vbs,
            0,
        )?;
        ops::moe_expert_silu_down_shared_t_rows(
            gpu,
            dkr,
            gate_out,
            up_out,
            dpk,
            dsc,
            gs2,
            down_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            (rows * TOP_K) as u32,
            sh_gi,
            sh_ui,
            &sh_dw,
            sh_do,
            N_DOWN as u32,
            K_DOWN as u32,
            TOP_K as u32,
            rows as u32,
            vbs,
            0,
        )?;
    }
    gpu.synchronize(0)?;
    let t_rows = t1.elapsed() / ITERS as u32;

    // compare bytes
    let mut identical = true;
    for (i, name) in [
        "gate_out",
        "up_out",
        "down_out",
        "sh_gate_out",
        "sh_up_out",
        "sh_down_out",
    ]
    .iter()
    .enumerate()
    {
        let (va, vb) = (&ref_host[i], &rows_host[i]);
        let inf = va
            .chunks_exact(2)
            .filter(|c| (c[0] == 0x7f || c[0] == 0xff) && (c[1] == 0x7f || c[1] == 0xff))
            .count();
        if inf > 0 {
            println!("  WARN {name}: {inf} inf/nan bytes in ref");
        }
        let diffs: Vec<usize> = va
            .iter()
            .zip(vb.iter())
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .map(|(i, _)| i)
            .collect();
        if !diffs.is_empty() {
            identical = false;
            let first = diffs[0];
            let (r, slot, n) = (
                first / (TOP_K * N_OUT * 2),
                (first / (N_OUT * 2)) % TOP_K,
                (first / 2) % N_OUT,
            );
            let fv = u16::from_le_bytes([va[first], va[first + 1]]);
            let rv = u16::from_le_bytes([vb[first], vb[first + 1]]);
            println!(
                "  MISMATCH {name}: {} bytes differ; first at byte {} (r{} slot{} n{}) ref={:#x} rows={:#x}",
                diffs.len(),
                first,
                r,
                slot,
                n,
                fv,
                rv
            );
            // does rows value appear anywhere in ref? (wrong-slot write)
            let mut found = vec![];
            for j in 0..va.len() / 2 {
                if u16::from_le_bytes([va[2 * j], va[2 * j + 1]]) == rv {
                    found.push(j);
                    if found.len() > 3 {
                        break;
                    }
                }
            }
            println!("    rows-val {rv:#x} at ref n-positions {:?}", found);
        }
    }
    let slots = rows * TOP_K;
    let eb = expert_bytes() as f64;
    let single_gbs = (rows * TOP_K) as f64 * eb / t_single.as_secs_f64() / 1e9;
    let rows_gbs = (union + if shared { 1 } else { 0 }) as f64 * eb / t_rows.as_secs_f64() / 1e9;
    println!(
        "RESULT rows={rows} union={union}/{slots} shared={shared} variant={vname}  single={:?} ({single_gbs:.1} GB/s) rows={:?} ({rows_gbs:.1} GB/s) ratio={:.3} identical={identical}",
        t_single,
        t_rows,
        t_rows.as_secs_f64() / t_single.as_secs_f64()
    );
    Ok(())
}
