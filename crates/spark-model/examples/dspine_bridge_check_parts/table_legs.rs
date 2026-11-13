// SPDX-License-Identifier: AGPL-3.0-only

//! The serial table/walk legs of `dspine_bridge_check`.

use super::*;

/// The serial table and walk: topk of random logits, then the table twice
/// (byte-identical), each block alone (== in batch), the walk against a CPU
/// walk of the GPU table (exact), and the table's picks against a CPU
/// replica in f64 (pick for pick).
pub(super) fn table_legs(g: &dyn GpuBackend, k: &Ks, rng: &mut Rng) -> Result<u32> {
    let mut fails = 0u32;
    let logits = rng.bf(ROWS * V, 8.0);
    let raw = rng.bf(ROWS * H, 2.0);
    let packed = rng.bf(ROWS * 3 * R, 2.0);
    let norm_w = rng.bf(H, 1.0);
    let bias = rng.bf(R, 1.0);
    let head = rng.bf(V * H, 0.05);
    let tg = rng.bf(V * R, 0.05);
    let tsv = rng.bf(V * 2 * R, 1.0);
    let rms: Vec<f32> = (0..ROWS).map(|_| rng.f().abs() + 0.5).collect();
    let anchors = [5u32, 77];
    let (lp, rawp, pkp, nwp, bp, hp, tgp, tsvp, ap) = (
        up(g, &logits)?,
        up(g, &raw)?,
        up(g, &packed)?,
        up(g, &norm_w)?,
        up(g, &bias)?,
        up(g, &head)?,
        up(g, &tg)?,
        up(g, &tsv)?,
        up_u32(g, &anchors)?,
    );
    let rp = g.alloc(ROWS * 4)?;
    let rb: Vec<u8> = rms.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&rb, rp)?;
    let run = |row0: usize, rows: usize| -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let lists = g.alloc(spark_model::layers::ops::dflash2_selector_lists_bytes(rows))?;
        let sbytes = spark_model::layers::ops::dflash2_selector_scratch_bytes(rows);
        let scratch = g.alloc(sbytes)?;
        g.memset(scratch, 0, sbytes)?;
        let (cand, idx, out) = (
            g.alloc(rows * K * 4)?,
            g.alloc(rows * K * 4)?,
            g.alloc(rows * 4)?,
        );
        let s = g.default_stream();
        let off = |p: DevicePtr, w: usize| p.offset(row0 * w);
        spark_model::layers::ops::dflash2_selector_topk(
            g,
            k.topk,
            off(lp, V * 2),
            V as u32,
            V as u32,
            0,
            rows as u32,
            lists,
            scratch,
            s,
        )?;
        KernelLaunch::new(g, k.table)
            .grid([rows as u32, 1, 1])
            .block([512, 1, 1])
            .arg_ptr(lists)
            .arg_ptr(off(rawp, H * 2))
            .arg_ptr(off(rp, 4))
            .arg_ptr(off(pkp, 3 * R * 2))
            .arg_ptr(nwp)
            .arg_ptr(bp)
            .arg_ptr(tgp)
            .arg_ptr(tsvp)
            .arg_ptr(ap.offset(row0 / G * 4))
            .arg_u32(1)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(0)
            .arg_ptr(hp)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_f32(0.0)
            .arg_u32(H as u32)
            .arg_ptr(cand)
            .arg_ptr(idx)
            .arg_u32(rows as u32)
            .arg_u32(G as u32)
            .arg_u32(R as u32)
            .arg_u32(V as u32)
            .arg_u32(u32::MAX)
            .arg_u32(u32::MAX)
            .arg_u32(u32::MAX)
            .arg_u32(u32::MAX)
            .launch(s)?;
        KernelLaunch::new(g, k.walk)
            .grid([(rows / G) as u32, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(lists)
            .arg_ptr(cand)
            .arg_ptr(idx)
            .arg_ptr(out)
            .arg_u32(rows as u32)
            .arg_u32(G as u32)
            .launch(s)?;
        let ids = down32(g, lists.offset(rows * K * 4), rows * K)?;
        Ok((
            down32(g, cand, rows * K)?,
            down32(g, idx, rows * K)?,
            down32(g, out, rows)?,
            ids,
        ))
    };
    let (cand, idx, out, ids) = run(0, ROWS)?;
    let again = run(0, ROWS)?;
    check(
        "table: run-to-run byte-identical",
        (cand.clone(), idx.clone(), out.clone()) == (again.0, again.1, again.2),
        &mut fails,
    );
    let solo = run(G, G)?;
    check(
        "table: block alone == block in batch",
        solo.0 == cand[G * K..] && solo.1 == idx[G * K..] && solo.2 == out[G..],
        &mut fails,
    );
    // CPU walk of the GPU table.
    let mut walk_ok = true;
    for b in 0..BLOCKS {
        let r0 = b * G;
        walk_ok &= out[r0] == ids[r0 * K];
        let mut j = 0usize;
        for t in 1..G {
            j = idx[(r0 + t) * K + j] as usize;
            walk_ok &= out[r0 + t] == cand[(r0 + t) * K + j];
        }
    }
    check(
        "walk: byte-exact vs CPU walk of the table",
        walk_ok,
        &mut fails,
    );
    // CPU replica of the table's picks (f64; near-ties excluded).
    let (mut agree, mut total, mut ties) = (0, 0, 0);
    for row in 0..ROWS {
        let t = row % G;
        if t == 0 {
            continue;
        }
        let mut c: Vec<u32> = ids[row * K..row * K + K].to_vec();
        c.sort();
        let send: Vec<u32> = if t == 1 {
            vec![anchors[row / G]; K]
        } else {
            let mut s = ids[(row - 1) * K..(row - 1) * K + K].to_vec();
            s.sort();
            s
        };
        let base: Vec<f64> = c
            .iter()
            .map(|&id| {
                (0..H)
                    .map(|i| {
                        (raw[row * H + i].to_f32() * norm_w[i].to_f32()) as f64
                            * head[id as usize * H + i].to_f64()
                    })
                    .sum()
            })
            .collect();
        for (j, &s) in send.iter().enumerate() {
            let m: Vec<f64> = (0..R)
                .map(|r| {
                    let z = packed[row * 3 * R + r].to_f64()
                        + tsv[s as usize * 2 * R + r].to_f64()
                        + bias[r].to_f64();
                    tsv[s as usize * 2 * R + R + r].to_f64() / (1.0 + (-z).exp())
                })
                .collect();
            let sc: Vec<f64> = c
                .iter()
                .enumerate()
                .map(|(kk, &id)| {
                    base[kk]
                        + rms[row] as f64
                            * (0..R)
                                .map(|r| m[r] * tg[id as usize * R + r].to_f64())
                                .sum::<f64>()
                })
                .collect();
            let mut order: Vec<usize> = (0..K).collect();
            order.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap());
            if (sc[order[0]] - sc[order[1]]).abs() < 1e-3 * sc[order[0]].abs().max(1.0) {
                ties += 1;
                continue;
            }
            total += 1;
            agree += (idx[row * K + j] as usize == order[0]) as usize;
        }
    }
    check(
        &format!("table: picks == CPU replica ({agree}/{total}, {ties} near-ties skipped)"),
        agree == total && total > 0,
        &mut fails,
    );
    ensure!(cand.len() == ROWS * K, "table output size");
    Ok(fails)
}
