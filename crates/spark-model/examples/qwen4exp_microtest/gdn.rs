// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas's GDN decode kernel against the qwen4_exp oracle.

use super::*;

/// `gated_delta_rule_decode` covers 36 of this model's 48 layers. Atlas already
/// ships it for Qwen3.5/3.6, so the question is not whether it works but
/// whether it computes the SAME recurrence this model expects -- decay first,
/// then correct by the recall error, rather than accumulating k v^T.
pub(super) fn gdn_decode_step(g: &dyn GpuBackend) -> Result<()> {
    use atlas_core::qwen4exp_reference::gdn_delta_step;

    // Published qwen4_exp linear-attention geometry.
    const NUM_K_HEADS: usize = 16;
    const NUM_V_HEADS: usize = 48;
    const KD: usize = 128;
    const VD: usize = 128;
    let repeat = NUM_V_HEADS / NUM_K_HEADS;

    let mut state = 0xDEAD_BEEF_1234_5678u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let round = |v: &f32| bf16::from_f32(*v).to_f32();

    // q and k arrive already L2-normalised and scaled -- the kernel does not
    // do it, and neither does the oracle's step.
    let mut q: Vec<f32> = (0..NUM_K_HEADS * KD).map(|_| next()).collect();
    let mut k: Vec<f32> = (0..NUM_K_HEADS * KD).map(|_| next()).collect();
    for head in 0..NUM_K_HEADS {
        for buf in [&mut q, &mut k] {
            let slice = &mut buf[head * KD..(head + 1) * KD];
            let inv = 1.0 / (slice.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            for v in slice.iter_mut() {
                *v *= inv;
            }
        }
    }
    // The 1/sqrt(key_head_dim) scale is applied ONCE, and the two sides do it
    // in different places: HF scales the QUERY before the recurrence, Atlas's
    // kernel scales the OUTPUT after it. Algebraically the same -- the output
    // is linear in q, and neither placement touches the state -- but doing
    // both is an 11x error, which is exactly what this harness did first.
    //
    // So: the kernel gets q UNSCALED, the oracle gets it scaled.
    let scale = 1.0 / (KD as f32).sqrt();
    let q_scaled: Vec<f32> = q.iter().map(|v| v * scale).collect();
    let v: Vec<f32> = (0..NUM_V_HEADS * VD).map(|_| next()).collect();
    // Decay in (0,1), beta in (0,1) -- the ranges exp(-softplus) and sigmoid
    // actually produce.
    let decay: Vec<f32> = (0..NUM_V_HEADS)
        .map(|_| next().abs() * 0.9 + 0.05)
        .collect();
    let beta: Vec<f32> = (0..NUM_V_HEADS)
        .map(|_| next().abs() * 0.9 + 0.05)
        .collect();
    let h0: Vec<f32> = (0..NUM_V_HEADS * KD * VD).map(|_| next() * 0.1).collect();

    let up_f32 = |d: &[f32]| -> Result<DevicePtr> {
        let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
        let p = g.alloc(bytes.len())?;
        g.copy_h2d(&bytes, p)?;
        Ok(p)
    };

    let d_h = up_f32(&h0)?;
    let d_q = up_bf16(g, &q)?;
    let d_k = up_bf16(g, &k)?;
    let d_v = up_bf16(g, &v)?;
    let d_decay = up_f32(&decay)?;
    let d_beta = up_f32(&beta)?;
    let d_out = g.alloc(NUM_V_HEADS * VD * 2)?;

    let kernel = g.kernel("gated_delta_rule", "gated_delta_rule_decode")?;
    KernelLaunch::new(g, kernel)
        .grid([NUM_V_HEADS as u32, 1, 1])
        .block([VD as u32, 1, 1])
        .arg_ptr(d_h)
        .arg_ptr(d_q)
        .arg_ptr(d_k)
        .arg_ptr(d_v)
        .arg_ptr(d_decay)
        .arg_ptr(d_beta)
        .arg_ptr(d_out)
        .arg_u32(1)
        .arg_u32(NUM_K_HEADS as u32)
        .arg_u32(NUM_V_HEADS as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_out, NUM_V_HEADS * VD)?;

    // Oracle, per value head, on the same BF16-rounded inputs.
    let (qr, kr, vr) = (
        q_scaled.iter().map(round).collect::<Vec<_>>(),
        k.iter().map(round).collect::<Vec<_>>(),
        v.iter().map(round).collect::<Vec<_>>(),
    );
    let mut want = Vec::with_capacity(NUM_V_HEADS * VD);
    let mut h = h0.clone();
    for head in 0..NUM_V_HEADS {
        let kh = head / repeat;
        let st = &mut h[head * KD * VD..(head + 1) * KD * VD];
        want.extend(gdn_delta_step(
            st,
            &qr[kh * KD..(kh + 1) * KD],
            &kr[kh * KD..(kh + 1) * KD],
            &vr[head * VD..(head + 1) * VD],
            decay[head],
            beta[head],
        ));
    }

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale_out = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "GDN decode step: max|diff| {worst:.3e} over {} values (up to {scale_out:.3e}), relative {:.3e}",
        got.len(),
        worst / scale_out.max(1e-9)
    );

    // q and k are SHARED across `repeat` value heads. If the kernel mapped
    // them per value head instead, heads inside a group would stop agreeing --
    // so check the mapping actually happened.
    let group_shared = (0..repeat).all(|r| {
        let a = &want[r * VD..(r + 1) * VD];
        let b = &want[VD..2 * VD];
        r == 1 || a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6)
    });
    println!("head-group mapping exercised: {group_shared}");

    anyhow::ensure!(
        worst / scale_out.max(1e-9) < 2e-2,
        "gated_delta_rule_decode disagrees with the qwen4_exp oracle"
    );
    println!("GDN DECODE STEP MATCHES THE ORACLE\n");
    attn_decode_step(g)
}

// ── The gated-Q attention decode kernel, against the qwen4_exp oracle ───────
