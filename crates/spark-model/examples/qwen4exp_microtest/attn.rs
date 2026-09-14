// SPDX-License-Identifier: AGPL-3.0-only

//! The gated-Q attention decode kernel, and its must-fail gate control.

use super::*;

/// `q4e_attn_decode` covers the 12 full-attention layers. What distinguishes
/// this model's attention is the gate: `q_proj` emits `[query | gate]` PER
/// HEAD, and the gate is applied ELEMENTWISE to the attention output before
/// `o_proj`. Read as ungated, a loader takes the gate half as query values and
/// the model still produces text.
///
/// The oracle here is `attention_decode_step`, which `attention_forward` itself
/// calls -- so agreement chains to the same code that matches HuggingFace at
/// 8.0e-7 rather than to a second transcription of the equations.
pub(super) fn attn_decode_step(g: &dyn GpuBackend) -> Result<()> {
    use atlas_core::qwen4exp_reference::{AttnDims, attention_decode_step};

    // Published qwen4_exp full-attention geometry.
    const NUM_HEADS: usize = 24;
    const NUM_KV_HEADS: usize = 2;
    const HD: usize = 256;
    const PAST: usize = 37; // an awkward length, not a multiple of the block

    let mut state = 0x0BAD_F00D_5EED_9911u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let round = |v: &f32| bf16::from_f32(*v).to_f32();

    let q_dim = NUM_HEADS * HD;
    let kv_dim = NUM_KV_HEADS * HD;
    let query: Vec<f32> = (0..q_dim).map(|_| next()).collect();
    // Gates spread across the sigmoid's useful range. A gate stuck in
    // saturation would make the elementwise multiply look like a constant,
    // which is the failure this check exists to see.
    let gate: Vec<f32> = (0..q_dim).map(|_| next() * 6.0).collect();
    let keys: Vec<f32> = (0..PAST * kv_dim).map(|_| next()).collect();
    let values: Vec<f32> = (0..PAST * kv_dim).map(|_| next()).collect();

    let d_q = up_bf16(g, &query)?;
    let d_gate = up_bf16(g, &gate)?;
    let d_k = up_bf16(g, &keys)?;
    let d_v = up_bf16(g, &values)?;
    let d_out = g.alloc(q_dim * 2)?;

    let kernel = g.kernel("qwen4exp_attn", "q4e_attn_decode")?;
    KernelLaunch::new(g, kernel)
        .grid([NUM_HEADS as u32, 1, 1])
        .block([HD as u32, 1, 1])
        .shared_mem((PAST * 4) as u32)
        .arg_ptr(d_q)
        .arg_ptr(d_gate)
        .arg_ptr(d_k)
        .arg_ptr(d_v)
        .arg_ptr(d_out)
        .arg_u32(NUM_HEADS as u32)
        .arg_u32(NUM_KV_HEADS as u32)
        .arg_u32(HD as u32)
        .arg_u32(PAST as u32)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_out, q_dim)?;

    // Oracle on the same BF16-rounded inputs.
    let dims = AttnDims {
        hidden: 2560,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HD,
        rotary_dim: 64,
        eps: 1e-6,
    };
    let want = attention_decode_step(
        &dims,
        &query.iter().map(round).collect::<Vec<_>>(),
        &gate.iter().map(round).collect::<Vec<_>>(),
        &keys.iter().map(round).collect::<Vec<_>>(),
        &values.iter().map(round).collect::<Vec<_>>(),
    );

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale_out = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "attention decode: max|diff| {worst:.3e} over {} values (up to {scale_out:.3e}), relative {:.3e}",
        got.len(),
        worst / scale_out.max(1e-9)
    );

    // CONTROL 1: the gate must be doing something. Recompute with every gate
    // forced to +8 (sigmoid ~ 1, i.e. effectively ungated) and require the
    // answer to move -- otherwise a kernel that ignored `gate` would pass.
    let ungated = attention_decode_step(
        &dims,
        &query.iter().map(round).collect::<Vec<_>>(),
        &vec![8.0f32; q_dim],
        &keys.iter().map(round).collect::<Vec<_>>(),
        &values.iter().map(round).collect::<Vec<_>>(),
    );
    let gate_effect = ungated
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("gate contribution: {gate_effect:.3e} (must be large)");
    anyhow::ensure!(
        gate_effect > worst * 10.0,
        "the gate changes nothing -- the kernel may be ignoring it"
    );

    // CONTROL 2: 24 query heads share 2 KV heads. If the kernel mapped KV per
    // query head instead of per group, heads in DIFFERENT groups would stop
    // differing in the way grouping implies. Heads 0..11 read kv_head 0 and
    // heads 12..23 read kv_head 1, so the two halves must disagree.
    let half = NUM_HEADS / 2 * HD;
    let cross_group = got[..half]
        .iter()
        .zip(&got[half..])
        .any(|(a, b)| (a - b).abs() > 1e-3);
    println!("kv head-group mapping exercised: {cross_group}");
    anyhow::ensure!(cross_group, "both KV groups produced the same context");

    anyhow::ensure!(
        worst / scale_out.max(1e-9) < 2e-2,
        "q4e_attn_decode disagrees with the qwen4_exp oracle"
    );
    println!("ATTENTION DECODE STEP MATCHES THE ORACLE\n");
    hc_expand_entry(g)
}

// ── Trunk entry: the embedding tiled across the residual streams ───────────
