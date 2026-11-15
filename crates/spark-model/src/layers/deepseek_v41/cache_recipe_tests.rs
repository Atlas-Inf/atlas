// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::numeric::f32_to_bf16;

use super::*;

fn bf16(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| f32_to_bf16(v)).collect()
}

fn f32s(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| bf16_to_f32(b)).collect()
}

fn hex_bf16(hex: &str) -> Vec<u16> {
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex");
    (0..hex.len() / 4)
        .map(|i| u16::from_le_bytes([byte(4 * i), byte(4 * i + 2)]))
        .collect()
}

/// Layer 2's compressor latents and the compressed KV DeepSeek's reference
/// held for them, from prefill (groups 0, 1, 350, 699) and from decode.
#[test]
fn compressed_kv_is_the_latent_rotated_and_quantized_to_fp4() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../test_data/deepseek_v41/compressed_kv_fixture.json"
    ))
    .expect("fixture");
    let freqs = rope_freqs(64, 65536, 160_000.0, 16.0, 32.0, 1.0);
    let rows = fixture["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 7);
    for row in rows {
        let position = row["first_position"].as_u64().expect("position") as usize;
        let mut kv = hex_bf16(row["latent"].as_str().expect("latent"));
        let cached = hex_bf16(row["cached"].as_str().expect("cached"));
        compressed_kv_from_latent(&mut kv, position, &freqs);
        assert!(kv == cached, "group at position {position}");
    }
}

#[test]
fn rope_without_yarn_is_the_plain_geometric_ladder() {
    let freqs = rope_freqs(64, 0, 10_000.0, 16.0, 32.0, 1.0);
    assert_eq!(freqs.len(), 32);
    assert_eq!(freqs[0], 1.0);
    assert!((freqs[16] - 0.01).abs() < 1e-6);
    // YaRN leaves the fast pairs alone and divides the slow ones by `factor`.
    let yarn = rope_freqs(64, 65536, 160_000.0, 16.0, 32.0, 1.0);
    assert_eq!(yarn[0], 1.0);
    let plain = rope_freqs(64, 0, 160_000.0, 16.0, 32.0, 1.0);
    assert!((yarn[31] - plain[31] / 16.0).abs() < plain[31] * 1e-6);
}

#[test]
fn rotation_turns_each_pair_by_its_angle() {
    // A quarter turn of (1, 0) is (0, 1); position 0 is no turn.
    let quarter = std::f32::consts::FRAC_PI_2;
    let mut x = bf16(&[1.0, 0.0, 2.0, 0.0]);
    rotate_pairs(&mut x, 1, &[quarter, 0.0]);
    let got = f32s(&x);
    assert!(
        got[0].abs() < 1e-6 && (got[1] - 1.0).abs() < 1e-6,
        "{got:?}"
    );
    assert_eq!(&got[2..], &[2.0, 0.0]);
    let mut still = bf16(&[0.75, -0.5]);
    rotate_pairs(&mut still, 0, &[quarter]);
    assert_eq!(f32s(&still), [0.75, -0.5]);
}

#[test]
fn fp8_keys_take_a_power_of_two_scale_per_32_values() {
    // amax 256 -> scale 1: E4M3 values are kept, others round to 3 mantissa bits.
    let mut block = vec![0.0f32; 64];
    block[0] = 256.0;
    block[1] = 17.0; // E4M3 steps by 2 here: 16, 18. A tie goes to the even mantissa, 16.
    block[2] = -3.25;
    block[3] = 0.0005; // under half the smallest step (2^-10): gone
    // second block: amax 3 -> scale 2^-7
    block[32] = 3.0;
    block[33] = 17.0 / 128.0;
    let mut x = bf16(&block);
    fp8_quant_dequant(&mut x);
    let got = f32s(&x);
    assert_eq!(&got[..4], &[256.0, 16.0, -3.25, 0.0]);
    assert_eq!(&got[32..34], &[3.0, 0.125]);
    // Already-quantized values come back unchanged.
    let again = x.clone();
    fp8_quant_dequant(&mut x);
    assert_eq!(x, again);
}

#[test]
fn fp8_scale_never_falls_below_the_reference_floor() {
    // amax is floored at 1e-4, so a block of tiny values shares the scale
    // 2^ceil(log2(1e-4 / 448)) = 2^-22, and anything under 2^-32 rounds away.
    let mut x = bf16(&[2.0f32.powi(-40); 32]);
    fp8_quant_dequant(&mut x);
    assert!(f32s(&x).iter().all(|&v| v == 0.0));
    let mut y = bf16(&[2.0f32.powi(-30); 32]);
    fp8_quant_dequant(&mut y);
    assert!(f32s(&y).iter().all(|&v| v == 2.0f32.powi(-30)));
}

#[test]
fn fp4_index_keys_take_a_power_of_two_scale_per_32_values() {
    // amax 6 -> scale 1; E2M1 holds 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    let mut block = vec![0.0f32; 32];
    block[..8].copy_from_slice(&[6.0, 5.0, 2.5, 1.25, 0.25, -0.75, 3.5, -1.75]);
    let mut x = bf16(&block);
    fp4_quant_dequant_e8m0(&mut x);
    assert_eq!(&f32s(&x)[..8], &[6.0, 4.0, 2.0, 1.0, 0.0, -1.0, 4.0, -2.0]);
    // amax 7 -> scale 2: 7 / 2 = 3.5 ties to 4, back to 8.
    let mut wide = vec![0.0f32; 32];
    wide[0] = 7.0;
    let mut y = bf16(&wide);
    fp4_quant_dequant_e8m0(&mut y);
    assert_eq!(f32s(&y)[0], 8.0);
}

#[test]
fn the_window_ring_holds_the_last_128_positions() {
    assert_eq!(window_slot(0), 0);
    assert_eq!(window_slot(127), 127);
    assert_eq!(window_slot(128), 0);
    assert_eq!(window_slot(1448), 1448 % 128);
    // Prefill: a query sees its own position and the 127 before it.
    assert_eq!(prefill_window(0), 0..1);
    assert_eq!(prefill_window(127), 0..128);
    assert_eq!(prefill_window(1400), 1273..1401);
    // Decode: slots oldest first; a slot not yet written is -1.
    assert_eq!(decode_window_slots(1), {
        let mut slots = vec![-1i32; WINDOW];
        slots[WINDOW - 2] = 0;
        slots[WINDOW - 1] = 1;
        slots
    });
    let slots = decode_window_slots(1448);
    assert_eq!(slots.len(), WINDOW);
    assert_eq!(slots[0], (1448 + 1) % 128);
    assert_eq!(slots[WINDOW - 1], 1448 % 128);
}

#[test]
fn a_compression_group_completes_on_its_last_token() {
    // ratio 2: positions 1, 3, 5 ... complete groups 0, 1, 2 ...
    assert_eq!(completed_group(0, 2), None);
    assert_eq!(completed_group(1, 2), Some(0));
    assert_eq!(completed_group(1401, 2), Some(700));
    // ratio 1: every token is a group.
    assert_eq!(completed_group(0, 1), Some(0));
    assert_eq!(completed_group(1448, 1), Some(1448));
    // A group's latent takes its first token's position.
    assert_eq!(group_position(700, 2), 1400);
    assert_eq!(group_position(5, 1), 5);
    // How many groups a query at `pos` may attend to.
    assert_eq!(visible_groups(0, 2), 0);
    assert_eq!(visible_groups(1, 2), 1);
    assert_eq!(visible_groups(1400, 2), 700);
    assert_eq!(visible_groups(1401, 2), 701);
}
