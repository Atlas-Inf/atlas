// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

const AT: At = At {
    rank: 1,
    request: 7,
    chunk_start: 8192,
    layer: 12,
};

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + i / 251) as u8).collect()
}

#[test]
fn hash_matches_the_independent_reference_vectors() {
    // Computed by a separate Python implementation of the documented scheme.
    assert_eq!(hash_bytes(b""), 0x13ab_e5de_04a5_2a3b);
    assert_eq!(hash_bytes(b"abc"), 0x2cbb_7314_b82f_444a);
    let ramp: Vec<u8> = (0..100).collect();
    assert_eq!(hash_bytes(&ramp), 0x0976_158b_c1f1_851a);
    assert_eq!(hash_bytes(&[0x7f; 64]), 0x86a7_99d7_b797_10a0);
}

#[test]
fn hash_does_not_depend_on_how_updates_split_the_bytes() {
    let data = pattern(1000);
    let whole = hash_bytes(&data);
    for cut in [0, 1, 7, 31, 32, 33, 64, 500, 999, 1000] {
        let mut h = Hasher::new();
        h.update(&data[..cut]);
        h.update(&data[cut..]);
        assert_eq!(h.finish(), whole, "cut {cut}");
    }
    let mut h = Hasher::new();
    for piece in data.chunks(13) {
        h.update(piece);
    }
    assert_eq!(h.finish(), whole);
}

#[test]
fn hash_sees_every_bit_the_length_and_paired_sign_flips() {
    let data = pattern(256);
    let base = hash_bytes(&data);
    for bit in 0..data.len() * 8 {
        let mut flipped = data.clone();
        flipped[bit / 8] ^= 1 << (bit % 8);
        assert_ne!(hash_bytes(&flipped), base, "bit {bit}");
    }
    // Trailing zeros change the length, not the padded block.
    assert_ne!(hash_bytes(&[0; 31]), hash_bytes(&[0; 32]));
    assert_ne!(hash_bytes(&[]), hash_bytes(&[0]));
    // Two top-bit flips in one lane (a BF16 sign each) must not cancel.
    let mut pair = data.clone();
    pair[7] ^= 0x80;
    pair[39] ^= 0x80;
    assert_ne!(hash_bytes(&pair), base);
    // Swapped blocks and swapped lanes are different inputs.
    let mut swapped = data.clone();
    swapped.swap(0, 8);
    assert_ne!(hash_bytes(&swapped), base);
    let mut blocks = data.clone();
    blocks[..64].rotate_left(32);
    assert_ne!(hash_bytes(&blocks), base);
}

#[test]
fn line_format_is_fixed() {
    assert_eq!(
        format_line(AT, "attn_red", (4096, 4096), 33_554_432, Some(0xabc)),
        "DET r=1 q=7 c=8192 L=12 s=attn_red r0=4096 n=4096 b=33554432 h=0000000000000abc"
    );
    assert_eq!(
        format_line(AT, "sel", (0, 3), 12, None),
        "DET r=1 q=7 c=8192 L=12 s=sel r0=0 n=3 b=12 h=ERR"
    );
}

#[test]
fn level_and_stage_list_parse() {
    assert_eq!(parse_level(None), 0);
    assert_eq!(parse_level(Some("0")), 0);
    assert_eq!(parse_level(Some("true")), 0);
    assert_eq!(parse_level(Some("1")), 1);
    assert_eq!(parse_level(Some("2")), 2);
    assert!(stage_listed(None, "out"));
    assert!(!stage_listed(None, "x_gated"));
    assert!(stage_listed(Some("in,x_gated,attn"), "x_gated"));
    assert!(stage_listed(Some("in, out,final"), "out"));
    assert!(!stage_listed(Some("in,final"), "out"));
    assert!(!stage_listed(Some("moe_in"), "moe"));
}

#[test]
fn device_line_hashes_exactly_the_named_rows_across_segments() {
    let gpu = MockGpuBackend::new();
    let data = pattern(10 * 48);
    let ptr = gpu.alloc(data.len()).unwrap();
    gpu.copy_h2d(&data, ptr).unwrap();
    // Rows 2..7 of a [10, 48] tensor.
    let want = hash_bytes(&data[2 * 48..7 * 48]);
    let line = device_line(AT, &gpu, 0, "out", ptr.offset(2 * 48), (2, 5), 48);
    assert_eq!(
        line,
        format!("DET r=1 q=7 c=8192 L=12 s=out r0=2 n=5 b=240 h={want:016x}")
    );
    assert_eq!(gpu.sync_count(), 1);
    // Staged copies of any size give the hash of the whole span.
    for segment in [1, 5, 32, 100, 240, 4096] {
        assert_eq!(
            device_hash(&gpu, 0, ptr.offset(2 * 48), 240, segment),
            Some(want),
            "segment {segment}"
        );
    }
    // A span the device cannot serve is reported, not hashed.
    assert_eq!(device_hash(&gpu, 0, DevicePtr(0x10), 16, 16), None);
}

#[test]
fn taps_outside_a_traced_chunk_touch_nothing() {
    // No traced chunk on this thread (and the tracer is off unless the
    // environment enables it): no synchronize, no copy, no request number.
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(64).unwrap();
    on_stream(&gpu, 0).tap("out", ptr, (0, 1), 64);
    tap_hashed("plogits", (0, 1), 64, 1);
    set_layer(3);
    assert_eq!(gpu.sync_count(), 0);
    assert_eq!(gpu.d2h_blocking_count(), 0);
    assert_eq!(CURRENT.with(Cell::get), None);
    assert!(take_lines().is_empty());
}
