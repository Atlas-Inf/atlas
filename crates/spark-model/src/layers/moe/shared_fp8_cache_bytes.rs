// SPDX-License-Identifier: AGPL-3.0-only
//! Independent, chunked resident-cache oracle. Decoded layout is native [N,K],
//! not the [K/2,N] layout of the separate packed W4A16 GEMM weight view.
use crate::weight_map::QuantizedWeight;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::OnceLock;

const CHUNK_ROWS: usize = 64;
const HOST_BOUND: usize = 512 * 1024;
const MAX_COEFFICIENTS: usize = 2048 * 4096;
const FP4: [f32; 16] = [
    0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
];

fn decode_e4m3(code: u8) -> f32 {
    let magnitude = code & 127;
    if magnitude == 127 {
        return f32::NAN;
    }
    let exponent = magnitude >> 3;
    let fraction = magnitude & 7;
    let value = if exponent == 0 {
        f32::from(fraction) / 512.
    } else {
        (1. + f32::from(fraction) / 8.) * 2f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 != 0 { -value } else { value }
}

// Independent nearest-code search in f64, not the CUDA conversion primitive.
// Values are exact binary fractions; midpoint ties select the even code.
fn encode_e4m3(value: f32) -> u8 {
    static VALUES: OnceLock<[f64; 127]> = OnceLock::new();
    let sign = if value.is_sign_negative() { 128 } else { 0 };
    if value.is_nan() {
        return 127;
    }
    if value.abs() >= 448. {
        return 126 | sign;
    }
    let values = VALUES.get_or_init(|| std::array::from_fn(|i| f64::from(decode_e4m3(i as u8))));
    let magnitude = f64::from(value.abs());
    let mut best = 0;
    let mut error = f64::INFINITY;
    for (code, &decoded) in values.iter().enumerate() {
        let distance = (magnitude - decoded).abs();
        if distance < error || (distance == error && code & 1 == 0) {
            best = code;
            error = distance;
        }
    }
    best as u8 | sign
}

fn checked_span(ptr: DevicePtr, bytes: usize) -> Result<std::ops::Range<u64>> {
    anyhow::ensure!(
        !ptr.is_null() && ptr.0.is_multiple_of(16),
        "shared FP8 byte oracle null/alignment"
    );
    Ok(ptr.0
        ..ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| anyhow::anyhow!("shared FP8 byte oracle address overflow"))?)
}

fn validate(original: &QuantizedWeight, decoded: DevicePtr, n: usize, k: usize) -> Result<usize> {
    // The loader independently enforces the exact GU/down production shapes.
    // This oracle accepts smaller bounded matrices for independent fixtures.
    anyhow::ensure!(
        n > 0 && n <= 4096 && k > 0 && k <= 4096 && k.is_multiple_of(16),
        "shared FP8 byte oracle dimensions"
    );
    let coefficients = n
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("shared FP8 byte oracle shape overflow"))?;
    anyhow::ensure!(
        coefficients <= MAX_COEFFICIENTS,
        "shared FP8 byte oracle coefficient bound"
    );
    anyhow::ensure!(
        original.weight_scale_2.is_finite() && !original.has_per_row_scale2(),
        "shared FP8 byte oracle scalar scale2"
    );
    let spans = [
        checked_span(original.weight, coefficients / 2)?,
        checked_span(original.weight_scale, coefficients / 16)?,
        checked_span(decoded, coefficients)?,
    ];
    for i in 0..3 {
        for j in i + 1..3 {
            anyhow::ensure!(
                spans[i].end <= spans[j].start || spans[j].end <= spans[i].start,
                "shared FP8 byte oracle source/destination alias"
            );
        }
    }
    let chunk = n.min(CHUNK_ROWS) * k;
    // Three staging arrays + conversion LUT + static codebook + conservative
    // overhead. No full resident weight snapshot or GPU allocation is made.
    let staging = chunk + chunk / 2 + chunk / 16 + 4096 + 127 * 8 + 4096;
    anyhow::ensure!(staging <= HOST_BOUND, "shared FP8 byte oracle host bound");
    Ok(chunk)
}

pub(super) fn verify_predecoded(
    gpu: &dyn GpuBackend,
    original: &QuantizedWeight,
    decoded: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !gpu.stream_is_capturing(stream),
        "shared FP8 byte oracle cannot run in graph capture"
    );
    let chunk = validate(original, decoded, n, k)?;
    let mut table = [[0u8; 16]; 256];
    for (scale, entries) in table.iter_mut().enumerate() {
        if scale & 127 == 127 {
            continue;
        }
        // Preserve the converter's TWO f32 multiplications in this order.
        let scaled = decode_e4m3(scale as u8) * original.weight_scale_2;
        for (nibble, value) in entries.iter_mut().enumerate() {
            *value = encode_e4m3(FP4[nibble] * scaled);
        }
    }
    let mut packed = vec![0u8; chunk / 2];
    let mut scales = vec![0u8; chunk / 16];
    let mut actual = vec![0u8; chunk];
    for first_row in (0..n).step_by(CHUNK_ROWS) {
        let count = (n - first_row).min(CHUNK_ROWS) * k;
        gpu.copy_d2h_on_stream(
            original.weight.offset(first_row * k / 2),
            &mut packed[..count / 2],
            stream,
        )?;
        gpu.copy_d2h_on_stream(
            original.weight_scale.offset(first_row * k / 16),
            &mut scales[..count / 16],
            stream,
        )?;
        gpu.copy_d2h_on_stream(decoded.offset(first_row * k), &mut actual[..count], stream)?;
        for i in 0..count {
            let scale = scales[i / 16];
            anyhow::ensure!(
                scale & 127 != 127,
                "shared FP8 byte oracle nonfinite block scale at group {}",
                (first_row * k + i) / 16
            );
            let nibble = (packed[i / 2] >> ((i & 1) * 4)) & 15;
            let expected = table[scale as usize][nibble as usize];
            anyhow::ensure!(
                expected & 127 != 127,
                "shared FP8 byte oracle nonfinite decoded reference at byte {}",
                first_row * k + i
            );
            anyhow::ensure!(
                actual[i] == expected,
                "shared FP8 cache mismatch byte {}: expected {expected:#04x}, actual {:#04x}",
                first_row * k + i,
                actual[i]
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "shared_fp8_cache_bytes_tests.rs"]
mod tests;
