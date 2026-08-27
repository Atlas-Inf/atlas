// SPDX-License-Identifier: AGPL-3.0-only

//! Shared image + audio decoding for the multimodal preprocessors.
//!
//! Image path extracted verbatim from the Qwen3-VL preprocessor
//! (`vision_preprocess.rs`) so every model family applies the same base64
//! data-URI handling and the same dimension/alloc limits *before* any pixel
//! buffer is reserved. Keep this file behavior-identical: Qwen tests assert
//! the exact limit errors.

use anyhow::{Context, Result};
use image::{DynamicImage, ImageFormat, ImageReader, Limits};
use std::io::Cursor;

/// Decoder limit: reject a header declaring more than this on either side
/// before a single pixel is allocated. Everything is resized down to each
/// model's own target geometry anyway, so this only has to be above any
/// real camera image; 16384 is ~4× the long side of a 50 MP photo.
pub const DECODE_MAX_SIDE: u32 = 16_384;

/// Decoder limit: bytes the decoder may hold at once for one image. The
/// `image` crate's own default is 512 MiB, which on GB10's UNIFIED 121 GB
/// CPU+GPU memory is a per-request budget competing directly with the KV
/// cache — and the request body arrives over HTTP from an unauthenticated
/// caller. 192 MiB still admits an 8000×8000 RGB image.
pub const DECODE_MAX_ALLOC: u64 = 192 * 1024 * 1024;

/// Decode a base64 data URI or raw base64 string into a `DynamicImage`.
pub fn decode_image(data_uri: &str) -> Result<DynamicImage> {
    // Strip optional "data:image/<fmt>;base64," prefix.
    let b64 = if let Some(pos) = data_uri.find(",base64,") {
        &data_uri[pos + 8..]
    } else if data_uri.starts_with("data:") {
        // "data:image/jpeg;base64,..."
        data_uri
            .find(',')
            .map(|p| &data_uri[p + 1..])
            .unwrap_or(data_uri)
    } else {
        data_uri
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;

    // Probe format from magic bytes.
    let fmt = image::guess_format(&bytes).unwrap_or(ImageFormat::Jpeg);
    // Decode through `ImageReader` rather than `load_from_memory_with_format`
    // so the limits are ours. (The free function is not unlimited — it applies
    // `Limits::default()`, i.e. 512 MiB alloc — but it sets NO dimension cap,
    // and the alloc cap is documented as non-strict.) A 40-byte PNG header can
    // declare 65535×65535; the dimension limit rejects that from the header,
    // before any buffer is reserved.
    let mut reader = ImageReader::new(Cursor::new(&bytes));
    reader.set_format(fmt);
    let mut limits = Limits::default();
    limits.max_image_width = Some(DECODE_MAX_SIDE);
    limits.max_image_height = Some(DECODE_MAX_SIDE);
    limits.max_alloc = Some(DECODE_MAX_ALLOC);
    reader.limits(limits);
    reader.decode().context("image decode failed")
}

/// Decoder limit: bytes of PCM audio the WAV decoder may hold at once. The
/// gemma-4-E2B front end caps a clip at `token_cap` mel frames (750) ≈ 12 s
/// at 16 kHz — a 60 s clip is ~1.9 MB of s16; 64 MiB is generous headroom
/// while bounding the unauthenticated HTTP allocation.
pub const DECODE_MAX_AUDIO_BYTES: u64 = 64 * 1024 * 1024;

/// Decode a base64 data-URI or raw base64 string holding a RIFF/WAVE file
/// into mono f32 samples (`[-1, 1]`), at the file's own sample rate.
///
/// Accepts PCM (s16/u8/s32) and IEEE-float WAVs; multi-channel audio is
/// averaged to mono (the mel front end requires mono 16 kHz — the caller
/// resamples). Fails fast on a non-RIFF header, a format we cannot decode,
/// or a payload over [`DECODE_MAX_AUDIO_BYTES`].
pub fn decode_wav(data_uri: &str) -> Result<(Vec<f32>, u32)> {
    let b64 = if let Some(pos) = data_uri.find(",base64,") {
        &data_uri[pos + 8..]
    } else if data_uri.starts_with("data:") {
        data_uri
            .find(',')
            .map(|p| &data_uri[p + 1..])
            .unwrap_or(data_uri)
    } else {
        data_uri
    };
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;
    if bytes.len() as u64 > DECODE_MAX_AUDIO_BYTES {
        anyhow::bail!("audio payload too large ({} bytes)", bytes.len());
    }
    parse_wav(&bytes)
}

/// Parse a RIFF/WAVE container: the `fmt ` chunk (format tag, channels,
/// sample rate, bits per sample) plus the `data` chunk. Chunk walking skips
/// unknown chunks (LIST, fact, ...) and pads to even offsets per the RIFF
/// spec. Only PCM (1) and IEEE float (3) encodings are decoded; anything
/// else (µ-law, ADPCM, ...) is rejected rather than mis-decoded (PCND).
fn parse_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("not a RIFF/WAVE file");
    }
    let mut fmt = None;
    let mut data = None;
    let mut off = 12usize;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        let body = off + 8;
        let end = body.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " if size >= 16 => {
                let tag = u16::from_le_bytes([bytes[body], bytes[body + 1]]);
                let ch = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
                let rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
                let bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
                fmt = Some((tag, ch as usize, rate, bits));
            }
            b"data" => {
                data = Some(&bytes[body..end]);
            }
            _ => {}
        }
        // RIFF chunks are word-aligned.
        off = end + (size & 1);
    }
    let Some((tag, channels, rate, bits)) = fmt else {
        anyhow::bail!("WAV has no fmt chunk");
    };
    let Some(pcm) = data else {
        anyhow::bail!("WAV has no data chunk");
    };
    if channels == 0 {
        anyhow::bail!("WAV declares 0 channels");
    }
    let bytes_per_sample = (bits as usize).div_ceil(8);
    let frame = bytes_per_sample * channels;
    if frame == 0 || pcm.len() % frame != 0 {
        anyhow::bail!("WAV data chunk is not a whole number of frames");
    }
    let n = pcm.len() / frame;
    let mut samples = Vec::with_capacity(n);
    for f in 0..n {
        let mut acc = 0.0f64;
        for c in 0..channels {
            let s =
                &pcm[(f * frame + c * bytes_per_sample)..(f * frame + (c + 1) * bytes_per_sample)];
            let v = match (tag, bits) {
                (1, 8) => (s[0] as f32 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
                (1, 24) => {
                    let i = ((s[2] as i32) << 16) | ((s[1] as i32) << 8) | (s[0] as i32);
                    (i as f32) / 8_388_608.0
                }
                (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
                (3, _) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                (t, b) => anyhow::bail!("unsupported WAV encoding (tag {t}, {b} bits)"),
            };
            acc += v as f64;
        }
        samples.push((acc / channels as f64) as f32);
    }
    Ok((samples, rate))
}
