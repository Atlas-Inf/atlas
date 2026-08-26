// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B audio input + the host-side geometry the audio tower needs
//! (Wave 4A of the multimodal bring-up).
//!
//! The mel frontend ([`super::mel`]) turns a waveform into
//! [`MelOutput`](super::mel::MelOutput); the server wraps that in
//! [`GemmaAudioInput`] with a per-frame validity mask and hands it to
//! `GemmaAudioEncoder::forward_batched`. Everything in this module is pure
//! host math — deterministic geometry that the GPU kernels (Wave 4C) must
//! reproduce, so it is unit-tested here as the contract.

use anyhow::Context;
use atlas_core::config::GemmaAudioConfig;

use super::{decode, mel};

/// Linear-interpolation resampler for the WAV→16 kHz front-end path. The
/// mel front end is defined at [`mel::SAMPLE_RATE`]; any other rate the WAV
/// declares is resampled here so a 44.1/48 kHz clip still produces the
/// correct log-mel (the HF processor's `torchaudio` resampler is a sinc
/// kernel; linear interpolation is a close, dependency-free stand-in).
fn resample_linear(src: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || src.is_empty() {
        return src.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n_out = (src.len() as f64 / ratio) as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f64 * ratio;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(src.len() - 1);
        let frac = (pos - lo as f64) as f32;
        out.push(src[lo] * (1.0 - frac) + src[hi] * frac);
    }
    out
}

/// One audio clip as consumed by the Gemma-4 E2B audio tower.
///
/// Mirrors the HF tower's `input_features [B, T_mel, 128]` +
/// `input_features_mask [B, T_mel]` (True = valid) pair.
#[derive(Debug, Clone)]
pub struct GemmaAudioInput {
    /// Log-mel features, `[n_frames × n_mels]` row-major (frame-major),
    /// produced by [`mel_spectrogram`](super::mel::mel_spectrogram).
    pub features: Vec<f32>,
    /// STFT frame count (the conv subsampling reduces this 4×).
    pub n_frames: usize,
    /// Mel bin count — must equal `cfg.mel_bins` (128 on the shipped tower).
    pub n_mels: usize,
    /// Per-frame validity, `len == n_frames`; 1 = valid, 0 = padding. The
    /// tower applies it multiplicatively in the subsample convs and drops
    /// invalid tokens before packing `buf_out`.
    pub mask: Vec<u8>,
}

/// Frame count after the two stride-2 subsample convs (pad 1, kernel 3):
/// `t → floor((t−1)/2)+1 → floor((t′−1)/2)+1`, which simplifies to `(t+3)/4`.
/// Verified against the HF conv geometry (`Gemma4AudioSubSampleConvProjection`):
/// `t=8 → 2`, `t=4 → 1`, `t=12 → 3`, `t=1 → 1`.
pub fn subsample_conv_len(t: usize) -> usize {
    t.div_ceil(4)
}

/// Subsample a per-frame validity mask through the two stride-2 convs.
///
/// HF (`Gemma4AudioSubSampleConvProjection`): the mask is `mask[:, ::2]`
/// after each conv, so the conv-output position `i` inherits validity from
/// input position `4i`: `out[i] = mask[4*i]`.
pub fn subsample_mask(mask: &[u8]) -> Vec<u8> {
    let n_out = subsample_conv_len(mask.len());
    (0..n_out).map(|i| mask[4 * i]).collect()
}

/// The relative position embedding of the audio tower, `[(context/2+1) ×
/// hidden]` f32 row-major: `[sin..., cos...]` concatenated, exactly the HF
/// `Gemma4AudioRelPositionalEncoding` (sinusoidal, min timescale 1,
/// max 10000, `num_timescales = hidden/2`).
///
/// Pure config math — the checkpoint ships no tensor for it (verified: no
/// `rel_pos_enc` keys under `model.audio_tower.*`).
pub fn rel_pos_embeddings(hidden: usize, context_size: usize) -> Vec<f32> {
    let num_timescales = hidden / 2;
    let log_increment = (10000.0f32).ln() / (num_timescales as f32 - 1.0).max(1.0);
    let inv: Vec<f32> = (0..num_timescales)
        .map(|k| (-(k as f32) * log_increment).exp())
        .collect();
    let n_pos = context_size / 2 + 1;
    let mut emb = vec![0.0f32; n_pos * hidden];
    for i in 0..n_pos {
        let pos = (context_size / 2 - i) as f32; // context/2 .. 0
        for k in 0..num_timescales {
            let a = pos * inv[k];
            emb[i * hidden + k] = a.sin();
            emb[i * hidden + num_timescales + k] = a.cos();
        }
    }
    emb
}

/// The blocked attention mask the chunked local attention consumes:
/// `[nblocks × chunk × context]` u8 (1 = attend, 0 = fill with the invalid
/// logit value), mirroring HF's `_convert_4d_mask_to_blocked_5d` over the
/// banded causal mask from `sliding_window_mask_function((left−1, right))`.
///
/// Block `b`, query `j` (global `q = b·chunk + j`), context slot `c`:
/// `kv = b·chunk + c − (left−1)`. Attend iff `0 ≤ kv < t`, `valid[kv]`,
/// `valid[q]`, and `q − (left−1) ≤ kv ≤ q + right` (causal window: with
/// `right = 0` a query attends to itself and up to `left−1` past tokens).
///
/// Returns `(mask_bytes, nblocks)`.
pub fn build_blocked_attn_mask(
    valid: &[u8],
    chunk: usize,
    max_past: usize,
    max_future: usize,
) -> (Vec<u8>, usize) {
    let t = valid.len();
    let context = chunk + max_past + max_future;
    let nblocks = t.div_ceil(chunk);
    let mut mask = vec![0u8; nblocks * chunk * context];
    for b in 0..nblocks {
        for j in 0..chunk {
            let q = b * chunk + j;
            for c in 0..context {
                let kv = (b * chunk + c) as i64 - max_past as i64;
                let attend = q < t
                    && valid[q] == 1
                    && kv >= 0
                    && (kv as usize) < t
                    && valid[kv as usize] == 1
                    && q as i64 - max_past as i64 <= kv
                    && kv <= q as i64 + max_future as i64;
                mask[(b * chunk + j) * context + c] = attend as u8;
            }
        }
    }
    (mask, nblocks)
}

/// Validate the audio-config geometry the encoder cannot stretch (PCND):
/// exactly two subsample conv stages, `mel_bins` divisible by 4 (the mel dim
/// halves twice), even `hidden_size` (sin/cos split), and a non-zero token
/// budget. Refused at `GemmaAudioEncoder::new`.
pub fn validate_audio_geometry(cfg: &GemmaAudioConfig) -> anyhow::Result<()> {
    use anyhow::bail;
    if cfg.subsampling_conv_channels.len() != 2 {
        bail!(
            "gemma audio: subsampling_conv_channels {:?} — the tower has exactly two \
             stride-2 conv stages",
            cfg.subsampling_conv_channels
        );
    }
    if !cfg.mel_bins.is_multiple_of(4) {
        bail!(
            "gemma audio: mel_bins {} not divisible by 4 (two stride-2 convs halve it twice)",
            cfg.mel_bins
        );
    }
    if !cfg.hidden_size.is_multiple_of(2) || cfg.hidden_size == 0 {
        bail!(
            "gemma audio: hidden_size {} must be positive and even (sin/cos rel-pos split)",
            cfg.hidden_size
        );
    }
    if cfg.num_attention_heads == 0 || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
        bail!(
            "gemma audio: {} heads must divide hidden_size {}",
            cfg.num_attention_heads,
            cfg.hidden_size
        );
    }
    if cfg.token_cap == 0 {
        bail!("gemma audio: token_cap must be > 0");
    }
    Ok(())
}

/// Build a [`GemmaAudioInput`] from a base64 WAV data URI.
///
/// Pipeline: [`decode::decode_wav`] → resample to the 16 kHz front end →
/// truncate to the `token_cap` mel budget → [`mel::mel_spectrogram`] → an
/// all-valid mask (the server rejects clips beyond the budget instead of
/// padding — HF truncates; we fail fast so the token expander's count is
/// authoritative). Returns a human-readable error for the HTTP layer.
pub fn gemma_audio_input_from_wav(
    data_uri: &str,
    cfg: &GemmaAudioConfig,
) -> anyhow::Result<GemmaAudioInput> {
    let (samples, rate) = decode::decode_wav(data_uri).context("audio decode failed")?;
    let wav = resample_linear(&samples, rate, mel::SAMPLE_RATE as u32);
    if wav.len() < cfg.frame_length {
        anyhow::bail!(
            "audio too short: {} samples < frame_length {}",
            wav.len(),
            cfg.frame_length
        );
    }
    // token_cap soft tokens ↔ n_frames mel frames (4× subsample reduction):
    // cap the waveform so the mel frame count never exceeds token_cap*4.
    let max_frames = cfg.token_cap * 4;
    let max_samples = cfg.frame_length + (max_frames - 1) * cfg.hop_length;
    let wav = if wav.len() > max_samples {
        &wav[..max_samples]
    } else {
        &wav[..]
    };
    let mel = mel::mel_spectrogram(wav, cfg).map_err(|e| anyhow::anyhow!("mel: {e}"))?;
    let mask = vec![1u8; mel.n_frames];
    Ok(GemmaAudioInput {
        features: mel.features,
        n_frames: mel.n_frames,
        n_mels: mel.n_mels,
        mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GemmaAudioConfig {
        GemmaAudioConfig {
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            subsampling_conv_channels: vec![8, 4],
            conv_kernel_size: 5,
            attention_chunk_size: 4,
            attention_context_left: 3,
            attention_context_right: 0,
            output_proj_dims: 1536,
            residual_weight: 0.5,
            use_clipped_linears: true,
            audio_token_id: 258_881,
            mel_bins: 16,
            frame_length: 320,
            hop_length: 160,
            fft_size: 512,
            mel_floor: 1e-3,
            mel_scale: "htk".to_string(),
            token_cap: 8,
            norm_eps: 1e-6,
            activation: "silu".to_string(),
            boa_token_id: 256_000,
            eoa_token_id: 258_883,
        }
    }

    /// The two stride-2 convs reduce the frame dim 4×: 8→2, 4→1, 12→3; a
    /// single frame survives as one token (pad 1 keeps it).
    #[test]
    fn subsample_len_matches_conv_math() {
        for &(t, out) in &[
            (8usize, 2usize),
            (4, 1),
            (12, 3),
            (1, 1),
            (2, 1),
            (6, 2),
            (9, 3),
            (0, 0),
        ] {
            assert_eq!(subsample_conv_len(t), out, "t={t}");
        }
    }

    /// `mask[:, ::2]` twice = every 4th element, with the conv output length.
    #[test]
    fn subsample_mask_takes_every_fourth() {
        assert_eq!(subsample_mask(&[1; 8]), vec![1, 1]);
        assert_eq!(subsample_mask(&[1, 1, 1, 1, 0, 0, 0, 0]), vec![1, 0]);
        assert_eq!(subsample_mask(&[0, 1, 1, 1, 1, 1, 1, 1]), vec![0, 1]);
        assert_eq!(subsample_mask(&[1; 5]), vec![1, 1]); // t=5 → 2 out
        assert_eq!(subsample_mask(&[1; 4]), vec![1]);
    }

    /// `rel_pos_embeddings` matches the HF sinusoid: positions `context/2..0`,
    /// `[sin..., cos...]` per row with inv-timescales `exp(−k·ln(10000)/N)`.
    #[test]
    fn rel_pos_matches_hf_sinusoid() {
        // hidden 4, context 8 → 2 timescales, inv = [1, 1e-4], 5 positions.
        let e = rel_pos_embeddings(4, 8);
        assert_eq!(e.len(), 5 * 4);
        // Position 4 (row 0): [sin(4·1), sin(4·1e-4), cos(4·1), cos(4·1e-4)].
        let (s4, c4) = (4.0f32.sin(), 4.0f32.cos());
        let (s4e, c4e) = ((4.0f32 * 1e-4).sin(), (4.0f32 * 1e-4).cos());
        assert!((e[0] - s4).abs() < 1e-6);
        assert!((e[1] - s4e).abs() < 1e-6);
        assert!((e[2] - c4).abs() < 1e-6);
        assert!((e[3] - c4e).abs() < 1e-6);
        // Position 0 (last row): [0, 0, 1, 1].
        assert!(e[16].abs() < 1e-6 && e[17].abs() < 1e-6);
        assert!((e[18] - 1.0).abs() < 1e-6 && (e[19] - 1.0).abs() < 1e-6);
    }

    /// Blocked mask: chunk 4, left 3 (max_past 2), right 0 → context 6, and
    /// the per-query attend windows `q−2 .. q` composed with the validity
    /// mask — hand-computed below.
    #[test]
    fn blocked_mask_causal_window_and_validity() {
        let valid = [1u8, 1, 0, 1];
        let (m, nb) = build_blocked_attn_mask(&valid, 4, 2, 0);
        assert_eq!(nb, 1);
        assert_eq!(m.len(), 4 * 6);
        let row = |q: usize| -> Vec<u8> { m[q * 6..q * 6 + 6].to_vec() };
        // q=0: kv ∈ {0} (window −2..0 ∧ valid) → c = kv+2 = 2.
        assert_eq!(row(0), vec![0, 0, 1, 0, 0, 0]);
        // q=1: kv ∈ {0,1} → c = 2,3.
        assert_eq!(row(1), vec![0, 0, 1, 1, 0, 0]);
        // q=2 is ITSELF invalid (valid[2]=0) → the whole row is masked, like
        // HF's 4D mask (query validity baked in).
        assert_eq!(row(2), vec![0, 0, 0, 0, 0, 0]);
        // q=3: kv ∈ {1,2,3}, valid[2]=0 → c = 3,5.
        assert_eq!(row(3), vec![0, 0, 0, 1, 0, 1]);
    }

    /// Two blocks: t=6 over chunk 4 — block 0 covers q 0..3, block 1 covers
    /// q 4..5 (+2 pad rows whose window still lands on real positions for
    /// the first rows; pad queries are never packed anyway).
    #[test]
    fn blocked_mask_two_blocks() {
        let valid = [1u8; 6];
        let (m, nb) = build_blocked_attn_mask(&valid, 4, 2, 0);
        assert_eq!(nb, 2);
        assert_eq!(m.len(), 2 * 4 * 6);
        // Block 1, q=4 (global): kv ∈ {2,3,4} → c = kv − 4 + 2 → {0,1,2}.
        assert_eq!(m[4 * 6..5 * 6], vec![1, 1, 1, 0, 0, 0]);
        // q=5: kv ∈ {3,4,5} → c = {1,2,3}.
        assert_eq!(m[5 * 6..6 * 6], vec![0, 1, 1, 1, 0, 0]);
        // Pad q=6,7: no attend (q ≥ t).
        assert_eq!(m[6 * 6..7 * 6], vec![0; 6]);
        assert_eq!(m[7 * 6..8 * 6], vec![0; 6]);
    }

    /// Geometry validation refuses what the encoder cannot stretch.
    #[test]
    fn validate_rejects_bad_geometry() {
        let c = cfg();
        assert!(validate_audio_geometry(&c).is_ok());
        let mut bad = cfg();
        bad.subsampling_conv_channels = vec![8];
        assert!(validate_audio_geometry(&bad).is_err());
        let mut bad = cfg();
        bad.mel_bins = 18;
        assert!(validate_audio_geometry(&bad).is_err());
        let mut bad = cfg();
        bad.hidden_size = 17;
        assert!(validate_audio_geometry(&bad).is_err());
        let mut bad = cfg();
        bad.num_attention_heads = 5;
        assert!(validate_audio_geometry(&bad).is_err());
    }
}
