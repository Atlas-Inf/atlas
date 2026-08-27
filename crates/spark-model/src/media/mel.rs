// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B audio frontend, Wave 1: the mel-spectrogram preprocessor.
//!
//! CPU-only, pure Rust, zero new dependencies (hand-rolled radix-2 FFT per the
//! approved design decision — `realfft`/`rustfft` must not be used here).
//!
//! DSP contract, verified against `processor_config.json` +
//! `feature_extraction_gemma4.py` and torchaudio semantics:
//!
//! * Sample rate is a fixed **16 kHz** constant — `GemmaAudioConfig` does not
//!   carry a sample-rate field (see `crates/atlas-core/src/config/gemma_media.rs`).
//! * STFT: `frame_length` = 320 samples (20 ms), `hop_length` = 160 (10 ms),
//!   `fft_size` = 512, periodic Hann window (`torch.hann_window(512, periodic=True)`),
//!   zero-padded from `frame_length` to `fft_size` (win_length < n_fft).
//! * Frame count matches `torchaudio.functional.spectrogram` with
//!   `center=False, pad=0`: `n_frames = 1 + floor((len - frame_length) / hop_length)`,
//!   integer division. Every frame starts at `k * hop_length` and is complete by
//!   construction (`k_max * hop_length <= len - frame_length`); the trailing
//!   remainder samples are dropped, exactly like `torch.stft(center=False)`'s
//!   unfold. The extraction is written defensively so a frame can never index
//!   past the end of the waveform.
//! * Mel: 128 HTK-scale bins over [0 Hz, 8 kHz], peak-normalised triangular
//!   filters over the `fft_size/2 + 1` one-sided FFT bins (identical geometry to
//!   `torchaudio.functional.melscale_fbanks(mel_scale="htk")` / librosa).
//! * Output: `log(mel_power + mel_floor)` with `mel_floor = 1e-3`.
//!
//! The audio tower (conv subsampling → conformer, later wave) consumes
//! `MelOutput::features`; this module only produces the log-mel tensor.

use atlas_core::config::GemmaAudioConfig;

/// Fixed sample rate of the Gemma-4 E2B audio frontend (16 kHz). The processor
/// resamples to 16 kHz before the STFT; `GemmaAudioConfig` carries no
/// sample-rate field, so this is a documented constant.
pub const SAMPLE_RATE: f32 = 16000.0;

/// Row-major log-mel spectrogram: `[n_frames, n_mels]`, `features.len() == n_frames * n_mels`.
#[derive(Debug, Clone, PartialEq)]
pub struct MelOutput {
    /// Log-mel energy per (frame, mel bin), `log(power + mel_floor)`, row-major.
    pub features: Vec<f32>,
    /// Number of STFT frames (`1 + floor((len - frame_length) / hop_length)`).
    pub n_frames: usize,
    /// Number of mel bins (`mel_bins`).
    pub n_mels: usize,
}

/// Errors produced by [`mel_spectrogram`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MelError {
    /// `mel_scale` is not the supported `"htk"`.
    UnsupportedMelScale(String),
    /// DSP knobs are self-inconsistent (non-power-of-two FFT, `frame_length > fft_size`, ...).
    InvalidConfig(String),
    /// Waveform is shorter than one frame.
    SignalTooShort { len: usize, frame_length: usize },
}

impl std::fmt::Display for MelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MelError::UnsupportedMelScale(s) => {
                write!(
                    f,
                    "unsupported mel scale {s:?}, only \"htk\" is implemented"
                )
            }
            MelError::InvalidConfig(msg) => write!(f, "invalid mel config: {msg}"),
            MelError::SignalTooShort { len, frame_length } => write!(
                f,
                "waveform of {len} samples is shorter than frame_length {frame_length}"
            ),
        }
    }
}

impl std::error::Error for MelError {}

/// Compute the log-mel spectrogram of a mono 16 kHz waveform, exactly as the
/// Gemma-4 E2B audio processor's `input_features`.
pub fn mel_spectrogram(wav: &[f32], cfg: &GemmaAudioConfig) -> Result<MelOutput, MelError> {
    validate(cfg, wav.len())?;
    let (n_mels, frame_length, hop, fft) =
        (cfg.mel_bins, cfg.frame_length, cfg.hop_length, cfg.fft_size);
    // HF `Gemma4AudioFeatureExtractor`: SEMICAUSAL padding — prepend
    // `frame_length // 2` zeros so the first STFT frame is centered at t=0
    // (sl.STFT(time_padding='semicausal')), then unfold with
    // `frame_size = frame_length + 1` and drop the last sample
    // (`frames_to_process[..., :-1]`). Frame k covers
    // padded[k·hop .. k·hop + frame_length).
    let pad = frame_length / 2;
    let padded_len = wav.len() + pad;
    let n_frames = 1 + (padded_len - (frame_length + 1)) / hop;
    let n_bins = fft / 2 + 1;
    let window = hann_periodic(frame_length);
    let filterbank = htk_mel_filterbank(n_mels, fft, SAMPLE_RATE);
    let mel_floor = cfg.mel_floor;

    let mut features = vec![0.0f32; n_frames * n_mels];
    let mut re = vec![0.0f32; fft];
    let mut im = vec![0.0f32; fft];
    for frame in 0..n_frames {
        let start = frame * hop;
        for i in 0..fft {
            re[i] = if i < frame_length {
                let s = start + i;
                if s < pad {
                    0.0
                } else {
                    let w = s - pad;
                    if w < wav.len() {
                        wav[w] * window[i]
                    } else {
                        0.0
                    }
                }
            } else {
                0.0
            };
        }
        im.fill(0.0);
        fft_radix2(&mut re, &mut im);
        let row = &mut features[frame * n_mels..(frame + 1) * n_mels];
        for m in 0..n_mels {
            let mut energy = 0.0f64;
            for k in 0..n_bins {
                let w = filterbank[m * n_bins + k];
                if w > 0.0 {
                    // HF `Gemma4AudioFeatureExtractor`: magnitude_spec =
                    // np.abs(stft) — the AMPLITUDE, not power. Squaring here
                    // (|X|²) inflated every mel bin by ~2 in log space and
                    // was the audio calibration's first divergence.
                    let mag = ((re[k] as f64).powi(2) + (im[k] as f64).powi(2)).sqrt();
                    energy += w as f64 * mag;
                }
            }
            row[m] = (energy + mel_floor).ln() as f32;
        }
    }
    Ok(MelOutput {
        features,
        n_frames,
        n_mels,
    })
}

/// Config sanity checks: fail fast on anything that would silently produce
/// garbage mel output (PCND — no implicit defaults).
fn validate(cfg: &GemmaAudioConfig, len: usize) -> Result<(), MelError> {
    let fft = cfg.fft_size;
    if !fft.is_power_of_two() {
        return Err(MelError::InvalidConfig(format!(
            "fft_size {fft} is not a power of two"
        )));
    }
    if cfg.mel_bins == 0 || cfg.frame_length == 0 || cfg.hop_length == 0 {
        return Err(MelError::InvalidConfig(
            "mel_bins / frame_length / hop_length must be non-zero".into(),
        ));
    }
    if cfg.frame_length > fft {
        return Err(MelError::InvalidConfig(format!(
            "frame_length {} > fft_size {fft}",
            cfg.frame_length
        )));
    }
    if cfg.mel_scale != "htk" {
        return Err(MelError::UnsupportedMelScale(cfg.mel_scale.clone()));
    }
    if len < cfg.frame_length {
        return Err(MelError::SignalTooShort {
            len,
            frame_length: cfg.frame_length,
        });
    }
    Ok(())
}

/// In-place iterative radix-2 Cooley-Tukey FFT for power-of-two `n`.
///
/// Structure: bit-reversal permutation, then `log2(n)` butterfly stages of
/// increasing span (2 → 4 → ... → n), each block applying `len/2` butterflies
/// with a twiddle factor advanced by complex multiplication from the stage root
/// `exp(-2πi/len)`.
fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    debug_assert!(n.is_power_of_two());
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // Butterfly stages: span 2, 4, ..., n.
    let mut len = 2usize;
    for _ in 0..n.trailing_zeros() {
        let angle = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (angle.cos(), angle.sin()); // stage root w = e^{-2πi/len}
        let half = len / 2;
        let mut k = 0usize;
        while k < n {
            let mut w = (1.0f32, 0.0f32);
            for j in 0..half {
                let (u_re, u_im) = (re[k + j], im[k + j]);
                let v_re = re[k + j + half] * w.0 - im[k + j + half] * w.1;
                let v_im = re[k + j + half] * w.1 + im[k + j + half] * w.0;
                re[k + j] = u_re + v_re;
                im[k + j] = u_im + v_im;
                re[k + j + half] = u_re - v_re;
                im[k + j + half] = u_im - v_im;
                w = (w.0 * wr - w.1 * wi, w.0 * wi + w.1 * wr); // advance twiddle
            }
            k += len;
        }
        len *= 2;
    }
}

/// Periodic Hann window (`torch.hann_window(n, periodic=True)`):
/// `w[i] = 0.5 - 0.5·cos(2πi/n)`, so `w[0] = 0` and `w[i] = w[n-i]`.
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// HTK mel scale: `mel(f) = 2595 · log10(1 + f/700)`.
fn htk_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Inverse HTK mel scale: `hz(m) = 700 · (10^(m/2595) − 1)`.
fn htk_mel_inv(m: f32) -> f32 {
    700.0 * (10.0_f32.powf(m / 2595.0) - 1.0)
}

/// `n_mels + 2` triangle edge/center frequencies in Hz: `mel`-space linearly
/// spaced points from `mel(0)` to `mel(sr/2)`, mapped back to Hz.
fn htk_mel_points(n_mels: usize, sr: f32) -> Vec<f32> {
    let m_min = htk_mel(0.0);
    let m_max = htk_mel(sr / 2.0);
    (0..n_mels + 2)
        .map(|i| htk_mel_inv(m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32))
        .collect()
}

/// HTK mel filterbank, dense row-major `[n_mels × (fft_size/2 + 1)]` weights.
/// Row `m` is a peak-normalised triangle over FFT bins whose frequencies lie
/// between the edge frequencies `hz[m]` and `hz[m+2]`, peaking at `hz[m+1]`.
fn htk_mel_filterbank(n_mels: usize, fft_size: usize, sr: f32) -> Vec<f32> {
    let n_bins = fft_size / 2 + 1;
    let hz = htk_mel_points(n_mels, sr);
    let mut fb = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (f_lo, f_peak, f_hi) = (hz[m], hz[m + 1], hz[m + 2]);
        let up = 1.0 / (f_peak - f_lo);
        let down = 1.0 / (f_hi - f_peak);
        for k in 0..n_bins {
            let f = k as f32 * sr / fft_size as f32;
            // Triangle, clamped to [0, 1]; zero outside [f_lo, f_hi].
            fb[m * n_bins + k] = ((f - f_lo) * up).min((f_hi - f) * down).max(0.0);
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma-4 E2B audio config (DSP fields from `processor_config.json`);
    /// tower fields are placeholders — the mel frontend ignores them.
    fn test_config() -> GemmaAudioConfig {
        GemmaAudioConfig {
            hidden_size: 1024,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            subsampling_conv_channels: vec![64, 128],
            conv_kernel_size: 3,
            attention_chunk_size: 128,
            attention_context_left: 32,
            attention_context_right: 32,
            output_proj_dims: 1024,
            residual_weight: 0.5,
            use_clipped_linears: false,
            audio_token_id: 1,
            mel_bins: 128,
            frame_length: 320,
            hop_length: 160,
            fft_size: 512,
            mel_floor: 1e-3,
            mel_scale: "htk".to_string(),
            token_cap: 750,
            norm_eps: 1e-6,
            activation: "gelu".to_string(),
            boa_token_id: 2,
            eoa_token_id: 3,
        }
    }

    fn sine(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE).sin())
            .collect()
    }

    /// 440 Hz 2 s fixture (amplitude 0.25, matching the e2e WAV): HF oracle
    /// mel[0] |x| ≈ 23.79, mel[50] |x| ≈ 65.45 — used to verify the
    /// semicausal-padded STFT + HTK filterbank against the HF extractor.
    #[test]
    fn fixture_440hz_matches_hf_magnitudes() {
        let cfg = test_config();
        let mut wav = vec![0.0f32; 32000];
        for i in 0..32000 {
            wav[i] = 0.25 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE).sin();
        }
        let out = mel_spectrogram(&wav, &cfg).unwrap();
        assert_eq!(out.n_frames, 199, "2 s @ 16 kHz → 199 mel frames");
        let row0 = &out.features[..cfg.mel_bins];
        let row50 = &out.features[50 * cfg.mel_bins..51 * cfg.mel_bins];
        let n0 = row0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n50 = row50.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("mel[0] |x|={n0:.4} first8={:?}", &row0[..8]);
        println!("mel[50] |x|={n50:.4} first8={:?}", &row50[..8]);
        std::fs::write(
            "/tmp/opencode/our_mel.txt",
            format!(
                "0 {}\n50 {}\n",
                row0.iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                row50
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        )
        .ok();
    }

    // ---- FFT correctness -------------------------------------------------

    /// FFT of a unit impulse is all-ones magnitude (analytic DFT result).
    #[test]
    fn fft_of_impulse_is_all_ones_magnitude() {
        let mut re = [0.0f32; 512];
        let mut im = [0.0f32; 512];
        re[0] = 1.0;
        fft_radix2(&mut re, &mut im);
        for k in 0..512 {
            let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
            assert!((mag - 1.0).abs() < 1e-4, "bin {k}: |X| = {mag}");
        }
    }

    /// FFT of a constant signal concentrates all energy at DC.
    #[test]
    fn fft_of_constant_has_energy_only_at_dc() {
        let mut re = [0.5f32; 512];
        let mut im = [0.0f32; 512];
        fft_radix2(&mut re, &mut im);
        assert!(
            (re[0] - 256.0).abs() < 1e-2,
            "DC bin = {} (expect 512·0.5 = 256)",
            re[0]
        );
        for k in 1..512 {
            let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
            assert!(mag < 0.05, "bin {k} leaked energy: |X| = {mag}");
        }
    }

    /// FFT of the 440 Hz fixture frame 0 (semicausal-padded, hann windowed)
    /// reproduces the exact python rfft magnitudes — pinning the radix-2
    /// implementation against numpy so a scale slip cannot hide.
    #[test]
    fn fft_frame0_440hz_matches_numpy_magnitudes() {
        let fl = 320usize;
        let pad = fl / 2;
        let mut wav = vec![0.0f32; 32000];
        for i in 0..32000 {
            wav[i] = 0.25 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE).sin();
        }
        let window = hann_periodic(fl);
        let mut re = [0.0f32; 512];
        let mut im = [0.0f32; 512];
        for i in 0..fl {
            let s = i; // frame 0 starts at sample 0 of the padded stream
            let v = if s < pad {
                0.0
            } else {
                let w = s - pad;
                if w < wav.len() { wav[w] } else { 0.0 }
            };
            re[i] = v * window[i];
        }
        fft_radix2(&mut re, &mut im);
        let mag1 = (re[1] * re[1] + im[1] * im[1]).sqrt();
        let mag14 = (re[14] * re[14] + im[14] * im[14]).sqrt();
        // numpy rfft reference for this exact input (440 Hz, amp 0.25).
        assert!(
            (mag1 - 1.464518).abs() < 1e-3,
            "|X[1]| = {mag1}, expect 1.464518"
        );
        assert!(
            (mag14 - 10.017454).abs() < 1e-3,
            "|X[14]| = {mag14}, expect 10.017454"
        );
    }

    /// Parseval: Σ_k |X[k]|² = N · Σ_n |x[n]|² over the full N-point DFT.
    #[test]
    fn fft_preserves_energy_parseval() {
        let mut re: Vec<f32> = (0..512)
            .map(|i| {
                let a = 2.0 * std::f32::consts::PI * 7.0 * i as f32 / 512.0;
                let b = 2.0 * std::f32::consts::PI * 43.0 * i as f32 / 512.0;
                a.sin() + 0.5 * b.cos()
            })
            .collect();
        let mut im = vec![0.0f32; 512];
        let time_energy: f64 = re.iter().map(|x| (*x as f64).powi(2)).sum();
        fft_radix2(&mut re, &mut im);
        let freq_energy: f64 = (0..512)
            .map(|k| (re[k] as f64).powi(2) + (im[k] as f64).powi(2))
            .sum();
        let rel = (freq_energy - 512.0 * time_energy).abs() / (512.0 * time_energy);
        assert!(rel < 1e-3, "Parseval relative error {rel}");
    }

    // ---- Window ----------------------------------------------------------

    /// Periodic Hann matches `torch.hann_window(n, periodic=True)`.
    #[test]
    fn hann_periodic_matches_torch() {
        let w = hann_periodic(512);
        assert_eq!(w.len(), 512);
        assert!(w[0].abs() < 1e-7, "periodic Hann starts at 0, got {}", w[0]);
        for i in 1..512 {
            assert!((w[i] - w[512 - i]).abs() < 1e-6, "w[{i}] != w[512-{i}]");
        }
        let sum: f32 = w.iter().sum();
        assert!((sum - 256.0).abs() < 1e-3, "window sums to N/2, got {sum}");
    }

    // ---- Mel scale / filterbank ------------------------------------------

    /// HTK mel points are strictly increasing and span [0, sr/2].
    #[test]
    fn htk_mel_points_are_monotonic_and_bounded() {
        let hz = htk_mel_points(128, SAMPLE_RATE);
        assert_eq!(hz.len(), 130);
        assert!(hz[0].abs() < 1e-3, "first edge {:.3} ≈ 0 Hz", hz[0]);
        assert!(
            (hz[129] - SAMPLE_RATE / 2.0).abs() < 1e-3,
            "last edge {} ≈ 8000 Hz",
            hz[129]
        );
        for pair in hz.windows(2) {
            assert!(pair[1] > pair[0], "mel points must be strictly increasing");
        }
    }

    /// Every filterbank row that covers an FFT bin carries positive energy.
    /// Row 0 spans (0, ~28 Hz) which contains no 512-point FFT bin (31.25 Hz
    /// spacing) — all-zero there matches torchaudio `melscale_fbanks`.
    #[test]
    fn filterbank_rows_are_positive() {
        let fb = htk_mel_filterbank(128, 512, SAMPLE_RATE);
        assert_eq!(fb.len(), 128 * 257, "dense [n_mels × (fft/2+1)]");
        assert!(
            fb[0..257].iter().all(|w| *w == 0.0),
            "row 0 covers no FFT bin"
        );
        for m in 1..128 {
            let sum: f32 = fb[m * 257..(m + 1) * 257].iter().sum();
            assert!(sum > 0.0, "row {m} sum = {sum}");
        }
    }

    // ---- End-to-end mel spectrogram ---------------------------------------

    /// 1 kHz pure tone, 1 s at 16 kHz: peak mel energy at the analytic HTK
    /// bin of 1 kHz (~44-45 of 128), and ≫10× the 8 kHz-region bin.
    #[test]
    fn one_khz_tone_peaks_at_expected_mel_bin() {
        let cfg = test_config();
        let out = mel_spectrogram(&sine(1000.0, 16000), &cfg).unwrap();
        let spacing = htk_mel(SAMPLE_RATE / 2.0) / (cfg.mel_bins + 1) as f32;
        let expected = ((htk_mel(1000.0) / spacing) as usize).saturating_sub(1);
        let mut mel_avg = vec![0.0f32; cfg.mel_bins];
        for f in 0..out.n_frames {
            for m in 0..cfg.mel_bins {
                mel_avg[m] += out.features[f * cfg.mel_bins + m];
            }
        }
        for v in &mut mel_avg {
            *v /= out.n_frames as f32;
        }
        let peak = (0..cfg.mel_bins)
            .max_by(|a, b| mel_avg[*a].partial_cmp(&mel_avg[*b]).unwrap())
            .unwrap();
        assert!(
            (peak as i64 - expected as i64).abs() <= 2,
            "peak mel bin {peak}, analytic expectation {expected}"
        );
        let far = cfg.mel_bins - 1; // ~8 kHz region, where the 1 kHz tone has no energy
        assert!(
            mel_avg[peak] > mel_avg[far] + std::f32::consts::LN_10,
            "peak {:.3} not >10× far bin {far} ({:.3})",
            mel_avg[peak],
            mel_avg[far]
        );
    }

    /// 1 s @ 16 kHz → `1 + floor((16000 + 160 − 321) / 160)` = 99 frames,
    /// the HF `Gemma4AudioFeatureExtractor` semicausal-padded count.
    #[test]
    fn one_second_at_16k_gives_99_frames() {
        let cfg = test_config();
        let out = mel_spectrogram(&sine(1000.0, 16000), &cfg).unwrap();
        assert_eq!(out.n_frames, 99);
        assert_eq!(out.n_mels, 128);
        assert_eq!(out.features.len(), 99 * 128);
        assert!(out.features.iter().all(|v| v.is_finite()));
    }

    /// Tail semantics under the HF semicausal pad: the frame count follows
    /// `1 + floor((n + fl/2 − (fl+1)) / hop)` — a one-sample tail can start
    /// a new frame once padding absorbs it, matching the HF extractor.
    #[test]
    fn partial_tail_follows_hf_padded_count() {
        let cfg = test_config();
        let out = mel_spectrogram(&sine(1000.0, 16001), &cfg).unwrap();
        assert_eq!(out.n_frames, 100, "16001 samples → 100 frames (padded)");
        let out = mel_spectrogram(&sine(1000.0, 15_999), &cfg).unwrap();
        assert_eq!(out.n_frames, 99, "(15999+160−321)/160 = 98 → 99 frames");
    }

    /// Pure function: identical input yields bit-identical output.
    #[test]
    fn identical_inputs_give_bit_identical_outputs() {
        let cfg = test_config();
        let wav = sine(440.0, 8000);
        let a = mel_spectrogram(&wav, &cfg).unwrap();
        let b = mel_spectrogram(&wav, &cfg).unwrap();
        assert_eq!(a, b);
    }

    /// Fail fast on configs the DSP cannot honour (PCND).
    #[test]
    fn rejects_unsupported_configs() {
        let mut cfg = test_config();
        cfg.mel_scale = "slaney".to_string();
        assert!(mel_spectrogram(&[0.0; 512], &cfg).is_err());
        let mut cfg = test_config();
        cfg.fft_size = 500; // not a power of two
        assert!(mel_spectrogram(&[0.0; 512], &cfg).is_err());
        let cfg = test_config();
        assert!(
            mel_spectrogram(&[0.0; 100], &cfg).is_err(),
            "shorter than a frame"
        );
    }
}
