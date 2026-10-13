// SPDX-License-Identifier: AGPL-3.0-only

//! Frame extraction for real-world containers, via ffmpeg.
//!
//! # Why a subprocess and not a linked decoder
//!
//! The alternative evaluated was `openh264` plus a pure-Rust MP4 demuxer. It
//! builds quickly (~9 s on aarch64) and needs no runtime dependency, but it
//! covers **H.264 only** — no H.265, VP9 or AV1, which is a large share of
//! what people actually send — and Cisco's royalty-free patent grant covers
//! the binaries *Cisco* distributes, not a source build redistributed by a
//! third party. That is a licensing question for the project to answer
//! deliberately, not one to settle by adding a dependency.
//!
//! ffmpeg as a subprocess needs no build or link dependency, decodes
//! everything, and can be swapped for an in-process decoder later without
//! changing this module's signature. The cost is a runtime binary and a
//! process spawn per request, so it is OPT-IN and the absence of the binary
//! is reported by name rather than as a decode failure.
//!
//! # What is bounded
//!
//! Everything the caller controls, because the input is an untrusted byte
//! blob from an HTTP request:
//!
//! - **No shell.** Arguments are passed as argv. Nothing is interpolated into
//!   a command string, so no input can become a flag or a second command.
//! - **No temp file.** The container goes in over stdin, so there is no path
//!   to traverse, collide on, or leave behind.
//! - **`-nostdin`, and stdin is the pipe** — ffmpeg cannot reach for a
//!   terminal or block waiting on one.
//! - **Frame count** capped with `-frames:v`, so a long clip cannot decode
//!   forever — but the cap is applied to a plan that SPANS the clip
//!   ([`crate::video_sample`]) rather than to a fixed-rate walk that would
//!   stop `max_frames / fps` seconds in. See [`decode_frames`].
//! - **Duration probed first**, with ffprobe, because a plan that covers the
//!   whole clip has to know where the clip ends. A duration that cannot be
//!   determined is an ERROR, never a silent fallback to the first frames.
//! - **Decoded pixel memory** bounded by scaling each frame before it is read
//!   back: the PNG-size cap bounds what is READ from the pipe, not what the
//!   returned `Vec<RgbImage>` holds once it is parsed.
//! - **Output size** capped while reading, and the child is killed the moment
//!   the cap is passed.
//! - **Wall clock** capped by a watchdog that kills the child; a decoder that
//!   hangs must not hold a request thread indefinitely.
//! - **Protocol whitelist** is moot because the input is a pipe, but
//!   `-f image2pipe` output and a `pipe:0` input mean ffmpeg is never asked
//!   to open a URL — the SSRF path that `remote_image` guards for stills
//!   simply does not exist here.

use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use crate::video_sample::{grid_step, sample_plan, wanted_frames};

mod probe;
use probe::probe_duration;

/// Operator policy for subprocess decoding.
#[derive(Debug, Clone)]
pub struct FfmpegPolicy {
    pub enabled: bool,
    /// Binary to run. A name is resolved on PATH; an absolute path is used
    /// as given, so a deployment can pin a known build.
    pub binary: String,
    /// ffprobe binary, used to determine the clip's duration. Defaults to
    /// `ffprobe`, which ships with ffmpeg; when that is not on PATH a sibling
    /// of [`Self::binary`] named `ffprobe` is tried before giving up.
    pub probe_binary: String,
    pub max_frames: usize,
    pub max_output_bytes: usize,
    pub timeout_secs: u64,
    /// Frames per second to sample at. The plan never exceeds it; a clip short
    /// enough for the frame cap is sampled at its OWN rate, so motion the cap
    /// can hold is not discarded.
    pub fps: f32,
    /// Ceiling, in pixels, on one decoded frame's area. Each planned frame is
    /// scaled to fit BEFORE it is read back as RGB, which is what bounds the
    /// returned frame collection at `max_frames × max_decoded_pixels`.
    pub max_decoded_pixels: usize,
}

/// Default decoded-frame area cap: 640×640.
///
/// At the 768-frame default this bounds the returned RGB at ~0.9 GiB
/// (768 × 640 × 640 × 3) where a 3840×2160 clip would otherwise cost ~10 GiB
/// inside a request. Well above the vision grid the preprocessor builds from
/// a frame, so it costs no resolution the encoder would have seen.
const DEFAULT_MAX_DECODED_PIXELS: usize = 640 * 640;

impl Default for FfmpegPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            binary: "ffmpeg".to_string(),
            probe_binary: "ffprobe".to_string(),
            max_frames: 768,
            // 768 frames of 1280x720 PNG is comfortably under this; it exists
            // to bound a pathological stream, not to size a normal one.
            max_output_bytes: 512 * 1024 * 1024,
            timeout_secs: 120,
            fps: 2.0,
            max_decoded_pixels: DEFAULT_MAX_DECODED_PIXELS,
        }
    }
}

/// The 8-byte PNG signature. Frames arrive concatenated on one stream, so
/// this is how they are separated.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// What a startup probe found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// Usable, with the version string it reported.
    Ready(String),
    /// Configured but not runnable, with why.
    Missing(String),
    /// Not asked for.
    Disabled,
}

/// Check at BOOT whether the configured decoder can actually run.
///
/// Worth doing eagerly rather than discovering it on the first video request:
/// a deployment that enabled video decoding and does not have the binary is
/// misconfigured, and the operator should learn that while reading the
/// startup log — not from a user's failed request an hour later. The check is
/// one `-version` invocation, so it costs nothing at boot.
pub fn probe(policy: &FfmpegPolicy) -> Availability {
    if !policy.enabled {
        return Availability::Disabled;
    }
    match Command::new(&policy.binary)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    {
        Ok(out) if out.status.success() => {
            let first = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("unknown version")
                .trim()
                .to_string();
            Availability::Ready(first)
        }
        Ok(out) => {
            Availability::Missing(format!("{:?} ran but exited {}", policy.binary, out.status))
        }
        Err(e) => Availability::Missing(format!("{:?} could not be run: {e}", policy.binary)),
    }
}

/// Decode `bytes` to RGB frames that SPAN the clip.
///
/// The clip's duration and native frame rate are probed first, a plan is
/// derived that covers the whole duration at no more than `target_fps`, and
/// ffmpeg is asked for exactly those frames. The result does not carry
/// timestamps here — [`decode_with_plan`] is the form callers that need them
/// (the preprocessor, for the prompt's per-group times) should use.
pub fn decode_frames(
    bytes: &[u8],
    target_fps: f32,
    policy: &FfmpegPolicy,
) -> Result<Vec<RgbImage>> {
    Ok(decode_with_plan(bytes, target_fps, policy)?.0)
}

/// What a probe found, or the error that says it could not be determined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipTiming {
    /// Duration in seconds, from the container.
    pub duration_secs: f32,
    /// The video stream's own rate: an average when the container declares
    /// one, otherwise the instantaneous rate. Used only to plan coverage.
    pub native_fps: f32,
}

/// Decode `bytes` to RGB frames spanning the clip, plus the plan that
/// produced them.
///
/// The returned timestamps are the times, in source seconds, of the frames
/// that came back — indexed exactly like the frames, so callers derive their
/// own group timestamps from the frames they actually kept rather than from
/// an assumed constant rate.
pub fn decode_with_plan(
    bytes: &[u8],
    target_fps: f32,
    policy: &FfmpegPolicy,
) -> Result<(Vec<RgbImage>, Vec<f32>)> {
    ensure!(
        policy.enabled,
        "this container needs ffmpeg to decode and subprocess decoding is disabled; \
         pass --video-allow-ffmpeg to enable it, or send an animated GIF"
    );
    // Check the decoder BEFORE probing the clip. Otherwise a missing binary
    // surfaces as an unprobeable clip, which sends the operator hunting for a
    // corrupt video when the real fix is to install ffmpeg. `probe` names the
    // binary and the remedy.
    if let Availability::Missing(detail) = probe(policy) {
        anyhow::bail!(
            "the ffmpeg decoder is not usable: {detail} (no ffmpeg installed at that path, or it \
             could not run) — install ffmpeg with its ffprobe companion, or point \
             --video-ffmpeg-path at a usable build"
        );
    }
    // A caller that passes a usable rate wins; anything else falls back to the
    // policy's, which is the operator's SSOT for video sampling.
    let policy_fps = crate::video_sample::finite_positive(policy.fps)
        .unwrap_or(crate::video_preprocess::DEFAULT_FPS);
    let fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        policy_fps
    };

    let timing = probe_duration(bytes, policy)?;
    let want = wanted_frames(
        timing.duration_secs,
        timing.native_fps,
        fps,
        policy.max_frames,
        fps,
    );
    // The plan converts the frame COUNT into evenly spaced positions across
    // the probed duration — this, and not the frame count, is what makes the
    // samples cover the whole clip instead of stopping `cap / fps` seconds in.
    let plan = sample_plan(want, timing.duration_secs)?;
    // The grid runs at the SOURCE rate, subdivided so the selector's indices
    // land on a dense-enough grid to name the plan's positions. The coverage
    // bound the operator's `fps` represents is enforced by how MANY samples
    // the plan asks for, not by ffmpeg dropping the rest — which is exactly
    // the distinction the old `fps=` filter could not make.
    let native = crate::video_sample::finite_positive(timing.native_fps).unwrap_or(fps);
    let step = grid_step(native, fps, fps);
    // `fps=A/B` samples one frame every `B/A` seconds, so the frame the
    // selector names for index K is the grid frame at time `K × B/A`.
    let grid = format!("fps={native}/{step}");
    let frames = run_ffmpeg(bytes, &plan, &grid, policy)?;

    // The frame-bearing timestamps come from the plan, clipped to the frames
    // that were actually produced: a container whose last frame lands earlier
    // than its declared duration must not be reported with timestamps for
    // frames the model never received.
    let times = plan.times.iter().copied().take(frames.len()).collect();
    Ok((frames, times))
}

/// Spawn ffmpeg with the plan's filter and collect the frames.
///
/// Split out so the argument construction (the part with the correctness
/// claims: `select` instead of a bare `fps`, and a `scale` that bounds decoded
/// RGB) is readable on its own.
fn run_ffmpeg(
    bytes: &[u8],
    plan: &crate::video_sample::SamplePlan,
    grid: &str,
    policy: &FfmpegPolicy,
) -> Result<Vec<RgbImage>> {
    // `select` first, `scale` after: at most `plan.len()` frames are ever
    // scaled, and the scale is what caps the RGB the returned Vec holds.
    let filter = format!(
        "{grid},{},scale='min({px},iw)':'min({px},ih)':force_original_aspect_ratio=decrease",
        plan.select_filter(),
        px = policy.max_decoded_pixels.max(1),
    );

    let mut child = Command::new(&policy.binary)
        .args([
            "-v",
            "error",
            // Never touch a terminal: without this ffmpeg can block on a
            // prompt when it thinks stdin is interactive.
            "-nostdin",
            "-i",
            "pipe:0",
            "-vf",
            &filter,
            "-frames:v",
            &plan.len().to_string(),
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "could not run {:?} — is ffmpeg installed and on PATH? \
                 (set --video-ffmpeg-path to point at it)",
                policy.binary
            )
        })?;

    let mut stdin = child.stdin.take().context("no stdin pipe")?;
    let mut stdout = child.stdout.take().context("no stdout pipe")?;
    let mut stderr = child.stderr.take().context("no stderr pipe")?;

    // Feed the container on a thread. It MUST be concurrent with reading:
    // ffmpeg writes output while still consuming input, so a write-then-read
    // sequence deadlocks as soon as the output pipe buffer fills.
    let input = bytes.to_vec();
    let writer = std::thread::spawn(move || {
        // A broken pipe here is normal — ffmpeg stops reading once it has the
        // frames it was asked for — so the error is deliberately dropped.
        let _ = stdin.write_all(&input);
        drop(stdin);
    });

    let child = Arc::new(Mutex::new(child));
    let watchdog = {
        let child = Arc::clone(&child);
        let secs = policy.timeout_secs.max(1);
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let mut guard = match child.lock() {
                    Ok(g) => g,
                    Err(_) => return false,
                };
                match guard.try_wait() {
                    Ok(Some(_)) => return false, // exited on its own
                    Ok(None) => {}
                    Err(_) => return false,
                }
                if std::time::Instant::now() >= deadline {
                    let _ = guard.kill();
                    return true; // we killed it
                }
            }
        })
    };

    // Read with a hard cap. `take` bounds it without trusting the child.
    let mut out = Vec::new();
    let read_res = (&mut stdout)
        .take(policy.max_output_bytes as u64 + 1)
        .read_to_end(&mut out);

    // KILL FIRST, then drain stderr. Order matters and getting it wrong
    // deadlocks: once the cap is hit we stop reading stdout, so the child
    // blocks writing into a full pipe and never closes stderr — and a
    // `read_to_string` on stderr then waits for an EOF that only the watchdog
    // will ever cause. Killing here turns a 120-second hang into an immediate
    // error. (Draining stdout instead would defeat the cap, which is the
    // thing being enforced.)
    let over_cap = out.len() > policy.max_output_bytes;
    if over_cap && let Ok(mut g) = child.lock() {
        let _ = g.kill();
    }

    let mut err_text = String::new();
    let _ = stderr.read_to_string(&mut err_text);
    let _ = writer.join();

    let status = {
        let mut g = child
            .lock()
            .map_err(|_| anyhow::anyhow!("decoder lock poisoned"))?;
        g.wait().context("waiting for the decoder")?
    };
    let timed_out = watchdog.join().unwrap_or(false);

    read_res.context("reading decoded frames")?;
    ensure!(
        !timed_out,
        "decoding exceeded {}s and was stopped",
        policy.timeout_secs
    );
    ensure!(
        !over_cap,
        "decoded output exceeded the {}-byte cap",
        policy.max_output_bytes
    );
    if !status.success() {
        let why = err_text.lines().next_back().unwrap_or("no detail").trim();
        bail!("decoder failed: {why}");
    }

    let frames = split_png_stream(&out)?;
    ensure!(
        !frames.is_empty(),
        "the container decoded to zero frames (is there a video stream?)"
    );
    Ok(frames)
}

/// Split a concatenated PNG stream into images.
///
/// Splitting on the signature rather than parsing IEND chunks: a PNG's
/// payload can legitimately contain the IEND byte pattern, whereas the
/// 8-byte signature only appears at a file start in a well-formed stream
/// produced by `image2pipe`.
fn split_png_stream(buf: &[u8]) -> Result<Vec<RgbImage>> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + PNG_MAGIC.len() <= buf.len() {
        if buf[i..i + PNG_MAGIC.len()] == PNG_MAGIC {
            starts.push(i);
            i += PNG_MAGIC.len();
        } else {
            i += 1;
        }
    }
    let mut frames = Vec::with_capacity(starts.len());
    for (n, &s) in starts.iter().enumerate() {
        let e = starts.get(n + 1).copied().unwrap_or(buf.len());
        let img = image::load_from_memory_with_format(&buf[s..e], image::ImageFormat::Png)
            .with_context(|| format!("frame {n} did not decode as PNG"))?;
        frames.push(img.to_rgb8());
    }
    Ok(frames)
}

#[cfg(test)]
#[path = "video_decode_ffmpeg_tests.rs"]
mod tests;
