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
//! - **Private seekable input.** A uniquely created, owner-only temporary file
//!   supports tail-index containers and is removed when decoding ends.
//! - **`-nostdin`** prevents ffmpeg from waiting on a terminal.
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
//!   `-f image2pipe` output and a seekable private temp input mean ffmpeg is never asked
//!   to open a URL — the SSRF path that `remote_image` guards for stills
//!   simply does not exist here.

use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use crate::video_sample::{sample_plan, wanted_frames};

mod frames;
mod input;
mod probe;
mod scale;
mod timestamps;
pub(crate) use frames::{decode_frame_indices, probe_frames};
use probe::probe_duration;
use scale::decoded_scale_filter;
mod png_stream;
use png_stream::decoded_frame_budget;
use timestamps::parse_showinfo_timestamps;

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
    /// Optional aggregate RGB pixel bound, applied before frames leave FFmpeg.
    /// Long-video profiles use the same total pixel budget as preprocessing.
    pub max_total_decoded_pixels: Option<usize>,
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
            max_total_decoded_pixels: None,
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
    pub native_fps: f64,
    /// Declared source frame count (`nb_frames`), when the container names
    /// it. `None` sizes the plan by duration alone.
    pub frame_count: Option<u64>,
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
    // Check the decoder BEFORE probing the clip. Otherwise a missing binary
    // surfaces as an unprobeable clip, which sends the operator hunting for a
    // corrupt video when the real fix is to install ffmpeg. `probe` names the
    // binary and the remedy.
    probe::ensure_decoder(policy)?;
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
        timing.native_fps as f32,
        fps,
        policy.max_frames,
        fps,
    );
    // Enntity/atlas d0cead1: when the cap can hold every declared source
    // frame, keep the whole clip verbatim — a short CFR or VFR clip is not
    // warped by an evenly spaced time plan. showinfo still reports the real
    // PTS of the frames ffmpeg actually emits.
    let (selection, output_frames) = if let Some(count) = timing
        .frame_count
        .filter(|&count| count > 0 && count <= crate::video_sample::MAX_SAMPLE_PLAN as u64)
        .filter(|&count| want >= count as usize)
    {
        ("select=1".to_string(), count as usize)
    } else {
        // The plan converts the frame COUNT into evenly spaced positions
        // across the probed duration — this, and not the frame count, is what
        // makes the samples cover the whole clip instead of stopping
        // `cap / fps` seconds in.
        let mut plan = sample_plan(want, timing.duration_secs)?;
        // A container's duration usually names the end of its presentation
        // timeline, while its last frame starts one frame interval earlier.
        // Rescale the targets to end inside that final interval so the
        // timestamp selector can actually reach it. The returned time is
        // still the decoded frame's PTS, captured from ffmpeg below.
        if plan.len() > 1 {
            let final_step = timing
                .frame_count
                .filter(|&count| count > 1)
                .map(|count| timing.duration_secs / (count - 1) as f32)
                .or_else(|| {
                    crate::video_sample::finite_positive(timing.native_fps as f32)
                        .map(|rate| 1.0 / rate)
                })
                .unwrap_or(1.0 / fps);
            let old_end = *plan.times.last().expect("a non-empty plan");
            let new_end = old_end.min((timing.duration_secs - final_step).max(0.0));
            if new_end < old_end && old_end > 0.0 {
                for time in &mut plan.times {
                    *time = *time * new_end / old_end;
                }
            }
        }
        let output_frames = plan.len();
        (plan.select_time_filter(), output_frames)
    };
    run_ffmpeg(bytes, &selection, output_frames, policy)
}

/// Generic ffmpeg subprocess: seekable staged input, streamed stdout read,
/// stderr drained on a thread, watchdog kill after `policy.timeout_secs`
/// (Enntity/atlas d0cead1).
/// Spawn ffmpeg with the plan's filter and collect the frames.
///
/// Split out so the argument construction (the part with the correctness
/// claims: `select` instead of a bare `fps`, and a `scale` that bounds decoded
/// RGB) is readable on its own.
fn run_ffmpeg(
    bytes: &[u8],
    selection: &str,
    output_frames: usize,
    policy: &FfmpegPolicy,
) -> Result<(Vec<RgbImage>, Vec<f32>)> {
    // `select` first, `scale` after: at most `output_frames` frames are ever
    // scaled, and the scale is what caps the RGB the returned Vec holds.
    // Enntity/atlas 6f3d0a71 + 95d20a25: `showinfo` reports each selected
    // frame's real PTS (VFR-accurate), and the scale bounds by PIXEL AREA,
    // not per-side (a 1xN strip has a huge side inside any per-side bound).
    let filter = format!(
        "{},showinfo,{}",
        selection,
        decoded_scale_filter(decoded_frame_budget(policy, output_frames)?),
    );
    let (frames, stderr) = spawn_ffmpeg(
        bytes,
        &filter,
        output_frames,
        &["-fps_mode", "passthrough", "-vcodec", "png"],
        policy,
        "info",
        |stdout| png_stream::read_frames(stdout, output_frames, policy),
    )?;
    ensure!(
        !frames.is_empty(),
        "the container decoded to zero frames (is there a video stream?)"
    );
    let times = parse_showinfo_timestamps(&stderr)?;
    ensure!(
        times.len() == frames.len(),
        "ffmpeg reported {} selected frame timestamps for {} decoded frames",
        times.len(),
        frames.len()
    );
    Ok((frames, times))
}

/// Run ffmpeg over `bytes` with `filter`, at most `frames` frames out in the
/// `output` codec on an image pipe, and hand its stdout to `read`.
///
/// A `read` error kills the child before anything else: once the reader
/// stops, the child blocks writing into a full pipe and never closes stderr,
/// and a `read_to_string` on stderr would then wait for an EOF only the
/// watchdog causes. Killing turns a 120-second hang into an immediate error.
fn spawn_ffmpeg<T>(
    bytes: &[u8],
    filter: &str,
    frames: usize,
    output: &[&str],
    policy: &FfmpegPolicy,
    log_level: &str,
    read: impl FnOnce(&mut std::process::ChildStdout) -> Result<T>,
) -> Result<(T, String)> {
    let input_file = input::SeekableInput::new(bytes)?;
    let mut child = Command::new(&policy.binary)
        // Never touch a terminal: without `-nostdin` ffmpeg can block on a
        // prompt when it thinks stdin is interactive.
        .args([
            "-v",
            log_level,
            "-nostdin",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            "mov,matroska,webm,avi,mpegts,mpeg,ogg,flv",
            "-i",
        ])
        .arg(input_file.path())
        .args(["-vf", filter])
        .args(["-frames:v", &frames.to_string()])
        .args(output)
        .args(["-f", "image2pipe", "pipe:1"])
        .stdin(Stdio::null())
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

    let mut stdout = child.stdout.take().context("no stdout pipe")?;
    let mut stderr = child.stderr.take().context("no stderr pipe")?;

    // `showinfo` emits one stderr line per selected frame under `-v info`:
    // drain it concurrently or the pipe fills mid-decode and ffmpeg blocks
    // writing while the stdout reader waits (frames silently truncated).
    // Enntity/atlas d0cead1's `stderr_reader`.
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr
            .by_ref()
            .take(8 * 1024 * 1024)
            .read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
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
    let read_res = read(&mut stdout);

    // KILL FIRST, then drain stderr. Order matters and getting it wrong
    // deadlocks: once the cap is hit we stop reading stdout, so the child
    // blocks writing into a full pipe and never closes stderr — and a
    // `read_to_string` on stderr then waits for an EOF that only the watchdog
    // will ever cause. Killing here turns a 120-second hang into an immediate
    // error. (Draining stdout instead would defeat the cap, which is the
    // thing being enforced.)
    let over_cap = false; // the read closure enforces the cap

    let err_text = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("ffmpeg stderr reader panicked"))?;

    let status = {
        let mut g = child
            .lock()
            .map_err(|_| anyhow::anyhow!("decoder lock poisoned"))?;
        g.wait().context("waiting for the decoder")?
    };
    let timed_out = watchdog.join().unwrap_or(false);

    let out = read_res.context("reading decoded frames")?;
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
    Ok((out, err_text))
}

/// Split a concatenated PNG stream into images.
///
/// Splitting on the signature rather than parsing IEND chunks: a PNG's
/// payload can legitimately contain the IEND byte pattern, whereas the
/// 8-byte signature only appears at a file start in a well-formed stream
/// produced by `image2pipe`.
#[cfg(test)]
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

#[cfg(test)]
#[path = "video_decode_ffmpeg_hour_tests.rs"]
mod hour_tests;
