// SPDX-License-Identifier: AGPL-3.0-only

//! ffprobe side of the subprocess decode backend: binary resolution and the
//! clip duration / native frame-rate probe.

use super::*;

/// Resolve the ffprobe binary to use.
///
/// `probe_binary` is the configured name/path. When it cannot be run at all
/// (a deployment that pinned `ffmpeg` to an absolute path and has no
/// `ffprobe` on PATH), the sibling of the configured `ffmpeg` binary is tried
/// — that is where a packaged install keeps it. Never guesses past those two.
fn probe_command(policy: &FfmpegPolicy) -> Command {
    let configured = Path::new(&policy.probe_binary);
    let sibling = Path::new(&policy.binary)
        .parent()
        .map(|d| d.join("ffprobe"));
    let mut cmd = Command::new(&policy.probe_binary);
    // Only swap to the sibling when the configured name is not an absolute
    // path that exists and the sibling does: otherwise the operator's choice
    // is authoritative and the failure should name it.
    if !configured.is_file()
        && let Some(sib) = sibling
        && sib.is_file()
    {
        cmd = Command::new(sib);
    }
    cmd
}

/// Read the clip's duration and native frame rate.
///
/// A plan that covers the whole clip has to know where the clip ends, so this
/// is a hard prerequisite: an unprobeable duration is an explicit error
/// naming `duration`, never a silent fall back to sampling the first frames
/// (which is the bug this module exists to remove). Duration comes from the
/// container's `format` entry; the rate is the stream's average when declared,
/// otherwise its instantaneous rate — used only to plan coverage, and NOT as a
/// substitute for a duration that could not be read.
pub(super) fn probe_duration(bytes: &[u8], policy: &FfmpegPolicy) -> Result<ClipTiming> {
    let input_file = input::SeekableInput::new(bytes)?;
    let child = probe_command(policy)
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            "mov,matroska,webm,avi,mpegts,mpeg,ogg,flv",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=avg_frame_rate,r_frame_rate,nb_frames",
            "-of",
            "default=nw=1",
        ])
        .arg(input_file.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "could not run {:?} to determine the clip's duration — is ffprobe installed \
                 and on PATH? (it ships with ffmpeg; set --video-ffmpeg-path to a build that \
                 has it)",
                policy.probe_binary
            )
        })?;

    let (out, err_text, status) = read_probe(child, policy.timeout_secs)?;

    let text = String::from_utf8_lossy(&out);
    let field = |name: &str| -> Option<String> {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix('=').map(str::to_string))
    };

    let duration = field("duration")
        .and_then(|v| v.parse::<f32>().ok())
        .and_then(crate::video_sample::finite_positive);
    let Some(duration_secs) = duration else {
        let why = err_text.lines().next_back().unwrap_or("no detail").trim();
        // The binary itself was checked before this point, so a probe that
        // cannot read the payload means the payload is not a decodable video.
        // Say which of the two it is instead of leaving that to be guessed.
        bail!(
            "decoder failed: the clip's duration could not be determined, so there is no span to \
             sample across (ffprobe exited {status}: {why}) — the payload may not be a decodable \
             video at all"
        );
    };

    // The rate picks a grid; when the container gives none the plan falls back
    // to the operator's fps, which is a sampling decision rather than a claim
    // about the container. It is never used in place of the duration.
    let native_fps = ["avg_frame_rate", "r_frame_rate"]
        .into_iter()
        .find_map(field)
        .and_then(|v| parse_rate(&v))
        .and_then(crate::video_sample::finite_positive)
        .unwrap_or(policy.fps);

    // Declared source frames (`nb_frames`): the decoder keeps the whole clip
    // verbatim when the cap can hold it, instead of warping a CFR/VFR clip
    // onto an evenly spaced plan. Enntity/atlas d0cead1.
    let frame_count = field("nb_frames")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&count| count > 0);

    Ok(ClipTiming {
        duration_secs,
        native_fps: f64::from(native_fps),
        frame_count,
    })
}

/// Parse ffprobe's `num/den` frame-rate form. Returns `None` for `0/0` and
/// other degenerate rates rather than dividing by zero.
fn parse_rate(s: &str) -> Option<f32> {
    let (num, den) = s.trim().split_once('/')?;
    let num: f32 = num.trim().parse().ok()?;
    let den: f32 = den.trim().parse().ok()?;
    (den != 0.0).then(|| num / den)
}

/// Read a probe child to completion under the same deadline the decoder
/// uses: a hung process is killed and reported, never held forever
/// (Enntity/atlas 8fe99bfe). Returns (stdout bytes, stderr text, exit status).
pub(super) fn read_probe(
    mut child: std::process::Child,
    timeout_secs: u64,
) -> Result<(Vec<u8>, String, std::process::ExitStatus)> {
    let mut stdout = child.stdout.take().context("no ffprobe stdout pipe")?;
    let mut stderr = child.stderr.take().context("no ffprobe stderr pipe")?;
    let child = std::sync::Arc::new(std::sync::Mutex::new(child));
    // A blocked stdout read can outlive any deadline that is only checked
    // after it; the watchdog runs in its own thread so a hung probe is
    // killed while the read is in flight, which is what lets the read
    // finish (Enntity/atlas 8fe99bfe).
    let watchdog = {
        let child = std::sync::Arc::clone(&child);
        let secs = timeout_secs.max(1);
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
            loop {
                let mut guard = child.lock().unwrap();
                match guard.try_wait() {
                    Ok(Some(_)) => return false,
                    Ok(None) if std::time::Instant::now() >= deadline => {
                        let _ = guard.kill();
                        let _ = guard.wait();
                        return true;
                    }
                    _ => {}
                }
                drop(guard);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        })
    };
    let mut out = Vec::new();
    let read_res = stdout.read_to_end(&mut out);
    let mut err_text = String::new();
    let _ = stderr.read_to_string(&mut err_text);
    let killed = watchdog.join().unwrap_or(true);
    ensure!(
        !killed,
        "probing the clip's duration exceeded {timeout_secs}s and was stopped"
    );
    read_res.context("reading the ffprobe report")?;
    let status = child
        .lock()
        .unwrap()
        .wait()
        .context("waiting for ffprobe")?;
    Ok((out, err_text, status))
}

/// Refuse before touching the container when the subprocess decoder is off or
/// its binary cannot run — the error names the flag or the missing install
/// rather than letting the failure surface later as a decode error.
pub(super) fn ensure_decoder(policy: &FfmpegPolicy) -> Result<()> {
    ensure!(
        policy.enabled,
        "this container needs ffmpeg to decode and subprocess decoding is disabled; \
         pass --video-allow-ffmpeg to enable it, or send an animated GIF"
    );
    if let Availability::Missing(detail) = probe(policy) {
        anyhow::bail!(
            "the ffmpeg decoder is not usable: {detail} (no ffmpeg installed at that path, or it \
             could not run) — install ffmpeg with its ffprobe companion, or point \
             --video-ffmpeg-path at a usable build"
        );
    }
    Ok(())
}
