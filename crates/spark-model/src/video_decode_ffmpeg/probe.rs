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
    let mut child = probe_command(policy)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=nw=1",
            "pipe:0",
        ])
        .stdin(Stdio::piped())
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

    // Same reason ffmpeg's input is fed on a thread: the child writes its
    // report while still draining stdin, so a write-then-read sequence can
    // deadlock on a large container.
    let mut stdin = child.stdin.take().context("no ffprobe stdin pipe")?;
    let input = bytes.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
        drop(stdin);
    });

    let mut out = Vec::new();
    let read_res = child
        .stdout
        .take()
        .context("no ffprobe stdout pipe")?
        .read_to_end(&mut out);

    // Bound the wait the same way the decoder is bounded: a probe that hangs
    // must not hold a request thread. The watchdog kills the child, which
    // closes its stdout and releases the reader above.
    let mut err_text = String::new();
    let status = {
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(policy.timeout_secs.max(1));
        loop {
            match child.try_wait().context("waiting for ffprobe")? {
                Some(s) => break s,
                None if std::time::Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!(
                        "probing the clip's duration exceeded {}s and was stopped",
                        policy.timeout_secs
                    );
                }
                None => std::thread::sleep(std::time::Duration::from_millis(50)),
            }
        }
    };
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut err_text);
    }
    let _ = writer.join();
    read_res.context("reading the ffprobe report")?;

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

    Ok(ClipTiming {
        duration_secs,
        native_fps,
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
