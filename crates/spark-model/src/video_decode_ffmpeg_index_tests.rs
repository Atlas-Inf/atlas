// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM frame-index decode and the probe's pipe bounds.

use super::tests::{enabled, have_ffmpeg, make_mp4};
use super::*;

// ── issue #239: MP4 with moov at the END (no +faststart) ─────────────────
// `pipe:0` was not seekable, so an MP4 whose moov box follows the media data
// could not be probed or decoded. `input::SeekableInput` makes the input a
// private seekable file removed on drop. This MP4 uses ffmpeg's default
// layout (moov at end) and must decode identically to a faststart one.

fn make_mp4_at(
    seconds: u32,
    fps: u32,
    size: &str,
    codec: &str,
    faststart: bool,
) -> Option<Vec<u8>> {
    static SEQ2: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "atlas-vid-moov-{}-{n}-{faststart}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join("clip.mp4");
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={size}:rate={fps}:duration={seconds}"),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]);
    if faststart {
        cmd.args(["-movflags", "+faststart"]);
    }
    cmd.arg(&out).stdout(Stdio::null()).stderr(Stdio::null());
    if !cmd.status().ok()?.success() {
        return None;
    }
    std::fs::read(&out).ok()
}

#[test]
fn a_moov_at_end_mp4_probes_and_decodes() {
    if !have_ffmpeg() {
        return;
    }
    let clip = make_mp4_at(2, 8, "64x64", "libx264", false).expect("generate MP4");
    let policy = enabled();
    let timing = super::probe::probe_duration(&clip, &policy).expect("moov-at-end probe");
    assert!(
        (timing.duration_secs - 2.0).abs() < 0.2,
        "duration should be ~2s, got {}",
        timing.duration_secs
    );
    let frames = super::decode_frames(&clip, 2.0, &policy).expect("moov-at-end decode");
    assert!(
        !frames.is_empty(),
        "moov-at-end clip should decode to frames"
    );
}

fn timing(frame_count: Option<u64>, duration_secs: f32, native_fps: f32) -> ClipTiming {
    ClipTiming {
        duration_secs,
        native_fps: f64::from(native_fps),
        frame_count,
    }
}

#[test]
fn declared_frames_reads_nb_frames_then_duration_times_rate() {
    assert_eq!(
        frames::declared_frames(&timing(Some(10), 1.0, 10.0), 1848).unwrap(),
        10
    );
    assert_eq!(
        frames::declared_frames(&timing(None, 2.0, 30.0), 1848).unwrap(),
        60
    );
}

#[test]
fn a_faststart_mp4_probes_and_decodes() {
    if !have_ffmpeg() {
        return;
    }
    let clip = make_mp4_at(2, 8, "64x64", "libx264", true).expect("generate MP4");
    let policy = enabled();
    let timing = super::probe::probe_duration(&clip, &policy).expect("faststart probe");
    assert!(
        (timing.duration_secs - 2.0).abs() < 0.2,
        "duration should be ~2s, got {}",
        timing.duration_secs
    );
    let frames = super::decode_frames(&clip, 2.0, &policy).expect("faststart decode");
    assert!(!frames.is_empty(), "faststart clip should decode to frames");
}

#[test]
fn a_frame_count_the_payload_cannot_hold_is_refused() {
    // ffprobe reported this count for a 1,848-byte MP4 with a patched `stts`.
    let err = frames::declared_frames(&timing(Some(4_000_000_000), 1.0, 10.0), 1848).unwrap_err();
    assert!(err.to_string().contains("4000000000"), "{err}");
    // Without `nb_frames` the same claim arrives as duration × rate.
    assert!(frames::declared_frames(&timing(None, f32::MAX, f32::MAX), 1848).is_err());
    assert_eq!(
        frames::declared_frames(&timing(Some(1848), 1.0, 10.0), 1848).unwrap(),
        1848
    );
}

fn ppm(width: u32, height: u32, value: u8) -> Vec<u8> {
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    out.extend(std::iter::repeat_n(value, (width * height * 3) as usize));
    out
}

#[test]
fn a_ppm_stream_splits_by_its_headers_and_is_bounded() {
    let mut stream = ppm(4, 3, 7);
    stream.extend(ppm(2, 2, 9));
    let mut frames = Vec::new();
    let n = frames::read_ppm_frames(&mut stream.as_slice(), 2, &mut |f| {
        frames.push(f);
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 2);
    assert_eq!(frames[0].dimensions(), (4, 3));
    assert_eq!(frames[1].get_pixel(1, 1).0, [9, 9, 9]);
    let err = frames::read_ppm_frames(&mut stream.as_slice(), 1, &mut |_| Ok(())).unwrap_err();
    assert!(err.to_string().contains("more than 1"), "{err:#}");
    let truncated = &stream[..stream.len() - 1];
    assert!(frames::read_ppm_frames(&mut &truncated[..], 2, &mut |_| Ok(())).is_err());
}

/// The index decode returns exactly the source frames it names, at their
/// native size: the same pixels as those positions of a decode of every frame.
#[test]
fn named_frames_decode_to_those_source_frames() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(3, 10, "320x240", "libx264").expect("encode");
    assert_eq!(probe_frames(&mp4, &enabled()).unwrap(), (30, 10.0));
    let decode = |indices: &[usize]| {
        let mut frames = Vec::new();
        let n = decode_frame_indices(&mp4, indices, &enabled(), &mut |f| {
            frames.push(f);
            Ok(())
        })
        .unwrap();
        assert_eq!(n, indices.len());
        frames
    };
    let all = decode(&(0..30).collect::<Vec<_>>());
    let picked = decode(&[0, 5, 11, 29]);
    assert_eq!(picked[0].dimensions(), (320, 240));
    for (frame, index) in picked.iter().zip([0, 5, 11, 29]) {
        assert_eq!(frame.as_raw(), all[index].as_raw(), "frame {index}");
    }
    assert_ne!(picked[0].as_raw(), picked[3].as_raw());
}

#[test]
fn a_rate_keeps_its_exact_value() {
    // 30000/1001 as f32 is off by 1.6e-7 Hz: over an hour of NTSC that moves
    // a frame across a whole-second timestamp.
    assert_eq!(
        super::probe::parse_rate("30000/1001"),
        Some(30000.0 / 1001.0)
    );
    assert_eq!(super::probe::parse_rate("0/0"), None);
}

fn sh_probe(script: &str) -> Result<(Vec<u8>, String, std::process::ExitStatus)> {
    let child = Command::new("sh")
        .args(["-c", script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("sh");
    super::probe::read_probe(child, 5)
}

#[test]
fn a_probe_that_floods_stdout_is_killed_at_the_bound() {
    let began = std::time::Instant::now();
    // `exec`, so the probe is one process as ffprobe is: dash would fork `yes`,
    // and the orphan would hold the pipes open after `sh` is killed.
    let err = sh_probe("exec yes").unwrap_err();
    assert!(err.to_string().contains("64 KiB"), "{err:#}");
    assert!(began.elapsed() < std::time::Duration::from_secs(4));
}

#[test]
fn a_probe_that_floods_stderr_does_not_stall_its_stdout() {
    let began = std::time::Instant::now();
    let (out, err, status) = sh_probe(
        "dd if=/dev/zero bs=1024 count=256 >&2 2>/dev/null; yes x | head -c 256 >&2; echo done",
    )
    .unwrap();
    assert!(status.success());
    assert_eq!(out, b"done\n");
    assert!(err.len() <= 64 * 1024);
    assert!(began.elapsed() < std::time::Duration::from_secs(4));
}
