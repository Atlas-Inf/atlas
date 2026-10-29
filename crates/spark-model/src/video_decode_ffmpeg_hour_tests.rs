// SPDX-License-Identifier: AGPL-3.0-only

//! Hour-video decode-budget and probe-watchdog tests (Enntity/atlas 8fe99bfe).

use super::*;

#[test]
fn hour_decode_budget_is_total_volume_not_frames_times_still_bound() {
    let p = FfmpegPolicy {
        max_frames: 7200,
        max_total_decoded_pixels: Some(469_762_048),
        ..Default::default()
    };
    let per_frame = tail::decoded_frame_budget(&p, 7200).unwrap();
    assert!(per_frame * 7200 <= 469_762_048);
    assert!(per_frame > 64_000);
}

#[cfg(unix)]
#[test]
fn probe_timeout_covers_a_blocked_stdout_read() {
    // Use an installed executable: executing a just-written script can race
    // with parallel fork/exec tests on overlayfs and yield ETXTBSY instead.
    let child = Command::new("sleep")
        .arg("30")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    let err = probe::read_probe(child, 1).unwrap_err();
    assert!(err.to_string().contains("exceeded 1s"), "{err:#}");
    assert!(start.elapsed().as_secs() < 5);
}
