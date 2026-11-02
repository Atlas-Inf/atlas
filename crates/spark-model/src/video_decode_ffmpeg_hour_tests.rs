// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn hour_decode_budget_is_total_volume_not_frames_times_still_bound() {
    let p = FfmpegPolicy {
        max_frames: 7200,
        max_total_decoded_pixels: Some(469_762_048),
        ..Default::default()
    };
    let per_frame = decoded_frame_budget(&p, 7200).unwrap();
    assert!(per_frame * 7200 <= 469_762_048);
    assert!(per_frame > 64_000);
}

#[test]
fn incremental_png_reader_checks_frame_count_bytes_and_geometry() {
    let img = image::RgbImage::from_pixel(32, 32, image::Rgb([9, 8, 7]));
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    let p = FfmpegPolicy::default();
    let frames = png_stream::read_frames(&mut png.as_slice(), 1, &p).unwrap();
    assert_eq!(frames[0].get_pixel(0, 0).0, [9, 8, 7]);
    let doubled = [png.clone(), png.clone()].concat();
    assert!(png_stream::read_frames(&mut doubled.as_slice(), 1, &p).is_err());
    let small = FfmpegPolicy {
        max_output_bytes: 20,
        ..p.clone()
    };
    assert!(png_stream::read_frames(&mut png.as_slice(), 1, &small).is_err());
    let small = FfmpegPolicy {
        max_decoded_pixels: 100,
        ..p
    };
    assert!(png_stream::read_frames(&mut png.as_slice(), 1, &small).is_err());
    assert!(png_stream::read_frames(&mut &png[..png.len() - 1], 1, &small).is_err());
}

#[test]
fn thousands_of_samples_cover_an_hour_without_an_argv_sized_filter() {
    if !tests::have_ffmpeg() {
        return;
    }
    // Low-resolution real MP4 spanning one hour: isolates planner, FFmpeg,
    // source timestamps and streamed framing without a GPU or huge fixture.
    let mp4 = tests::make_mp4(3600, 3, "32x32", "libx264").expect("encode");
    let p = FfmpegPolicy {
        enabled: true,
        max_frames: 7200,
        max_total_decoded_pixels: Some(7200 * 1024),
        ..Default::default()
    };
    let (frames, times) = decode_with_plan(&mp4, 2.0, &p).unwrap();
    assert_eq!(frames.len(), 7200);
    assert_eq!(times.len(), frames.len());
    assert_eq!(times[0], 0.0);
    assert!(times[7199] > 3599.3);
    assert!(times.windows(2).all(|x| x[1] > x[0]));
}

#[cfg(unix)]
#[test]
fn probe_timeout_covers_a_blocked_stdout_read() {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::temp_dir().join(format!("atlas-hung-probe-{}", std::process::id()));
    std::fs::write(&path, "#!/bin/sh\nexec sleep 30\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let p = FfmpegPolicy {
        probe_binary: path.to_string_lossy().into_owned(),
        timeout_secs: 1,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let err = probe::probe_duration(b"invalid", &p).unwrap_err();
    let _ = std::fs::remove_file(path);
    assert!(err.to_string().contains("exceeded 1s"), "{err:#}");
    assert!(start.elapsed().as_secs() < 5);
}
