// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};

/// Extract the source PTS that `showinfo` reports for each selected frame.
///
/// The decoder intentionally does not reconstruct these values from the
/// sample plan: on CFR they can differ by a rounding tick, and on VFR they
/// can differ materially. `showinfo` runs before scaling, so these are the
/// timestamps of the actual source frames returned to the model.
pub(super) fn parse_showinfo_timestamps(stderr: &str) -> Result<Vec<f32>> {
    let mut times = Vec::new();
    for line in stderr.lines() {
        if !line.contains("showinfo") {
            continue;
        }
        for token in line.split_whitespace() {
            let Some(value) = token.strip_prefix("pts_time:") else {
                continue;
            };
            let timestamp = value
                .parse::<f32>()
                .with_context(|| format!("ffmpeg returned an invalid frame timestamp {value:?}"))?;
            ensure!(
                timestamp.is_finite() && timestamp >= 0.0,
                "ffmpeg returned an invalid frame timestamp {timestamp}"
            );
            times.push(timestamp);
            break;
        }
    }
    Ok(times)
}
