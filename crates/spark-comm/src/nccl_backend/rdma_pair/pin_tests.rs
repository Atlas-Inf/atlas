// SPDX-License-Identifier: AGPL-3.0-only

//! Proxy core selection against sysfs-shaped fixtures.

use super::*;
use std::fs;
use std::path::{Path, PathBuf};

/// A `/sys/devices/system/cpu`-shaped root: `cpu<N>/cpufreq/cpuinfo_max_freq`
/// holds each entry's text; a `None` core has no cpufreq directory.
fn cores(name: &str, freqs: &[Option<&str>]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("atlas-rdma-pin-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    for (cpu, freq) in freqs.iter().enumerate() {
        let dir = root.join(format!("cpu{cpu}"));
        fs::create_dir_all(&dir).unwrap();
        if let Some(freq) = freq {
            fs::create_dir_all(dir.join("cpufreq")).unwrap();
            fs::write(dir.join("cpufreq/cpuinfo_max_freq"), format!("{freq}\n")).unwrap();
        }
    }
    root
}

fn cleanup(root: &Path) {
    let _ = fs::remove_dir_all(root);
}

#[test]
fn picks_the_last_big_core_of_a_gb10() {
    // GB10: ten little Cortex-A725 (2.808 GHz) then ten big Cortex-X925
    // (3.900 GHz), cpuinfo_max_freq in kHz.
    let freqs: Vec<_> = (0..20)
        .map(|cpu| Some(if cpu < 10 { "2808000" } else { "3900000" }))
        .collect();
    let root = cores("gb10", &freqs);
    assert_eq!(fastest_core(&root, 0..20), Some((3_900_000, 19)));
    cleanup(&root);
}

#[test]
fn considers_only_the_allowed_cores() {
    let root = cores(
        "allowed",
        &[
            Some("2808000"),
            Some("3900000"),
            Some("2808000"),
            Some("3900000"),
        ],
    );
    assert_eq!(fastest_core(&root, [0, 1, 2]), Some((3_900_000, 1)));
    assert_eq!(fastest_core(&root, [0, 2]), Some((2_808_000, 2)));
    cleanup(&root);
}

#[test]
fn skips_cores_whose_frequency_does_not_read() {
    let root = cores("unreadable", &[Some("2808000"), None, Some("fast")]);
    assert_eq!(fastest_core(&root, 0..3), Some((2_808_000, 0)));
    assert_eq!(fastest_core(&root, 1..3), None);
    // Allowed cores beyond those the root describes read as nothing.
    assert_eq!(fastest_core(&root, [5]), None);
    cleanup(&root);
}
