// SPDX-License-Identifier: AGPL-3.0-only

// RoCE GID-index discovery against sysfs-shaped fixtures. The GID table is
// rebuilt whenever a port's IP addresses are re-added (a link flap, a peer
// reboot), and a slot still referenced by live QPs is not reused, so the
// RoCE v2 IPv4 entry does not stay at a fixed index. The fixtures reproduce
// two DGX Sparks observed on 2026-09-28: one with the entry at index 3, one
// with a hole at 3 and the entry at 4 after the peer rebooted.

use atlas_rdma::gid::roce_v2_ipv4_index_in;
use std::fs;
use std::path::{Path, PathBuf};

const LINK_LOCAL: &str = "fe80:0000:0000:0000:4ebb:47ff:fe2e:6f1a";
const IPV4: &str = "0000:0000:0000:0000:0000:ffff:0a64:c001";
const ZERO: &str = "0000:0000:0000:0000:0000:0000:0000:0000";

/// A sysfs root holding `dev` port 1 with `entries` as `(gid, type)`; a
/// `None` type is a slot whose type read fails, as the kernel does for an
/// unused entry.
fn table(name: &str, dev: &str, entries: &[(&str, Option<&str>)]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("atlas-rdma-gid-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let port = root.join(dev).join("ports/1");
    fs::create_dir_all(port.join("gids")).unwrap();
    fs::create_dir_all(port.join("gid_attrs/types")).unwrap();
    for (i, (gid, kind)) in entries.iter().enumerate() {
        fs::write(port.join("gids").join(i.to_string()), format!("{gid}\n")).unwrap();
        if let Some(kind) = kind {
            fs::write(
                port.join("gid_attrs/types").join(i.to_string()),
                format!("{kind}\n"),
            )
            .unwrap();
        }
    }
    root
}

fn cleanup(root: &Path) {
    let _ = fs::remove_dir_all(root);
}

#[test]
fn finds_the_roce_v2_ipv4_entry_at_its_usual_index() {
    let root = table(
        "usual",
        "rocep1s0f0",
        &[
            (LINK_LOCAL, Some("IB/RoCE v1")),
            (LINK_LOCAL, Some("RoCE v2")),
            (IPV4, Some("IB/RoCE v1")),
            (IPV4, Some("RoCE v2")),
            (ZERO, None),
        ],
    );
    assert_eq!(roce_v2_ipv4_index_in(&root, "rocep1s0f0").unwrap(), 3);
    cleanup(&root);
}

#[test]
fn follows_the_entry_past_a_hole_left_by_a_re_added_address() {
    let root = table(
        "hole",
        "rocep1s0f0",
        &[
            (LINK_LOCAL, Some("IB/RoCE v1")),
            (LINK_LOCAL, Some("RoCE v2")),
            (IPV4, Some("IB/RoCE v1")),
            (ZERO, None),
            (IPV4, Some("RoCE v2")),
        ],
    );
    assert_eq!(roce_v2_ipv4_index_in(&root, "rocep1s0f0").unwrap(), 4);
    cleanup(&root);
}

#[test]
fn a_port_without_an_ipv4_address_is_an_error_naming_the_override() {
    let root = table(
        "noipv4",
        "rocep1s0f0",
        &[
            (LINK_LOCAL, Some("IB/RoCE v1")),
            (LINK_LOCAL, Some("RoCE v2")),
            (IPV4, Some("IB/RoCE v1")),
        ],
    );
    let err = roce_v2_ipv4_index_in(&root, "rocep1s0f0").unwrap_err();
    assert!(err.to_string().contains("ATLAS_RDMA_GID"), "{err}");
    cleanup(&root);
}

#[test]
fn a_missing_device_is_an_error_naming_the_device() {
    let root = table("nodev", "rocep1s0f0", &[]);
    let err = roce_v2_ipv4_index_in(&root, "roceP2p1s0f0").unwrap_err();
    assert!(err.to_string().contains("roceP2p1s0f0"), "{err}");
    cleanup(&root);
}
