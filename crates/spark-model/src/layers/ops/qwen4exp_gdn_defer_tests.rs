// SPDX-License-Identifier: AGPL-3.0-only

//! Launch plumbing of the exact deferred GDN commit. The arithmetic is
//! checked on the GPU by `scripts/dev/qwen4exp_gdn_defer_bench.cu`.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn layer(i: u64) -> GdnCommitLayer {
    GdnCommitLayer {
        h: DevicePtr(0x1000 + i),
        conv: DevicePtr(0x2000 + i),
        stage_qkv: DevicePtr(0x3000 + i),
        stage_gb: DevicePtr(0x4000 + i),
        conv_w: DevicePtr(0x5000 + i),
    }
}

#[test]
fn defer_table_is_the_kernels_struct() {
    let seqs = [
        GdnDeferSeq {
            h: DevicePtr(1),
            conv: DevicePtr(2),
            stage_qkv: DevicePtr(3),
            stage_gb: DevicePtr(4),
            row0: 5,
            k: 4,
            fuse_n: DevicePtr::NULL,
        },
        GdnDeferSeq {
            h: DevicePtr(11),
            conv: DevicePtr(12),
            stage_qkv: DevicePtr(13),
            stage_gb: DevicePtr(14),
            row0: 9,
            k: 8,
            fuse_n: DevicePtr(15),
        },
    ];
    let t = defer_table(&seqs);
    // QdfDeferSeq is 48 bytes: four pointers, row0 and k as u32s, then the
    // pending-commit word (NULL: the unfused kernel).
    assert_eq!(t.len() * 8, 48 * GDN_ROWS_MAX);
    assert_eq!(&t[..6], &[1, 2, 3, 4, 5 | 4 << 32, 0]);
    assert_eq!(&t[6..12], &[11, 12, 13, 14, 9 | 8 << 32, 15]);
    assert!(t[12..].iter().all(|&w| w == 0), "unused sequences are null");
}

#[test]
fn commit_table_is_the_kernels_struct() {
    let layers: Vec<_> = (0..3).map(layer).collect();
    let t = commit_table(&layers);
    // QdfCommitLayers: 48 x 40 bytes, well inside the 4 KiB parameter space.
    assert_eq!(t.len() * 8, 40 * GDN_COMMIT_LAYERS);
    assert!(t.len() * 8 + 20 <= 4096);
    assert_eq!(&t[5..10], &[0x1001, 0x2001, 0x3001, 0x4001, 0x5001]);
    assert!(t[15..].iter().all(|&w| w == 0));
}

#[test]
fn commit_is_one_launch_per_48_layers_over_the_value_heads() {
    let gpu = MockGpuBackend::new();
    let layers: Vec<_> = (0..50).map(layer).collect();
    gdn_commit_layers(&gpu, &layers, 3, 8, 24, 128, 128, 4, 1e-6, 0).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2);
    assert_eq!((l[0].grid, l[0].block), ([24, 48, 1], [128, 1, 1]));
    assert_eq!(l[1].grid, [24, 2, 1]);
}

#[test]
fn commit_refuses_what_the_staging_cannot_hold() {
    let gpu = MockGpuBackend::new();
    let layers = [layer(0)];
    for n in [0, GDN_VERIFY_KMAX as u32 + 1] {
        assert!(gdn_commit_layers(&gpu, &layers, n, 8, 24, 128, 128, 4, 1e-6, 0).is_err());
    }
    // A geometry the kernel does not serve is an error, never a skipped commit.
    assert!(gdn_commit_layers(&gpu, &layers, 2, 8, 16, 128, 128, 4, 1e-6, 0).is_err());
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn defer_verify_declines_without_staging() {
    let gpu = MockGpuBackend::new();
    let seq = GdnDeferSeq {
        h: DevicePtr(0x1000),
        conv: DevicePtr(0x2000),
        stage_qkv: DevicePtr::NULL,
        stage_gb: DevicePtr(0x4000),
        row0: 0,
        k: 4,
        fuse_n: DevicePtr::NULL,
    };
    let rows = GdnDeferRows {
        seqs: &[seq],
        qkvz: DevicePtr(0x10000),
        qkvz_stride: 10240,
        conv_w: DevicePtr(0x20000),
        gates: DevicePtr(0x30000),
        norm_w: DevicePtr(0x40000),
        out: DevicePtr(0x50000),
    };
    let launched = gdn_verify_defer_rows(&gpu, &rows, 8, 24, 128, 128, 4, 1e-6, 1e-6, 0).unwrap();
    assert!(!launched);
    assert_eq!(gpu.launch_count(), 0);
    let ok = GdnDeferRows {
        seqs: &[GdnDeferSeq {
            stage_qkv: DevicePtr(0x3000),
            ..seq
        }],
        ..rows
    };
    assert!(gdn_verify_defer_rows(&gpu, &ok, 8, 24, 128, 128, 4, 1e-6, 1e-6, 0).unwrap());
    assert_eq!(gpu.launches_snapshot()[0].grid, [24, 1, 1]);
}
