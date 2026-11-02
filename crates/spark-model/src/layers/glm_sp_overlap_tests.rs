// SPDX-License-Identifier: AGPL-3.0-only
//! The overlapped seam on both ranks at once (mock GPUs joined by the
//! uneven tests' mailbox pair, which lands each exchange immediately): block
//! by block it must leave every byte the whole reduce-scatter, site and
//! all-gather leave.

use std::sync::Arc;

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::layers::glm_sp_uneven::tests::{LoopPair, Wire, bf16, with_ctx};

const W: usize = 4;

#[test]
fn overlap_switch_parses_strictly() {
    assert_eq!(parse_blocks(None).unwrap(), 0);
    assert_eq!(parse_blocks(Some("0")).unwrap(), 0);
    assert_eq!(parse_blocks(Some("1")).unwrap(), 4);
    assert_eq!(parse_blocks(Some("3")).unwrap(), 3);
    for bad in ["9", "true", "-1"] {
        assert!(parse_blocks(Some(bad)).is_err(), "{bad}");
    }
}

#[test]
fn blocks_tile_the_rows_and_keep_the_seam_form() {
    assert_eq!(
        cut(4096, 4).unwrap(),
        [(0, 1024), (1024, 1024), (2048, 1024), (3072, 1024)]
    );
    assert_eq!(
        cut(2694, 4).unwrap(),
        [(0, 674), (674, 674), (1348, 674), (2022, 672)]
    );
    // A block under the seam's 512 rows refuses the split.
    assert!(cut(2048, 5).is_none());
    assert!(cut(4096, 1).is_none());
}

/// The site: each row block's values replaced by a function of the reduced
/// rows (what the mHC seam does to `normed`), on the host.
fn site_on_host(gpu: &MockGpuBackend, buf: DevicePtr, first: usize, (r0, rows): (usize, usize)) {
    let at = buf.offset((first + r0) * W * 2);
    let mut b = vec![0u8; rows * W * 2];
    gpu.copy_d2h(at, &mut b).unwrap();
    for v in b.chunks_exact_mut(2) {
        let x = f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16);
        v.copy_from_slice(&bf16(x * 0.5 + 3.0).to_le_bytes());
    }
    gpu.copy_h2d(&b, at).unwrap();
}

fn run_pair(total: usize, overlapped: bool) -> [Vec<u8>; 2] {
    let wire = Arc::new(Wire::default());
    let run = |rank: usize| {
        let gpu = MockGpuBackend::new();
        let pair = LoopPair {
            gpu: &gpu,
            rank,
            wire: wire.clone(),
        };
        let sp = SpRows::for_rank(total, rank);
        let start: Vec<u8> = (0..total * W)
            .flat_map(|i| bf16(i as f32 * 0.11 + rank as f32 * 7.0 - 50.0).to_le_bytes())
            .collect();
        let buf = gpu.alloc(start.len()).unwrap();
        gpu.copy_h2d(&start, buf).unwrap();
        with_ctx(&gpu, &pair, |ctx| {
            if overlapped {
                let blocks = cut(sp.rows, 4).unwrap();
                pipelined(sp, &blocks, Some(buf), Some(buf), W, ctx, 0, |b| {
                    site_on_host(&gpu, buf, sp.row0, b);
                    Ok(())
                })
                .unwrap();
            } else {
                sp.reduce_scatter(buf, W, ctx, 0).unwrap();
                site_on_host(&gpu, buf, sp.row0, (0, sp.rows));
                sp.all_gather(buf, W, ctx, 0).unwrap();
            }
        });
        let mut out = vec![0u8; start.len()];
        gpu.copy_d2h(buf, &mut out).unwrap();
        out
    };
    std::thread::scope(|s| {
        let a = s.spawn(|| run(0));
        let b = s.spawn(|| run(1));
        [a.join().unwrap(), b.join().unwrap()]
    })
}

#[test]
fn overlapped_seam_leaves_the_whole_seams_bytes() {
    for total in [4096usize, 8192, 5388] {
        let whole = run_pair(total, false);
        let blocked = run_pair(total, true);
        assert!(whole[0] == whole[1], "{total}: ranks agree");
        assert!(blocked == whole, "{total}: overlapped seam differs");
    }
}
