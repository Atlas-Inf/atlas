// SPDX-License-Identifier: AGPL-3.0-only

//! The specialized four-rank ops through the real NIC (see the parent): the
//! all-to-all, the FP8 reduce-scatter and the fused reduce-scatter.

use super::*;

/// The KDA o_proj exchange of an `n`-row chunk (8192, a chunk at
/// `--max-prefill-tokens=8192`): each rank holds `[n, 2048]` E4M3 rows of its heads and
/// their tiled UE8M0 scales (128 x 4 per 512-byte tile); part `d` of the rows
/// belongs to rank `3 - d`, which lands every origin's part as the K slice
/// `[o * 2048, ..)` of its `[n / 4, 8192]` operand and the scale tiles at
/// that slice's tile columns. Checked byte for byte, then timed back to back.
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_all_to_all() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let n = 8192usize;
    let (kl, rows) = (2048usize, n / RANKS);
    let (q_part, sf_part) = (rows * kl, rows * kl / 32);
    let tile_row = 16 * 512; // one 128-row band of a 2048-column slice
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    // Distinct bytes per rank and position (any u8 is an E4M3 / UE8M0 code).
    let fill = |rank: usize, salt: u64, bytes: usize| -> Vec<u16> {
        (0..bytes / 2)
            .map(|i| mix(salt ^ (rank as u64) << 40 ^ i as u64) as u16)
            .collect()
    };
    let srcs: Vec<(u64, u64, Vec<u16>, Vec<u16>)> = (0..RANKS)
        .map(|r| {
            let (q, sf) = (alloc(RANKS * q_part), alloc(RANKS * sf_part));
            let (hq, hs) = (fill(r, 1, RANKS * q_part), fill(r, 2, RANKS * sf_part));
            upload(&g, q, &hq);
            upload(&g, sf, &hs);
            (q, sf, hq, hs)
        })
        .collect();
    let dsts: Vec<(u64, u64)> = (0..RANKS)
        .map(|_| (alloc(rows * RANKS * kl), alloc(rows * RANKS * kl / 32)))
        .collect();
    ck!(cuStreamSynchronize(g.stream));
    let planes = |r: usize| {
        let (q, sf, _, _) = &srcs[r];
        let (a, a_sf) = dsts[r];
        // Part d (for rank 3 - d) at d parts in: rank d's part counts down.
        [
            crate::AllToAllPlane {
                src: q + ((RANKS - 1) * q_part) as u64,
                src_stride: -(q_part as i64),
                dst: a,
                dst_stride: kl as i64,
                rows,
                row_bytes: kl,
                pitch: RANKS * kl,
            },
            crate::AllToAllPlane {
                src: sf + ((RANKS - 1) * sf_part) as u64,
                src_stride: -(sf_part as i64),
                dst: a_sf,
                dst_stride: tile_row as i64,
                rows: rows / 128,
                row_bytes: tile_row,
                pitch: RANKS * tile_row,
            },
        ]
    };
    let issue = || {
        for (r, chain) in chains.iter().enumerate() {
            assert!(chain.all_to_all(&planes(r), streams[r]).unwrap());
        }
    };
    // The shape needs ops of 1/16 of the chunk at least (two planes of whole
    // rows): at the default 1 MiB one-shot maximum an 8K chunk does not fit.
    let fits: Vec<bool> = (0..RANKS)
        .map(|r| chains[r].all_to_all(&planes(r), streams[r]).unwrap())
        .collect();
    if !fits.iter().all(|&f| f) {
        assert!(fits.iter().all(|&f| !f), "the ranks disagree on the shape");
        eprintln!(
            "{n}-row all-to-all exceeds the one-shot maximum: skipped (ATLAS_RDMA_ONESHOT_MAX)"
        );
        return;
    }
    sync(&streams, &chains);
    issue();
    sync(&streams, &chains);
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut wrong = 0usize;
    for (r, &(a, a_sf)) in dsts.iter().enumerate() {
        let got_q = bytes(&download(&g, a, rows * RANKS * kl / 2));
        let got_sf = bytes(&download(&g, a_sf, rows * RANKS * kl / 64));
        let part = RANKS - 1 - r;
        for (o, src) in srcs.iter().enumerate() {
            let (q, sf) = (bytes(&src.2), bytes(&src.3));
            for row in 0..rows {
                let want = &q[part * q_part + row * kl..][..kl];
                let at = row * RANKS * kl + o * kl;
                wrong += got_q[at..at + kl]
                    .iter()
                    .zip(want)
                    .filter(|(x, y)| x != y)
                    .count();
            }
            for t in 0..rows / 128 {
                let want = &sf[part * sf_part + t * tile_row..][..tile_row];
                let at = t * RANKS * tile_row + o * tile_row;
                wrong += got_sf[at..at + tile_row]
                    .iter()
                    .zip(want)
                    .filter(|(x, y)| x != y)
                    .count();
            }
        }
    }
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        issue();
    }
    sync(&streams, &chains);
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let poison: Vec<_> = chains.iter().map(|c| c.oneshot().poisoned()).collect();
    println!(
        "chain NIC loopback all-to-all: {n} rows, {} KiB a pair a rank, {wrong} wrong bytes, poison {poison:?}; {us:.0} us back to back",
        (q_part + sf_part) >> 10
    );
    assert_eq!(wrong, 0);
    assert!(poison.iter().all(Option::is_none));
}

/// The lossy FP8 reduce-scatter (`reduce_scatter_fp8_async`): every rank's
/// own segment is within the quantization's bound of the exact sum of the
/// four BF16 partials (each of at most three E4M3 roundings on a path off by
/// at most 1/16 of its value, plus the BF16 rounding), close on average, and
/// the same bits on a second run of the same inputs; one op and several.
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_reduce_scatter_fp8() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let max = chains[0].oneshot().max_bytes();
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    let bf16 = |v: u16| f32::from_bits(u32::from(v) << 16) as f64;
    // Magnitudes 2^-4 .. 2^4 within a row, either sign.
    let fill = |rank: usize, salt: u64, n: usize| -> Vec<u16> {
        (0..n as u64)
            .map(|i| {
                let x = mix(salt << 40 ^ (rank as u64) << 36 ^ i);
                let v = ((x % 2001) as f32 - 1000.0) / 1000.0 * 2f32.powi((x >> 20) as i32 % 8 - 4);
                (v.to_bits() >> 16) as u16
            })
            .collect()
    };
    let sizes = [
        4 * 8192,
        4 * 65_536,
        4 * 1_048_576,
        4 * (max / 4 / 8192 * 8192 + 8192),
    ];
    let mut worst = 0f64;
    for (salt, &bytes) in sizes.iter().enumerate() {
        let (n, part) = (bytes / 2, bytes / RANKS);
        let inputs: Vec<Vec<u16>> = (0..RANKS).map(|r| fill(r, salt as u64, n)).collect();
        let bufs: Vec<u64> = (0..RANKS).map(|_| alloc(bytes)).collect();
        let run = || -> Vec<Vec<u16>> {
            for r in 0..RANKS {
                upload(&g, bufs[r], &inputs[r]);
            }
            ck!(cuStreamSynchronize(g.stream));
            for (r, chain) in chains.iter().enumerate() {
                let top = bufs[r] + ((RANKS - 1) * part) as u64;
                assert!(
                    chain
                        .reduce_scatter(top, -(part as i64), part, true, streams[r])
                        .unwrap()
                );
            }
            sync(&streams, &chains);
            (0..RANKS).map(|r| download(&g, bufs[r], n)).collect()
        };
        let (first, second) = (run(), run());
        let (mut bad, mut rel, mut count) = (0usize, 0f64, 0usize);
        for rank in 0..RANKS {
            let from = (RANKS - 1 - rank) * n / RANKS;
            assert_eq!(
                first[rank][from..][..n / RANKS],
                second[rank][from..][..n / RANKS],
                "rank {rank}: a second run of the same inputs differs ({bytes} bytes)"
            );
            for i in from..from + n / RANKS {
                let xs: Vec<f64> = (0..RANKS).map(|r| bf16(inputs[r][i])).collect();
                let exact: f64 = xs.iter().sum();
                let mag: f64 = xs.iter().map(|x| x.abs()).sum();
                let got = bf16(first[rank][i]);
                let err = (got - exact).abs();
                // Three roundings of at most 1/16 of a value no larger than
                // `mag`, the BF16 rounding, and E4M3's floor under a group
                // whose largest value is at most 16 * 4.
                if err > 3.0 * mag / 16.0 + exact.abs() / 128.0 + 64.0 / 448.0 / 512.0 * 3.0 {
                    bad += 1;
                }
                if exact.abs() > 0.5 {
                    rel += err / exact.abs();
                    count += 1;
                }
            }
        }
        let mean = rel / count.max(1) as f64;
        worst = worst.max(mean);
        eprintln!(
            "fp8 reduce-scatter {bytes} B: {bad} values out of bound, mean relative error {mean:.4} over {count}"
        );
        assert_eq!(bad, 0, "{bytes} bytes: values outside the FP8 bound");
    }
    assert!(worst < 0.1, "mean relative error {worst:.4}");
}

/// The fused reduce-scatter (`rs_fused_begin`, `rdma_rs_fused_copy` as the
/// producer, `reduce_scatter_fused_async`): each rank's own segment is the
/// plain reduce-scatter's words, its completion words show the op's epoch,
/// over seams of several sizes with an all-reduce between them (a fused op
/// right after another is refused), the producer and the chain kernel on
/// two streams with nothing ordering them, up to the chain's whole segment
/// (`oneshot::config::RS_FUSED_SEG`).
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_reduce_scatter_fused() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let producer = load("rdma_oneshot.cubin", "rdma_rs_fused_copy");
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    let sides = [new_stream(), new_stream(), new_stream(), new_stream()];
    let cap = crate::nccl_backend::rdma_pair::oneshot::RS_FUSED_SEG;
    let mut wrong = 0;
    for (k, seg) in [65_536usize, 1 << 20, cap / 2, cap].into_iter().enumerate() {
        let k = k as u64 * 7 + 3;
        let step = Step::new(4 * seg, Kind::ReduceScatter, 0);
        step.prepare(&g, k);
        let fused: Vec<crate::RsFused> = chains
            .iter()
            .enumerate()
            .map(|(r, c)| {
                let top = step.buf[r] + (3 * seg) as u64;
                c.rs_fused_begin(top, -(seg as i64), seg, 16)
                    .unwrap()
                    .expect("fused op refused")
            })
            .collect();
        for (r, chain) in chains.iter().enumerate() {
            let mut plan = fused[r].plan;
            let mut params: [*mut c_void; 1] = [(&raw mut plan).cast()];
            ck!(cuLaunchKernel(
                producer,
                8,
                1,
                1,
                256,
                1,
                1,
                0,
                streams[r],
                params.as_mut_ptr(),
                std::ptr::null_mut()
            ));
            assert!(chain.reduce_scatter_fused(&fused[r], sides[r]).unwrap());
            // A second fused op before an all-reduce is refused.
            let top = step.buf[r] + (3 * seg) as u64;
            assert!(
                chain
                    .rs_fused_begin(top, -(seg as i64), seg, 16)
                    .unwrap()
                    .is_none()
            );
        }
        sync(&sides, &chains);
        sync(&streams, &chains);
        let bad = step.check(&g, k);
        for (r, f) in fused.iter().enumerate() {
            let mut done = vec![0u32; f.pieces];
            ck!(cuMemcpyDtoHAsync_v2(
                done.as_mut_ptr().cast(),
                f.done,
                4 * f.pieces,
                g.stream
            ));
            ck!(cuStreamSynchronize(g.stream));
            assert!(
                done.iter().all(|&d| d == f.epoch),
                "rank {r}: done words {done:?}, epoch {}",
                f.epoch
            );
        }
        eprintln!("fused reduce-scatter of {seg}-byte segments: {bad} wrong");
        wrong += bad;
        // An all-reduce between fused ops.
        let ar = Step::new(65_536, Kind::Sum, 0);
        ar.prepare(&g, k + 1);
        ar.issue(&chains, &streams);
        sync(&streams, &chains);
        wrong += ar.check(&g, k + 1);
    }
    assert_eq!(wrong, 0);
}
