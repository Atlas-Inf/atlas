// SPDX-License-Identifier: AGPL-3.0-only

//! Geometry and the exchange + assembly of the qwen4_exp LM-head split, on
//! the mock GPU with a communicator that hands each rank its peer's staging.

use std::sync::Mutex;

use anyhow::{Result, bail};
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

/// The checkpoint's tokenizer vocabulary (odd).
const FLASH_NEXT_VOCAB: usize = 248077;

#[test]
fn flash_next_split_is_aligned_and_covers_the_vocabulary_once() {
    let s = Split::new(FLASH_NEXT_VOCAB).unwrap();
    assert_eq!((s.start(1), s.width()), (124032, 124045));
    assert_eq!(s.start(1) % 8, 0, "rank 1's weight rows start on 16 bytes");
    assert_eq!(s.start(1) + s.width(), FLASH_NEXT_VOCAB);
    // 248 KB a row: four rows per exchange stay under the 1 MiB one-shot.
    assert_eq!(s.chunk_rows(8, 1 << 20), 4);
}

#[test]
fn every_vocabulary_partitions_into_two_owned_ranges() {
    for vocab in (16..300).chain([154856, 248320, FLASH_NEXT_VOCAB]) {
        let s = Split::new(vocab).unwrap();
        let mut seen = vec![0u8; vocab];
        for rank in 0..2 {
            assert!(s.owned(rank) <= s.width(), "vocab {vocab} rank {rank}");
            assert!(
                s.start(rank) + s.width() <= vocab,
                "vocab {vocab} rank {rank}"
            );
            for c in s.start(rank)..s.start(rank) + s.owned(rank) {
                seen[c] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "vocab {vocab}");
    }
    assert_eq!(Split::new(7), None);
}

#[test]
fn chunks_follow_the_one_shot_cap() {
    let s = Split::new(1000).unwrap(); // width 504, 1008 bytes a row
    assert_eq!(s.chunk_rows(5, 0), 5, "no one-shot: one exchange");
    assert_eq!(s.chunk_rows(5, 1007), 5, "a row over the cap: one exchange");
    assert_eq!(s.chunk_rows(5, 1008), 1);
    assert_eq!(s.chunk_rows(5, 3 * 1008 + 1), 3);
    assert_eq!(s.chunk_rows(2, 1 << 20), 2);
}

#[test]
fn shard_arithmetic_matches_the_unsplit_ladder() {
    // `lm_head_batched` on a BF16 head: two GEMVs at 2 rows, the scalar
    // GEMM otherwise (1 row included).
    assert_eq!(HeadArith::batched(2, false), HeadArith::Gemv);
    for rows in [1, 3, 4, 8, 16] {
        assert_eq!(
            HeadArith::batched(rows, false),
            HeadArith::Gemm,
            "rows {rows}"
        );
    }
}

#[test]
fn exact_verify_projects_every_verify_width_as_gemv_rows() {
    // ATLAS_QWEN4EXP_EXACT_VERIFY: serial decode's head is `dense_gemv_bf16`;
    // the scalar tile GEMM accumulates K sequentially and rounds differently.
    for rows in [2, 3, 4] {
        assert_eq!(
            HeadArith::batched(rows, true),
            HeadArith::Gemv,
            "rows {rows}"
        );
    }
    assert_eq!(HeadArith::batched(1, true), HeadArith::Gemm);
}

#[test]
fn staging_is_off_without_the_switch() {
    let gpu = MockGpuBackend::new();
    if enabled() {
        return; // the test environment opted in
    }
    let staged = HeadSplit::alloc("qwen4_exp", FLASH_NEXT_VOCAB, None, 8, &gpu).unwrap();
    assert!(staged.is_none());
}

/// One rank of a pair: `peer_exchange_async` lands the peer's staging at the
/// same offset as this rank's send.
struct Pair {
    gpu: &'static MockGpuBackend,
    rank: usize,
    own_send: DevicePtr,
    peer_send: DevicePtr,
    max: usize,
    calls: Mutex<Vec<usize>>,
}

impl CommBackend for Pair {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn supports_peer_exchange_async(&self) -> bool {
        true
    }
    fn capturable_all_reduce_max_bytes(&self) -> usize {
        self.max
    }
    fn peer_exchange_async(&self, send: u64, recv: u64, bytes: usize, _: u64) -> Result<()> {
        self.calls.lock().unwrap().push(bytes);
        let off = (send - self.own_send.0) as usize;
        self.gpu
            .copy_d2d(self.peer_send.offset(off), DevicePtr(recv), bytes)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected receive")
    }
}

/// The full head's logit for (row, column), as BF16 bits.
fn full(row: usize, col: usize) -> u16 {
    (row * 7919 + col * 31 + 1) as u16
}

fn bytes(words: impl IntoIterator<Item = u16>) -> Vec<u8> {
    words.into_iter().flat_map(u16::to_le_bytes).collect()
}

/// Both ranks project their shard (rank 0's spare columns hold garbage),
/// swap and assemble; both must then hold exactly the full rows.
fn run_pair(vocab: usize, rows: usize, max: usize) -> Vec<usize> {
    let gpu: &'static MockGpuBackend = Box::leak(Box::new(MockGpuBackend::new()));
    let geom = Split::new(vocab).unwrap();
    let w = geom.width();
    let staging: Vec<(DevicePtr, DevicePtr, DevicePtr)> = (0..2)
        .map(|rank| {
            let send = gpu.alloc(rows * w * BF16).unwrap();
            let shard = (0..rows).flat_map(|r| {
                (0..w).map(move |c| {
                    if c < geom.owned(rank) {
                        full(r, geom.start(rank) + c)
                    } else {
                        0xDEAD
                    }
                })
            });
            gpu.copy_h2d(&bytes(shard), send).unwrap();
            let recv = gpu.alloc(rows * w * BF16).unwrap();
            let logits = gpu.alloc(rows * vocab * BF16).unwrap();
            gpu.copy_h2d(&vec![0xAB; rows * vocab * BF16], logits)
                .unwrap();
            (send, recv, logits)
        })
        .collect();
    let want = bytes((0..rows).flat_map(|r| (0..vocab).map(move |c| full(r, c))));
    let mut calls = Vec::new();
    for rank in 0..2 {
        let (send, recv, logits) = staging[rank];
        let comm = Pair {
            gpu,
            rank,
            own_send: send,
            peer_send: staging[1 - rank].0,
            max,
            calls: Mutex::default(),
        };
        exchange(&comm, gpu, geom, (send, recv), rows, 0).unwrap();
        assemble(gpu, geom, rank, (send, recv), logits, rows, 0).unwrap();
        let mut got = vec![0u8; want.len()];
        gpu.copy_d2h(logits, &mut got).unwrap();
        assert!(got == want, "vocab {vocab} rows {rows} rank {rank}");
        let mine = comm.calls.into_inner().unwrap();
        if rank == 1 {
            assert_eq!(mine, calls, "both ranks issue the same exchanges");
        }
        calls = mine;
    }
    calls
}

#[test]
fn both_ranks_assemble_the_full_rows() {
    for vocab in [37, 64, 101, 1000] {
        for rows in [1, 2, 3, 4] {
            assert_eq!(
                run_pair(vocab, rows, 0).len(),
                1,
                "one exchange without one-shot"
            );
        }
    }
}

#[test]
fn chunked_exchanges_assemble_the_same_rows() {
    let row = Split::new(101).unwrap().width() * BF16;
    assert_eq!(run_pair(101, 5, row), vec![row; 5]);
    assert_eq!(run_pair(101, 5, 2 * row), vec![2 * row, 2 * row, row]);
    assert_eq!(run_pair(101, 4, 1 << 20), vec![4 * row]);
}
