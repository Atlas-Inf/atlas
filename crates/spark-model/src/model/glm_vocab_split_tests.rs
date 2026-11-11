// SPDX-License-Identifier: AGPL-3.0-only
//! The split head's shard argmax launch: unchanged without row masks, and
//! under a strict verify's masks each rank reads its own slice of every row.
use super::launch_shard_argmax;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../layers/moe/gate_up_btile_test_gpu.rs"]
mod recording;
use recording::{Arg, Event};

const VALUE: KernelHandle = KernelHandle(7);
const ALLOW: KernelHandle = KernelHandle(9);
/// GLM-5.3: 154,880 tokens, two shards of 77,440 (2,420 mask words each).
const VOCAB: usize = 154_880;
const SHARD: usize = VOCAB / 2;
const BAN: [u32; 4] = [11, 12, u32::MAX, u32::MAX];

fn ptr(p: u64) -> Arg {
    Arg::Ptr(DevicePtr(p))
}
fn u32a(v: u32) -> Arg {
    Arg::Bytes(v.to_le_bytes().to_vec())
}

fn launch(rank: usize, allow: Option<(KernelHandle, DevicePtr, usize)>) -> Vec<Event> {
    let gpu = recording::Gpu::new();
    launch_shard_argmax(
        &gpu,
        VALUE,
        allow,
        DevicePtr(0x1000),
        DevicePtr(0x2000),
        9,
        (rank * SHARD, SHARD, VOCAB),
        BAN,
        3,
    )
    .unwrap();
    gpu.trace()
}

#[test]
fn an_unmasked_verify_launches_the_shard_argmax_as_before() {
    let abi = [
        ptr(0x1000),
        ptr(0x2000),
        u32a(SHARD as u32),
        u32a(VOCAB as u32),
    ]
    .into_iter()
    .chain(BAN.map(u32a))
    .collect();
    assert_eq!(
        launch(1, None),
        [Event::Launch(7, [9, 1, 1], [1024, 1, 1], 0, 3, abi)]
    );
}

#[test]
fn a_masked_verify_reads_each_rows_mask_from_the_ranks_first_vocabulary_bit() {
    // GLM-5.3's real split (154,856 tokens, 77,428 per rank) starts rank 1
    // inside a mask word: the kernel gets the row's mask and the bit offset.
    for (vocab, rank) in [(VOCAB, 0), (VOCAB, 1), (154_856, 0), (154_856, 1)] {
        let (shard, words) = (vocab / 2, vocab.div_ceil(32));
        let gpu = recording::Gpu::new();
        launch_shard_argmax(
            &gpu,
            VALUE,
            Some((ALLOW, DevicePtr(0x9000), words)),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            9,
            (rank * shard, shard, vocab),
            BAN,
            3,
        )
        .unwrap();
        let abi = [
            ptr(0x1000),
            ptr(0x2000),
            u32a(shard as u32),
            u32a(vocab as u32),
            ptr(0x9000),
            u32a(words as u32),
            u32a((rank * shard) as u32),
        ]
        .into_iter()
        .chain(BAN.map(u32a))
        .collect();
        assert_eq!(
            gpu.trace(),
            [Event::Launch(9, [9, 1, 1], [1024, 1, 1], 0, 3, abi)],
            "vocab {vocab} rank {rank}"
        );
    }
}
