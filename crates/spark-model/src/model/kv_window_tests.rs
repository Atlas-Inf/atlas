// SPDX-License-Identifier: AGPL-3.0-only

//! A sequence's window holds through decode, verify rollback, prefill chunks,
//! truncation and snapshot pins, on the production window pool.

use super::KvWindow;
use crate::traits::SequenceState;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache, WindowPlan};

const BS: usize = 16;
const WINDOW: usize = 64;

fn cache(gpu: &MockGpuBackend, local: usize, pins: usize) -> PagedKvCache {
    let config = KvCacheConfig {
        block_size: BS,
        num_kv_heads: 1,
        head_dim: 8,
        num_layers: 2,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let plan = WindowPlan {
        layers: vec![1],
        window: WINDOW,
        blocks: local,
        pins,
    };
    PagedKvCache::new_windowed(config, 256, plan, gpu).unwrap()
}

/// A sequence holding primary blocks for positions `..tokens`.
fn seq_with(kv: &mut PagedKvCache, tokens: usize) -> SequenceState {
    let mut seq = SequenceState::host_only(0);
    seq.block_table = (0..tokens.div_ceil(BS))
        .map(|_| kv.alloc_block().unwrap())
        .collect();
    seq
}

fn mapped(kv: &PagedKvCache, seq: &SequenceState) -> Vec<usize> {
    (0..seq.block_table.len())
        .filter(|&i| kv.window_mapped(seq.block_table[i]))
        .collect()
}

#[test]
fn decode_holds_the_window_and_lets_go_below_it() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 0);
    let window = KvWindow::new(&kv);
    let mut seq = seq_with(&mut kv, 256);
    window
        .sync(&mut seq, 0, 199 / BS, &mut kv, &gpu, 0)
        .unwrap();
    for pos in 200..=230 {
        seq.seq_len = pos;
        window
            .sync(&mut seq, pos, pos / BS, &mut kv, &gpu, 0)
            .unwrap();
        // keys from floor32(pos + 1 - 64)
        let first = (pos + 1 - WINDOW) / 32 * 32 / BS;
        assert_eq!(seq.window_held, first..pos / BS + 1, "pos {pos}");
        assert_eq!(mapped(&kv, &seq), (first..pos / BS + 1).collect::<Vec<_>>());
    }
}

#[test]
fn a_rejected_draft_never_loses_a_window_row() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 0);
    let window = KvWindow::new(&kv);
    let mut seq = seq_with(&mut kv, 320);
    window
        .sync(&mut seq, 0, 159 / BS, &mut kv, &gpu, 0)
        .unwrap();
    // a verify of 16 rows from 160, then the drafts after one are rejected
    window
        .sync(&mut seq, 160, 175 / BS, &mut kv, &gpu, 0)
        .unwrap();
    let held = seq.window_held.clone();
    window
        .sync(&mut seq, 162, 162 / BS, &mut kv, &gpu, 0)
        .unwrap();
    // the floor stays put and the rejected rows' block stays lent
    assert_eq!(seq.window_held, held);
}

#[test]
fn a_prefill_chunk_holds_its_rows_then_trims_to_the_window() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 0);
    let window = KvWindow::new(&kv);
    let mut seq = seq_with(&mut kv, 1024);
    window
        .sync(&mut seq, 0, 511 / BS, &mut kv, &gpu, 0)
        .unwrap();
    assert_eq!(seq.window_held, 0..32);
    seq.seq_len = 512;
    window.trim(&mut seq, &mut kv, &gpu, 0).unwrap();
    assert_eq!(seq.window_held, (512 + 1 - WINDOW) / 32 * 32 / BS..32);
    window
        .sync(&mut seq, 512, 1023 / BS, &mut kv, &gpu, 0)
        .unwrap();
    assert_eq!(kv.window_free_blocks(), 63 - 32 - 4);
}

#[test]
fn rows_already_written_must_still_be_mapped() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 0);
    let window = KvWindow::new(&kv);
    // a sequence that resumes at 160 with no window rows behind it
    let mut seq = seq_with(&mut kv, 200);
    let err = window.sync(&mut seq, 160, 160 / BS, &mut kv, &gpu, 0);
    assert!(format!("{:#}", err.unwrap_err()).contains("window rows"));
    assert_eq!(seq.window_held, 0..0);
    assert_eq!(kv.window_free_blocks(), 63);
}

#[test]
fn truncation_releases_the_dropped_blocks_holds() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 0);
    let window = KvWindow::new(&kv);
    let mut seq = seq_with(&mut kv, 96);
    window.sync(&mut seq, 0, 5, &mut kv, &gpu, 0).unwrap();
    window.truncate(&mut seq, 4, &mut kv, &gpu, 0).unwrap();
    assert_eq!(seq.window_held, 0..4);
    assert_eq!(mapped(&kv, &seq), vec![0, 1, 2, 3]);
    window.truncate(&mut seq, 0, &mut kv, &gpu, 0).unwrap();
    assert_eq!(kv.window_free_blocks(), 63);
}

#[test]
fn a_pinned_snapshot_window_outlives_its_sequence_for_a_resume() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 4);
    let window = KvWindow::new(&kv);
    let mut seq = seq_with(&mut kv, 160);
    window
        .sync(&mut seq, 0, 159 / BS, &mut kv, &gpu, 0)
        .unwrap();
    seq.seq_len = 160;
    // aux collection queues the pin without the cache lock
    window.queue_pin(&seq);
    let prefix = seq.block_table.clone();
    for &b in &prefix {
        kv.inc_ref(b); // the prefix cache's own ref
    }
    window.truncate(&mut seq, 0, &mut kv, &gpu, 0).unwrap();
    kv.free_blocks(&seq.block_table);
    // a new sequence adopts the cached prefix and resumes at 160
    let mut next = SequenceState::host_only(1);
    for &b in &prefix {
        kv.inc_ref(b);
    }
    next.block_table = prefix;
    next.seq_len = 160;
    window
        .sync(&mut next, 160, 159 / BS, &mut kv, &gpu, 0)
        .unwrap();
    assert_eq!(next.window_held, (161 - WINDOW) / 32 * 32 / BS..10);
}

#[test]
fn cached_chunks_below_a_resume_lend_no_rows() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 4);
    let window = KvWindow::new(&kv);
    // a finished sequence left a snapshot at 160 with its window pinned
    let mut done = seq_with(&mut kv, 160);
    window
        .sync(&mut done, 0, 159 / BS, &mut kv, &gpu, 0)
        .unwrap();
    done.seq_len = 160;
    window.queue_pin(&done);
    let prefix = done.block_table.clone();
    prefix.iter().for_each(|&b| kv.inc_ref(b));
    window.truncate(&mut done, 0, &mut kv, &gpu, 0).unwrap();
    kv.free_blocks(&done.block_table);
    // a resume there adopts the prefix and the snapshot's window
    let mut seq = SequenceState::host_only(1);
    prefix.iter().for_each(|&b| kv.inc_ref(b));
    seq.block_table = prefix;
    seq.block_table
        .extend((0..10).map(|_| kv.alloc_block().unwrap()));
    window
        .sync(&mut seq, 160, 159 / BS, &mut kv, &gpu, 0)
        .unwrap();
    seq.marconi_skip_to = 160;
    let adopted = seq.window_held.clone();
    // the chunk loop then reserves the cached chunk 0..128 from position 0
    window
        .sync(&mut seq, 0, 127 / BS, &mut kv, &gpu, 0)
        .unwrap();
    seq.seq_len = 128;
    window.trim(&mut seq, &mut kv, &gpu, 0).unwrap();
    // no cached row is lent fresh (it would read as zeros)
    assert_eq!(seq.window_held, adopted);
    assert_eq!(mapped(&kv, &seq), adopted.clone().collect::<Vec<_>>());
    // the chunk that computes from 160 lends its own rows
    window
        .sync(&mut seq, 128, 255 / BS, &mut kv, &gpu, 0)
        .unwrap();
    assert_eq!(seq.window_held, adopted.start..16);
}

#[test]
fn a_resume_without_its_window_is_refused_lending_nothing() {
    let gpu = MockGpuBackend::new();
    let mut kv = cache(&gpu, 64, 0);
    let window = KvWindow::new(&kv);
    let mut seq = seq_with(&mut kv, 320);
    seq.marconi_skip_to = 160;
    assert!(
        window
            .sync(&mut seq, 0, 127 / BS, &mut kv, &gpu, 0)
            .is_err()
    );
    assert_eq!(seq.window_held, 0..0);
    assert_eq!(kv.window_free_blocks(), 63);
}

/// Two prompts prefilled chunk by chunk in turn, each chunk held then
/// trimmed, keep both windows in a pool sized for two sequences and one
/// forward's rows.
#[test]
fn interleaved_chunked_prefills_keep_both_windows() {
    const CHUNK: usize = 128;
    let gpu = MockGpuBackend::new();
    let local = WindowPlan::size_blocks(WINDOW, BS, 2, 8, CHUNK, 0);
    let mut kv = cache(&gpu, local, 0);
    let window = KvWindow::new(&kv);
    let mut seqs = [seq_with(&mut kv, 4 * CHUNK), seq_with(&mut kv, 4 * CHUNK)];
    for chunk in 0..4 {
        for seq in &mut seqs {
            let end = (chunk + 1) * CHUNK;
            let first = seq.seq_len;
            window
                .sync(seq, first, (end - 1) / BS, &mut kv, &gpu, 0)
                .unwrap();
            seq.seq_len = end;
            window.trim(seq, &mut kv, &gpu, 0).unwrap();
        }
    }
    for seq in &mut seqs {
        let first = seq.seq_len;
        window
            .sync(seq, first, first / BS, &mut kv, &gpu, 0)
            .unwrap();
    }
}

/// Every prefill path reserves its blocks through the model's
/// `ensure_prefill_blocks` (or `reserve_prefill_blocks`), which hold the
/// window rows the forward writes. A direct `ensure_blocks_through_prefill`
/// leaves them unmapped, and the next step finds them gone.
#[test]
fn prefill_blocks_are_reserved_with_their_window() {
    fn scan(dir: &std::path::Path, offenders: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if path.is_dir() {
                scan(&path, offenders);
            } else if name.ends_with(".rs")
                && !name.ends_with("tests.rs")
                && !matches!(name.as_str(), "block_mgmt.rs" | "kv_admission.rs")
            {
                let mut source = std::fs::read_to_string(&path).unwrap();
                source.retain(|c| !c.is_whitespace());
                if source.contains("ensure_blocks_through_prefill(") {
                    offenders.push(path.display().to_string());
                }
            }
        }
    }
    let mut offenders = Vec::new();
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    scan(&src, &mut offenders);
    assert!(offenders.is_empty(), "{offenders:?}");
}
