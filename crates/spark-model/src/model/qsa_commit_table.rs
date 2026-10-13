// SPDX-License-Identifier: AGPL-3.0-only

//! The wide step's staged QSA commits as one launch
//! (`ATLAS_QWEN4EXP_QSA_COMMIT_TABLE=1`, default off; needs
//! `ATLAS_QWEN4EXP_QSA_COMMIT_ROWS=1`).
//!
//! After a wide step's layer loop the model commits every indexer layer's
//! staged raw keys per sequence (`decode_pieces::qsa_commit_staged`): one
//! pitched copy and one `qsa_block_pool` launch per (layer, sequence) -- 96
//! eager ops at C8 over the 12 QSA layers, ~1.03 ms a step of which ~0.7 is
//! launch and dependency gaps (nsys, rank 0, 2026-10-07). The host
//! bookkeeping (window room, counters) is unchanged and still runs per
//! commit; the device work is collected (`layers::qsa::CommitTable`) and runs
//! as ONE `qsa_commit_table` launch over a table uploaded with one copy:
//! each entry's copy then its pool, the pool arithmetic `qsa_block_pool`'s
//! own, so the raw keys, pooled keys and counters are the bytes and values
//! of the per-commit launches. Admitted only when every (layer, sequence)
//! has one commit, so entries touch disjoint state and their order is free.
//! `scripts/dev/qsa_commit_table_bench.cu`: byte-equal raw windows and
//! pooled keys over 1032 random entries; 766 -> 15 us at 96 entries.

use std::sync::OnceLock;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::pinned_upload::PinnedUpload;
use crate::layers::qsa::CommitEntry;

/// `ATLAS_QWEN4EXP_QSA_COMMIT_TABLE=1`, read once.
pub(crate) fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_QSA_COMMIT_TABLE").as_deref() == Ok("1"))
}

/// Bytes of one `QsaCommitEntry` (qsa_indexer.cu): five pointers, four u32.
const ENTRY_BYTES: usize = 56;

/// The kernel's table for `entries`, little-endian.
pub(crate) fn table_bytes(entries: &[CommitEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * ENTRY_BYTES);
    for e in entries {
        for p in [e.src, e.dst, e.raw_origin, e.k_norm_w, e.block_keys] {
            out.extend_from_slice(&p.0.to_le_bytes());
        }
        for w in [e.src_pitch, e.count, e.first_block, e.n_new] {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

/// Whether `entries` can run as one launch: one pooling shape and kernel.
pub(crate) fn one_launch(entries: &[CommitEntry]) -> bool {
    entries.first().is_some_and(|f| {
        entries
            .iter()
            .all(|e| e.shape == f.shape && e.kernel.0 == f.kernel.0)
    })
}

/// The device table, grown on demand, and its page-locked upload buffer
/// (process-wide: one model a process). Pageable, the upload would stage
/// synchronously and hold the host behind the step's graphs.
static TABLE: Mutex<(DevicePtr, usize, PinnedUpload)> =
    Mutex::new((DevicePtr(0), 0, PinnedUpload::EMPTY));

/// Launch `entries` on `stream`: one upload, one kernel (module docs).
pub(crate) fn launch(gpu: &dyn GpuBackend, entries: &[CommitEntry], stream: u64) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    // The upload is a host copy into the table: a capture would replay it.
    anyhow::ensure!(
        !gpu.stream_is_capturing(stream),
        "QSA commit table: launched inside a graph capture"
    );
    let bytes = table_bytes(entries);
    let mut t = TABLE.lock();
    if t.1 < bytes.len() {
        // Every earlier launch reading the old table must have finished.
        gpu.synchronize(stream)?;
        if t.0.0 != 0 {
            gpu.free(t.0)?;
            // Forget it before the alloc can fail: never upload into freed memory.
            t.0 = DevicePtr(0);
            t.1 = 0;
        }
        let cap = bytes.len().next_multiple_of(64 * ENTRY_BYTES);
        t.0 = gpu.alloc(cap)?;
        t.1 = cap;
    }
    t.2.stage(gpu, bytes.len())?.copy_from_slice(&bytes);
    let dev = t.0;
    t.2.send(gpu, bytes.len(), dev, stream)?;
    // Layers of different pooling shapes (none on this checkpoint): one
    // launch per entry, from its own row of the table.
    let groups: Vec<(usize, usize)> = if one_launch(entries) {
        vec![(0, entries.len())]
    } else {
        (0..entries.len()).map(|i| (i, 1)).collect()
    };
    for (first, n) in groups {
        let e = &entries[first];
        let (ratio, hd, rot, theta, eps) = e.shape;
        KernelLaunch::new(gpu, e.kernel)
            .grid([n as u32, 1, 1])
            .block([hd, 1, 1])
            .shared_mem((hd + 32) * 4)
            .arg_ptr(t.0.offset(first * ENTRY_BYTES))
            .arg_u32(ratio)
            .arg_u32(hd)
            .arg_u32(rot)
            .arg_f32(f32::from_bits(theta))
            .arg_f32(f32::from_bits(eps))
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::KernelHandle;

    fn entry(i: u64) -> CommitEntry {
        CommitEntry {
            kernel: KernelHandle(7),
            src: DevicePtr(0x100 + i),
            dst: DevicePtr(0x200 + i),
            raw_origin: DevicePtr(0x300 + i),
            k_norm_w: DevicePtr(0x400 + i),
            block_keys: DevicePtr(0x500 + i),
            src_pitch: 640,
            count: 4,
            first_block: 9,
            n_new: 1,
            shape: (4, 128, 64, 10000f32.to_bits(), 1e-6f32.to_bits()),
        }
    }

    #[test]
    fn the_table_is_the_kernels_struct() {
        let t = table_bytes(&[entry(0), entry(1)]);
        assert_eq!(t.len(), 2 * ENTRY_BYTES);
        let w = |at: usize| u64::from_le_bytes(t[at..at + 8].try_into().unwrap());
        let h = |at: usize| u32::from_le_bytes(t[at..at + 4].try_into().unwrap());
        assert_eq!(
            [w(0), w(8), w(16), w(24), w(32)],
            [0x100, 0x200, 0x300, 0x400, 0x500]
        );
        assert_eq!([h(40), h(44), h(48), h(52)], [640, 4, 9, 1]);
        assert_eq!(w(ENTRY_BYTES), 0x101);
    }

    #[test]
    fn one_launch_needs_one_shape() {
        let mut b = entry(1);
        assert!(one_launch(&[entry(0), b]));
        b.shape.2 = 32;
        assert!(!one_launch(&[entry(0), b]));
        assert!(!one_launch(&[]));
    }
}
