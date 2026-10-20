// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: the windowed raw keys pool to the same BYTES as one contiguous
//! buffer of every raw key pooled at once (the layout before the window),
//! across window slides, a speculative rewind and a Marconi aux
//! snapshot/restore. Synthetic weights and activations; no checkpoint.
//!
//! `#[ignore]` per repo convention. On a GB10:
//! ```text
//! ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo test -p spark-model --release --lib qsa_window_gpu -- --ignored --nocapture
//! ```

use super::super::*;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const N_HEADS: usize = 4;
const HD: usize = 128;
const RATIO: usize = 4;
const HIDDEN: usize = 2560;

/// Deterministic BF16 values in about `[-scale, scale)`.
fn bf16s(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

struct Run<'a> {
    qsa: &'a QsaIndexer,
    g: &'a dyn GpuBackend,
    hidden: DevicePtr,
    /// Every raw key the engine wrote, by position (the contiguous layout).
    keys: Vec<u8>,
}

impl Run<'_> {
    /// Ingest `n` tokens (at most one slab, so they are all still in the
    /// window afterwards) and record their raw keys.
    fn ingest(&mut self, st: &mut QsaSeqState, n: usize) {
        let (g, row) = (self.g, HD * 2);
        let start = st.ingested;
        assert!(n <= INGEST_SLAB);
        let rows = self.hidden.offset(start * HIDDEN * 2);
        self.qsa
            .prefill_ingest(st, rows, n, start, g, g.default_stream())
            .unwrap();
        g.synchronize(g.default_stream()).unwrap();
        let got = download(g, self.qsa.raw_slot(st, start), n * row);
        self.keys[start * row..(start + n) * row].copy_from_slice(&got);
    }

    /// The old layout's block keys: pool every complete block at once.
    fn reference(&self, blocks: usize) -> Vec<u8> {
        let g = self.g;
        let raw = upload(g, &self.keys);
        let out = g.alloc(blocks.max(1) * HD * 2).unwrap();
        ops::qsa_block_pool(
            g,
            self.qsa.k_pool_k,
            raw,
            self.qsa.k_norm_w,
            out,
            0,
            blocks as u32,
            self.qsa.ratio,
            self.qsa.hd,
            self.qsa.rot,
            self.qsa.theta,
            self.qsa.eps,
            g.default_stream(),
        )
        .unwrap();
        g.synchronize(g.default_stream()).unwrap();
        download(g, out, blocks * HD * 2)
    }
}

#[test]
#[ignore]
fn qsa_window_gpu_pools_like_a_contiguous_buffer() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let total = 12_288usize;
    let qsa = QsaIndexer::new(
        upload(g, &bf16s((N_HEADS + 1) * HD * HIDDEN, 1, 0.02)),
        upload(g, &bf16s(HD, 2, 0.5)),
        upload(g, &bf16s(HD, 3, 0.5)),
        N_HEADS,
        HD,
        RATIO,
        /*budget*/ 2048,
        /*max_seq_len*/ total,
        /*rot*/ 64,
        /*theta*/ 1e7,
        /*eps*/ 1e-6,
        HIDDEN,
        /*nkv_attn*/ 2,
        /*hd_attn*/ 256,
        g,
    )
    .unwrap();
    let mut run = Run {
        qsa: &qsa,
        g,
        hidden: upload(g, &bf16s(total * HIDDEN, 7, 1.0)),
        keys: vec![0u8; total * HD * 2],
    };
    let mut st = qsa.new_seq_state(g).unwrap();
    qsa.reserve(&mut st, total, g, 0).unwrap();

    // Prefill-shaped chunks, then decode-shaped single rows across the first
    // slide, with a 4-row verify rewound to one row (and re-ingested) on the
    // way.
    run.ingest(&mut st, 2048);
    run.ingest(&mut st, 953);
    while st.ingested < 5003 {
        if st.ingested.is_multiple_of(97) {
            for _ in 0..4 {
                run.ingest(&mut st, 1);
            }
            qsa.rewind_verify(&mut st, 3).unwrap();
        }
        run.ingest(&mut st, 1);
    }
    assert!(st.raw.base > 0, "the window slid");

    // Marconi: snapshot here, restore into a fresh carry, run both on.
    let blob = qsa.snapshot_aux(&st, g, 0).unwrap();
    let mut back = qsa.new_seq_state(g).unwrap();
    qsa.reserve(&mut back, total, g, 0).unwrap();
    qsa.restore_aux(&mut back, &blob, g, 0).unwrap();
    let mut twin = Run {
        qsa: &qsa,
        g,
        hidden: run.hidden,
        keys: run.keys.clone(),
    };
    for n in [2048, 2048, 1, 1, 1, 2000] {
        run.ingest(&mut st, n);
        twin.ingest(&mut back, n);
    }
    assert_eq!(st.ingested, back.ingested);
    assert_eq!(run.keys, twin.keys, "same raw keys either way");

    let blocks = st.pooled;
    assert_eq!(blocks, st.ingested / RATIO);
    let want = run.reference(blocks);
    let row = HD * 2;
    let got = download(g, st.block_keys, blocks * row);
    let restored = download(g, back.block_keys, blocks * row);
    let first_diff = |a: &[u8]| {
        a.chunks(row)
            .zip(want.chunks(row))
            .position(|(x, y)| x != y)
    };
    assert_eq!(
        first_diff(&got),
        None,
        "windowed pooling differs from contiguous"
    );
    assert_eq!(first_diff(&restored), None, "restored carry differs");
    println!(
        "  {blocks} blocks bit-identical; window base {} cap {}",
        st.raw.base, st.raw.cap
    );
}
