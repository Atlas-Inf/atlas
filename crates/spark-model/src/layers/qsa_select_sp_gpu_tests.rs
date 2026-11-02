// SPDX-License-Identifier: AGPL-3.0-only

//! The split selection on CUDA, both ranks at once: two threads, each with
//! its own backend and copy of the inputs, swapping lists through a host
//! mailbox. Each rank's attention output must be byte-identical to the
//! unsplit loop's, over three slabs (the odd last one rides the spare slot).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Result, bail};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::glm_sp::{self, SpRows};
use crate::layers::glm_sp_uneven::tests::with_model_ctx;
use crate::layers::qsa::QsaIndexer;

#[derive(Default)]
struct Wire {
    queues: Mutex<[VecDeque<Vec<u8>>; 2]>,
    cv: Condvar,
}

struct GpuPair<'a> {
    gpu: &'a dyn GpuBackend,
    rank: usize,
    wire: Arc<Wire>,
}

impl CommBackend for GpuPair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn exchange_async(&self, send: u64, dst: u64, n: usize, add: bool, s: u64) -> Result<bool> {
        assert!(!add, "list swaps copy");
        self.gpu.synchronize(s)?;
        let mut out = vec![0u8; n];
        self.gpu.copy_d2h(DevicePtr(send), &mut out)?;
        let mut q = self.wire.queues.lock().unwrap();
        q[self.rank].push_back(out);
        self.wire.cv.notify_all();
        let got = loop {
            if let Some(p) = q[1 - self.rank].pop_front() {
                break p;
            }
            q = self.wire.cv.wait(q).unwrap();
        };
        drop(q);
        assert_eq!(got.len(), n, "both ranks move the same bytes");
        self.gpu.copy_h2d(&got, DevicePtr(dst))?;
        Ok(true)
    }
    fn supports_exchange_async(&self, _: usize) -> bool {
        true
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
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

const HIDDEN: usize = 2560;
const HEADS: usize = 4;
const HD: usize = 128;
const RATIO: usize = 4;
const BUDGET: usize = 2048;
const NQ: usize = 4;
const NKV: usize = 2;
const HD_ATTN: usize = 256;
const BS: usize = 16;
const ROWS: usize = 2048;
/// Three slabs past the bound: 2048 + 2048 + 300.
const T_ALL: usize = BUDGET + RATIO - 1 + 2 * ROWS + 300;

/// Deterministic BF16 values in about [-scale, scale].
fn bf16s(seed: u32, n: usize, scale: f32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale;
            ((x.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// One run's selective attention output: unsplit with `rank == None`, else
/// that rank of the split pair.
fn run(rank: Option<(usize, Arc<Wire>)>) -> Vec<u8> {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let up = |bytes: Vec<u8>| {
        let p = g.alloc(bytes.len()).unwrap();
        g.copy_h2d(&bytes, p).unwrap();
        p
    };
    let qsa = QsaIndexer::new(
        up(bf16s(1, (HEADS + 1) * HD * HIDDEN, 0.05)),
        up(bf16s(2, HD, 1.0)),
        up(bf16s(3, HD, 1.0)),
        HEADS,
        HD,
        RATIO,
        BUDGET,
        8192,
        64,
        1.0e7,
        1e-6,
        HIDDEN,
        NKV,
        HD_ATTN,
        g,
    )
    .unwrap();
    let hidden = up(bf16s(4, T_ALL * HIDDEN, 1.0));
    let mut st = qsa.new_seq_state(g).unwrap();
    qsa.prefill_ingest(&mut st, hidden, T_ALL, 0, g, stream)
        .unwrap();
    let q_row = NQ * HD_ATTN;
    let q_roped = up(bf16s(5, T_ALL * q_row, 1.0));
    let attn_ctx = up(vec![0u8; T_ALL * q_row * 2]);
    let pages = T_ALL.div_ceil(BS);
    let kpool = up(bf16s(6, pages * BS * NKV * HD_ATTN, 1.0));
    let vpool = up(bf16s(7, pages * BS * NKV * HD_ATTN, 1.0));
    let table: Vec<u32> = (0..pages as u32).collect();
    let qkw = (HEADS + 1) * HD;
    let stride = T_ALL.div_ceil(RATIO);
    let topk = BUDGET / RATIO;
    let scratch = g
        .alloc(ROWS * qkw * 2 + ROWS * HEADS * HD * 4 + ROWS * stride * 4 + ROWS * topk * 4)
        .unwrap();
    let mut select = |ctx: Option<&crate::layer::ForwardContext>| {
        qsa.prefill_select(
            &mut st,
            hidden,
            q_roped,
            attn_ctx,
            kpool,
            vpool,
            &table,
            0,
            T_ALL,
            NQ as u32,
            BS as u32,
            1.0 / (HD_ATTN as f32).sqrt(),
            scratch,
            ctx,
            g,
            stream,
        )
    };
    match rank {
        None => select(None).unwrap(),
        Some((rank, wire)) => {
            let pair = GpuPair { gpu: g, rank, wire };
            with_model_ctx(g, &pair, Some("qwen4_exp"), |ctx| {
                let _sp = glm_sp::enter(SpRows::split_at(T_ALL, T_ALL / 2, rank));
                select(Some(ctx)).unwrap();
            });
            assert_ne!(
                super::SLOTS.with(|s| s.get().1),
                0,
                "rank {rank}: split taken"
            );
        }
    }
    g.synchronize(stream).unwrap();
    let mut out = vec![0u8; T_ALL * q_row * 2];
    g.copy_d2h(attn_ctx, &mut out).unwrap();
    out
}

#[test]
#[ignore]
fn split_selection_is_byte_identical() {
    // SAFETY: set before any thread reads the environment.
    unsafe { std::env::set_var("ATLAS_QWEN4EXP_PREFILL_QSA_SPLIT", "1") };
    let want = run(None);
    let q_row = NQ * HD_ATTN;
    let first = (BUDGET + RATIO - 1) * q_row * 2;
    assert!(
        want[first..].chunks_exact(2).any(|c| c != [0, 0]),
        "the selective rows were written"
    );
    let wire = Arc::new(Wire::default());
    let got: Vec<Vec<u8>> = std::thread::scope(|s| {
        let h: Vec<_> = (0..2)
            .map(|r| {
                let w = wire.clone();
                s.spawn(move || run(Some((r, w))))
            })
            .collect();
        h.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (rank, g) in got.iter().enumerate() {
        let diff = g.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff, 0,
            "rank {rank}: {diff} bytes differ from the unsplit loop"
        );
    }
}
