// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE layer: ids -> NVMe row gather -> projections -> gate -> dilated
//! conv -> highway add.
//!
//! Runs on ONE model layer (layer 1 here) and injects into the `hc_mult`-wide
//! hyper-connection highway BEFORE that layer's attention hyper-connection,
//! matching `Qwen4ExpTextDecoderLayer.forward`'s
//! `hidden_states = hidden_states + self.ple(...)`.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::ids::{PleIdDims, ple_ngram_ids};
use crate::layer::ForwardContext;
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

#[path = "seq_state.rs"]
mod seq_state;
pub use seq_state::PleSeqState;

/// Verify windows this layer can roll back: slot `t` = state after `t` rows,
/// so K rows need K+1 slots. K is `num_drafts + 1` and the batched MoE arms
/// stop at 8, which is the bound this matches.
const VERIFY_SNAP_SLOTS: usize = 9;

/// Bisection hatch: `ATLAS_PLE_VERIFY_SNAPSHOTS=0` restores the single batched
/// conv launch, which leaves nothing for `rollback_verify` to restore — the
/// PLE carry then keeps rejected drafts. Debug only.
pub(crate) fn verify_snapshots_enabled() -> bool {
    std::env::var("ATLAS_PLE_VERIFY_SNAPSHOTS").ok().as_deref() != Some("0")
}

pub struct PleLayer {
    dims: PleIdDims,
    head_dim: usize,
    hidden: usize,
    hc_mult: usize,
    state_len: usize,
    k_size: usize,
    dilation: usize,
    eps: f32,

    key_proj: DenseWeight,
    value_proj: DenseWeight,
    norm_key: DenseWeight,
    norm_query: DenseWeight,
    norm_conv: DenseWeight,
    conv1d: DenseWeight,
    /// Behind a mutex because the NVMe cache RESOLVES (and faults, and
    /// evicts) on the forward path, which needs `&mut`, while layers are
    /// invoked through `&self`. `Arc` so a per-sequence prefill warm worker
    /// (`warm.rs`) can prefetch rows while earlier chunks/layers compute.
    table: std::sync::Arc<std::sync::Mutex<NgramTable>>,

    embed_k: KernelHandle,
    /// The dequant gather, used when `scale_va` is `Some`.
    embed_fp8_k: KernelHandle,
    /// Device VA of the table's per-slot f32 dequant scale array, `None` for a
    /// BF16 table. Its presence is what selects the FP8 gather.
    scale_va: Option<u64>,
    /// Packed EXL3 n-gram rows. Host-dequanted into `emb` before the GEMM.
    trellis: Option<ngram_trellis::NgramTrellis>,
    gemm_k: KernelHandle,
    gate_k: KernelHandle,
    conv_k: KernelHandle,
    add_k: KernelHandle,

    /// Scratch, sized once for `scratch_tokens` — see `forward_with_ids`.
    emb: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gated: DevicePtr,
    gated_normed: DevicePtr,
    out: DevicePtr,
    slots_dev: DevicePtr,
    max_tokens: usize,
    /// Width the scratch above was allocated for. Wider forwards run the
    /// pipeline in `scratch_tokens` spans — every buffer is per-token, and
    /// the one sequential piece (the conv carry) lives in `st.conv`, which
    /// threads across calls exactly as the per-row verify path relies on.
    /// Never below VERIFY_SNAP_SLOTS when `max_tokens` allows it: a
    /// verify-width forward must stay one span, because the snapshot loop
    /// indexes the whole window.
    scratch_tokens: usize,
    /// Event recorded after the gather kernel (see `gather_embed`), so
    /// `release_prev_pins` waits on THAT kernel instead of the whole stream.
    /// 0 until lazily created; 0 falls back to a full stream sync.
    gather_done: std::sync::Mutex<u64>,
    /// Pinned staging for the slot upload: (host pointer as usize, capacity
    /// in bytes). (0, 0) until the first `Cached` gather grows it.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    slots_staging: std::sync::Mutex<(usize, usize)>,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: PleIdDims,
        head_dim: usize,
        hidden: usize,
        hc_mult: usize,
        k_size: usize,
        dilation: usize,
        eps: f32,
        weights: PleWeights,
        table: NgramTable,
        max_tokens: usize,
        scratch_tokens: usize,
        trellis: Option<ngram_trellis::NgramTrellis>,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        dims.validate()?;
        let heads = dims.ngram_heads();
        anyhow::ensure!(
            heads * head_dim == hidden,
            "PLE: {heads} heads x {head_dim} dims = {} != ple_embed_dim {hidden}. \
             The head slices are CONCATENATED (not summed as LongCat's are), so \
             this product is the embedding width and a mismatch means the \
             geometry is not what we think.",
            heads * head_dim
        );
        let c = hc_mult * hidden;
        let state_len = (k_size - 1) * dilation;
        let w = bounded_scratch(scratch_tokens, max_tokens);

        // Fail closed when element width and gather disagree — see
        // `gather_guard`, and the bug it exists for.
        #[cfg(feature = "cuda")]
        if let NgramTable::Cached(cache) = &table {
            let stride = cache.row_stride();
            let scaled = cache.scale_dev_va()?.is_some();
            if let Some(t) = trellis.as_ref() {
                ngram_trellis::check_packed_stride(stride, t, scaled)?;
            } else {
                gather_matches_element_size(stride, head_dim, scaled)?;
            }
        }
        Ok(Self {
            dims,
            head_dim,
            hidden,
            hc_mult,
            state_len,
            k_size,
            dilation,
            eps,
            key_proj: weights.key_proj,
            value_proj: weights.value_proj,
            norm_key: weights.norm_key,
            norm_query: weights.norm_query,
            norm_conv: weights.norm_conv,
            conv1d: weights.conv1d,
            // The scale arena is allocated once at load and its VA never
            // moves, so resolve it here rather than reaching back through the
            // table mutex on every gather.
            scale_va: match &table {
                #[cfg(feature = "cuda")]
                NgramTable::Cached(c) => c.scale_dev_va()?,
                _ => None,
            },
            trellis,
            embed_fp8_k: gpu.kernel("embed_from_argmax", "batched_embed_fp8")?,
            table: std::sync::Arc::new(std::sync::Mutex::new(table)),
            embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            gemm_k: projection_gemm(gpu)?,
            gate_k: gpu.kernel("ple", "ple_gate")?,
            conv_k: gpu.kernel("ple", "ple_conv")?,
            add_k: gpu.kernel("ple", "ple_add_highway")?,
            emb: gpu.alloc(w * hidden * 2)?,
            key: gpu.alloc(w * c * 2)?,
            value: gpu.alloc(w * hidden * 2)?,
            gated: gpu.alloc(w * c * 4)?,
            gated_normed: gpu.alloc(w * c * 4)?,
            out: gpu.alloc(w * c * 4)?,
            slots_dev: gpu.alloc(w * heads * 4)?,
            max_tokens,
            scratch_tokens: w,
            gather_done: std::sync::Mutex::new(0),
            slots_staging: std::sync::Mutex::new((0, 0)),
        })
    }

    /// Allocate one sequence's PLE carry (conv buffer + empty history).
    /// `reset` runs on first use (`fresh`), so contents start undefined.
    pub fn new_seq_state(&self, gpu: &dyn GpuBackend) -> Result<PleSeqState> {
        Ok(PleSeqState {
            conv: gpu.alloc(self.state_len * self.hc_mult * self.hidden * 4)?,
            verify_snaps: gpu.alloc(VERIFY_SNAP_SLOTS * self.conv_bytes())?,
            verify_snap_rows: 0,
            history_ckpt: Vec::new(),
            verify_tokens: Vec::new(),
            history: Vec::new(),
            prestaged_va: None,
            prestaged_n: 0,
            last_staged_va: 0,
            warm: None,
        })
    }

    // `reset` + `prestage` live in `aux_state.rs` (≤500 LoC split).

    // Marconi aux-state (snapshot_aux / restore_aux) moved to
    // `aux_state.rs` (≤500 LoC split).

    // The public forward entry points (`forward`, `forward_rows`) live in
    // `forward.rs` and the multi-sequence `forward_seqs` with the shared
    // stages in `batch.rs` (≤500 LoC split).

    #[allow(clippy::too_many_arguments)]
    fn forward_with_ids(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
        ids_override: Option<&[u32]>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            num_tokens <= self.max_tokens,
            "PLE: {num_tokens} tokens exceeds the {}-token forward width. \
             Raise ATLAS_PLE_MAX_TOKENS (the scratch stays bounded by \
             ATLAS_PLE_CHUNK — {} tokens here) or lower \
             --max-num-batched-tokens.",
            self.max_tokens,
            self.scratch_tokens
        );
        let c = self.hc_mult * self.hidden;
        let heads = self.dims.ngram_heads();
        let gpu = ctx.gpu;

        // The ids are a pure function of TOKEN IDS, computed on the host.
        // Prefer `ctx.host_token_ids` — the very slice the caller uploaded
        // into the device buffer — over reading the device copy back: the D2H
        // was a synchronous round trip per DECODE STEP for bytes the caller
        // had in hand, and inside a CUDA-graph capture region it is a
        // capture-unsupported op (STREAM_CAPTURE_INVALIDATED, 901).
        let tokens: Vec<u32> = if let Some(ov) = ids_override {
            anyhow::ensure!(ov.len() == num_tokens, "PLE: ids_override length");
            ov.to_vec()
        } else if let Some(host) = ctx.host_token_ids {
            anyhow::ensure!(
                host.len() >= num_tokens,
                "PLE: host_token_ids has {} ids for {num_tokens} tokens",
                host.len()
            );
            host[..num_tokens].to_vec()
        } else {
            // Fallback for passes that did not thread the host slice. Never
            // legal under capture — refuse rather than invalidate the graph.
            anyhow::ensure!(
                !ctx.graph_capture,
                "PLE: no host_token_ids and a D2H readback is \
                 capture-unsupported; thread the host ids through this pass"
            );
            let tok_dev = ctx.token_ids.ok_or_else(|| {
                anyhow::anyhow!(
                    "PLE needs token ids (host or device); this pass staged \
                     neither"
                )
            })?;
            let mut raw = vec![0u8; num_tokens * 4];
            gpu.copy_d2h(tok_dev, &mut raw)?;
            raw.chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        // A prestage staged for a REPLAYED decode step is consumed by the
        // graph, not by this forward (which never runs on replay) — so a
        // `Some` here on a prefill call is ordinary leftover from the
        // previous request's last replayed step, not an error. Only a
        // single-token, non-fresh decode may consume it.
        // The staging must have been built for exactly this many rows: see
        // `prestaged_n`. A verify step (K rows) and a decode step (1 row) both
        // qualify — what does not is consuming one for the other.
        let staged_n = st.prestaged_n;
        let prestaged = st
            .prestaged_va
            .take()
            .filter(|_| !fresh && staged_n == num_tokens);
        if fresh || st.history.len() != self.dims.context_len() {
            self.reset(st, gpu, stream)?;
        }

        // `flat` = this forward's row ids, `[token][head]` row-major. A
        // prestaged step hashed its own in `prestage`; its slots already sit
        // in `slots_dev`.
        let mut flat: Vec<u64> = Vec::new();
        if let Some(table_va) = prestaged {
            // The host half already ran from `decode_prestage`, before graph
            // replay/capture: slots sit in `slots_dev`, history has advanced.
            // Only the capture-safe kernel half remains.
            self.gather_embed(table_va, num_tokens, heads, gpu, stream)?;
        } else {
            anyhow::ensure!(
                !ctx.graph_capture,
                "PLE: un-prestaged forward inside CUDA graph capture — the \
                 pageable slot upload would invalidate the recording (901); \
                 the scheduler must call decode_prestage every step"
            );
            flat = self.stage_window(st, &tokens);
        }

        // The pipeline runs in `scratch_tokens` spans — the bounded
        // micro-batch discipline llama.cpp applies to its compute buffers —
        // so a `max_tokens`-wide forward needs only `scratch_tokens`-wide
        // scratch (~1.2 GB at the 8192-token default, not ~4.7 GB at a
        // 32K --max-prefill-tokens). Every stage is per-token except the
        // conv, whose carry lives in `st.conv` and threads across calls —
        // the per-row verify path already relies on that composability.
        // A span's NVMe fault-in also overlaps the previous span's kernels.
        let mut base = 0;
        while base < num_tokens {
            let n = (num_tokens - base).min(self.scratch_tokens);
            if prestaged.is_none() {
                self.gather(
                    &flat[base * heads..(base + n) * heads],
                    n,
                    heads,
                    gpu,
                    stream,
                )?;
                // Pace the prefill warm worker: these positions are now
                // consumed, so its lookahead window slides forward. No-op
                // without a session (decode, verify, resident table).
                if let Some(w) = st.warm.as_ref() {
                    w.note(n);
                }
            }

            self.project(n, gpu, stream)?;
            self.inject(
                st,
                highway.offset(base * c * 4),
                (0, Some(base)),
                n,
                num_tokens,
                gpu,
                stream,
            )?;
            base += n;
        }

        Ok(())
    }
}

/// The scratch width: `scratch` clamped into `[min(VERIFY_SNAP_SLOTS,
/// max_tokens), max_tokens]`. The floor is the verify contract — a
/// verify-width forward must never split across spans, because the per-row
/// conv snapshot path indexes the whole window. When `max_tokens` itself is
/// below VERIFY_SNAP_SLOTS the floor relaxes to it, so every legal forward
/// still fits in one span.
pub(crate) fn bounded_scratch(scratch: usize, max_tokens: usize) -> usize {
    scratch.clamp(VERIFY_SNAP_SLOTS.min(max_tokens), max_tokens)
}

/// The dense weights of one PLE site.
pub struct PleWeights {
    pub key_proj: DenseWeight,
    pub value_proj: DenseWeight,
    pub norm_key: DenseWeight,
    pub norm_query: DenseWeight,
    pub norm_conv: DenseWeight,
    pub conv1d: DenseWeight,
}

// Child module (not sibling): the aux fns read PleLayer private
// fields, and only a CHILD module sees them. Same #[path] trick
// qsa.rs uses for its tests.
#[path = "ckpt.rs"]
mod ckpt;
#[path = "verify.rs"]
mod verify;

#[path = "batch.rs"]
mod batch;
pub use batch::PleSeqRows;

#[path = "aux_state.rs"]
mod aux_state;

#[path = "forward.rs"]
mod forward;

// `pub(crate)`: `qwen3_ssm::trait_layer` calls `PleLayer::prefill_warm`
// through here, and `ple.rs` re-exports `warm_ahead_tokens` for the loader's
// cache sizing.
#[path = "warm.rs"]
pub(crate) mod warm;

#[path = "gather_guard.rs"]
mod gather_guard;
#[path = "ngram_trellis.rs"]
pub(crate) mod ngram_trellis;
#[cfg(feature = "cuda")]
use gather_guard::gather_matches_element_size;

/// `ATLAS_QWEN4EXP_PREFILL_GEMM_RASTER=1`: the key/value projections run
/// `dense_gemm_bf16_pipelined_g8`, the default GEMM with its output tiles
/// visited in groups of 8 M-tiles. The key projection's weight ([4 x hidden,
/// hidden], 52 MB) is wider than L2, and the default order re-streams it from
/// DRAM once per M-tile: 20.6 -> 7.6 ms at an 8192-row PLE pass on GB10
/// (`scripts/dev/qwen4exp_ple_gemm_bench.cu`). Every tile is the default's
/// tile body, so every output byte is the default's; rank-local.
fn projection_gemm(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON
        .get_or_init(|| std::env::var("ATLAS_QWEN4EXP_PREFILL_GEMM_RASTER").as_deref() == Ok("1"));
    let g8 = if on {
        crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined_g8")
    } else {
        KernelHandle(0)
    };
    if g8.0 != 0 {
        return Ok(g8);
    }
    gpu.kernel("gemm", "dense_gemm_bf16_pipelined")
}
