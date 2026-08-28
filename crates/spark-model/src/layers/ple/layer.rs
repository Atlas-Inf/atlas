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

/// Per-SEQUENCE carry: the dilated conv's 9 steps and the token history the
/// id hash needs. Owned by the sequence's [`crate::layer::SsmLayerState`]
/// (Avarok #753 item B: concurrency needs one of these per in-flight
/// sequence, not a layer singleton).
pub struct PleSeqState {
    /// `[(k-1)*dilation, channels]` FP32, device.
    conv: DevicePtr,
    /// The last `context_len` token ids, EOS-filled at a sequence start.
    history: Vec<u32>,
    /// Set by `prestage`: the n-gram table's device VA, recorded when the
    /// step's host work (hash + fault-in + slot upload) already ran BEFORE
    /// graph replay/capture. `forward` consumes it and enqueues kernels only.
    prestaged_va: Option<u64>,
    /// How many token rows `prestaged_va` was staged for. A verify step
    /// stages K of them; a decode step stages 1. `forward` consumes the
    /// staging only when this matches its own `num_tokens` — a K=1 staging
    /// consumed by a K=2 forward would gather one row and read the second
    /// from whatever followed it.
    prestaged_n: usize,
    /// Per-row conv snapshots for the speculative verify in flight, slot `t`
    /// holding the state after `t` rows (slot 0 = before the window). The
    /// conv carry advances once per token and is NOT part of the SSM
    /// checkpoint set, so a partially accepted verify would otherwise leave
    /// it conditioned on rejected drafts — the n-gram injection for every
    /// later token then reads a history that never happened.
    verify_snaps: DevicePtr,
    /// Rows with a valid snapshot. 0 = none (a prefill-width forward skips
    /// the per-row split), so a rollback then has nothing to restore.
    verify_snap_rows: usize,
    /// `history` as it stood before the window, plus the window's ids —
    /// together these rebuild the history for any accepted prefix.
    history_ckpt: Vec<u32>,
    verify_tokens: Vec<u32>,
    /// The last VA `prestage` staged, never cleared. `rearm` restores it when
    /// a failed capture attempt re-runs the step eagerly: the slots are still
    /// in `slots_dev` and history has already advanced, so re-hashing would
    /// double-count the token — re-arming is the only correct recovery.
    last_staged_va: u64,
}

/// Verify windows this layer can roll back: slot `t` = state after `t` rows,
/// so K rows need K+1 slots. K is `num_drafts + 1` and the batched MoE arms
/// stop at 8, which is the bound this matches.
const VERIFY_SNAP_SLOTS: usize = 9;

/// Bisection hatch: `ATLAS_PLE_VERIFY_SNAPSHOTS=0` restores the single batched
/// conv launch, which leaves nothing for `rollback_verify` to restore — the
/// PLE carry then keeps rejected drafts. Debug only.
fn verify_snapshots_enabled() -> bool {
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
    /// invoked through `&self`.
    table: std::sync::Mutex<NgramTable>,

    embed_k: KernelHandle,
    /// The dequant gather, used when `scale_va` is `Some`.
    embed_fp8_k: KernelHandle,
    /// Device VA of the table's per-slot f32 dequant scale array, `None` for a
    /// BF16 table. Its presence is what selects the FP8 gather.
    scale_va: Option<u64>,
    gemm_k: KernelHandle,
    gate_k: KernelHandle,
    conv_k: KernelHandle,
    add_k: KernelHandle,

    /// Scratch, sized once for `max_tokens`.
    emb: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gated: DevicePtr,
    gated_normed: DevicePtr,
    out: DevicePtr,
    slots_dev: DevicePtr,
    max_tokens: usize,
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

        // Fail closed when element width and gather disagree — see
        // `gather_guard`, and the bug it exists for.
        #[cfg(feature = "cuda")]
        if let NgramTable::Cached(cache) = &table {
            let stride = cache.row_stride();
            let scaled = cache.scale_dev_va()?.is_some();
            gather_matches_element_size(stride, head_dim, scaled)?;
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
            embed_fp8_k: gpu.kernel("embed_from_argmax", "batched_embed_fp8")?,
            table: std::sync::Mutex::new(table),
            embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            gate_k: gpu.kernel("ple", "ple_gate")?,
            conv_k: gpu.kernel("ple", "ple_conv")?,
            add_k: gpu.kernel("ple", "ple_add_highway")?,
            emb: gpu.alloc(max_tokens * hidden * 2)?,
            key: gpu.alloc(max_tokens * c * 2)?,
            value: gpu.alloc(max_tokens * hidden * 2)?,
            gated: gpu.alloc(max_tokens * c * 4)?,
            gated_normed: gpu.alloc(max_tokens * c * 4)?,
            out: gpu.alloc(max_tokens * c * 4)?,
            slots_dev: gpu.alloc(max_tokens * heads * 4)?,
            max_tokens,
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
        })
    }

    // `reset` + `prestage` live in `aux_state.rs` (≤500 LoC split).

    // Marconi aux-state (snapshot_aux / restore_aux) moved to
    // `aux_state.rs` (≤500 LoC split).

    /// Restore the prestaged state after a failed CUDA-graph capture attempt
    /// (the eager replay re-runs `forward`, which consumed `prestaged_va`).
    /// Bytes in one conv carry — `[(k-1)*dilation, channels]` FP32.
    fn conv_bytes(&self) -> usize {
        self.state_len * self.hc_mult * self.hidden * 4
    }

    /// Rewind this sequence's PLE carry to the `num_kept` verify tokens that
    /// were accepted (the bonus token included).
    ///
    /// Both halves of the carry move: the device conv window comes back from
    /// the per-row snapshot taken during the forward, and the host n-gram
    /// history is rebuilt from the pre-window copy plus the accepted ids.
    /// Rebuilt rather than truncated because `history` is a FIXED-WIDTH
    /// window — tokens fall off the front as it advances, so the dropped ones
    /// have to come back too.
    pub fn rollback_verify(
        &self,
        st: &mut PleSeqState,
        num_accepted: usize,
        k: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            num_accepted <= k,
            "PLE rollback: {num_accepted} accepted of a {k}-row verify"
        );
        if num_accepted == k {
            // Full accept: the live carry IS the committed one. Still finalize
            // the window so a later rollback cannot restore against it.
            st.verify_snap_rows = 0;
            st.verify_tokens.clear();
            return Ok(());
        }
        // Reaching here means rows were REJECTED, so the carry has advanced
        // past the committed prefix and must come back. If the forward took
        // the batched conv (no per-row carry) there is nothing to come back
        // to — failing loudly beats leaving the carry conditioned on drafts
        // that were thrown away, which is invisible until the n-gram
        // injections have drifted for hundreds of tokens.
        anyhow::ensure!(
            st.verify_snap_rows == k,
            "PLE rollback for a {k}-row verify but {} row(s) were snapshotted \
             — the carry advanced past the accepted prefix with nothing to \
             restore from. K must stay under VERIFY_SNAP_SLOTS ({}) and \
             ATLAS_PLE_VERIFY_SNAPSHOTS must not be 0 when MTP is on.",
            st.verify_snap_rows,
            VERIFY_SNAP_SLOTS
        );
        anyhow::ensure!(
            st.verify_tokens.len() == k,
            "PLE rollback: {} staged id(s) for a {k}-row verify — the history \
             rebuild would be short and silently wrong.",
            st.verify_tokens.len()
        );
        let num_kept = num_accepted;
        let cb = self.conv_bytes();
        gpu.copy_d2d_async(st.verify_snaps.offset(num_kept * cb), st.conv, cb, stream)?;

        let keep = self.dims.context_len();
        let mut window = st.history_ckpt.clone();
        window.extend_from_slice(&st.verify_tokens[..num_kept]);
        if window.len() >= keep {
            st.history = window[window.len() - keep..].to_vec();
        } else {
            // Shorter than one window only before the first reset; leave the
            // history invalid-length so `forward` re-seeds it via `reset`.
            st.history = window;
        }
        // A staging built for the rejected window must not be consumed by the
        // next step: it hashed ids that are no longer the sequence's history.
        st.prestaged_va = None;
        st.prestaged_n = 0;
        st.verify_snap_rows = 0;
        st.verify_tokens.clear();
        Ok(())
    }

    pub fn rearm(&self, st: &mut PleSeqState) {
        if st.last_staged_va != 0 {
            st.prestaged_va = Some(st.last_staged_va);
        }
    }

    /// Inject into `highway` `[T, hc_mult*hidden]` FP32, in place.
    ///
    /// `fresh` starts a new sequence (prefill from position 0).
    /// One highway ROW with an explicit id — the multi-seq decode entry
    /// (`ctx.host_token_ids` holds the whole batch; the caller slices).
    pub fn forward_row(
        &self,
        st: &mut PleSeqState,
        highway_row: DevicePtr,
        ids: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway_row, 1, false, Some(ids), ctx, stream)
    }

    /// Multi-token forward against an EXPLICIT id slice — the batched verify
    /// path, where the rows of one sequence are a sub-slice of the batch's
    /// host ids rather than its prefix. `fresh` is false: a verify step never
    /// starts a sequence.
    pub fn forward_rows(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        ids: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway, ids.len(), false, Some(ids), ctx, stream)
    }

    pub fn forward(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway, num_tokens, fresh, None, ctx, stream)
    }

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
            "PLE: {num_tokens} tokens exceeds the {} this layer was sized for. \
             Raise ATLAS_PLE_MAX_TOKENS (costs tokens*10240*14 bytes of \
             scratch) or lower the prefill chunk size.",
            self.max_tokens
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
            // history ++ tokens, hashed together, then keep the new tokens'
            // rows — the same slice the reference takes with
            // `[:, -input_ids.shape[1]:]`.
            // Same bookkeeping as `prestage` — see `rollback_verify`.
            st.history_ckpt = st.history.clone();
            st.verify_tokens = tokens.clone();
            let mut window = st.history.clone();
            window.extend_from_slice(&tokens);
            let all = ple_ngram_ids(&self.dims, &window);
            let rows = &all[all.len() - num_tokens..];
            let flat: Vec<u64> = rows.iter().flat_map(|r| r.iter().copied()).collect();

            self.gather(&flat, num_tokens, heads, gpu, stream)?;

            // Carry the last `context_len` tokens for the next step.
            let keep = self.dims.context_len();
            st.history = window[window.len() - keep..].to_vec();
        }

        // Projections off the concatenated n-gram embedding.
        //
        // `dense_gemm_bf16_pipelined`, NOT `dense_gemm`: the ops wrapper and
        // the kernel are a PAIR. `dense_gemm` launches grid
        // [ceil(n,16), ceil(m,16)] block 16x16 for the scalar kernel, while
        // the pipelined one wants [ceil(n,128), ceil(m,128)] block 256.
        // Handing the pipelined kernel to the scalar launcher reads far out of
        // bounds and produced NaN through the whole highway.
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.key_proj,
            self.key,
            num_tokens as u32,
            c as u32,
            self.hidden as u32,
            stream,
        )
        .context("PLE key_proj")?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.value_proj,
            self.value,
            num_tokens as u32,
            self.hidden as u32,
            self.hidden as u32,
            stream,
        )
        .context("PLE value_proj")?;

        ops::ple_gate(
            gpu,
            self.gate_k,
            highway,
            self.key,
            self.value,
            self.norm_query.weight,
            self.norm_key.weight,
            self.norm_conv.weight,
            self.gated,
            self.gated_normed,
            num_tokens as u32,
            self.hidden as u32,
            self.hc_mult as u32,
            self.eps,
            stream,
        )?;
        // The conv carry is the one piece of PLE state a speculative verify
        // has to be able to rewind, so at verify widths the launch is split
        // per row and each row's resulting carry is parked. At prefill widths
        // that would be thousands of launches for a carry nothing rolls back,
        // so the batched form stays and `verify_snap_rows` says "no snapshots".
        let cb = self.conv_bytes();
        if num_tokens < VERIFY_SNAP_SLOTS && verify_snapshots_enabled() {
            gpu.copy_d2d_async(st.conv, st.verify_snaps, cb, stream)?;
            for t in 0..num_tokens {
                let row = t * c * 4; // [T, c] FP32
                ops::ple_conv(
                    gpu,
                    self.conv_k,
                    self.gated_normed.offset(row),
                    self.gated.offset(row),
                    self.conv1d.weight,
                    st.conv,
                    self.out.offset(row),
                    1,
                    c as u32,
                    self.k_size as u32,
                    self.dilation as u32,
                    stream,
                )?;
                gpu.copy_d2d_async(st.conv, st.verify_snaps.offset((t + 1) * cb), cb, stream)?;
            }
            st.verify_snap_rows = num_tokens;
        } else {
            ops::ple_conv(
                gpu,
                self.conv_k,
                self.gated_normed,
                self.gated,
                self.conv1d.weight,
                st.conv,
                self.out,
                num_tokens as u32,
                c as u32,
                self.k_size as u32,
                self.dilation as u32,
                stream,
            )?;
            st.verify_snap_rows = 0;
        }
        ops::ple_add_highway(
            gpu,
            self.add_k,
            self.out,
            highway,
            (num_tokens * c) as u32,
            stream,
        )?;

        Ok(())
    }

    /// Resolve row ids to cache slots and gather them into `self.emb`.
    ///
    /// `T * ngram_heads` rows of `head_dim` land contiguously, which IS the
    /// `[T, ngram_heads * head_dim]` concatenation the projections expect —
    /// so `batched_embed` needs no PLE-specific variant.
    fn gather(
        &self,
        ids: &[u64],
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let table_va = self.gather_host(ids, gpu, stream)?;
        self.gather_embed(table_va, num_tokens, heads, gpu, stream)
    }

    /// The HOST half of `gather`: NVMe fault-in + slot upload into the
    /// stable `slots_dev` buffer. Capture-illegal (pageable H2D), so under
    /// CUDA graphs it runs from `prestage` BEFORE replay/capture. Returns
    /// the table's device VA for the kernel half.
    fn gather_host(&self, ids: &[u64], gpu: &dyn GpuBackend, stream: u64) -> Result<u64> {
        let mut table = self
            .table
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE table mutex poisoned"))?;

        // Release the PREVIOUS batch's pins. See `release_prev_pins`.
        Self::release_prev_pins(&mut table, gpu, stream)?;
        let table_va = match &mut *table {
            #[cfg(feature = "cuda")]
            NgramTable::Cached(cache) => {
                // Host resolves row -> slot (the ids are host-side anyway) and
                // faults missing rows off NVMe into the pinned, GPU-addressable
                // arena. The gather kernel then reads the arena BY SLOT.
                let mut slots = Vec::with_capacity(ids.len());
                let (h0, m0, _) = cache.stats();
                let t0 = std::time::Instant::now();
                cache.resolve(ids, &mut slots)?;
                // Prefill-scale gathers log the fault profile at info: the
                // misses are SERIAL blocking preads today (QD=1 under this
                // mutex), so miss-count x latency IS the prefill stall.
                // Decode-scale (16 ids) stays at debug.
                let (h1, m1, _) = cache.stats();
                let (dh, dm) = (h1 - h0, m1 - m0);
                let us = t0.elapsed().as_micros();
                if ids.len() > 64 {
                    tracing::info!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                } else {
                    tracing::debug!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                }
                let bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                let va = cache.table_dev_va()?;
                // NOTE: the pins are NOT released here. They are released at
                // the TOP of the next `gather_host`, after a stream sync — see
                // the comment there. Releasing them at this point is what
                // produced garbage input embeddings.
                DevicePtr(va)
            }
            NgramTable::Bf16(w) => {
                // Fully resident table (small fixtures / tests): the "slot" IS
                // the row id, so upload the ids truncated to u32.
                let bytes: Vec<u8> = ids.iter().flat_map(|r| (*r as u32).to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                w.weight
            }
            NgramTable::Fp8(_) => anyhow::bail!(
                "PLE: FP8 n-gram tables are not wired. This checkpoint ships BF16 \
                 rows, which are both simpler and more accurate (on LongCat, BF16 \
                 measured 0.0050 error vs FP8's 0.0247)."
            ),
        };
        Ok(table_va.0)
    }

    /// The KERNEL half of `gather`: reads `slots_dev` and the table arena —
    /// both stable device addresses — so it is graph-capture-safe.
    fn gather_embed(
        &self,
        table_va: u64,
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.gather_embed_dispatch(table_va, num_tokens, heads, gpu, stream)
    }
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
#[path = "aux_state.rs"]
mod aux_state;

#[path = "gather_guard.rs"]
mod gather_guard;
#[cfg(feature = "cuda")]
use gather_guard::gather_matches_element_size;
