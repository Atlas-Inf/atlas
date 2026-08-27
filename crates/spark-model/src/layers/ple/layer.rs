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

/// Per-sequence carry: the dilated conv's 9 steps and the token history the
/// id hash needs.
///
/// **One sequence at a time.** v1 refuses the batched and multi-sequence
/// decode paths (`refuse_batched_under_hc`), so a single state here is
/// correct rather than merely convenient — and `reset` at every prefill start
/// is what keeps one request's lexical context out of the next one's.
struct PleState {
    /// `[(k-1)*dilation, channels]` FP32, device.
    conv: DevicePtr,
    /// The last `context_len` token ids, EOS-filled at a sequence start.
    history: Vec<u32>,
    /// Set by `prestage`: the n-gram table's device VA, recorded when the
    /// step's host work (hash + fault-in + slot upload) already ran BEFORE
    /// graph replay/capture. `forward` consumes it and enqueues kernels only.
    prestaged_va: Option<u64>,
    /// The last VA `prestage` staged, never cleared. `rearm` restores it when
    /// a failed capture attempt re-runs the step eagerly: the slots are still
    /// in `slots_dev` and history has already advanced, so re-hashing would
    /// double-count the token — re-arming is the only correct recovery.
    last_staged_va: u64,
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

    state: std::sync::Mutex<PleState>,
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
            state: std::sync::Mutex::new(PleState {
                conv: gpu.alloc(state_len * c * 4)?,
                history: Vec::new(),
                prestaged_va: None,
                last_staged_va: 0,
            }),
        })
    }

    /// Fresh sequence: EOS-filled history and a zeroed conv state.
    fn reset(&self, st: &mut PleState, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        st.history = vec![self.dims.eos_token_id; self.dims.context_len()];
        st.prestaged_va = None;
        let zeros = vec![0u8; self.state_len * self.hc_mult * self.hidden * 4];
        gpu.copy_h2d_async(&zeros, st.conv, stream)?;
        Ok(())
    }

    /// Hoisted per-step HOST work for decode under CUDA graphs: the n-gram
    /// hash, the NVMe fault-in and the slot upload into the stable
    /// `slots_dev` buffer. All three are capture-illegal (the upload reads
    /// pageable memory, which invalidates a recording graph with status
    /// 901), so the scheduler calls this BEFORE graph replay/capture — the
    /// same phasing decode_a already gives the `token_ids` upload. `forward`
    /// then consumes `prestaged_va` and enqueues only stable-buffer kernels.
    ///
    /// History advances HERE; the prestaged `forward` must not advance it
    /// again.
    pub fn prestage(&self, tokens: &[u32], gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let mut st = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE state mutex poisoned"))?;
        if st.history.len() != self.dims.context_len() {
            self.reset(&mut st, gpu, stream)?;
        }
        let mut window = st.history.clone();
        window.extend_from_slice(tokens);
        let all = ple_ngram_ids(&self.dims, &window);
        let rows = &all[all.len() - tokens.len()..];
        let flat: Vec<u64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let va = self.gather_host(&flat, gpu, stream)?;
        let keep = self.dims.context_len();
        st.history = window[window.len() - keep..].to_vec();
        st.prestaged_va = Some(va);
        st.last_staged_va = va;
        Ok(())
    }

    // Marconi aux-state (snapshot_aux / restore_aux) moved to
    // `aux_state.rs` (≤500 LoC split).

    /// Restore the prestaged state after a failed CUDA-graph capture attempt
    /// (the eager replay re-runs `forward`, which consumed `prestaged_va`).
    pub fn rearm(&self) {
        if let Ok(mut st) = self.state.lock()
            && st.last_staged_va != 0
        {
            st.prestaged_va = Some(st.last_staged_va);
        }
    }

    /// Inject into `highway` `[T, hc_mult*hidden]` FP32, in place.
    ///
    /// `fresh` starts a new sequence (prefill from position 0).
    pub fn forward(
        &self,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
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
        let tokens: Vec<u32> = if let Some(host) = ctx.host_token_ids {
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

        let mut st = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE state mutex poisoned"))?;
        // A prestage staged for a REPLAYED decode step is consumed by the
        // graph, not by this forward (which never runs on replay) — so a
        // `Some` here on a prefill call is ordinary leftover from the
        // previous request's last replayed step, not an error. Only a
        // single-token, non-fresh decode may consume it.
        let prestaged = st.prestaged_va.take().filter(|_| num_tokens == 1 && !fresh);
        if fresh || st.history.len() != self.dims.context_len() {
            self.reset(&mut st, gpu, stream)?;
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
                cache.end_batch();
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
        ops::batched_embed(
            gpu,
            self.embed_k,
            self.slots_dev,
            DevicePtr(table_va),
            self.emb,
            (num_tokens * heads) as u32,
            self.head_dim as u32,
            stream,
        )
        .context("PLE row gather")
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
