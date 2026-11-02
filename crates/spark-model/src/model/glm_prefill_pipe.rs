// SPDX-License-Identifier: AGPL-3.0-only
//! Two-half pipelined GLM prefill over the four-rank chain
//! (`ATLAS_GLM_PREFILL_PIPE2=1`, default off).
//!
//! A sequence-parallel chunk's seams wait on their collectives: on the chain
//! about 9 ms of reduce-scatter and gather a seam at 8K rows, a third of the
//! chunk, with nothing to run meanwhile. Here the chunk runs as its two
//! halves, each exactly the half-chunk a halved `--max-prefill-tokens` would
//! run (so the bytes are those two chunks'), in lockstep a layer apart: half
//! A runs layer `i` on the model's stream and arena, half B layer `i` on its
//! own stream and arena once A's layer `i` has finished its attention (B
//! reads A's KDA state and K/V; `glm_sp::pipe_fences` fences at the
//! layer's first reduce-scatter), so B's attention runs while A reduces,
//! runs its seams and MoE, and each half's collectives overlap the other's
//! compute; B's MoE waits on A's (fenced at A's second reduce-scatter). (Fenced at A's layer end instead, the halves ran in phase and
//! sat in their collectives at the same time.) The host issues them in that
//! order too (A's layer `i + 1` before B's layer `i`), so a layer that waits
//! on its own stream from the host (the MoE's exact tiles) never holds back
//! the other half's next layer.
//!
//! The halves never share a byte in flight: B has its own activation arena
//! (allocated with the model's, before the KV pool is sized), its own
//! one-shot channel (`spark_comm::chain_channel`, so each half's collectives
//! keep one order on every rank) and its own scratch lane
//! (`spark_runtime::lane`: cuBLASLt/CUTLASS workspaces, op-cache scratch).
//! A layer's own buffers serve one half at a time (B's layer `i` follows A's).
//! Its metadata is the chunk's, offset to its rows, with its own attention
//! sequence lengths. Every input to the choice is mirrored on every rank.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::glm_sp::{self, SpRows};
use crate::traits::SequenceState;

/// `ATLAS_GLM_PREFILL_PIPE2=1`. Read once.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PREFILL_PIPE2").as_deref() == Ok("1"))
}

/// `ATLAS_GLM_PREFILL_PIPE2_TRACE=1`: log every pass's choice and, for a
/// pipelined one, when each half finished (host clock, syncing at the end).
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PREFILL_PIPE2_TRACE").as_deref() == Ok("1"))
}

/// The second half-chunk's arena, stream and the fence between the halves.
pub(crate) struct PrefillPipe {
    buffers: BufferArena,
    stream: u64,
    /// Per layer: recorded on the model's stream once half A's layer has
    /// finished its attention, which half B's layer waits on.
    a_done: Vec<u64>,
    /// Per layer: half A's MoE done, which half B's MoE waits on.
    a_moe: Vec<u64>,
    /// Recorded on B's stream at its end; the model's stream waits on it.
    b_done: u64,
}

/// Half B's arena when the pipeline is requested for a GLM-5 TP4 chain
/// model: rows for half of `max_batch_tokens`, and the same attached
/// scratch as the model's arena. Allocate with the model's, before the KV
/// pool is sized.
pub(crate) fn arena(
    config: &ModelConfig,
    max_batch_tokens: usize,
    max_seq_len: usize,
    kv_block_size: usize,
    max_batch_size: usize,
    glm_latent_tokens: Option<usize>,
    gpu: &dyn GpuBackend,
) -> Result<Option<BufferArena>> {
    if !requested() || config.model_type != "glm5_next" || config.tp_world_size != 4 {
        return Ok(None);
    }
    let mut arena = BufferArena::new(
        config,
        max_batch_tokens.div_ceil(2),
        max_seq_len,
        kv_block_size,
        max_batch_size,
        gpu,
    )?;
    if let Some(tokens) = glm_latent_tokens {
        arena.attach_glm_latent_scratch(tokens, gpu)?;
    }
    tracing::info!(
        "GLM two-half prefill pipeline: half-chunk arena for {} rows",
        max_batch_tokens.div_ceil(2)
    );
    Ok(Some(arena))
}

/// One half's rows, metadata and rows split.
struct Half<'a> {
    ctx: ForwardContext<'a>,
    hidden: DevicePtr,
    residual: DevicePtr,
    start: usize,
    sp: SpRows,
    stream: u64,
    /// Lane and channel 1 for half B.
    second: bool,
}

/// `ctx` for one half: its arena, metadata and token ids (a plain pass has
/// no mid-chunk capture).
fn half_ctx<'a>(
    ctx: &ForwardContext<'a>,
    buffers: &'a BufferArena,
    meta: AttnMetadataDev,
    token_ids: Option<DevicePtr>,
) -> ForwardContext<'a> {
    ForwardContext {
        buffers,
        gpu: ctx.gpu,
        config: ctx.config,
        dispatch: ctx.dispatch,
        derived: ctx.derived,
        levers: ctx.levers,
        stats: ctx.stats,
        attn_metadata: Some(meta),
        ssm_batch: ctx.ssm_batch,
        profile: ctx.profile,
        comm: ctx.comm,
        graph_capture: ctx.graph_capture,
        gdn_exact_replay: ctx.gdn_exact_replay,
        token_ids,
        host_token_ids: ctx.host_token_ids,
        routed_lora_layers: ctx.routed_lora_layers,
        midchunk_capture: None,
        moe_lora_route: ctx.moe_lora_route,
    }
}

impl TransformerModel {
    pub(crate) fn set_prefill_pipe(&mut self, buffers: BufferArena) -> Result<()> {
        let gpu = self.gpu.as_ref();
        self.prefill_pipe = Some(PrefillPipe {
            buffers,
            stream: gpu.create_stream()?,
            a_done: (0..self.layers.len())
                .map(|_| gpu.create_event())
                .collect::<Result<_>>()?,
            a_moe: (0..self.layers.len())
                .map(|_| gpu.create_event())
                .collect::<Result<_>>()?,
            b_done: gpu.create_event()?,
        });
        Ok(())
    }

    /// The two halves' rows splits when this pass runs pipelined: the
    /// pipeline is set up and the comm has the second channel, the pass is a
    /// plain cold one (`plain`: no passengers, decode path, mid-chunk
    /// capture, prefix rows or write floor), and each half runs
    /// sequence-parallel (with DFlash's window gathered, `w > rows / 4`).
    pub(super) fn glm_pipe2_rows(
        &self,
        proc_count: usize,
        plain: bool,
        ctx: &ForwardContext,
    ) -> Option<(SpRows, SpRows)> {
        let pipe = self.prefill_pipe.as_ref()?;
        let comm = self.comm.as_ref()?;
        let half = proc_count / 2;
        let why = if !plain {
            "not a plain cold pass"
        } else if !proc_count.is_multiple_of(2) {
            "odd rows"
        } else if ctx.profile || crate::det_trace::on() {
            "profiling or tracing"
        } else if comm.collective_channels() < 2 {
            "one collective channel (ATLAS_RDMA_ONESHOT_CHANNELS=2)"
        } else if half > pipe.buffers.max_batch_tokens() {
            "half beyond the second arena"
        } else if !(self.config.dflash_capture_layers.is_empty()
            || self.config.dflash_ctx_window.is_some_and(|w| w > half / 4))
        {
            "DFlash window inside a half's rows"
        } else {
            match (
                self.glm_prefill_sp_rows(half, false, ctx),
                self.glm_prefill_sp_rows(half, false, ctx),
            ) {
                (Some(a), Some(b)) => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::info!(
                            "GLM two-half prefill pipeline engaged: {proc_count} rows as 2 x {half}"
                        )
                    });
                    if trace() {
                        tracing::info!("PIPE2 pass of {proc_count} rows: pipelined");
                    }
                    return Some((a, b));
                }
                _ => "a half is not sequence-parallel",
            }
        };
        if trace() {
            tracing::info!("PIPE2 pass of {proc_count} rows: whole ({why})");
        }
        None
    }

    /// Run the pass of `proc_count` rows at `start` (embedded in the model's
    /// arena, metadata uploaded) as two pipelined halves (module docs);
    /// `ctx` is the pass's context on the model's arena.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_prefill_pipe2(
        &self,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        (sp_a, sp_b): (SpRows, SpRows),
        start: usize,
        proc_count: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let pipe = self
            .prefill_pipe
            .as_ref()
            .expect("pipelined without a pipe");
        let gpu = self.gpu.as_ref();
        let (h, half) = (self.config.hidden_size, proc_count / 2);
        let row = h * 2;
        let meta = ctx
            .attn_metadata
            .expect("pipelined prefill needs its metadata");
        let _pipelined = glm_sp::enter_pipelined();
        let t0 = trace()
            .then(|| gpu.synchronize(stream).map(|_| std::time::Instant::now()))
            .transpose()?;

        // Half B's arena: zeroed as a chunk's, then its embedded rows.
        self.warm_zero_arena_of(&pipe.buffers, stream)?;
        let hidden_a = self.buffers.hidden_states();
        let hidden_b = pipe.buffers.hidden_states();
        gpu.copy_d2d_async(hidden_a.offset(half * row), hidden_b, half * row, stream)?;
        pipe.buffers.note_rows(half);
        // Each half's attention lengths (`upload_chunk_seq_lens`): A's over
        // the chunk's, B's in a buffer of its own.
        let lens_b = gpu.op_cache().scratch(
            gpu,
            "glm_pipe2_seq_len",
            4 * (1 + crate::layer::PREFILL_MAX_SUB_CHUNKS),
        )?;
        let paged = !meta.seq_len.is_null();
        if paged {
            self.upload_seq_lens_at(seq, start, half, meta.seq_len, stream)?;
            self.upload_seq_lens_at(seq, start + half, half, lens_b, stream)?;
        }
        // The metadata uploads come from host memory: complete before use.
        gpu.synchronize(stream)?;
        let meta_b = AttnMetadataDev {
            positions: meta.positions.offset(half * 4),
            positions_h: meta.positions_h.offset(half * 4),
            positions_w: meta.positions_w.offset(half * 4),
            slot: meta.slot.offset(half * 8),
            seq_len: if paged { lens_b } else { meta.seq_len },
            ..meta
        };
        let halves = [
            Half {
                ctx: half_ctx(ctx, &self.buffers, meta, ctx.token_ids),
                hidden: hidden_a,
                residual: self.buffers.residual(),
                start,
                sp: sp_a,
                stream,
                second: false,
            },
            Half {
                ctx: half_ctx(
                    ctx,
                    &pipe.buffers,
                    meta_b,
                    ctx.token_ids.map(|t| t.offset(half * 4)),
                ),
                hidden: hidden_b,
                residual: pipe.buffers.residual(),
                start: start + half,
                sp: sp_b,
                stream: pipe.stream,
                second: true,
            },
        ];
        let layers = self.layers.len();
        // A's layer i, then B's layer i - 1: (half, layer) in issue order.
        let order = (0..=layers).flat_map(|i| {
            [(0, i), (1, i.wrapping_sub(1))]
                .into_iter()
                .filter(move |&(_, l)| l < layers)
        });
        for (x, i) in order {
            let x = &halves[x];
            let _lane = x.second.then(|| spark_runtime::lane::enter(1));
            let _channel = x.second.then(|| spark_comm::chain_channel::enter(1));
            if x.second {
                gpu.stream_wait_event(x.stream, pipe.a_done[i])?;
            }
            let _sp = glm_sp::enter(x.sp);
            // A marks its attention and MoE done; B's MoE input gather
            // waits on A's MoE (the layer's MoE buffers serve one half).
            if x.second {
                glm_sp::pipe_fences([None; 2], [None, Some(pipe.a_moe[i])]);
            } else {
                glm_sp::pipe_fences([Some(pipe.a_done[i]), Some(pipe.a_moe[i])], [None; 2]);
            }
            self.layers[i]
                .prefill(
                    x.hidden,
                    x.residual,
                    half,
                    seq.layer_states[i].as_mut(),
                    kv_cache,
                    x.start,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    0,
                    &x.ctx,
                    x.stream,
                )
                .map_err(|e| anyhow::anyhow!("Pipelined prefill layer {i} failed: {e}"))?;
            let gathered = self.glm_sp_capture_gather(Some(x.sp), i, &x.ctx, x.stream)?;
            self.try_dflash_prefill_capture_layer(seq, i, x.start, half, x.stream)?;
            drop(gathered);
            // A layer without a reduce (or a MoE) marks its end instead.
            for event in glm_sp::take_pipe_fences().into_iter().flatten() {
                gpu.record_event(event, x.stream)?;
            }
        }
        // Each half's contracted rows, then B's after A's in the model's arena.
        for x in &halves {
            let _lane = x.second.then(|| spark_runtime::lane::enter(1));
            let _channel = x.second.then(|| spark_comm::chain_channel::enter(1));
            let _sp = glm_sp::enter(x.sp);
            x.sp.all_gather(x.hidden, h, &x.ctx, x.stream)?;
        }
        gpu.copy_d2d_async(
            hidden_b,
            hidden_a.offset(half * row),
            half * row,
            pipe.stream,
        )?;
        gpu.record_event(pipe.b_done, pipe.stream)?;
        gpu.stream_wait_event(stream, pipe.b_done)?;
        if let Some(t0) = t0 {
            let issued = t0.elapsed();
            gpu.event_synchronize(pipe.a_moe[layers - 1])?;
            let a = t0.elapsed();
            gpu.event_synchronize(pipe.b_done)?;
            tracing::info!(
                "PIPE2 {proc_count} rows: issued {:.1} ms, half A done {:.1} ms, half B done {:.1} ms",
                issued.as_secs_f64() * 1e3,
                a.as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        ensure!(
            halves[0].sp.total() == half && halves[1].sp.total() == half,
            "pipelined halves split {half} rows"
        );
        Ok(())
    }
}
