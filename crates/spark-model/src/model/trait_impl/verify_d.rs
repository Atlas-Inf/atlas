// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//! H2D POD byte slices follow the safety contract documented in `verify_c.rs`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use atlas_core::config::LayerType;
use std::time::Instant;

use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

#[path = "verify_d_graph_policy.rs"]
mod graph_policy;
#[path = "verify_d_meta.rs"]
mod meta;
#[path = "verify_d_oracle.rs"]
mod oracle;
#[path = "verify_d_pieces.rs"]
mod pieces;

impl TransformerModel {
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
        // Strict structured-output row masks staged for this verify
        // (`glm_verify_masks`): eager only, and the split head must serve.
        allow: Option<spark_runtime::gpu::DevicePtr>,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let ban = seq.eos_ban;
        let ban_rows = ban.row_mask(seq.seq_len, k);
        if self.lightning_dspark_identity.policy().is_some()
            && std::env::var("ATLAS_LIGHTNING_VERIFY_SERIAL_M1").as_deref() == Ok("1")
        {
            anyhow::ensure!(allow.is_none(), "masked verify on the serial-M1 lane");
            return self.decode_verify_serial_m1_dispatch(tokens, seq, _stream);
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;

        // Canonical h_state is rewound in place by commit_accepted_prefix.

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed K tokens
        for t in 0..k {
            self.embed(tokens[t], hidden.offset(t * h * fp32), stream)?;
        }

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..k {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
            self.reserve_decode_blocks(seq, blocks_needed - 1, &mut kv_cache, stream)?;
        }

        // 1c. Upload K-entry attention metadata (see `kgamma_upload_meta`).
        let metadata = self.kgamma_upload_meta(seq, k, bs, stream)?;

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1 forces eager (no graph capture) so
        // CUDA_LAUNCH_BLOCKING=1 reports the exact failing kernel — used
        // to localize K=γ illegal-address crashes downstream of SSM.
        // Product Lightning serves froze this switch at admission: the
        // admitted policy rejects any presence of the variable, so the
        // product path never consults the environment here. Generic and
        // diagnostic serves keep the legacy read.
        let force_eager = if self.lightning_dspark_identity.policy().is_some() {
            false
        } else {
            crate::model::graph_flags::dflash_debug_no_graph()
        };
        // ATLAS_LORA_EAGER: LoRA graph-vs-eager debugging hatch (see decode_a).
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // Host-maintained per-layer state must not freeze during graph replay.
        let layer_veto = self.layers.iter().any(|l| l.decode_graph_unsupported());
        // Preserve K5 opt-in; repaired C1 K2 has a separate default-off gate.
        let glm_tp_graphs = (self.config.model_type == "glm5_next"
            && k == 5
            && self.config.tp_world_size == 2
            && crate::model::graph_flags::glm_tp_verify_graph())
            || graph_policy::mtp1_graph_allowed(
                crate::model::graph_flags::glm_mtp1_verify_graph(),
                &self.config.model_type,
                self.config.tp_world_size,
                self.config.ep_world_size,
                self.levers.max_decode_seqs,
                k,
                crate::speculative::glm_repair_policy::enabled(),
                crate::model::graph_flags::k2_diag(),
            );
        let use_graphs = (self.comm.is_none() || glm_tp_graphs)
            && allow.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && !super::verify_layer_trace::enabled()
            && !lora_eager
            && !layer_veto;
        // The optional M16 oracle must never perform D2H inside capture.
        crate::layers::moe::validate_m16_gate_up_graphs(&self.config.model_type, use_graphs)?;
        crate::layers::moe::validate_shared_fp8_cache_graphs(&self.config.model_type, use_graphs)?;
        crate::layers::moe::validate_m5_projection_graphs(&self.config.model_type, k, use_graphs)?;
        let verify_profile = std::env::var("ATLAS_GLM_VERIFY_PROFILE").ok().as_deref() == Some("1")
            && self.config.model_type == "glm5_next"
            && !use_graphs;

        // PLE's host half (n-gram hash + NVMe fault-in + slot upload) for the
        // WHOLE draft window, hoisted before capture/replay exactly as
        // `decode_a` hoists the single decode token. Without it the verify
        // forward has no staging to consume and falls back to a D2H readback,
        // which invalidates a recording graph (901) — the "PLE: no
        // host_token_ids ... capture-unsupported" refusal. #753 item B.
        for (li, l) in self.layers.iter().enumerate() {
            l.verify_prestage(
                tokens,
                seq.layer_states[li].as_mut(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: verify_profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: Some(tokens),
            routed_lora_layers: None, // #30: decode/verify never routes prefill.
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(), // route-aware: base(Skip) decodes; adapter refuses
        };

        // GLM DFlash lane: a verify block is an ordinary causal prefill chunk
        // of k rows continuing at seq_len, so full-attention (MLA) layers take
        // the multi-row prefill path (batched projections, row-tiled index
        // selection and sparse attention) instead of one decode chain per row.
        // Rejected rows' K/V and per-token index entries are rewritten by the
        // next block, which re-finalizes the pools it touches. KDA layers keep
        // the verify path that snapshots per-row recurrent state.
        let glm_prefill_ctx = (self.config.model_type == "glm5_next"
            && crate::speculative::glm_repair_policy::dflash_enabled()
            && crate::speculative::glm_repair_policy::dflash_prefill_verify()
            && !hss_engaged
            && !use_graphs
            && k >= 2)
            .then(|| ForwardContext {
                attn_metadata: Some(AttnMetadataDev {
                    positions: metadata.positions,
                    positions_h: metadata.positions,
                    positions_w: metadata.positions,
                    slot: metadata.slot,
                    // Chunk-total length: the last row's causal extent.
                    seq_len: metadata.seq_len.offset((k - 1) * 4),
                    block_table: metadata.block_table,
                    max_blocks_per_seq: metadata.max_blocks_per_seq,
                    num_seqs: 1,
                    seq_slot: metadata.seq_slot,
                    moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
                }),
                midchunk_capture: None,
                ..ctx
            });

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        let cache_key = (seq.slot_idx, k);
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            let mut attn_us = 0u128;
            let mut kda_us = 0u128;
            let mut attn_layers = 0usize;
            let mut kda_layers = 0usize;
            // Product Lightning serves carry force_eager=false from the
            // frozen admission, so this timing hatch only arms on
            // diagnostic/generic serves (it requires eager anyway).
            let time_layers = force_eager
                && std::env::var("ATLAS_DFLASH_LAYER_TIMING").ok().as_deref() == Some("1");
            let mut t_attn = 0u128;
            let mut t_moe = 0u128;
            let mut t_lin = 0u128;
            // ATLAS_GLM_VERIFY_GRAPH: KDA runs replay piecewise graphs; the
            // sparse-MLA layers and the head stay eager (host seq_len/ban).
            let pieces = pieces::admitted(
                crate::model::verify_pieces::requested(),
                &self.config.model_type,
                self.config.tp_world_size,
                use_graphs
                    || self.comm.is_none()
                    || self.lora.is_some()
                    || self
                        .suppress_graphs
                        .load(std::sync::atomic::Ordering::Relaxed)
                    || hss_engaged
                    || force_eager
                    || layer_veto
                    || verify_profile
                    || super::verify_layer_trace::enabled()
                    || pieces::diagnostics_sync(),
            );

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if pieces && self.kgamma_kda_run(layer_idx, k, seq, &mut kv_cache, &ctx, stream)? {
                    continue;
                }
                let layer_type = self.config.layer_type(layer_idx);
                let layer_started = if verify_profile {
                    self.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if time_layers {
                    self.gpu.synchronize(stream)?;
                }
                let t0 = Instant::now();

                if layer_type == LayerType::FullAttention
                    && let Some(ref prefill_ctx) = glm_prefill_ctx
                {
                    layer.prefill(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        0,
                        prefill_ctx,
                        stream,
                    )?;
                } else if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS path: decode_multi_seq's paged-decode kernel
                        // reads K/V from HBM only, missing the long-context
                        // history on disk. Fall back to decode_batched
                        // (sequential single-token decodes via the HSS
                        // orchestrator). See verify_b.rs for full rationale.
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        // k ROWS of ONE sequence, not k sequences: per-sequence
                        // aux state (the QSA indexer) must advance once per row
                        // against this sequence's own state. See
                        // `decode_multi_seq_rows`.
                        let mut seq_state_arr: [&mut (dyn LayerState + 'static); 1] =
                            [seq.layer_states[layer_idx].as_mut()];
                        let row_owner = vec![0usize; k];
                        layer.decode_multi_seq_rows(
                            hidden,
                            residual,
                            k,
                            &mut seq_state_arr,
                            &row_owner,
                            &mut kv_cache,
                            &seq_lens_vec,
                            &block_tables_vec,
                            &ctx,
                            stream,
                        )?;
                    }
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
                self.trace_lightning_hidden_rows("k4", seq.seq_len, layer_idx, hidden, k, stream)?;
                self.kgamma_dflash_capture(layer_idx, k, stream)?;
                if let Some(started) = layer_started {
                    self.gpu.synchronize(stream)?;
                    let elapsed = started.elapsed().as_micros();
                    if layer_type == LayerType::FullAttention {
                        attn_us += elapsed;
                        attn_layers += 1;
                    } else {
                        kda_us += elapsed;
                        kda_layers += 1;
                    }
                }
                if time_layers {
                    self.gpu.synchronize(stream)?;
                    let dt = t0.elapsed().as_micros();
                    match layer_type {
                        LayerType::FullAttention | LayerType::SlidingAttention => t_attn += dt,
                        LayerType::Moe => t_moe += dt,
                        LayerType::LinearAttention => t_lin += dt,
                    }
                }
            }

            if verify_profile {
                tracing::info!(
                    "GLM Kgamma layer profile K={k}: kda={:.2}ms({}L) mla={:.2}ms({}L)",
                    kda_us as f64 / 1000.0,
                    kda_layers,
                    attn_us as f64 / 1000.0,
                    attn_layers,
                );
            }
            if time_layers {
                tracing::info!(
                    "DFLASH LAYER_TIMING K={k}: attn={:.1}ms moe={:.1}ms mamba={:.1}ms",
                    t_attn as f64 / 1000.0,
                    t_moe as f64 / 1000.0,
                    t_lin as f64 / 1000.0
                );
            }

            // Final norm [K, H]
            let normed = self.buffers.norm_output();
            self.final_norm_rows(hidden, normed, k as u32, stream)?;

            // LM head + argmax for K tokens, inside the graph (fixed scratch
            // addresses — graph-safe).
            let argmax_out = self.buffers.scratch();
            if !self.glm_split_head_argmax(
                normed,
                k,
                argmax_out,
                (ban_rows, &ban),
                allow,
                stream,
            )? {
                anyhow::ensure!(
                    allow.is_none(),
                    "masked verify needs the GLM vocab-split head, which declined"
                );
                self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;
                let vocab = self.config.vocab_size;
                for t in 0..k {
                    let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                    let out_t = argmax_out.offset(t * 4);
                    ops::argmax_bf16(
                        self.gpu.as_ref(),
                        self.argmax_kernel,
                        logits_t,
                        out_t,
                        vocab as u32,
                        stream,
                    )?;
                }
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // The diagnostic compares against an unmasked head; masked rows differ.
        if allow.is_none() {
            self.check_glm_k5_bf16_head(k, &out, stream)?;
        }

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }
}
