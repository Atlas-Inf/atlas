// SPDX-License-Identifier: AGPL-3.0-only

//! Piecewise CUDA-graph capture of the qwen4_exp (Qwen3.8-Flash-Next)
//! decode and MTP verify steps (`ATLAS_QWEN4EXP_DECODE_GRAPH=1`, default
//! off). The GLM verify mechanism (`verify_pieces.rs`,
//! `docs/glm-verify-graphs.md`) applied to this model's layer stack.
//!
//! A whole-step graph is vetoed for this model: the QSA indexer on every
//! full-attention layer keeps host state (its ingest counter) and launches
//! with position-dependent geometry (`select_geometry`'s `n_sel`), and the
//! PLE layer hashes n-grams and faults rows in on the host and gathers into
//! a per-SEQUENCE conv carry. The device radix top-k (now the default) took
//! the host sort out of QSA, but not the counter or the geometry. The other
//! 35 GDN layers, each an mHC-bracketed gated-delta mixer plus an mHC-
//! bracketed 512-expert MoE, launch nothing that depends on the step. So
//! each maximal run of GDN layers that does not veto graphs
//! (`TransformerLayer::decode_graph_unsupported`, true for the PLE layer)
//! goes through [`super::verify_pieces::Pieces`]: eager warm-up on the first
//! visit, capture on the second, replay after. Embed, the metadata upload,
//! the PLE layer, the 12 attention layers and the head stay eager. Runs on
//! the 48-layer `[GDN, GDN, GDN, attention] x 12` stack with PLE at layer 1:
//! `[0]`, `[2]`, then `[4i..4i+3)` for i in 1..12 -- 13 runs, 35 layers.
//!
//! Steps (`decode_pieces_steps.rs` and `decode_a2`): the single-row decode
//! (`decode_a3`), the K=2/3/4 verify of one sequence (`verify_b`, `verify_c`,
//! `verify_c2` -- the exact-verify / BATCH_FAST rows included), the batched
//! decode of 2..N sequences (`decode_a2`) and the batched multi-sequence
//! verify (`verify_e`).
//!
//! Keys: `(step, wide, SSM pool slot per row, rows, first layer)`. A run
//! bakes the fixed arenas (hidden, highway streams, scratch, MoE buffers,
//! the decode metadata at its fixed address) and the SSM pool addresses of
//! its rows' slots (h/conv state, verify intermediates, deferred-commit
//! staging), all a function of the slot; the step (decode / batched decode /
//! verify / batched verify) and the row count pick the kernels (a batched
//! decode's padding rows carry the dummy slot, a batched verify's vector
//! each sequence's row count too). Nothing per-sequence (the PLE carry, QSA
//! keys) is in a run, so runs survive `free_sequence` like the GLM ones.
//!
//! Wide runs (`ATLAS_QWEN4EXP_DECODE_GRAPH_WIDE=1`, needs the switch above;
//! default off). Inside the QSA inert bound (every row of the step at a
//! position < 2051) an indexer selects nothing: its only step-dependent work
//! is the ingest of the row's raw key, which `layers/qsa_staged.rs` splits
//! into a graph half (the same qk projection, parked in layer-owned staging)
//! and a host half the model runs after the layer loop
//! ([`TransformerModel::qsa_commit_staged`]). Everything else an attention
//! layer launches reads the step's metadata from its fixed address. On such
//! a step the 12 attention layers join the runs, so the stack is two runs,
//! `[0]` and `[2..48)`, around the eager PLE layer. A step with any row past
//! the bound uses the GDN-only runs above; `wide` is part of the key.
//!
//! Collectives. Default: every collective splits the capture and replays
//! eagerly between graphs (`Recorder`), so both TP ranks issue the same
//! collective calls in the same order whatever either rank captured.
//! `ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES=1` instead records the RDMA
//! one-shot all-reduce inside the graph when the communicator offers it
//! (`ATLAS_RDMA_ONESHOT=1`). That stays rank-symmetric: (1) the eager
//! `all_reduce_async` already takes the one-shot first under the same
//! size-only eligibility, so a captured reduce is the very op the eager
//! pass would have enqueued, with the same arguments; (2) the one-shot
//! protocol advances its sequence on the device and the proxy counts
//! staged ops as they EXECUTE, so a graph replay, an eager call and a mix
//! across ranks are indistinguishable to the peer; (3) a capture pass
//! executes nothing, and a refused or failed capture re-runs eagerly, so
//! each rank executes each step's ops exactly once and in program order;
//! (4) a size the one-shot refuses splits as in the default. A rank-local
//! difference (diagnostics, budget) costs speed, never pairing.
//!
//! Lossless: `ctx.graph_capture` stays false inside a run, so every layer
//! takes the eager kernels and collective calls; the stream order is the
//! eager order.

use anyhow::{Result, ensure};
use atlas_core::config::LayerType;
use spark_runtime::kv_cache::PagedKvCache;

use super::types::TransformerModel;
use super::verify_pieces::Pieces;
use crate::layer::{ForwardContext, LayerState};

#[path = "decode_pieces_steps.rs"]
mod steps;

/// `ATLAS_QWEN4EXP_DECODE_GRAPH=1`; read once, from the profile both ranks
/// share (`startup_parity`).
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_DECODE_GRAPH").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES=1`: record capturable one-shot
/// all-reduces inside the graphs (module docs). Only read under
/// [`requested`].
pub(crate) fn collectives_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES").as_deref() == Ok("1")
    })
}

/// `ATLAS_QWEN4EXP_DECODE_GRAPH_WIDE=1`: runs span the QSA attention layers
/// on an all-inert step (module docs). Only read under [`requested`].
pub(crate) fn wide_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_DECODE_GRAPH_WIDE").as_deref() == Ok("1"))
}

/// Which step a run belongs to: a GDN layer launches different kernels for
/// a single decode row (`decode`), a batch of sequences
/// (`decode_multi_seq`), K rows of one sequence (`decode_batched`) and
/// ragged rows of several (`decode_verify_multi`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PieceStep {
    Decode,
    Batch,
    Verify,
    VerifyBatch,
}

/// `(step, wide, SSM pool slot per row owner, rows, first layer of the run)`.
/// The batched verify's slot vector is `verify_batched_graph_key`'s (each
/// sequence's slot and row count, and the WY-table sentinel).
pub(crate) type DecodePieceKey = (PieceStep, bool, Vec<u32>, usize, usize);

/// The model's cache. Budget: `ATLAS_QWEN4EXP_DECODE_GRAPH_MAX_GRAPHS`,
/// default 600 (~100-150 MiB on GB10, see `VerifyPieces`). Split at
/// collectives, a TP2 key is 13 runs + one graph per reduce: about 83 for decode
/// and the K=2/3 verify or C2/C3 (35 GDN layers x 2 reduces), 188 for C4
/// (the per-row MoE loop reduces 4 times a layer). 13 with the collectives
/// captured, and at TP1. A warm key over the budget stays eager.
pub(crate) fn new_cache() -> Pieces<DecodePieceKey> {
    let budget = std::env::var("ATLAS_QWEN4EXP_DECODE_GRAPH_MAX_GRAPHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    Pieces::new(
        "decode",
        "(step, wide, ssm slots, rows, layer)",
        budget,
        requested() && collectives_requested(),
    )
}

/// Admission. `eager_only` folds every reason the step must stay eager or
/// already runs as a whole graph.
pub(crate) fn admitted(requested: bool, model_type: &str, eager_only: bool) -> bool {
    requested && model_type == "qwen4_exp" && !eager_only
}

/// Diagnostics that sync or read back inside a GDN run. Under a capture
/// they would refuse every key, so they keep the step eager. Rank-local is
/// safe: a rank that stays eager issues the same collective calls.
fn diagnostics_sync(k4_diag: bool) -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        [
            "ATLAS_QWEN4EXP_DECODE_PROF",
            "ATLAS_QWEN4EXP_VERIFY_PROF",
            "ATLAS_SSM_DETAIL_PROFILE",
            "ATLAS_SSM_MS_PROFILE",
            "ATLAS_CONC_HSD",
            "ATLAS_DFLASH_CAPTURE_TRACE",
        ]
        .iter()
        .any(|k| std::env::var(k).is_ok_and(|v| v == "1" || v == "true"))
    });
    env || k4_diag
        || super::graph_flags::k2_diag()
        || super::graph_flags::ms_profile()
        || super::graph_flags::ssm_save_dump()
        || super::graph_flags::verify_layer_trace()
        // Arms the MoE and SSM readbacks keyed on `ctx.graph_capture`,
        // which pieces leave false.
        || tracing::level_filters::LevelFilter::current() >= tracing::Level::DEBUG
}

/// Where layer `i` sits among the maximal runs of `capturable` layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunPos {
    /// Not capturable: the caller runs it eagerly.
    Eager,
    /// First layer of the run `i..end`.
    Start(usize),
    /// Inside a run its first layer already handled.
    Inside,
}

pub(crate) fn run_pos(capturable: impl Fn(usize) -> bool, i: usize, layers: usize) -> RunPos {
    if !capturable(i) {
        return RunPos::Eager;
    }
    if i > 0 && capturable(i - 1) {
        return RunPos::Inside;
    }
    RunPos::Start((i..layers).find(|&j| !capturable(j)).unwrap_or(layers))
}

/// [`TransformerModel::decode_pieces_wide`] as a pure function of the switch,
/// each staging layer's inert bound (`None`: unbounded is not an indexer,
/// refuse) and the step's last row position.
fn wide_admitted(
    requested: bool,
    mut bounds: impl Iterator<Item = Option<usize>>,
    max_pos: usize,
) -> bool {
    let mut any = false;
    requested
        && bounds.all(|b| {
            any = true;
            b.is_some_and(|bound| max_pos < bound)
        })
        && any
}

impl TransformerModel {
    /// Whether this step takes the pieces. `use_graphs`: the step already
    /// runs as one graph. Every input is shared configuration except the
    /// diagnostics, which only cost speed when they differ (module docs).
    pub(super) fn decode_pieces_admitted(
        &self,
        use_graphs: bool,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
    ) -> bool {
        // Switch and model first: every other model pays nothing per step.
        admitted(requested(), &self.config.model_type, false)
            && !(use_graphs
                || self.profile
                || ctx.profile
                || ctx.ssm_batch.is_some()
                || ctx.midchunk_capture.is_some()
                // NVMe-backed KV does host I/O per step.
                || kv_cache.config().cache_blocks_per_seq.is_some()
                // As for the GLM pieces: adapters stay eager.
                || self.lora.is_some()
                || diagnostics_sync(self.levers.k4_diag))
    }

    /// Whether layer `i` may sit inside a captured run: a GDN layer without
    /// a veto, and on a `wide` step an indexer layer that stages its ingest.
    fn piece_capturable(&self, i: usize, wide: bool) -> bool {
        match self.config.layer_type(i) {
            LayerType::LinearAttention => !self.layers[i].decode_graph_unsupported(),
            LayerType::FullAttention => wide && self.layers[i].qsa_inert_capturable(),
            _ => false,
        }
    }

    /// Whether this step's runs are wide (module docs): the switch, an
    /// indexer layer to stage, and every row inside every such indexer's
    /// inert bound. `max_pos` is the step's last 0-based row position. Both
    /// ranks see the same positions and decide alike (and a rank-local
    /// difference would still pair: each rank issues the same collectives in
    /// the same order whichever runs it took).
    pub(super) fn decode_pieces_wide(&self, max_pos: usize) -> bool {
        wide_admitted(
            wide_requested(),
            self.layers
                .iter()
                .filter(|l| l.qsa_inert_capturable())
                .map(|l| l.verify_context_limit_multi_seq()),
            max_pos,
        )
    }

    /// Run the maximal run of capturable layers starting at `layer_idx`
    /// through the piecewise cache, calling `layer(li, ctx)` for each of its
    /// layers with the context the run must use. `Ok(false)`: `layer_idx` is
    /// not capturable, the caller runs it. `Ok(true)`: handled now, or with
    /// its run's first layer. A `wide` run stages its indexer ingest, so the
    /// caller owes [`Self::qsa_commit_staged`] after its layer loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn piece_run(
        &self,
        layer_idx: usize,
        step: PieceStep,
        wide: bool,
        slots: &[u32],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
        mut layer: impl FnMut(usize, &ForwardContext) -> Result<()>,
    ) -> Result<bool> {
        let end = match run_pos(
            |i| self.piece_capturable(i, wide),
            layer_idx,
            self.layers.len(),
        ) {
            RunPos::Eager => return Ok(false),
            RunPos::Inside => return Ok(true),
            RunPos::Start(end) => end,
        };
        ensure!(
            ctx.midchunk_capture.is_none(),
            "piecewise decode graph: decode has no mid-chunk capture"
        );
        let key = (step, wide, slots.to_vec(), rows, layer_idx);
        self.decode_pieces
            .run_with(key, self.gpu.as_ref(), ctx.comm, stream, |comm| {
                // The decode contexts carry no mid-chunk capture (the one
                // field that is not `Copy`).
                let ctx = ForwardContext {
                    comm,
                    midchunk_capture: None,
                    ..*ctx
                };
                let _staged = crate::layers::qsa::StagedIngest::enter(wide);
                (layer_idx..end).try_for_each(|li| layer(li, &ctx))
            })?;
        Ok(true)
    }

    /// The host half of a wide step's staged ingest (`layers/qsa_staged.rs`):
    /// for every indexer layer the wide runs hold, row `r` of the step
    /// (`rows[r] = (owner, pos)`; `owners[owner]` is that sequence's layer
    /// states) commits its staged raw key at `pos`, in row order. Call once,
    /// after the layer loop of a step whose runs were wide.
    pub(super) fn qsa_commit_staged(
        &self,
        rows: &[(usize, usize)],
        owners: &mut [&mut Vec<Box<dyn LayerState>>],
        stream: u64,
    ) -> Result<()> {
        for li in 0..self.layers.len() {
            if self.config.layer_type(li) != LayerType::FullAttention
                || !self.piece_capturable(li, true)
            {
                continue;
            }
            for (row, &(owner, pos)) in rows.iter().enumerate() {
                let states = owners
                    .get_mut(owner)
                    .ok_or_else(|| anyhow::anyhow!("staged QSA commit: no owner {owner}"))?;
                self.layers[li].qsa_commit_staged(
                    states[li].as_mut(),
                    row,
                    pos,
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_qwen4exp_only() {
        assert!(admitted(true, "qwen4_exp", false));
        assert!(!admitted(false, "qwen4_exp", false));
        assert!(!admitted(true, "qwen4_exp", true));
        for model in ["glm5_next", "qwen3_next", ""] {
            assert!(!admitted(true, model, false));
        }
    }

    /// The Flash-Next stack: `[GDN, GDN, GDN, attention] x 12`, PLE (a
    /// graph veto) on layer 1.
    fn flash_next(i: usize) -> bool {
        i % 4 != 3 && i != 1
    }

    #[test]
    fn flash_next_runs() {
        let starts: Vec<(usize, usize)> = (0..48)
            .filter_map(|i| match run_pos(flash_next, i, 48) {
                RunPos::Start(end) => Some((i, end)),
                _ => None,
            })
            .collect();
        let mut want = vec![(0, 1), (2, 3)];
        want.extend((1..12).map(|b| (4 * b, 4 * b + 3)));
        assert_eq!(starts, want);
        let captured: usize = starts.iter().map(|(s, e)| e - s).sum();
        assert_eq!(captured, 35, "every GDN layer but the PLE one");
        for i in [1, 3, 47] {
            assert_eq!(run_pos(flash_next, i, 48), RunPos::Eager);
        }
        for i in [5, 6, 46] {
            assert_eq!(run_pos(flash_next, i, 48), RunPos::Inside);
        }
    }

    #[test]
    fn wide_runs_span_the_attention_layers() {
        // Wide: the attention layers join, only the PLE layer stays eager.
        let wide = |i: usize| i != 1;
        let starts: Vec<(usize, usize)> = (0..48)
            .filter_map(|i| match run_pos(wide, i, 48) {
                RunPos::Start(end) => Some((i, end)),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![(0, 1), (2, 48)]);
        assert_eq!(run_pos(wide, 3, 48), RunPos::Inside);
        assert_eq!(run_pos(wide, 47, 48), RunPos::Inside);
    }

    #[test]
    fn wide_only_inside_every_inert_bound() {
        let two = [Some(2051), Some(2051)];
        assert!(wide_admitted(true, two.into_iter(), 2050));
        assert!(!wide_admitted(true, two.into_iter(), 2051));
        assert!(!wide_admitted(false, [Some(2051)].into_iter(), 10));
        assert!(
            !wide_admitted(true, std::iter::empty(), 10),
            "no indexer layer to stage"
        );
        assert!(!wide_admitted(true, [Some(2051), None].into_iter(), 10));
    }

    #[test]
    fn a_run_reaching_the_last_layer_ends_there() {
        assert_eq!(run_pos(|i| i >= 2, 2, 5), RunPos::Start(5));
        assert_eq!(run_pos(|_| true, 0, 3), RunPos::Start(3));
        assert_eq!(run_pos(|_| false, 0, 3), RunPos::Eager);
    }
}
