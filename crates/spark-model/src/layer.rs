// SPDX-License-Identifier: AGPL-3.0-only

//! Composable transformer layer traits (SDD).
//!
//! Decouples the generic model loop (embed -> layers -> norm -> lm_head)
//! from layer-specific logic (attention vs SSM, MoE vs dense FFN).
//! Adding a new architecture only requires implementing [`TransformerLayer`]
//! for each layer type, not duplicating the model loop.

use std::any::Any;

use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub mod glm_long_owner;
pub mod glm_packed;
mod prefill_pieces;
pub mod ssm_batch;
pub use prefill_pieces::{
    PREFILL_ATTENTION_ROWS, PREFILL_MAX_SUB_CHUNKS, prefill_attention_pieces,
};
mod multi_seg;
pub use multi_seg::{MultiPass, MultiSeg};
mod prefill_meta;
pub use prefill_meta::{AttnMetadataDev, BatchedAttnMetadata, GdnPrefillBuffers};
mod transformer_layer;
pub use transformer_layer::{
    KvRows, TransformerLayer, VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS,
    VERIFY_WY_TABLE_STRIDE_BYTES, VERIFY_WY_TABLES_PER_LAYER,
};

/// Per-layer persistent state tracked across decode steps.
///
/// Attention layers use [`EmptyLayerState`] (KV lives in `PagedKvCache`).
/// SSM layers use [`SsmLayerState`] (recurrent h_state + conv_state).
/// Custom layers can implement this trait for arbitrary state.
pub trait LayerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Empty state for layers that store all persistent state externally
/// (e.g., attention layers where KV is in `PagedKvCache`).
pub struct EmptyLayerState;

/// Attention-layer per-sequence state. KV lives in `PagedKvCache`; the only
/// resident piece is the QSA indexer carry on the 12 qwen4_exp
/// full-attention layers (Avarok #753 item B).
#[derive(Default)]
pub struct AttnLayerState {
    pub qsa: Option<crate::layers::qsa::QsaSeqState>,
}

impl LayerState for AttnLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl LayerState for EmptyLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// SSM layer state: recurrent hidden state + conv1d sliding window.
///
/// Used by Mamba, Gated Delta Net (GDN), and similar recurrent layers.
pub struct SsmLayerState {
    /// Recurrent hidden state: [num_v_heads, v_dim, k_dim] in f32.
    pub h_state: DevicePtr,
    /// Conv1d sliding window state: [d_inner, d_conv] in f32.
    pub conv_state: DevicePtr,
    /// Checkpoint buffer for h_state (allocated lazily for speculative decode).
    pub h_state_checkpoint: Option<DevicePtr>,
    /// Checkpoint buffer for conv_state (allocated lazily for speculative decode).
    pub conv_state_checkpoint: Option<DevicePtr>,
    /// Intermediate h_state snapshots during batched verification.
    /// Element i holds h_state after processing verification token i.
    /// Used by rollback_ssm_states to restore to the correct position.
    pub h_state_intermediates: Vec<DevicePtr>,
    /// Intermediate conv_state snapshots during batched verification.
    pub conv_state_intermediates: Vec<DevicePtr>,
    /// Deferred-commit input staging (ATLAS_GDN_DEFERRED_COMMIT): the
    /// verify-row conv-out q/k/v (`[K_cap][conv_dim]` BF16) and gate/beta
    /// (`[K_cap][2*nv]` FP32) rows the wyN kernel consumed, copied here so
    /// `gated_delta_rule_commit` can replay the accepted prefix after the
    /// shared ctx.buffers scratch is reused. POOL-STABLE per (layer, slot)
    /// — repointed at slot-bind alongside `h_state`; `DevicePtr(0)` when
    /// the lever is off.
    pub gdn_commit_qkv: DevicePtr,
    pub gdn_commit_gb: DevicePtr,
    /// Set at the VERIFY DISPATCH entry (host side — never inside a
    /// captured forward) when this layer's wyN verify ran deferred:
    /// `h_state` still holds H0 and the accept path must commit via
    /// `gated_delta_rule_commit` rather than the intermediates index-select.
    /// Cleared at commit / rollback.
    pub gdn_commit_pending: bool,
    /// `ATLAS_QWEN4EXP_GDN_COMMIT_FUSE`: this slot's pending-commit word,
    /// which the deferred verify reads (`qwen4exp_gdn_verify_defer_rows`);
    /// POOL-STABLE per slot like the staging. NULL when off.
    pub gdn_fuse_n: DevicePtr,
    /// GLM KDA fold records of a verify (`--ssm-rollback-mode records`): the
    /// verify leaves `h_state` untouched and the commit folds the accepted
    /// rows into it (NULL, with snapshots in `h_state_intermediates`, off).
    pub kda_records: DevicePtr,
    /// Storage dtype of `h_state`: `false` = FP32, `true` = FP16
    /// (`--ssm-h-dtype f16`).
    ///
    /// This is the single source of truth for the h-state format. Which edge
    /// sets it depends on the POOL width:
    ///
    /// * FP32-sized pool (stage 1/2, `h_prefill_stage == None`): prefill is
    ///   the only FP32 writer and writes the slot in place, so the flag
    ///   starts `false` and the decode mixer
    ///   (`TransformerModel::ssm_h_to_f16_dispatch`) flips it exactly once
    ///   per sequence, on the first decode step. No caller has to know where
    ///   the prefill->decode edge is.
    /// * f16-SIZED pool (stage 3, `h_prefill_stage == Some`): the slot is
    ///   physically 2 bytes/element and can NEVER hold FP32, so the flag is
    ///   `true` from allocation onwards and the decode mixer is a no-op.
    ///   Prefill's FP32 kernels run over [`Self::h_prefill_stage`] instead.
    ///
    /// It rides through swap-out/swap-in because `state_io` mutates these
    /// states in place rather than rebuilding them.
    pub h_is_f16: bool,
    /// Stage-3 f16-SIZED pool ONLY (`--ssm-h-dtype f16-pool`): the FP32
    /// staging blob for THIS sequence's slot, which the GDN prefill widens
    /// `h_state` into before its FP32 kernels run and narrows back after.
    ///
    /// `None` — every configuration before stage 3 — means "the slot IS
    /// FP32-wide": prefill writes `h_state` in place exactly as it always
    /// has, and not one byte moves. The same blob is shared by every layer
    /// of the sequence (see `SsmStatePool::h_prefill_stage`).
    pub h_prefill_stage: Option<DevicePtr>,
    /// PLE per-sequence carry (n-gram history + dilated-conv state), present
    /// only on the layer that hosts a `PleLayer` (Avarok #753 item B: one per
    /// in-flight sequence, lazily created on the sequence's first pass).
    pub ple: Option<crate::layers::ple::PleSeqState>,
}

impl SsmLayerState {
    /// Whether this sequence's `h_state_intermediates` back a wyN launch
    /// at `num_tokens = k`: needs ≥ k−1 entries at the pool-contiguous
    /// `h_bytes` stride the kernel assumes (`Hi_t` at
    /// `base + t·h_bytes`). Shared SSOT — the layer's dispatch filter and
    /// the host-side `mark_gdn_deferred_commit` must agree exactly, or the
    /// pending flag and the launched kernel diverge under graph replay.
    pub fn h_inter_pool_layout(&self, k: usize, h_bytes: usize) -> bool {
        if self.h_state_intermediates.len() < k.saturating_sub(1) || k < 2 {
            return false;
        }
        let h_base = self.h_state_intermediates[0];
        self.h_state_intermediates
            .iter()
            .take(k - 1)
            .enumerate()
            .all(|(t, p)| p.0 == h_base.0 + (t * h_bytes) as u64)
    }
}

impl LayerState for SsmLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Shared context for a single forward pass step.
///
/// Provides access to GPU, buffers, and config without coupling
/// layer implementations to the model struct.
pub struct ForwardContext<'a> {
    /// Pre-allocated scratch buffers.
    pub buffers: &'a BufferArena,
    /// GPU backend for kernel launches and memory ops.
    pub gpu: &'a dyn GpuBackend,
    /// Model configuration (dimensions, hyperparameters).
    pub config: &'a ModelConfig,
    /// Which GEMM implementation each projection takes. Carried rather than
    /// read from a static so it cannot outlive the model whose flags it
    /// encodes — see `layers::ops::GemmDispatch`.
    pub dispatch: &'a crate::layers::ops::GemmDispatch,
    /// Re-encoded copies of this model's weights, memoized for this model's
    /// lifetime. Carried rather than kept in a static keyed by device pointer,
    /// where a recycled address would HIT after a model swap.
    pub derived: &'a crate::layers::ops::DerivedWeights,
    /// Kernel-path levers for this model — the SSM/GDN variant, FFN routing,
    /// MoE quantization, LoRA mode, diagnostics. The non-GEMM half of the
    /// lever set; `dispatch` is the GEMM half.
    pub levers: &'a crate::layers::ops::ModelLevers,
    /// This model's diagnostic counters and one-shot dump latches. Carried
    /// for the same reason as `levers`: a counter that spans a model swap
    /// averages two models and describes neither, and a one-shot latch that
    /// already fired swallows the next model's dump.
    pub stats: &'a crate::layers::ops::ModelStats,
    /// Pre-uploaded attention metadata (None if no attention layers).
    pub attn_metadata: Option<AttnMetadataDev>,
    /// Independent live FP32 SSM rows, validated and refreshed before replay.
    /// None for prefill, temporal verification, and unsupported decode scopes.
    pub ssm_batch: Option<ssm_batch::SsmBatchView<'a>>,
    /// Profile mode: sync+time per-operation within layers.
    pub profile: bool,
    /// Communication backend for expert parallelism (EP) all-reduce.
    /// None when running single-GPU (no distributed communication).
    pub comm: Option<&'a dyn spark_comm::CommBackend>,
    /// True when inside CUDA graph capture (between begin_capture/end_capture).
    /// MoE layers use sync all_reduce (capturable) instead of async (event-based).
    pub graph_capture: bool,
    /// True when this prefill pass continues from a restored Marconi SSM
    /// snapshot (warm prefix-cache hit). GDN layers must then take the
    /// bit-faithful WY4 recurrence instead of the FLA chunked kernel: FLA's
    /// chunk grid is anchored at the (arbitrary) snapshot offset and its
    /// bf16 intermediates drift vs the pass that originally produced the
    /// cached K/V, and the replay range [snap_tok, matched) is rewritten
    /// into SHARED prefix-cache blocks — non-exact recompute poisons them
    /// and the drift ratchets across turns (2026-06-10 warm-hit stutter).
    /// Exact path: register-resident token-sequential kernel (default), WY4
    /// fallback. Off under the prefill grid (see `grid_replay_is_cold`).
    pub gdn_exact_replay: bool,
    /// Device `[num_tokens]` u32 token IDs for the tokens being processed this
    /// pass, in the SAME order the per-token MoE loop visits them. Required by
    /// DeepSeek-V4 hash-MoE layers (static `tid2eid[token_id]` routing); `None`
    /// for models without hash routing. Must be a STABLE address across the
    /// layer loop (and, under CUDA-graph decode, uploaded before each replay).
    pub token_ids: Option<DevicePtr>,
    /// HOST copy of the same token ids, when the caller had them in hand
    /// (decode always does — it uploads `token_ids` FROM this value; chunked
    /// prefill likewise). PLE computes its n-gram ids on the host, and
    /// reading them back off the device costs a synchronous D2H per decode
    /// step — pure overhead, and capture-unsupported inside a CUDA graph.
    pub host_token_ids: Option<&'a [u32]>,
    /// #30 (routed-prefill precision): the REQUEST slot's per-layer LoRA pairs,
    /// GLOBAL-layer-indexed (`len == num_hidden_layers`), set ONLY at the prefill
    /// entries and ONLY when the request routes to a NON-active slot. `Some` makes
    /// the K/V/O prefill apply sites select the request slot's pair and fold it
    /// through the SAME dense `apply_lora_delta` (dense_gemm_tc) the ACTIVE adapter
    /// uses — numerically identical to serving that adapter active, instead of the
    /// per-row bgmv (whose fp accumulation order tips razor-margin tokens). `None`
    /// (active/base request, no LoRA, and every decode/verify/mtp/moe pass) leaves
    /// the installed-active-pair path byte-identical. Prefill runs eager
    /// (`graph_capture: false`) so this per-pass CPU borrow is safe.
    pub routed_lora_layers: Option<&'a [Option<crate::lora::LoraLayerWeights>]>,
    /// Default-ON mid-chunk SSM tail capture (opt-out `ATLAS_SSM_TAIL_MIDCHUNK=0`).
    ///
    /// `Some` only on the single prefill pass whose local token range spans
    /// the block-floored matched-prefix boundary `tb`. GDN/SSM layers then
    /// split their recurrent (h_state) and conv (conv_state) kernels at
    /// `cap_local` and copy the @tb state into the reserved snapshot slot.
    /// `None` (default) => no split, byte-identical to prior behavior.
    pub midchunk_capture: Option<MidchunkCapture<'a>>,
    /// Feature-1 MoE-LoRA per-request fold decision for this forward pass,
    /// resolved by `TransformerModel::moe_lora_route` from the owning request's
    /// `adapter_slot`. Governs the prefill router/expert fold hooks
    /// (`layers/moe/lora.rs`). Ignored when no MoE adapter is installed
    /// (`self.lora == None` short-circuits first — byte-identical off). Default
    /// `Fold` keeps legacy single-request call sites unchanged.
    pub moe_lora_route: MoeLoraRoute,
}

/// Per-pass descriptor for mid-chunk SSM tail capture. Points at the reserved
/// Marconi snapshot slot's per-SSM-layer destination buffers (already offset to
/// the slot) plus the split point in local (chunk) token coordinates.
///
/// `ssm_layer_counter` is a fresh per-pass counter: each SSM layer's prefill
/// increments it once, in model order, so the value indexes `h_dsts`/`conv_dsts`
/// (which are in the same SSM-layer order as the snapshot pool).
pub struct MidchunkCapture<'a> {
    /// Split point in local token coordinates: capture state AFTER this many
    /// tokens (== `tb - proc_start`).
    pub cap_local: usize,
    /// Per-SSM-layer h_state snapshot destination (offset to the reserved slot).
    pub h_dsts: &'a [DevicePtr],
    /// Per-SSM-layer conv_state snapshot destination (offset to the reserved slot).
    pub conv_dsts: &'a [DevicePtr],
    /// Bytes per layer of h_state.
    pub h_bytes: usize,
    /// Bytes per layer of conv_state.
    pub conv_bytes: usize,
    /// Fresh per-pass SSM-layer ordinal counter (model order == pool order).
    pub ssm_layer_counter: &'a std::sync::atomic::AtomicUsize,
    /// Optional SECOND capture one KV block earlier, at `tb - block_size`
    /// (local split point `cap_local - block_size`). `Some` only when the pass
    /// also covers that point. On ~5/19 warm turns the next turn's block-floored
    /// `matched_tokens` lands exactly `tb - block_size` (generation-suffix /
    /// retokenize divergence), one block short of the tail; registering this
    /// earlier restore point makes those turns zero-replay too.
    pub cap_local_early: Option<usize>,
    /// Per-SSM-layer h_state dst for the `tb - block_size` slot (offset applied).
    pub h_dsts_early: &'a [DevicePtr],
    /// Per-SSM-layer conv_state dst for the `tb - block_size` slot.
    pub conv_dsts_early: &'a [DevicePtr],
    /// qwen4_exp only: more points of the pass (`layers::qwen4exp_ckpt`).
    pub extra: &'a [crate::layers::qwen4exp_ckpt::CapPoint],
}

/// Feature-1 MoE-LoRA fold decision for a single forward pass.
///
/// The MoE router/expert delta is a SINGLE globally-installed adapter (phase 1):
/// this gate makes the prefill fold per-request without a device kernel, exactly
/// mirroring the attention BGMV `seq_slot < 0` skip but at request granularity.
/// A base request pays nothing; a packed/mixed batch refuses loudly rather than
/// fold one adapter onto every row (the device-side per-row fold that would let
/// a mixed batch skip base rows individually is the documented follow-up —
/// `docs/design/lora-solid.md` Incr 1/3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoeLoraRoute {
    /// Single-request pass whose request owns the installed active MoE adapter:
    /// FOLD. Genuine single-seq decode now folds the router + expert gate/up/down
    /// deltas at this altitude (mirroring prefill); the remaining back-compat
    /// paths (multi-seq / verify) still bail via `reject_decode_lora` before the
    /// fold, so their `Fold` value is inert there. Also the default.
    #[default]
    Fold,
    /// Single-request pass that is base (`adapter_slot < 0`, no adapter) or
    /// routes to a different, non-installed adapter: SKIP the fold entirely.
    /// Base tokens pay nothing (request-granularity mirror of the attention
    /// BGMV `seq_slot < 0` early-return).
    Skip,
    /// Multi-request / packed / codispatch batch whose per-row adapter identity
    /// cannot be honored without the device-side per-row fold (follow-up): the
    /// fold REFUSES loudly rather than mis-apply one adapter to every row.
    Refuse,
}

/// A single transformer layer performing the full per-layer computation.
///
/// Each layer encapsulates:
/// 1. Pre-norm -> attention/SSM -> residual add
/// 2. Post-norm -> FFN/MoE -> residual add
///
/// The generic model loop iterates `layers` whatever each one is.
#[cfg(test)]
mod tests;
