// SPDX-License-Identifier: AGPL-3.0-only

//! Device-side metadata the layers read during a pass: decode attention
//! pointers, batched (stacked-stream) attention geometry and the GDN prefill
//! buffers. Split out of `layer.rs` (500-line cap); re-exported there.

use spark_runtime::gpu::DevicePtr;

/// Pre-uploaded attention metadata device pointers.
///
/// Uploaded once per decode step in the model loop, reused across all
/// 12 attention layers. Eliminates 44 redundant H2D copies per step.
///
/// For batched decode (num_seqs > 1), arrays are contiguous:
/// - positions: `[N]` u32
/// - slots: `[N]` i64
/// - seq_lens: `[N]` i32
/// - block_table: `[N * max_blocks_per_seq]` i32 (row-major)
#[derive(Clone, Copy)]
pub struct AttnMetadataDev {
    /// Position values: `[N]` u32 at this device address. For multi-modal
    /// MRoPE this is the temporal (T) stream; callers set
    /// `positions_h`/`positions_w` to distinct buffers only when the token
    /// stream contains image or video patches.
    pub positions: DevicePtr,
    /// Height (H) position stream for MRoPE-interleaved. When identical
    /// to `positions` (same pointer) the rope reduces to scalar RoPE.
    /// Default: same as `positions`.
    pub positions_h: DevicePtr,
    /// Width (W) position stream for MRoPE-interleaved. Same fallback as
    /// `positions_h`.
    pub positions_w: DevicePtr,
    /// Slot mappings: `[N]` i64 at this device address.
    pub slot: DevicePtr,
    /// Sequence lengths (+1): `[N]` i32 at this device address.
    pub seq_len: DevicePtr,
    /// Block tables: `[N * max_blocks_per_seq]` i32 at this device address.
    pub block_table: DevicePtr,
    /// Number of blocks per sequence row in block_table.
    pub max_blocks_per_seq: u32,
    /// Number of sequences in this batch (1 for single-sequence decode).
    pub num_seqs: u32,
    /// M2 per-request LoRA routing: `[num_seqs]` i32 at this device address,
    /// one adapter SLOT index per row (`< 0` = base / no delta; pad rows are
    /// `-1`). Uploaded each decode step to a stable address (like positions /
    /// block_table), so the batched bgmv stays inside the captured decode
    /// graph. `DevicePtr(0)` on every non-routed path (single-seq decode,
    /// prefill, verify, MLA, MTP) — the bgmv apply sites no-op when it is null.
    pub seq_slot: DevicePtr,
    /// SOLID Incr-4 (batched decode MoE fold): `[num_seqs]` i32 per-row adapter
    /// map for the MoE expert gather-BGMV fold, at this device address. MoE
    /// semantics (distinct from `seq_slot`): `< 0` = base / no fold (device
    /// kernel skips the row); `>= 0` = fold the installed active adapter's
    /// per-expert delta on that row. Built by
    /// [`crate::lora::build_moe_row_adapter_decode`] and uploaded each decode
    /// step to a stable address (a dedicated fixed-address buffer,
    /// `TransformerModel::moe_row_adapter_buf`, alloc'd once at init), so the
    /// batched fold stays inside the captured decode graph and is route-agnostic across
    /// replays (base rows no-op individually). `DevicePtr(0)` when no adapter is
    /// resident and on every non-batched path (the fold hooks then fall back to
    /// the request-granularity `moe_route_gate`). NOT the `seq_slot` buffer —
    /// that resolves `-1 → active` (attention defer-to-active), which would fold
    /// the adapter onto base rows here.
    pub moe_row_adapter: DevicePtr,
}

/// Q12 batched-prefill device-side metadata.
///
/// The single-stream `AttnMetadataDev` collapses per-stream pointers into
/// concrete device pointers because there's only one stream. For Q12 we
/// dispatch N concurrent prefilling streams through one batched kernel,
/// and the kernel takes:
///   - stacked positions / slot tables (one big buffer with all streams'
///     data concatenated in cu_seqlens order), and
///   - per-stream pointer arrays for block_table / seq_len / h_state.
///
/// Built once per `prefill_batch_chunk_dispatch` call by
/// `stage_batched_attn_metadata`; threaded through the model-level
/// per-layer batched dispatch (`prefill_attn_batched_layer`,
/// `prefill_ssm_batched_layer`) — see `model/trait_impl/prefill_b/batch.rs`.
pub struct BatchedAttnMetadata {
    /// Stacked positions across all streams: `[total_tokens]` u32 at this
    /// address. For MRoPE interleaved this is the temporal (T) stream.
    pub positions_stacked: DevicePtr,
    /// MRoPE H position stream, stacked. Equal to `positions_stacked` when
    /// MRoPE is disabled.
    pub positions_h_stacked: DevicePtr,
    /// MRoPE W position stream, stacked. Equal to `positions_stacked` when
    /// MRoPE is disabled.
    pub positions_w_stacked: DevicePtr,
    /// Stacked slot indices for KV writes: `[total_tokens]` i64.
    pub slot_stacked: DevicePtr,
    /// Per-stream block_table pointer array: `[batch_size]` of `DevicePtr`,
    /// each element pointing to a stream's chunked-prefill block_table.
    /// Used by `prefill_attention_paged_*_batched` kernels.
    pub block_table_ptrs: DevicePtr,
    /// Per-stream seq_len pointer array: `[batch_size]` of `DevicePtr`.
    pub seq_len_ptrs: DevicePtr,
    // Note: `h_state_ptrs` is NOT cached in BatchedAttnMetadata because
    // it's per-layer (each SSM layer's SsmLayerState has its own h_state
    // allocation). `prefill_ssm_batched_layer` stages h_state_ptrs JIT
    // per-layer-call into the model's scratch buffer.
    /// Number of batched streams.
    pub batch_size: u32,
    /// Per-stream chunk_len. In the legacy same-length path this is uniform; in
    /// the VARLEN path (`cu_seqlens` populated) it is the MAX per-stream length
    /// (retained only for buffer-bound/debug use — per-stream lengths come from
    /// `cu_seqlens`).
    pub chunk_len: u32,
    /// Total tokens stacked across streams. Legacy: `batch_size * chunk_len`.
    /// VARLEN: `Σ per-stream lengths` (= `cu_seqlens_host[batch_size]`).
    pub total_tokens: u32,
    /// VARLEN geometry: `[batch_size+1]` i32 prefix-sum of per-request token
    /// counts, on device (read by the GDN kernel + FlashInfer). `DevicePtr::NULL`
    /// in the legacy same-length path (callers fall back to `b*chunk_len`).
    pub cu_seqlens: DevicePtr,
    /// Host copy of `cu_seqlens` (`[batch_size+1]` i32) — FlashInfer's PrefillPlan
    /// dereferences the indptr on the CPU, and per-request slice offsets are
    /// computed host-side. Empty in the legacy path.
    pub cu_seqlens_host: Vec<i32>,
    /// VARLEN geometry: per-stream KV length `[batch_size]` i32 on device,
    /// `kv_lens[b] = chunk_start + per-stream token count`. The batched paged
    /// attention kernels need this per stream: a single scalar at the MAX
    /// makes short streams index their block_table past the blocks they
    /// actually own, and applies the wrong causal bound. `DevicePtr::NULL` in
    /// the legacy same-length path (kernels fall back to the scalar `kv_len`).
    pub kv_lens: DevicePtr,
    /// Host copy of `kv_lens`. Empty in the legacy path.
    pub kv_lens_host: Vec<i32>,
    /// Per-stream prefix-bidirectional state end `[batch_size]` i32 on device
    /// (`-1` = causal), read by the batched paged attention kernels; see
    /// `prefill_paged_compute.cuh`. `DevicePtr::NULL` (every stream causal)
    /// everywhere except the score path's `prefix_bidir` requests.
    pub state_end: DevicePtr,
    /// Score pass: the BF16 HDIM-256 layers run GQA-packed 128-row tiles at
    /// any length (`rows128.rs`), so every request of a pass takes the same
    /// kernel alone or merged. False everywhere else.
    pub gqa_tiles: bool,
    /// Maximum block_table length across the batch (kernel uses for
    /// bounds checking; per-stream block_table reads via the pointer
    /// array dereference).
    pub max_blocks_per_seq: u32,
    /// Exact byte footprint of this metadata block within the scratch
    /// buffer (from `scratch_offset_bytes` to the end of `seq_len_ptrs`).
    /// SSOT for the caller's scratch-cursor advance — the per-SSM-layer
    /// `h_state_ptrs` slot is placed at `scratch_cursor + staged_bytes`, so
    /// an under-estimate here would overwrite the live `slot_stacked` array
    /// with device pointers and produce wild KV-cache slots (#110 bug #2).
    pub staged_bytes: usize,
}

/// Device pointers to full-sequence GDN input/output buffers.
///
/// Used by the two-phase SSM prefill: phase 1 writes GDN inputs here,
/// phase 2 reads them for the single-launch GDN kernel, phase 3 reads output.
///
/// Uses a **packed QKV layout** matching the conv1d output: each token occupies
/// `conv_dim` contiguous BF16 elements as `[Q(key_dim) | K(key_dim) | V(value_dim)]`.
/// This allows simple contiguous memcpy from per-chunk conv1d output buffers.
/// The GDN kernel reads Q/K/V via stride parameters (`qk_stride = conv_dim`,
/// `v_stride = conv_dim`) to index into the packed layout.
pub struct GdnPrefillBuffers {
    /// Packed Q/K/V: [total_len, conv_dim] BF16.
    /// Layout per token: [Q(key_dim) | K(key_dim) | V(value_dim)].
    pub qkv: DevicePtr,
    /// Interleaved gate/beta: [total_len, 2*num_v_heads] FP32.
    /// Layout per token: [gate(nv) | beta(nv)].
    pub gate_beta: DevicePtr,
    /// GDN recurrence output: [total_len, value_dim] BF16.
    pub output: DevicePtr,
    /// Z gate for gated RMS norm: [total_len, z_stride] BF16, value_dim used.
    pub z: DevicePtr,
    /// Row pitch of `z` in elements: value_dim for the dedicated buffer, the
    /// QKVZ row when `z` reads the projection output in place (score pass).
    pub z_stride: usize,
    /// Total number of tokens across all chunks.
    pub total_len: usize,
}
