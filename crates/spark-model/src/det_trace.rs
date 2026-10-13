// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill determinism tracer (`ATLAS_GLM_DET_TRACE`), a debugging aid.
//!
//! The same prompt on the same binary can prefill to different bytes on
//! different server boots. This logs one line per (request, chunk, layer,
//! stage) with a 64-bit hash of the exact bytes of that stage's tensor, on
//! every rank, so two logs diff down to the first stage that differs:
//!
//! ```text
//! DET r=<rank> q=<request> c=<chunk start> L=<layer> s=<stage> r0=<first row> n=<rows> b=<bytes> h=<hash>
//! ```
//!
//! * `q` counts requests since boot (mirrored on both ranks), so the same
//!   request order gives the same numbers on every boot.
//! * `c` is the sequence position of the chunk's first computed row, `r0`
//!   the hashed rows' first row within the chunk. Under sequence-parallel
//!   prefill a rank holds half the rows of the row-local stages.
//! * Stages in forward order: `emb` (the embeddings, before layer 0); per
//!   layer `in` (block input after the mHC pre-mix), on MLA layers `kv_lat`
//!   (latents written to the KV cache) and one `sel` per sparse attention
//!   piece (selected token ids), `attn` (attention/KDA output before the TP
//!   reduce), `attn_red` (after it), `moe_in`, `rt_ids` and `rt_w` (router
//!   top-k), `moe_local` (before the EP reduce), `moe_red` (after it),
//!   `moe_sh` (shared expert), `moe` (after its blend), `out` (the mHC
//!   highway); then at `L = layers` `final` (hidden), `plogits` (every
//!   scored row's logits, scoring requests only) and `logits` (last row).
//!   `attn` and `moe_local` hold this rank's partial and differ by rank.
//!
//! `ATLAS_GLM_DET_TRACE=1` also makes every request recompute its whole
//! prompt (no prefix-cache reuse, as for prompt-logprob scoring), so a
//! repeated prompt is traced in full; `=2` traces without that.
//! `ATLAS_GLM_DET_TRACE_STAGES=a,b` keeps only the named stages, and is the
//! only way to get the `x_` stages that split a block further: KDA `x_qkv`
//! (Q/K/V planes), `x_g` (gate planes), `x_gated` (recurrence + gated norm,
//! the `o_proj` input); MLA `x_q_lat` (query latent) and per piece `x_q_abs`
//! (absorbed queries), `x_attn_lat` (attention output before W_uv/`o_proj`);
//! MoE `x_gate` (router logits).
//!
//! Every tap synchronizes the stream and copies the tensor to the host, so
//! a traced prefill is several times slower and its stream timing differs.
//! Unset, the taps return on one cached flag and nothing else changes.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use atlas_tier::hash::mix64;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Distinct odd multipliers, one per lane.
const LANE: [u64; 4] = [
    0x9E37_79B9_7F4A_7C15,
    0xBF58_476D_1CE4_E5B9,
    0x94D0_49BB_1331_11EB,
    0xD6E8_FEB8_6659_FD93,
];
const BLOCK: usize = 32;
/// Host staging bytes per device copy.
const SEGMENT: usize = 32 << 20;
const SLOTS: usize = 64;

/// Streaming 64-bit hash of exact bytes: four independent multiply-xorshift
/// lanes over little-endian words (fast enough for multi-GiB tensors), folded
/// with the byte length. The result does not depend on how `update` calls
/// split the bytes. Each lane step is a bijection, so two inputs differing
/// in one word never collide.
#[derive(Clone)]
pub struct Hasher {
    lanes: [u64; 4],
    len: u64,
    tail: [u8; BLOCK],
    held: usize,
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    pub fn new() -> Self {
        Self {
            lanes: LANE,
            len: 0,
            tail: [0; BLOCK],
            held: 0,
        }
    }

    #[inline(always)]
    fn block(lanes: &mut [u64; 4], block: &[u8]) {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let mut word = [0u8; 8];
            word.copy_from_slice(&block[i * 8..i * 8 + 8]);
            let x = (*lane ^ u64::from_le_bytes(word)).wrapping_mul(LANE[i]);
            *lane = x ^ (x >> 29);
        }
    }

    pub fn update(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len() as u64;
        if self.held > 0 {
            let take = (BLOCK - self.held).min(bytes.len());
            self.tail[self.held..self.held + take].copy_from_slice(&bytes[..take]);
            self.held += take;
            bytes = &bytes[take..];
            if self.held < BLOCK {
                return;
            }
            Self::block(&mut self.lanes, &self.tail);
            self.held = 0;
        }
        let mut blocks = bytes.chunks_exact(BLOCK);
        for block in &mut blocks {
            Self::block(&mut self.lanes, block);
        }
        let rest = blocks.remainder();
        self.tail[..rest.len()].copy_from_slice(rest);
        self.held = rest.len();
    }

    pub fn finish(mut self) -> u64 {
        if self.held > 0 {
            self.tail[self.held..].fill(0);
            Self::block(&mut self.lanes, &self.tail);
        }
        self.lanes.iter().fold(self.len, |h, &lane| mix64(h, lane))
    }
}

/// [`Hasher`] over one slice.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finish()
}

/// Where a trace line sits: rank, request ordinal, chunk start, layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct At {
    pub rank: usize,
    pub request: u64,
    pub chunk_start: usize,
    pub layer: usize,
}

/// The trace line for `rows` rows (`bytes` bytes) of `stage` from chunk row
/// `row0`; `hash` is `None` when the tensor could not be read.
pub fn format_line(
    at: At,
    stage: &str,
    (row0, rows): (usize, usize),
    bytes: usize,
    hash: Option<u64>,
) -> String {
    let hash = hash.map_or_else(|| "ERR".to_owned(), |h| format!("{h:016x}"));
    format!(
        "DET r={} q={} c={} L={} s={stage} r0={row0} n={rows} b={bytes} h={hash}",
        at.rank, at.request, at.chunk_start, at.layer
    )
}

/// `ATLAS_GLM_DET_TRACE`: 0 off, 1 trace and recompute every prompt, 2 trace.
fn parse_level(value: Option<&str>) -> u8 {
    match value {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    }
}

fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| parse_level(std::env::var("ATLAS_GLM_DET_TRACE").ok().as_deref()))
}

/// Whether the tracer is enabled.
#[inline]
pub fn on() -> bool {
    level() != 0
}

/// Whether `stage` passes an `ATLAS_GLM_DET_TRACE_STAGES` list. Without a
/// list every stage passes except the `x_` ones, which must be named.
fn stage_listed(list: Option<&str>, stage: &str) -> bool {
    match list {
        Some(list) => list.split(',').any(|s| s.trim() == stage),
        None => !stage.starts_with("x_"),
    }
}

fn stage_selected(stage: &str) -> bool {
    static LIST: OnceLock<Option<String>> = OnceLock::new();
    let list = LIST.get_or_init(|| std::env::var("ATLAS_GLM_DET_TRACE_STAGES").ok());
    stage_listed(list.as_deref(), stage)
}

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(0);
static SLOT_REQUEST: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];

thread_local! {
    static CURRENT: Cell<Option<At>> = const { Cell::new(None) };
    static STAGING: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// The lines this thread logged, for tests of the traced call sites.
    #[cfg(test)]
    static LINES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn emit(line: String) {
    #[cfg(test)]
    LINES.with(|lines| lines.borrow_mut().push(line.clone()));
    tracing::warn!("{line}");
}

/// Drain the lines this thread logged.
#[cfg(test)]
pub(crate) fn take_lines() -> Vec<String> {
    LINES.with(|lines| std::mem::take(&mut *lines.borrow_mut()))
}

/// A request starts prefilling on sequence slot `slot` (its prefix lookup,
/// once per request on every rank): number it. Returns whether the request
/// must skip prefix-cache reuse and recompute its whole prompt.
pub fn begin_request(slot: usize) -> bool {
    if !on() {
        return false;
    }
    let request = NEXT_REQUEST.fetch_add(1, Ordering::Relaxed) + 1;
    SLOT_REQUEST[slot % SLOTS].store(request, Ordering::Relaxed);
    level() == 1
}

/// Ends the traced chunk when dropped.
pub struct Scope(());

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(None));
    }
}

/// Trace this thread's taps as the chunk of slot `slot`'s request whose
/// first computed row is sequence position `chunk_start`, until the scope
/// drops. `None` (and no taps) when the tracer is off.
pub fn enter(rank: usize, slot: usize, chunk_start: usize) -> Option<Scope> {
    if !on() {
        return None;
    }
    let request = SLOT_REQUEST[slot % SLOTS].load(Ordering::Relaxed);
    CURRENT.with(|c| {
        c.set(Some(At {
            rank,
            request,
            chunk_start,
            layer: 0,
        }))
    });
    Some(Scope(()))
}

/// The following taps belong to `layer` (the layer count after the last).
pub fn set_layer(layer: usize) {
    if on() {
        CURRENT.with(|c| c.set(c.get().map(|at| At { layer, ..at })));
    }
}

fn current(stage: &str) -> Option<At> {
    if !on() || !stage_selected(stage) {
        return None;
    }
    CURRENT.with(Cell::get)
}

fn device_hash(
    gpu: &dyn GpuBackend,
    stream: u64,
    ptr: DevicePtr,
    bytes: usize,
    segment: usize,
) -> Option<u64> {
    gpu.synchronize(stream).ok()?;
    STAGING.with(|staging| {
        let mut staging = staging.borrow_mut();
        let mut hasher = Hasher::new();
        let mut done = 0;
        while done < bytes {
            let n = (bytes - done).min(segment);
            if staging.len() < n {
                staging.resize(n, 0);
            }
            gpu.copy_d2h(ptr.offset(done), &mut staging[..n]).ok()?;
            hasher.update(&staging[..n]);
            done += n;
        }
        Some(hasher.finish())
    })
}

/// The line for `rows` rows of `row_bytes` bytes at `ptr`, read after `stream`.
fn device_line(
    at: At,
    gpu: &dyn GpuBackend,
    stream: u64,
    stage: &str,
    ptr: DevicePtr,
    rows: (usize, usize),
    row_bytes: usize,
) -> String {
    let bytes = rows.1 * row_bytes;
    let hash = device_hash(gpu, stream, ptr, bytes, SEGMENT);
    format_line(at, stage, rows, bytes, hash)
}

/// Taps reading after `stream` (see [`on_stream`]).
#[derive(Clone, Copy)]
pub struct Taps<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
}

/// A handle for the taps of one forward pass on `stream`.
pub fn on_stream(gpu: &dyn GpuBackend, stream: u64) -> Taps<'_> {
    Taps { gpu, stream }
}

impl Taps<'_> {
    /// Hash `rows.1` rows of `row_bytes` bytes at `ptr` (chunk rows from
    /// `rows.0`) as `stage` of the current chunk and layer. Does nothing
    /// outside a traced chunk, so the verify and decode callers of shared
    /// layer code are silent, and nothing for an empty span.
    pub fn tap(&self, stage: &str, ptr: DevicePtr, rows: (usize, usize), row_bytes: usize) {
        if rows.1 * row_bytes == 0 {
            return;
        }
        if let Some(at) = current(stage) {
            emit(device_line(
                at,
                self.gpu,
                self.stream,
                stage,
                ptr,
                rows,
                row_bytes,
            ));
        }
    }
}

/// Log a `hash` of `bytes` host bytes already read for `stage`.
pub fn tap_hashed(stage: &str, rows: (usize, usize), bytes: usize, hash: u64) {
    if let Some(at) = current(stage) {
        emit(format_line(at, stage, rows, bytes, Some(hash)));
    }
}

#[cfg(test)]
#[path = "det_trace_tests.rs"]
mod tests;
