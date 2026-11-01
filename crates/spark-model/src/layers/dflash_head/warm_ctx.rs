// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_WARM_CTX`: the drafter's prompt window after a prefill,
//! cold or restored from the prefix cache.
//!
//! The drafter attends over the target's hidden rows at the last
//! `max_ctx_len` prompt positions (`ctx_hidden_acc`, slot `i` = position
//! `window_start + i`). The prefill capture writes the rows each pass
//! computes; `seed_prefill_ctx` then labels the window. Two things go wrong
//! with the legacy placement:
//!
//! * The capture anchors each pass's window at `seq.tokens.len().max(end)`,
//!   which during a chunked prefill is the END OF THAT PASS (the scheduler
//!   appends a chunk's tokens after it runs), while the label anchors at the
//!   prompt end. Every prompt longer than the window whose window spans two
//!   passes (the tail split alone makes that every prompt over
//!   `max_ctx_len` tokens) holds rows from earlier positions than its labels
//!   say, and misses the rows the later pass's offset overwrote.
//! * A prefix-cache hit restores the target's state at `skip` and computes
//!   only `[skip, prompt)`. The label still covers the whole window, so
//!   every slot under `skip` is a zero row (a fresh accumulator is zeroed):
//!   on a warm agentic turn the drafter attends mostly zeros.
//!
//! Values (strict: anything else fails startup on every rank):
//!
//! * `legacy` (default): unchanged.
//! * `exact`: every pass captures against the prompt end, so slot `i` holds
//!   position `window_start + i` exactly. Warm turns keep their zero rows.
//! * `seed`: `exact`, and on a prefix-cache hit the window starts at the
//!   first computed position. A shorter context, but no zero rows.
//! * `restore`: `seed`, and the window rows under `skip` come back from
//!   [`WarmCtxStore`](super::WarmCtxStore): the rows a prefill checkpoint held when it was saved,
//!   keyed by the content of the token prefix the checkpoint covers. A miss,
//!   or an entry that lacks some rows, falls back to `seed` for those rows.
//!
//! The store is rank-0 only (the accumulator is) and adds no collective, so
//! ranks need not agree on it; the target's restore depth is agreed as
//! before. What it stores, per prefill: one window of up to `max_ctx_len`
//! rows of `capture layers x hidden x BF16` (GLM-5.3-Flash with DFlash2:
//! 5 x 4096 x 2 = 40 KiB a row, 80 MiB for a full 2048-row window) in a
//! device slab of `ATLAS_DFLASH_WARM_CTX_SLOTS` windows (default 8, 640 MiB
//! on GLM) reserved before the KV pool is sized. Every SSM checkpoint the
//! prefill saves at the end of a pass (the tail split, a branch point, the
//! prompt leaf) keys the sub-range of that window ending there; a later
//! prefill that restores the target at exactly that position reads it
//! (`warm_ctx_store` docs: eviction, keys, ordering). A branch checkpoint below the prompt's
//! own window would find nothing in the accumulator, so under `restore` the
//! pass that ends there captures its last rows against its own end first
//! (the window's rows all come later and overwrite them).

use anyhow::Result;

/// Strict startup switch: an unknown value is an error, never a default.
const WARM_CTX_ENV: &str = "ATLAS_DFLASH_WARM_CTX";
/// Windows the `restore` store holds (0 = no store: `restore` acts as `seed`).
const STORE_SLOTS_ENV: &str = "ATLAS_DFLASH_WARM_CTX_SLOTS";
const DEFAULT_STORE_SLOTS: usize = 8;
/// Debug: write the drafter window of every request's first propose here.
const DUMP_DIR_ENV: &str = "ATLAS_DFLASH_CTX_DUMP_DIR";

/// How a prefill fills the drafter's prompt window (module docs).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WarmCtx {
    #[default]
    Legacy,
    Exact,
    Seed,
    Restore,
}

impl WarmCtx {
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        Ok(match raw {
            None | Some("legacy") => Self::Legacy,
            Some("exact") => Self::Exact,
            Some("seed") => Self::Seed,
            Some("restore") => Self::Restore,
            Some(other) => {
                anyhow::bail!("{WARM_CTX_ENV}={other:?} is not one of legacy|exact|seed|restore")
            }
        })
    }

    /// Startup only, on every rank before any weight load (with
    /// [`store_slots_from_env`]); the head then freezes it (`for_head`).
    pub fn from_env() -> Result<Self> {
        let raw = match std::env::var(WARM_CTX_ENV) {
            Ok(raw) => Some(raw),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => anyhow::bail!("{WARM_CTX_ENV}: {error}"),
        };
        Self::parse(raw.as_deref())
    }

    /// Read at head construction and logged once.
    pub fn for_head() -> Result<Self> {
        let variant = Self::from_env()?;
        tracing::info!(
            "DFlash warm context: {variant:?} ({WARM_CTX_ENV}=legacy|exact|seed|restore)"
        );
        Ok(variant)
    }

    /// Every pass captures against the prompt end.
    pub(crate) fn exact(self) -> bool {
        self != Self::Legacy
    }

    /// A prefix-cache hit starts the window at the first computed position.
    pub(crate) fn seeds(self) -> bool {
        matches!(self, Self::Seed | Self::Restore)
    }

    /// The rows under the restore point come back from the store.
    pub fn restores(self) -> bool {
        self == Self::Restore
    }
}

/// `ATLAS_DFLASH_WARM_CTX_SLOTS`: unset = 8; a non-integer is an error.
pub fn parse_store_slots(raw: Option<&str>) -> Result<usize> {
    match raw.map(str::trim) {
        None | Some("") => Ok(DEFAULT_STORE_SLOTS),
        Some(v) => v
            .parse()
            .map_err(|_| anyhow::anyhow!("{STORE_SLOTS_ENV}={v:?} is not a window count")),
    }
}

pub fn store_slots_from_env() -> Result<usize> {
    parse_store_slots(std::env::var(STORE_SLOTS_ENV).ok().as_deref())
}

/// `ATLAS_DFLASH_CTX_DUMP_DIR`, read once.
pub(crate) fn dump_dir() -> Option<&'static std::path::Path> {
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var_os(DUMP_DIR_ENV).map(Into::into))
        .as_deref()
}

/// One prefill pass, as the capture sees it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pass {
    /// Prompt tokens recorded before this pass (`seq.tokens.len()`).
    pub tokens_len: usize,
    /// The whole prompt (`seq.prompt_len`; 0 = unknown).
    pub prompt_len: usize,
    /// Absolute position of the pass's first computed row.
    pub start: usize,
    pub rows: usize,
    /// This rank's first row of a sequence-parallel pass (0 otherwise).
    pub sp_row0: usize,
    /// A planned branch checkpoint (`seq.pc_branch_at`).
    pub branch_at: Option<usize>,
}

impl Pass {
    pub(crate) fn end(&self) -> usize {
        self.start + self.rows
    }
}

/// Where a pass's rows land: positions `[first, end)` at slots
/// `first - window_start ..`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapturePlan {
    pub window_start: usize,
    pub first: usize,
    pub end: usize,
}

impl CapturePlan {
    /// The pass's hidden row holding position `first` (an SP pass keeps its
    /// rows from `sp_row0` on, compacted at row 0).
    pub(crate) fn src_row(&self, pass: &Pass) -> usize {
        self.first - pass.start - pass.sp_row0
    }

    pub(crate) fn dst_slot(&self) -> usize {
        self.first - self.window_start
    }

    pub(crate) fn rows(&self) -> usize {
        self.end - self.first
    }
}

/// The window a pass captures into; `None` = nothing of it is kept. `floor`
/// is the lowest position the window may start at (`ctx_floor`).
pub(crate) fn capture_plan(
    mode: WarmCtx,
    max_ctx: usize,
    floor: usize,
    pass: &Pass,
) -> Option<CapturePlan> {
    let end = pass.end();
    let anchor = if !mode.exact() {
        pass.tokens_len.max(end)
    } else if mode.restores() && pass.branch_at == Some(end) && end + max_ctx <= pass.prompt_len {
        // A branch checkpoint below the window: the accumulator is free
        // scratch until the window's own passes, so keep this pass's tail
        // for the checkpoint's store entry.
        end
    } else {
        pass.prompt_len.max(pass.tokens_len).max(end)
    };
    let window_start = anchor.saturating_sub(max_ctx).max(floor);
    let first = window_start.max(pass.start + pass.sp_row0);
    (first < end).then_some(CapturePlan {
        window_start,
        first,
        end,
    })
}

/// The accumulator rows holding real target rows: positions `[lo, hi)` in
/// the layout whose slot 0 is position `window_start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CtxRun {
    pub window_start: usize,
    pub lo: usize,
    pub hi: usize,
}

impl CtxRun {
    /// Rows `[lo, hi)` were just written in layout `window_start`. A write
    /// that extends the run keeps it; a gap or another layout starts anew.
    pub(crate) fn note(run: &mut Option<CtxRun>, window_start: usize, lo: usize, hi: usize) {
        if lo >= hi {
            return;
        }
        *run = match *run {
            Some(r) if r.window_start == window_start && r.lo <= lo && lo <= r.hi => Some(CtxRun {
                hi: r.hi.max(hi),
                ..r
            }),
            _ => Some(CtxRun {
                window_start,
                lo,
                hi,
            }),
        };
    }

    /// The real rows that end exactly at position `end`.
    pub(crate) fn rows_to(run: Option<CtxRun>, end: usize) -> Option<CtxRun> {
        run.filter(|r| r.hi == end && r.lo < end)
    }
}

/// `ATLAS_DFLASH_CTX_DUMP_DIR`: write the window a request's first propose
/// attends (`ctx-<pid>-<n>.bin`, `ctx_len` BF16 slots, and a `.json` with
/// the positions) so a warm turn can be compared with a cold prefill of the
/// same prompt (`scripts/dflash_warm_ctx_check.py`). Synchronous; debug only.
pub(crate) fn dump_first_window(
    d: &super::DflashProposerState,
    position: usize,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    stream: u64,
) -> Result<()> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let Some(dir) = dump_dir() else {
        return Ok(());
    };
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stem = dir.join(format!("ctx-{}-{n:06}", std::process::id()));
    let mut rows = vec![0u8; d.ctx_len * d.ctx_slot_bytes];
    gpu.synchronize(stream)?;
    gpu.copy_d2h(d.ctx_hidden_acc, &mut rows)?;
    std::fs::write(stem.with_extension("bin"), &rows)?;
    let meta = serde_json::json!({
        "position": position,
        "ctx_len": d.ctx_len,
        "ctx_slot_bytes": d.ctx_slot_bytes,
        "ctx_floor": d.ctx_floor,
        "warm_ctx": format!("{:?}", d.warm_ctx),
        "positions": d.ctx_positions,
    });
    std::fs::write(stem.with_extension("json"), meta.to_string())?;
    Ok(())
}
