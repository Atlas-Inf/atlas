// SPDX-License-Identifier: AGPL-3.0-only

//! Startup-frozen diagnostic switches for the DSpark draft head. Split out
//! of `product_policy.rs` for the file-size cap; semantically one module
//! with it (same visibility, re-exported from the parent).

/// Startup-static diagnostic switches for the draft head hot paths.
///
/// Every field defaults to off; each mirrors exactly one legacy
/// environment probe (value semantics preserved: `=1` booleans, parsed
/// integers with the legacy default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkDiagnostics {
    /// `ATLAS_DFLASH_DEBUG_CTX_OFF=1`.
    pub debug_ctx_off: bool,
    /// `ATLAS_DFLASH_DEBUG_CTX_USED=<usize>`.
    pub debug_ctx_used: Option<usize>,
    /// `ATLAS_DFLASH_PRECOMPUTE=1`.
    pub precompute_probe: bool,
    /// `ATLAS_DFLASH_PRECOMPUTE_COMMIT=1`.
    pub precompute_commit: bool,
    /// `ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1`.
    pub force_pattern: bool,
    /// `ATLAS_DFLASH_DEBUG_DUMP_FULL=1`.
    pub dump_full: bool,
    /// `ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1`.
    pub force_noise_pattern: bool,
    /// `ATLAS_DFLASH_PROPOSE_WARMUP_N=<usize>` (legacy default 2).
    pub propose_warmup_n: usize,
    /// `ATLAS_DFLASH_BLOCK_DUMP=1`.
    pub block_dump: bool,
    /// `ATLAS_DFLASH_BLOCK_DUMP_AT_POS=<usize>` (legacy default 0).
    pub block_dump_at_pos: usize,
    /// `ATLAS_DFLASH_LOG_DRAFTS=1`.
    pub log_drafts: bool,
    /// `ATLAS_DFLASH_CTX_PARITY_DUMP=1`.
    pub ctx_parity_dump: bool,
    /// `ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND=1`.
    pub no_decode_append: bool,
    /// `ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE=1`.
    pub full_precompute: bool,
    /// `ATLAS_DFLASH_CTXLEN_PROBE=1`.
    pub ctxlen_probe: bool,
    /// `ATLAS_DFLASH_VERIFY_TRACE=1`.
    pub verify_trace: bool,
    /// `ATLAS_DFLASH_BATCH_PARITY=1` — native-vs-serial fail-closed gate.
    pub batch_parity: bool,
    /// `ATLAS_DFLASH_PRECOMPUTE_DUMP=1`.
    pub precompute_dump: bool,
    /// `ATLAS_DFLASH_OPTION_B_DIAG=1`.
    pub option_b_diag: bool,
}

impl Default for DsparkDiagnostics {
    fn default() -> Self {
        Self {
            debug_ctx_off: false,
            debug_ctx_used: None,
            precompute_probe: false,
            precompute_commit: false,
            force_pattern: false,
            dump_full: false,
            force_noise_pattern: false,
            propose_warmup_n: 2,
            block_dump: false,
            block_dump_at_pos: 0,
            log_drafts: false,
            ctx_parity_dump: false,
            no_decode_append: false,
            full_precompute: false,
            ctxlen_probe: false,
            verify_trace: false,
            batch_parity: false,
            precompute_dump: false,
            option_b_diag: false,
        }
    }
}

impl DsparkDiagnostics {
    /// Legacy lenient parse of every hot-path diagnostic probe, executed
    /// exactly once. Malformed integers keep their legacy defaults.
    pub fn from_env_lenient() -> Self {
        fn one(name: &str) -> bool {
            std::env::var(name).ok().as_deref() == Some("1")
        }
        fn num(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .and_then(|raw| raw.parse().ok())
                .unwrap_or(default)
        }
        Self {
            debug_ctx_off: one("ATLAS_DFLASH_DEBUG_CTX_OFF"),
            debug_ctx_used: std::env::var("ATLAS_DFLASH_DEBUG_CTX_USED")
                .ok()
                .and_then(|raw| raw.parse().ok()),
            precompute_probe: one("ATLAS_DFLASH_PRECOMPUTE"),
            precompute_commit: one("ATLAS_DFLASH_PRECOMPUTE_COMMIT"),
            force_pattern: one("ATLAS_DFLASH_DEBUG_FORCE_PATTERN"),
            dump_full: one("ATLAS_DFLASH_DEBUG_DUMP_FULL"),
            force_noise_pattern: one("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN"),
            propose_warmup_n: num("ATLAS_DFLASH_PROPOSE_WARMUP_N", 2),
            block_dump: one("ATLAS_DFLASH_BLOCK_DUMP"),
            block_dump_at_pos: num("ATLAS_DFLASH_BLOCK_DUMP_AT_POS", 0),
            log_drafts: one("ATLAS_DFLASH_LOG_DRAFTS"),
            ctx_parity_dump: one("ATLAS_DFLASH_CTX_PARITY_DUMP"),
            no_decode_append: one("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND"),
            full_precompute: one("ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE"),
            ctxlen_probe: one("ATLAS_DFLASH_CTXLEN_PROBE"),
            verify_trace: one("ATLAS_DFLASH_VERIFY_TRACE"),
            batch_parity: one("ATLAS_DFLASH_BATCH_PARITY"),
            precompute_dump: one("ATLAS_DFLASH_PRECOMPUTE_DUMP"),
            option_b_diag: one("ATLAS_DFLASH_OPTION_B_DIAG"),
        }
    }
}
