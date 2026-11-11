// SPDX-License-Identifier: AGPL-3.0-only

/// Presence kill switch for the wide (9..=VERIFY_ROW_CAP row) batched LM head arm:
/// `ATLAS_NO_LMHEAD_BATCHED_WIDE` restores the M64-tile `w4a16_gemm` fallback.
/// Presence, not value — `=0` is NOT "off" (see `atlas_env_presence_check_trap`).
pub(super) fn lmhead_batched_wide_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_LMHEAD_BATCHED_WIDE").is_err())
}

/// Exact five-row BF16 vocabulary projection for GLM's K=5 verifier. The
/// existing batched GEMV streams each BF16 weight row once and reuses it for
/// all five hidden rows; the generic tiny-M GEMM is retained as an A/B fallback.
pub(super) fn glm_k5_bf16_lmhead_batchm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_GLM_K5_BF16_LMHEAD_BATCHM")
            .ok()
            .as_deref()
            == Some("1")
    })
}
