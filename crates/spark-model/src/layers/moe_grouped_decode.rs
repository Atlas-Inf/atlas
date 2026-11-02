// SPDX-License-Identifier: AGPL-3.0-only

//! Width gate for the grouped-GEMM MoE decode arm (threshold, env switches, decider).

/// Optional kernel lookup: `KernelHandle(0)` instead of an error.
///
/// `#[track_caller]` so the audit names the DISPATCH SITE — this helper stands
/// between ~500 call sites and `GpuBackend::kernel`, and without it every
/// optional lookup in the binary would be reported against this one line.
///
/// A zero handle is a SILENT slower path, so a lookup that lands here for a
/// model that genuinely needs the kernel is a bug. Either gate the call on the
/// model's config so it is never issued, or declare it in the target's
/// MODEL.toml `[expected_absent]` with a reason; the boot gate
/// (`kernel_audit::classify_failures`) fails closed on anything else.
/// Minimum rows in flight for the grouped-GEMM MoE decode arm. SSOT for the
/// SSM stack (`qwen3_ssm::trait_decode_multi_seq`) and the attention layers
/// (`qwen3_attention::…::multi_seq::ffn`), which must agree — they are the
/// same trade on the same weights.
///
/// The arm reads each routed expert ONCE instead of once per token, so it
/// wins when there are enough tokens to amortise the expert sort/permute
/// launch overhead, and loses when there are not. Both ends are measured:
///
/// | n | verdict | measurement |
/// |---|---|---|
/// | 4 | LOSS | 31 vs 56 tok/s on Holo — the fixed per-layer sort/permute dominates at small N |
/// | >=16 | WIN | SSM-side alone C=32 172.7 -> 216.2 tok/s (+25%); #415's attention-side extension +7.9% at C=32 / +9.7% at C=64 on Qwen3.6-35B-A3B-NVFP4, paired gsm8k n=200 strict 0.960 vs 0.900 baseline, zero regressions |
///
/// 16 is the smallest width measured on the winning side. n=5..15 is
/// UNMEASURED, not a known win — it sits on the losing side of this gate on
/// purpose, because the one thing we know about the gap is that the loss at
/// n=4 is large (-45%) and the win at n=16 is smaller (+25%).
pub fn moe_grouped_decode_min_rows() -> usize {
    16
}

/// Kill switch for the grouped-GEMM MoE decode arm. PRESENCE check per the
/// house convention (`ATLAS_NO_MOE_GROUPED_DECODE=0` is NOT off), read once
/// per process — this predicate sits in the decode path, and the `env::var`
/// it replaces ran on every dispatch for MoE models.
pub fn moe_grouped_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_MOE_GROUPED_DECODE").is_none())
}

/// Force the grouped arm BELOW `moe_grouped_decode_min_rows()`. Diagnostic
/// only — it exists so the n=5..15 gap can be measured without a rebuild, and
/// it is the same var #415's measurements used, kept working on purpose.
/// Never a production setting: if forcing wins at a width, move the THRESHOLD.
pub fn moe_grouped_decode_forced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_GROUPED_DECODE").as_deref() == Ok("1"))
}

/// Whether the grouped-GEMM MoE decode arm should run for `n` rows —
/// PURE, so both polarities are testable without touching process env or the
/// `OnceLock`s below (which latch, and would make the tests order-dependent).
pub fn moe_grouped_decode_decide(n: usize, enabled: bool, forced: bool) -> bool {
    enabled && (n >= moe_grouped_decode_min_rows() || forced)
}

/// Whether the grouped-GEMM MoE decode arm should run for `n` rows.
pub fn moe_grouped_decode_for(n: usize) -> bool {
    moe_grouped_decode_decide(n, moe_grouped_decode_enabled(), moe_grouped_decode_forced())
}

#[cfg(test)]
#[path = "moe_grouped_decode_tests.rs"]
mod moe_grouped_decode_tests;
