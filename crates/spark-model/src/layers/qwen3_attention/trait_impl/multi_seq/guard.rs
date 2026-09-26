// SPDX-License-Identifier: AGPL-3.0-only

//! The pre-mutation QSA plan for the batched multi-row path.
//!
//! Split out of `mod.rs` for the file-size cap, and because this is the one
//! thing in that entry point that reads a property of the LAYER rather than
//! of the step. It runs before any ingest mutation and answers one of three
//! ways: every row is INERT (today's batched attention serves it), an ACTIVE
//! row is present and the per-row phase (`qsa_rows.rs`) can serve the step, or
//! the step is refused — here, cleanly, instead of as an error after earlier
//! layers have advanced their state.
//!
//! Serving an active selection is an ALLOW-list, not a deny-list: exactly the
//! combination the serial path supports and the per-row phase mirrors. The
//! static half also decides `verify_context_limit()`, so the scheduler's view
//! and this guard are the same function and cannot disagree.

use anyhow::Result;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;

/// `ATLAS_QSA_VERIFY_ACTIVE`: serve an ACTIVE QSA selection on the batched
/// multi-row path (speculative verify past the inert bound). Default ON for
/// NVIDIA since 2026-09-26 (#70): GB10 job 250 measured MTP K=2 + verify-active
/// at 5.1k context at 28.7 tok/s vs 18.4 serial (+56 %); job 257 measured
/// needles 10/12 = 10/12 vs serial and deterministic (M vs M2 top-1 100 %,
/// JSD 0) — NOT output-identical to serial on long generations (top-1 43.7 %,
/// the same class as existing short-context MTP divergence). TODO(#70): ST-995
/// A/B and agentic perf leg numbers land here. `=0`/`=false` restores the old
/// inert-bound refusal; gfx1151 (`atlas_scale`) keeps default OFF (unmeasured
/// there).
pub(in crate::layers::qwen3_attention) fn verify_active_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        verify_active_decision(
            std::env::var("ATLAS_QSA_VERIFY_ACTIVE").ok().as_deref(),
            cfg!(atlas_scale),
        )
    })
}

/// Pure decision (table-tested): explicit `1`/`true` or `0`/`false` wins;
/// anything else, including unset, defaults ON for NVIDIA and OFF on gfx1151.
pub(crate) fn verify_active_decision(env: Option<&str>, gfx: bool) -> bool {
    match env {
        Some("1") | Some("true") => true,
        Some("0") | Some("false") => false,
        _ => !gfx,
    }
}

/// Everything the allow-list looks at, as plain facts, so the decision is a
/// pure function of them.
#[derive(Debug, Clone, Copy)]
pub(super) struct RowsFacts {
    /// `ATLAS_QSA_VERIFY_ACTIVE` resolved (default ON for NVIDIA —
    /// see [`verify_active_enabled`]).
    pub switch_on: bool,
    /// mHC highway layer — the only batched body the per-row phase is wired into.
    pub highway: bool,
    /// MLA layers take `ms_mla_decode`, which has no selection hook.
    pub mla: bool,
    /// The gather copies raw NHD rows: plain BF16 K and V only (no FP8, no
    /// TurboQuant/WHT — which also means no WHT bookends are owed).
    pub bf16_kv: bool,
    /// Every row belongs to ONE sequence (a verify window). Rows of different
    /// sequences are concurrent decode, which `decode_a2` routes per-sequence.
    pub single_owner: bool,
    /// Expert/tensor-parallel comm present: `decode_a2` returns into the
    /// batched path BEFORE its active-QSA gate there, so this guard is the
    /// only thing between an EP serve and an unaudited path.
    pub comm: bool,
    /// `--high-speed-swap` engaged: the gather reads the HBM pool.
    pub high_speed_swap: bool,
}

impl RowsFacts {
    /// Why the per-row phase cannot serve this step, if it cannot.
    fn refusal(&self) -> Option<&'static str> {
        if !self.switch_on {
            Some("ATLAS_QSA_VERIFY_ACTIVE is off")
        } else if !self.highway {
            Some("not an mHC highway layer")
        } else if self.mla {
            Some("MLA attention has no selection hook")
        } else if !self.bf16_kv {
            Some("the KV cache is not plain BF16")
        } else if !self.single_owner {
            Some("the rows span several sequences")
        } else if self.comm {
            Some("a parallel comm is present")
        } else if self.high_speed_swap {
            Some("--high-speed-swap is engaged")
        } else {
            None
        }
    }
}

/// The plan for one step. `first_active` is the first row whose selection is
/// ACTIVE, as `(row, pos)`. `Ok(false)`: all rows inert, batched attention.
/// `Ok(true)`: per-row phase. `Err`: refused, nothing mutated.
pub(super) fn plan_rows(
    first_active: Option<(usize, usize)>,
    bound: usize,
    facts: &RowsFacts,
) -> Result<bool> {
    let Some((row, pos)) = first_active else {
        return Ok(false);
    };
    match facts.refusal() {
        None => Ok(true),
        Some(why) => anyhow::bail!(
            "VerifyUnsupportedWithActiveQsa: row {row} at pos {pos} >= inert bound {bound} ({why})"
        ),
    }
}

impl Qwen3AttentionLayer {
    /// The STATIC half of the allow-list: what this layer knows about itself
    /// at load. `verify_context_limit()` lifts the bound exactly when this
    /// holds, so the scheduler never dispatches a verify this guard refuses
    /// for a static reason.
    pub(in crate::layers::qwen3_attention) fn qsa_rows_static_ok(&self) -> bool {
        let (k, v) = self.kv_dtype.kv_pair();
        self.qsa.is_some()
            && self.hc.is_some()
            && self.mla.is_none()
            && matches!(k, KvCacheDtype::Bf16)
            && matches!(v, KvCacheDtype::Bf16)
            && verify_active_enabled()
    }
}

/// Plan this step's attention before ANY mutation. See the module docs.
pub(super) fn plan_qsa_rows(
    layer: &Qwen3AttentionLayer,
    seq_lens: &[usize],
    num_seqs: usize,
    row_owner: Option<&[usize]>,
    kv_cache: &PagedKvCache,
    ctx: &ForwardContext,
) -> Result<bool> {
    let Some(qsa) = layer.qsa.as_ref() else {
        return Ok(false);
    };
    // `seq_lens[i]` is row i's 0-based position (`pos + 1` tokens visible).
    let first_active = seq_lens
        .iter()
        .take(num_seqs)
        .enumerate()
        .find(|&(_, &pos)| qsa.is_active_at(pos))
        .map(|(row, &pos)| (row, pos));
    if first_active.is_none() {
        return Ok(false);
    }
    let (k, v) = layer.kv_dtype.kv_pair();
    let single_owner = match row_owner {
        Some(map) => map.len() >= num_seqs && map.iter().take(num_seqs).all(|o| *o == map[0]),
        None => num_seqs == 1,
    };
    let facts = RowsFacts {
        switch_on: verify_active_enabled(),
        highway: layer.hc.is_some(),
        mla: layer.mla.is_some(),
        bf16_kv: matches!(k, KvCacheDtype::Bf16) && matches!(v, KvCacheDtype::Bf16),
        single_owner,
        comm: ctx.comm.is_some(),
        high_speed_swap: layer.high_speed_swap_engaged(kv_cache),
    };
    plan_rows(first_active, qsa.inert_bound(), &facts)
}

#[cfg(test)]
mod tests {
    use super::{RowsFacts, plan_rows, verify_active_decision};

    const SERVABLE: RowsFacts = RowsFacts {
        switch_on: true,
        highway: true,
        mla: false,
        bf16_kv: true,
        single_owner: true,
        comm: false,
        high_speed_swap: false,
    };

    #[test]
    fn inert_steps_take_the_batched_path_whatever_the_facts() {
        let nothing = RowsFacts {
            switch_on: false,
            highway: false,
            mla: true,
            bf16_kv: false,
            single_owner: false,
            comm: true,
            high_speed_swap: true,
        };
        assert!(!plan_rows(None, 2051, &nothing).unwrap());
        assert!(!plan_rows(None, 2051, &SERVABLE).unwrap());
    }

    #[test]
    fn an_active_row_is_served_only_by_the_exact_allow_list() {
        assert!(plan_rows(Some((1, 2051)), 2051, &SERVABLE).unwrap());
        // Flip each fact alone: every single one must refuse, with its reason.
        let cases: [(RowsFacts, &str); 7] = [
            (
                RowsFacts {
                    switch_on: false,
                    ..SERVABLE
                },
                "ATLAS_QSA_VERIFY_ACTIVE",
            ),
            (
                RowsFacts {
                    highway: false,
                    ..SERVABLE
                },
                "highway",
            ),
            (
                RowsFacts {
                    mla: true,
                    ..SERVABLE
                },
                "MLA",
            ),
            (
                RowsFacts {
                    bf16_kv: false,
                    ..SERVABLE
                },
                "BF16",
            ),
            (
                RowsFacts {
                    single_owner: false,
                    ..SERVABLE
                },
                "several sequences",
            ),
            (
                RowsFacts {
                    comm: true,
                    ..SERVABLE
                },
                "comm",
            ),
            (
                RowsFacts {
                    high_speed_swap: true,
                    ..SERVABLE
                },
                "high-speed-swap",
            ),
        ];
        for (facts, why) in cases {
            let err = plan_rows(Some((1, 2051)), 2051, &facts)
                .unwrap_err()
                .to_string();
            assert!(
                err.starts_with(
                    "VerifyUnsupportedWithActiveQsa: row 1 at pos 2051 >= inert bound 2051"
                ),
                "{err}"
            );
            assert!(err.contains(why), "refusal for {why:?} reads: {err}");
        }
    }

    #[test]
    fn verify_active_decision_table() {
        // Unset: default ON for NVIDIA, OFF on gfx1151.
        assert!(verify_active_decision(None, false));
        assert!(!verify_active_decision(None, true));
        // Explicit off wins on NVIDIA; explicit on wins on gfx1151.
        assert!(!verify_active_decision(Some("0"), false));
        assert!(!verify_active_decision(Some("false"), false));
        assert!(verify_active_decision(Some("1"), false));
        assert!(verify_active_decision(Some("true"), true));
        // Unrecognised values behave as unset.
        assert!(verify_active_decision(Some("yes"), false));
        assert!(!verify_active_decision(Some("yes"), true));
    }
}
