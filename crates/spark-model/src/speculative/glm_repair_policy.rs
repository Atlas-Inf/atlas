// SPDX-License-Identifier: AGPL-3.0-only

//! Rollout gating only. Actual positions, state ownership and storage are
//! validated separately immediately before each existing E1 command.

use anyhow::{Result, ensure};

pub fn enabled() -> bool {
    std::env::var("ATLAS_GLM_MTP_REPAIR").as_deref() == Ok("1")
}

/// GLM DFlash lane: block verification through the long-context sparse path
/// without the MTP repair machinery (`ATLAS_GLM_DFLASH=1` with `--dflash`).
pub fn dflash_enabled() -> bool {
    std::env::var("ATLAS_GLM_DFLASH").as_deref() == Ok("1")
}

/// The DFlash lane verifies a block through the multi-row prefill path;
/// `ATLAS_GLM_DFLASH_PREFILL_VERIFY=0` reverts to one decode chain per row.
pub fn dflash_prefill_verify() -> bool {
    std::env::var("ATLAS_GLM_DFLASH_PREFILL_VERIFY").as_deref() != Ok("0")
}

/// `ATLAS_GLM_MTP_DISTRIBUTED=1`: the appended MTP body's experts are loaded
/// on both ranks instead of rank 0 alone (split-vocabulary MTP). Every rank
/// must run the same value (`model::startup_parity`): the worker would not
/// hold the body the head runs across the pair.
pub fn mtp_distributed() -> bool {
    std::env::var("ATLAS_GLM_MTP_DISTRIBUTED").ok().as_deref() == Some("1")
}

/// A retained long-context GLM verify owner is active: repaired MTP or DFlash.
pub fn long_lane_enabled() -> bool {
    enabled() || dflash_enabled()
}

/// Widest DFlash block the GLM long-context verify admits (bonus + drafts).
pub const MAX_DFLASH_VERIFY_ROWS: usize = 8;

pub fn long_context_enabled() -> bool {
    std::env::var("ATLAS_GLM_MTP_LONG_CONTEXT").as_deref() == Ok("1")
}

/// Default repaired long-context domain; also the native sparse prefill
/// plan's qualified context, which stays at this value.
pub const MAX_LONG_CONTEXT: usize = 32_768;
/// Largest configurable repaired domain (the served 262K profile).
pub const MAX_REPAIR_CONTEXT: usize = 262_144;

/// The repaired verifier's indexed domain: `ATLAS_GLM_MTP_REPAIR_CONTEXT`
/// (a multiple of 16 in 32768..=262144), else [`MAX_LONG_CONTEXT`]. Requests
/// whose prompt+output budget crosses it are fenced into native decode; the
/// MTP prompt capture and the drafters' private caches are sized to it.
pub fn max_long_context() -> usize {
    static CONTEXT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CONTEXT.get_or_init(|| {
        std::env::var("ATLAS_GLM_MTP_REPAIR_CONTEXT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| (MAX_LONG_CONTEXT..=MAX_REPAIR_CONTEXT).contains(&v) && v % 16 == 0)
            .unwrap_or(MAX_LONG_CONTEXT)
    })
}

/// Bound a serving arena's context before it is handed to the repaired
/// verifier. The native serving lane may expose a larger context; requests
/// beyond this indexed domain are admitted with MTP disabled.
pub fn repair_context(context: usize) -> usize {
    context.min(max_long_context())
}

/// The arena domain the repaired verifier actually indexes: the bounded repair
/// window only on the opt-in long-context GLM lane, and the plain served
/// context everywhere else (other models, repair off, long context off). Both
/// the prompt-capture allocation and the private-cache quote derive from this
/// one function so the two cannot disagree.
pub fn arena_context(model_type: &str, repair_long_enabled: bool, served_context: usize) -> usize {
    if model_type == "glm5_next" && repair_long_enabled {
        repair_context(served_context)
    } else {
        served_context
    }
}

/// The legacy prompt capture has one writer. Retained owner tails and repair
/// staging allow other requests to decode while that one prompt is chunked.
pub fn new_prompt_capacity(active: usize, prefilling: usize, capacity: usize) -> usize {
    usize::from(prefilling == 0 && active < capacity)
}

pub fn parse(raw: Option<&str>) -> Result<bool> {
    ensure!(
        matches!(raw, None | Some("0") | Some("1")),
        "ATLAS_GLM_MTP_REPAIR must be 0 or 1"
    );
    Ok(raw == Some("1"))
}

#[derive(Clone, Copy, Debug)]
pub struct RepairPolicy<'a> {
    pub model_type: &'a str,
    pub world: usize,
    pub tp: usize,
    pub ep: usize,
    pub active: usize,
    pub admitted: usize,
    pub context: usize,
    pub drafts: usize,
    pub native_only: bool,
    pub bf16: bool,
    pub prefix_reuse: bool,
    pub force: bool,
    pub long_context: bool,
}

impl RepairPolicy<'_> {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.model_type == "glm5_next"
                && self.world == 2
                && self.tp == 2
                && self.ep == 2
                && ((self.active == 1 && self.admitted == 1)
                    || (self.long_context && self.active == 4 && self.admitted == self.active))
                && matches!(self.drafts, 1 | 2 | 4)
                && (!self.long_context || self.drafts == 2)
                && self.native_only
                && self.bf16
                && !self.prefix_reuse,
            "GLM repair requires native MTP1/MTP2/MTP4, GLM TP2/EP2, C1 or opt-in MTP2 C4, BF16, and no prefix reuse"
        );
        ensure!(
            self.context >= 4
                && self.context
                    <= if self.long_context {
                        max_long_context()
                    } else {
                        2044
                    },
            "GLM repair requires context4..=2044 or opt-in MTP2 context within the repaired domain"
        );
        ensure!(self.force, "GLM repair requires resolved MTP gate force");
        Ok(())
    }
}

/// Explicit resolved drafter policy matters: old CARRY=0 variables are ignored.
pub fn validate_environment() -> Result<()> {
    ensure!(
        matches!(
            std::env::var("ATLAS_GLM_MTP_LONG_CONTEXT").ok().as_deref(),
            None | Some("0") | Some("1")
        ),
        "ATLAS_GLM_MTP_LONG_CONTEXT must be 0 or 1"
    );
    if long_context_enabled() {
        for name in [
            "ATLAS_GLM_MTP1_VERIFY_GRAPH",
            "ATLAS_GLM_TP_VERIFY_GRAPH",
            "ATLAS_GLM_C4_DECODE",
            "ATLAS_GLM_C4_SPARSE",
            "ATLAS_GLM_INDEPENDENT_DECODE",
            "ATLAS_GLM_MULTI_SEQ_SPARSE",
            "ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS",
        ] {
            ensure!(
                matches!(std::env::var(name).ok().as_deref(), None | Some("0")),
                "GLM long-context MTP2 requires {name} disabled"
            );
        }
    }
    for name in [
        "ATLAS_GLM_MTP_DISTRIBUTED",
        "ATLAS_GLM_MTP_BATCHED_PREFILL",
        "ATLAS_MTP_SPEC_THINK",
        "ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE",
    ] {
        ensure!(
            std::env::var(name).as_deref() == Ok("1"),
            "GLM repair requires {name}=1"
        );
    }
    for name in [
        "ATLAS_GLM_MTP_SERIAL_PREFILL",
        "ATLAS_NO_MTP_DRAFTER_CONTEXT",
        "ATLAS_MTP_SINGLE_DEPTH_ADAPT",
        "ATLAS_DFLASH_ADAPTIVE",
        "ATLAS_DFLASH_RESUME_GUARD",
        "ATLAS_MTP_CATCHUP",
        "ATLAS_MTP_REFEED_ACCEPTED",
    ] {
        ensure!(
            matches!(std::env::var(name).ok().as_deref(), None | Some("0")),
            "GLM repair requires {name} disabled"
        );
    }
    ensure!(
        crate::speculative::draft_conf_tau() == 0.0,
        "GLM repair cannot discard unverified drafts by confidence"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy() -> RepairPolicy<'static> {
        RepairPolicy {
            model_type: "glm5_next",
            world: 2,
            tp: 2,
            ep: 2,
            active: 1,
            admitted: 1,
            context: 2044,
            drafts: 4,
            native_only: true,
            bf16: true,
            prefix_reuse: false,
            force: true,
            long_context: false,
        }
    }
    #[test]
    fn exact_lane_and_overflow_checked() {
        assert!(policy().validate().is_ok());
        assert!(
            RepairPolicy {
                drafts: 1,
                ..policy()
            }
            .validate()
            .is_ok()
        );
        assert!(
            RepairPolicy {
                drafts: 2,
                ..policy()
            }
            .validate()
            .is_ok()
        );
        for drafts in [0, 3, 5] {
            assert!(RepairPolicy { drafts, ..policy() }.validate().is_err());
        }
        for context in [0, 3, 2045, usize::MAX] {
            assert!(
                RepairPolicy {
                    context,
                    ..policy()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            RepairPolicy {
                admitted: 2,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepairPolicy {
                prefix_reuse: true,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepairPolicy {
                force: false,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(
            RepairPolicy {
                drafts: 3,
                ..policy()
            }
            .validate()
            .is_err()
        );
        assert!(parse(Some("true")).is_err());
        assert!(!parse(None).unwrap());
    }

    #[test]
    fn long_context_requires_explicit_fixed_mtp2_and_bounded_owners() {
        let p = RepairPolicy {
            drafts: 2,
            context: MAX_LONG_CONTEXT,
            long_context: true,
            ..policy()
        };
        for active in [1, 4] {
            assert!(
                RepairPolicy {
                    active,
                    admitted: active,
                    ..p
                }
                .validate()
                .is_ok()
            );
        }
        for invalid in [
            RepairPolicy { drafts: 1, ..p },
            RepairPolicy { drafts: 4, ..p },
            RepairPolicy {
                context: MAX_LONG_CONTEXT + 1,
                ..p
            },
            RepairPolicy {
                long_context: false,
                ..p
            },
            RepairPolicy {
                active: 4,
                admitted: 5,
                ..p
            },
            RepairPolicy {
                active: 5,
                admitted: 5,
                ..p
            },
        ] {
            assert!(invalid.validate().is_err());
        }
        assert_eq!(new_prompt_capacity(0, 0, 4), 1);
        assert_eq!(new_prompt_capacity(3, 0, 4), 1);
        assert_eq!(new_prompt_capacity(0, 1, 4), 0);
        assert_eq!(new_prompt_capacity(2, 1, 4), 0);
        assert_eq!(new_prompt_capacity(4, 0, 4), 0);
    }

    #[test]
    fn repair_context_bounds_larger_serving_profiles() {
        assert_eq!(repair_context(MAX_LONG_CONTEXT), MAX_LONG_CONTEXT);
        assert_eq!(repair_context(MAX_LONG_CONTEXT + 4096), MAX_LONG_CONTEXT);
    }
    #[test]
    fn arena_context_only_bounds_selected_glm_repair() {
        for context in [2048, 32768, 36864, 262144, 524288] {
            assert_eq!(
                arena_context("glm5_next", true, context),
                context.min(32768)
            );
            assert_eq!(arena_context("glm5_next", false, context), context);
            assert_eq!(arena_context("qwen3", true, context), context);
        }
    }
}
