// SPDX-License-Identifier: AGPL-3.0-only

use std::fmt;

pub const LIGHTNING_ALGORITHM: &str = "DSpark";
pub const LIGHTNING_MODEL_IDENTITY: &str = "Qwen3DSparkModel";
pub const LIGHTNING_CHECKPOINT_BLOCK_SIZE: usize = 8;
pub const LIGHTNING_PHYSICAL_KV_PAGE_SIZE: usize = 16;
pub const LIGHTNING_SERVED_GAMMA: usize = 4;
pub const LIGHTNING_NUM_DRAFTS: usize = 3;
pub const LIGHTNING_TAPS: [usize; 6] = [1, 5, 19, 29, 41, 51];
pub const LIGHTNING_MARKOV_RANK: usize = 512;
pub const LIGHTNING_SWA_WINDOW: usize = 1024;
pub const LIGHTNING_TP: usize = 1;
pub const LIGHTNING_EP: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    Fp8,
    Bf16,
}

impl fmt::Display for KvDtype {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Fp8 => "FP8",
            Self::Bf16 => "BF16",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointLayout {
    pub block_size: usize,
    pub physical_kv_page_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionLayout {
    pub causal: bool,
    pub use_swa: bool,
    pub swa_window: usize,
    pub attention_sink_bias: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkovLayout {
    pub rank: usize,
    pub w1_present: bool,
    pub w2_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BonusLayout {
    pub bonus_anchor: bool,
    pub sample_from_anchor: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfidenceLayout {
    pub head_present: bool,
    pub adaptive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayout {
    pub target: KvDtype,
    pub drafter: KvDtype,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelismLayout {
    pub tensor_parallel: usize,
    pub expert_parallel: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightningDsparkProfile {
    pub algorithm: String,
    pub model_identity: String,
    pub checkpoint: CheckpointLayout,
    pub served_gamma: usize,
    pub num_drafts: usize,
    pub taps: Vec<usize>,
    pub attention: AttentionLayout,
    pub markov: MarkovLayout,
    pub bonus: BonusLayout,
    pub confidence: ConfidenceLayout,
    pub kv: KvLayout,
    pub parallelism: ParallelismLayout,
}

impl LightningDsparkProfile {
    pub fn lightning() -> Self {
        Self {
            algorithm: LIGHTNING_ALGORITHM.to_owned(),
            model_identity: LIGHTNING_MODEL_IDENTITY.to_owned(),
            checkpoint: CheckpointLayout {
                block_size: LIGHTNING_CHECKPOINT_BLOCK_SIZE,
                physical_kv_page_size: LIGHTNING_PHYSICAL_KV_PAGE_SIZE,
            },
            served_gamma: LIGHTNING_SERVED_GAMMA,
            num_drafts: LIGHTNING_NUM_DRAFTS,
            taps: LIGHTNING_TAPS.to_vec(),
            attention: AttentionLayout {
                causal: true,
                use_swa: true,
                swa_window: LIGHTNING_SWA_WINDOW,
                attention_sink_bias: true,
            },
            markov: MarkovLayout {
                rank: LIGHTNING_MARKOV_RANK,
                w1_present: true,
                w2_present: true,
            },
            bonus: BonusLayout {
                bonus_anchor: true,
                sample_from_anchor: false,
            },
            confidence: ConfidenceLayout {
                head_present: false,
                adaptive: false,
            },
            kv: KvLayout {
                target: KvDtype::Fp8,
                drafter: KvDtype::Bf16,
            },
            parallelism: ParallelismLayout {
                tensor_parallel: LIGHTNING_TP,
                expert_parallel: LIGHTNING_EP,
            },
        }
    }

    pub fn validate(&self) -> Result<(), LightningDsparkContractError> {
        check_identity("algorithm", &self.algorithm, LIGHTNING_ALGORITHM)?;
        check_identity(
            "model_identity",
            &self.model_identity,
            LIGHTNING_MODEL_IDENTITY,
        )?;
        check_usize(
            "checkpoint.block_size",
            self.checkpoint.block_size,
            LIGHTNING_CHECKPOINT_BLOCK_SIZE,
        )?;
        check_usize(
            "physical_kv.page_size",
            self.checkpoint.physical_kv_page_size,
            LIGHTNING_PHYSICAL_KV_PAGE_SIZE,
        )?;
        check_usize("served_gamma", self.served_gamma, LIGHTNING_SERVED_GAMMA)?;
        check_usize("num_drafts", self.num_drafts, LIGHTNING_NUM_DRAFTS)?;
        if self.taps != LIGHTNING_TAPS {
            return Err(LightningDsparkContractError::Taps {
                expected: LIGHTNING_TAPS,
                found: self.taps.clone(),
            });
        }
        check_bool("attention.causal", self.attention.causal, true)?;
        check_bool("attention.use_swa", self.attention.use_swa, true)?;
        check_usize(
            "attention.swa_window",
            self.attention.swa_window,
            LIGHTNING_SWA_WINDOW,
        )?;
        check_bool(
            "attention.attention_sink_bias",
            self.attention.attention_sink_bias,
            true,
        )?;
        check_usize("markov.rank", self.markov.rank, LIGHTNING_MARKOV_RANK)?;
        check_presence("markov.w1_present", self.markov.w1_present, true)?;
        check_presence("markov.w2_present", self.markov.w2_present, true)?;
        check_bool("bonus.bonus_anchor", self.bonus.bonus_anchor, true)?;
        check_bool(
            "bonus.sample_from_anchor",
            self.bonus.sample_from_anchor,
            false,
        )?;
        check_presence(
            "confidence.head_present",
            self.confidence.head_present,
            false,
        )?;
        check_bool("confidence.adaptive", self.confidence.adaptive, false)?;
        check_kv("kv.target", self.kv.target, KvDtype::Fp8)?;
        check_kv("kv.drafter", self.kv.drafter, KvDtype::Bf16)?;
        check_usize(
            "parallelism.tensor_parallel",
            self.parallelism.tensor_parallel,
            LIGHTNING_TP,
        )?;
        check_usize(
            "parallelism.expert_parallel",
            self.parallelism.expert_parallel,
            LIGHTNING_EP,
        )?;
        Ok(())
    }
}

fn check_identity(
    field: &'static str,
    found: &str,
    expected: &'static str,
) -> Result<(), LightningDsparkContractError> {
    if found == expected {
        Ok(())
    } else {
        Err(LightningDsparkContractError::Identity {
            field,
            expected,
            found: found.to_owned(),
        })
    }
}

fn check_usize(
    field: &'static str,
    found: usize,
    expected: usize,
) -> Result<(), LightningDsparkContractError> {
    if found == expected {
        Ok(())
    } else {
        Err(LightningDsparkContractError::ExactUsize {
            field,
            expected,
            found,
        })
    }
}

fn check_bool(
    field: &'static str,
    found: bool,
    expected: bool,
) -> Result<(), LightningDsparkContractError> {
    if found == expected {
        Ok(())
    } else {
        Err(LightningDsparkContractError::ExactBool {
            field,
            expected,
            found,
        })
    }
}

fn check_presence(
    field: &'static str,
    found: bool,
    expected: bool,
) -> Result<(), LightningDsparkContractError> {
    if found == expected {
        Ok(())
    } else {
        Err(LightningDsparkContractError::WeightPresence {
            field,
            expected_present: expected,
            found_present: found,
        })
    }
}

fn check_kv(
    field: &'static str,
    found: KvDtype,
    expected: KvDtype,
) -> Result<(), LightningDsparkContractError> {
    if found == expected {
        Ok(())
    } else {
        Err(LightningDsparkContractError::KvDtype {
            field,
            expected,
            found,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LightningDsparkContractError {
    Identity {
        field: &'static str,
        expected: &'static str,
        found: String,
    },
    ExactUsize {
        field: &'static str,
        expected: usize,
        found: usize,
    },
    ExactBool {
        field: &'static str,
        expected: bool,
        found: bool,
    },
    Taps {
        expected: [usize; 6],
        found: Vec<usize>,
    },
    WeightPresence {
        field: &'static str,
        expected_present: bool,
        found_present: bool,
    },
    KvDtype {
        field: &'static str,
        expected: KvDtype,
        found: KvDtype,
    },
}

impl fmt::Display for LightningDsparkContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "Lightning DSpark contract rejected {field}={found:?}; expected {expected:?}"
            ),
            Self::ExactUsize {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "Lightning DSpark contract rejected {field}={found}; expected {expected}"
            ),
            Self::ExactBool {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "Lightning DSpark contract rejected {field}={found}; expected {expected}"
            ),
            Self::Taps { expected, found } => write!(
                formatter,
                "Lightning DSpark contract rejected taps={found:?}; expected ordered taps {expected:?}"
            ),
            Self::WeightPresence {
                field,
                expected_present,
                found_present,
            } => write!(
                formatter,
                "Lightning DSpark contract rejected {field}={}; expected {}",
                presence(*found_present),
                presence(*expected_present),
            ),
            Self::KvDtype {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "Lightning DSpark contract rejected {field}={found}; expected {expected}"
            ),
        }
    }
}

fn presence(value: bool) -> &'static str {
    if value { "present" } else { "absent" }
}

impl std::error::Error for LightningDsparkContractError {}
