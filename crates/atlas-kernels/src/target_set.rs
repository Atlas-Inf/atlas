// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::target::KernelTarget;
use super::{ModelBehavior, ModelTypeMatch, SamplingPresets};

/// Drafter configuration for models paired with an speculative proposer.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    /// HuggingFace id (or local path) of the drafter checkpoint.
    pub draft_model: &'static str,
    /// Block size γ (parallel draft tokens per step). Defaults to 16.
    pub gamma: usize,
    /// Drafter sliding-window size in tokens. 0 = full attention.
    pub window_size: usize,
    /// Token id used to fill the γ "to-be-predicted" positions during
    /// drafter forward. From the drafter's `dflash_config.mask_token_id`.
    pub mask_token_id: u32,
    /// Target-side layer indices to capture intermediate hidden states from
    /// (shallow-to-deep). The drafter's `fc` projection consumes the stack
    /// of these hiddens. From the drafter's `dflash_config.target_layer_ids`.
    pub target_layer_ids: &'static [usize],
}

/// Kernel modules hyperoptimized for a specific (H, M_q) target.
///
/// Each blob is the compiled kernel for one module, emitted uniformly as
/// `&'static [u8]` by build.rs (`include_bytes!`). NVIDIA PTX is ASCII
/// text but valid as bytes; SCALE/AMD and Metal produce binary objects.
/// The runtime registry sniffs text-vs-binary per blob at load time.
pub struct TargetPtxSet {
    pub target: KernelTarget,
    pub modules: Vec<(&'static str, &'static [u8])>,
    pub sampling: SamplingPresets,
    pub behavior: ModelBehavior,
    pub model_type_matches: Vec<ModelTypeMatch>,
    /// `[model] match_names` needles from MODEL.toml — case-insensitive
    /// substrings of the checkpoint reference (HF id / `--model-name` /
    /// resolved model dir) that identify checkpoints THIS target serves.
    /// Consulted only to break a tie when several targets declare the same
    /// `(model_type, hidden_size)` (e.g. qwen3.6-27b vs qwen3.8-27b, whose
    /// configs are bit-identical); see [`resolve::resolve_target`]. Empty
    /// for targets that never collide — `build.rs` panics if a colliding
    /// target omits them.
    pub match_names: &'static [&'static str],
    /// DFlash drafter pairing for this model. `None` when the MODEL.toml has
    /// no `[dflash]` section. Consumed by spark-server when `--dflash` is
    /// set without an explicit `--draft-model` flag.
    pub dflash: Option<DflashConfig>,
    /// `(module, kernel)` pairs this model's kernel files DROPPED by shadowing
    /// their `common/` namesakes — the kernel exists in `common/` but this
    /// model's fork of the file does not define it, so it is not compiled here.
    ///
    /// Shadowing is whole-file, so a fork that predates a kernel added to
    /// `common/` silently loses it: `try_kernel` returns handle 0 and whatever
    /// depends on it fails CLOSED. The startup audit joins this against the
    /// kernels the model actually looked up, which separates the two classes of
    /// missing kernel — dropped-by-fork (a build defect) from
    /// never-built-for-this-architecture (expected, e.g. MLA on a Qwen model).
    pub shadowed_dropped: &'static [(&'static str, &'static str)],
    /// `(module, kernel)` lookups this model's dispatch may issue and fail to
    /// resolve WITHOUT that being an error, declared in the model's MODEL.toml
    /// `[expected_absent]` with a mandatory stated reason per entry.
    ///
    /// The boot audit (`kernel_audit::classify_failures`) fails CLOSED on every
    /// unresolved lookup that is not in this list, so the list is the entire
    /// difference between "this model is known to run this way" and "nobody has
    /// looked". It is TRANSITIONAL: the right fix for a lookup that can never
    /// resolve is to gate it on config so it is never issued (see
    /// `qwen3_attention::init_arch_gates`), which removes it from here.
    pub expected_absent: &'static [(&'static str, &'static str)],
}
