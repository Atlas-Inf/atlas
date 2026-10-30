// SPDX-License-Identifier: AGPL-3.0-only
//! Atlas-native KDA prefill, compiled from credited MIT device kernels.
//! FP32 persistent state is retained in the engine layout; the tensor-core
//! algorithm uses BF16 internally and remains explicitly opt-in pending quality
//! qualification. Decode and speculative verification keep native FP32 kernels.
//!
//! `ATLAS_KDA_FLASH_LIBRARY` (SparkGLM's switch for its FlashKDA bridge)
//! selects SparkGLM 0.2.0's FP32-recurrence FlashKDA instead
//! ([`sparkglm`]), compiled into spark-runtime; the path is not opened.
use anyhow::{Result, bail, ensure};
use spark_runtime::glm_kda;
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;

#[path = "flash_prefill_sparkglm.rs"]
mod sparkglm;

pub(super) enum FlashPrefill {
    /// Atlas's H32 operator (BF16 recurrent tiles).
    Native,
    /// SparkGLM's bridge (FP32 recurrence) at 16 or 32 local heads.
    SparkGlm(sparkglm::SparkGlm),
}

/// The KDA prefill operator the environment asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Choice {
    Off,
    Native,
    /// `planes`: `ATLAS_GLM_KDA_FLASH_PLANES=1`, the convolution writes the
    /// bridge's q / k / v planes (the same values, no packed copy).
    SparkGlm {
        planes: bool,
    },
}

impl Choice {
    /// The startup-parity word: every rank must run the same arithmetic.
    pub(crate) fn word(self) -> u64 {
        match self {
            Self::Off => 0,
            Self::Native => 1,
            Self::SparkGlm { planes } => 2 + planes as u64,
        }
    }
}

/// `ATLAS_KDA_FLASH_PREFILL` turns the operator on; a nonempty
/// `ATLAS_KDA_FLASH_LIBRARY` picks SparkGLM's, whose
/// `ATLAS_GLM_KDA_FLASH_PLANES=1` it alone reads (as SparkGLM did).
fn choose(prefill: Option<&str>, library: Option<&str>, planes: Option<&str>) -> Result<Choice> {
    if !enabled(prefill)? {
        return Ok(Choice::Off);
    }
    Ok(match library {
        Some(path) if !path.is_empty() => Choice::SparkGlm {
            planes: planes == Some("1"),
        },
        _ => Choice::Native,
    })
}

pub(crate) fn requested() -> Result<Choice> {
    let var = |name| std::env::var(name).ok();
    choose(
        var("ATLAS_KDA_FLASH_PREFILL").as_deref(),
        var("ATLAS_KDA_FLASH_LIBRARY").as_deref(),
        var("ATLAS_GLM_KDA_FLASH_PLANES").as_deref(),
    )
}

fn enabled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => bail!("ATLAS_KDA_FLASH_PREFILL must be 0 or 1, got {value:?}"),
    }
}

/// Initial bounded native operator range; larger prefills carry state across pieces.
const PIECE_ROWS: std::ops::RangeInclusive<usize> = glm_kda::MIN_ROWS..=glm_kda::MAX_ROWS;

pub(super) fn eligible(tokens: usize, decode: bool, graph_capture: bool) -> bool {
    !decode && !graph_capture && tokens >= *PIECE_ROWS.start()
}

/// Near-equal consecutive pieces within `PIECE_ROWS` covering `tokens`
/// rows; the recurrent state carries across calls in place.
fn pieces(tokens: usize) -> impl Iterator<Item = (usize, usize)> {
    let n = tokens.div_ceil(*PIECE_ROWS.end());
    let piece = tokens.div_ceil(n);
    (0..n).map(move |i| (i * piece, piece.min(tokens - i * piece)))
}

impl FlashPrefill {
    pub(super) fn load(heads: usize, dim: usize, lower: f32) -> Result<Option<Self>> {
        Self::load_with(heads, dim, lower, requested()?, glm_kda::preload)
    }

    fn load_with(
        heads: usize,
        dim: usize,
        lower: f32,
        choice: Choice,
        preload: impl FnOnce() -> std::result::Result<(), glm_kda::Error>,
    ) -> Result<Option<Self>> {
        match choice {
            Choice::Off => return Ok(None),
            Choice::SparkGlm { planes } => {
                return sparkglm::SparkGlm::load(heads, dim, lower, planes)
                    .map(|s| Some(Self::SparkGlm(s)));
            }
            Choice::Native => {}
        }
        ensure!(
            heads == 32 && dim == 128 && lower == -5.0,
            "Atlas native KDA supports only H32/D128/lower_bound=-5"
        );

        // Loading layers precedes the factory's KV free-memory snapshot.
        // Resolve CUDA code residency here so the first request cannot grow it.
        preload()?;
        tracing::info!(
            "Atlas native KDA prefill enabled for H32/D128 pieces 2048..=4100; FP32 storage/BF16 internal recurrence"
        );
        Ok(Some(Self::Native))
    }

    /// Whether [`Self::forward_planes`] serves: the convolution then writes
    /// q / k / v planes for it.
    pub(super) fn takes_planes(&self) -> bool {
        matches!(self, Self::SparkGlm(s) if s.planes)
    }

    /// [`Self::forward`] over q / k / v planes (`3 x [tokens, heads * 128]`
    /// at `planes`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_planes(
        &self,
        planes: DevicePtr,
        gate: DevicePtr,
        beta: DevicePtr,
        a: DevicePtr,
        bias: DevicePtr,
        state: DevicePtr,
        output: DevicePtr,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Self::SparkGlm(s) = self else {
            bail!("Native KDA takes interleaved q / k / v, not planes");
        };
        ensure!(
            eligible(tokens, false, ctx.graph_capture),
            "Unsupported native KDA invocation"
        );
        s.forward_planes(
            [planes, gate, beta, a, bias, state, output],
            tokens,
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        qkv: DevicePtr,
        gate: DevicePtr,
        beta: DevicePtr,
        a: DevicePtr,
        bias: DevicePtr,
        state: DevicePtr,
        output: DevicePtr,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            eligible(tokens, false, ctx.graph_capture),
            "Unsupported native KDA invocation"
        );
        if let Self::SparkGlm(s) = self {
            return s.forward(
                [qkv, gate, beta, a, bias, state, output],
                tokens,
                ctx,
                stream,
            );
        }
        let plane_row = 32 * 128 * 2;
        for (row, rows) in pieces(tokens) {
            self.forward_piece(
                qkv.offset(row * 3 * plane_row),
                gate.offset(row * plane_row),
                beta.offset(row * 32 * 2),
                a,
                bias,
                state,
                output.offset(row * plane_row),
                rows,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_piece(
        &self,
        qkv: DevicePtr,
        gate: DevicePtr,
        beta: DevicePtr,
        a: DevicePtr,
        bias: DevicePtr,
        state: DevicePtr,
        output: DevicePtr,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(PIECE_ROWS.contains(&tokens), "Unsupported native KDA piece");
        let params = glm_kda::Params::new(
            qkv.0,
            gate.0,
            beta.0,
            a.0,
            bias.0,
            state.0,
            output.0,
            ctx.buffers.expert_gate_out().0,
            ctx.buffers.sizes().expert_gate_out,
            tokens,
            stream,
        )?;
        // SAFETY: the layer owns these device allocations and their shapes;
        // Params checks byte ranges, alignment and scratch capacity. They stay
        // live on the caller's stream through recurrence and the following MoE.
        // Never retry a failed launch: recurrent state may already be updated.
        unsafe { glm_kda::launch(&params) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enabled_load_preloads_before_returning_the_operator() {
        let calls = std::cell::Cell::new(0);
        let loaded = FlashPrefill::load_with(32, 128, -5.0, Choice::Native, || {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap();
        assert!(loaded.is_some());
        assert_eq!(calls.get(), 1);
    }
    #[test]
    fn preload_errors_abort_load_and_disabled_or_invalid_shapes_never_touch_cuda() {
        let result = FlashPrefill::load_with(32, 128, -5.0, Choice::Native, || {
            Err(glm_kda::Error::Preload(701))
        });
        let error = result
            .err()
            .expect("CUDA preload failure must stop model construction");
        assert_eq!(
            error.downcast_ref::<glm_kda::Error>(),
            Some(&glm_kda::Error::Preload(701))
        );
        let disabled = FlashPrefill::load_with(32, 128, -5.0, Choice::Off, || {
            panic!("disabled operator must not touch CUDA")
        })
        .unwrap();
        assert!(disabled.is_none());
        assert!(
            FlashPrefill::load_with(16, 128, -5.0, Choice::Native, || {
                panic!("invalid shape must fail before CUDA")
            })
            .is_err()
        );
    }
    #[test]
    fn configuration_is_explicit_and_strict() {
        assert!(!enabled(None).unwrap());
        assert!(!enabled(Some("0")).unwrap());
        assert!(enabled(Some("1")).unwrap());
        assert!(enabled(Some("true")).is_err());
    }
    #[test]
    fn sparkglm_library_switch_selects_its_bridge_and_planes() {
        let lib = Some("/opt/atlas/native/libatlas_mango_flash.so");
        assert_eq!(choose(None, lib, Some("1")).unwrap(), Choice::Off);
        assert_eq!(choose(Some("0"), lib, None).unwrap(), Choice::Off);
        assert_eq!(choose(Some("1"), None, Some("1")).unwrap(), Choice::Native);
        assert_eq!(choose(Some("1"), Some(""), None).unwrap(), Choice::Native);
        assert_eq!(
            choose(Some("1"), lib, None).unwrap(),
            Choice::SparkGlm { planes: false }
        );
        assert_eq!(
            choose(Some("1"), lib, Some("true")).unwrap(),
            Choice::SparkGlm { planes: false }
        );
        assert_eq!(
            choose(Some("1"), lib, Some("1")).unwrap(),
            Choice::SparkGlm { planes: true }
        );
        assert!(choose(Some("yes"), lib, None).is_err());
        let words = [
            Choice::Off,
            Choice::Native,
            Choice::SparkGlm { planes: false },
            Choice::SparkGlm { planes: true },
        ]
        .map(Choice::word);
        assert_eq!(words, [0, 1, 2, 3]);
    }
    #[test]
    fn sparkglm_load_checks_shape_then_build_and_never_preloads_native() {
        let never = || -> std::result::Result<(), glm_kda::Error> {
            panic!("SparkGLM's bridge must not preload Atlas's operator")
        };
        for (heads, dim, lower) in [(8, 128, -5.0), (32, 64, -5.0), (16, 128, -4.0)] {
            let error = FlashPrefill::load_with(
                heads,
                dim,
                lower,
                Choice::SparkGlm { planes: false },
                never,
            )
            .err()
            .expect("unqualified shape");
            assert!(error.to_string().contains("H16/H32"), "{error}");
        }
        for heads in [16, 32] {
            let loaded =
                FlashPrefill::load_with(heads, 128, -5.0, Choice::SparkGlm { planes: true }, never);
            if glm_kda::available() {
                assert!(loaded.unwrap().unwrap().takes_planes());
            } else {
                let error = loaded.err().expect("uncompiled bridge must stop startup");
                assert_eq!(
                    error.downcast_ref::<glm_kda::Error>(),
                    Some(&glm_kda::Error::Unavailable)
                );
            }
        }
    }
    #[test]
    fn qualification_excludes_decode_capture_and_short_tails() {
        for n in [2048, 4100, 4101, 8196, 16388] {
            assert!(eligible(n, false, false));
        }
        for n in [0, 1, 3, 2047] {
            assert!(!eligible(n, false, false));
        }
        assert!(!eligible(4096, true, false));
        assert!(!eligible(4096, false, true));
    }
    #[test]
    fn pieces_tile_rows_within_the_native_range() {
        for tokens in [
            2048, 4096, 4100, 4101, 4104, 6000, 8192, 8196, 12292, 16388, 65520,
        ] {
            let mut next = 0;
            for (row, rows) in pieces(tokens) {
                assert_eq!(row, next);
                assert!(PIECE_ROWS.contains(&rows), "{tokens}: piece {rows}");
                next += rows;
            }
            assert_eq!(next, tokens);
        }
    }
}
