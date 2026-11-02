// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN projection format, and the opt-in FP8 one (`ATLAS_QWEN4EXP_FP8_GDN`).
//!
//! The checkpoint ships `in_proj_qkv`/`in_proj_z`/`out_proj` BF16 on all 36
//! GDN layers, and so does Qwen's own FP8 release (`Qwen/Qwen3.8-Flash-Next-
//! FP8` lists all three in `modules_to_not_convert`). At C=1 decode their BF16
//! GEMVs are bandwidth-bound: 72 launches a token, ~14 ms of a ~39 ms token on
//! the TP2 pair. `ATLAS_QWEN4EXP_BF16_GDN=0` halves that again with NVFP4 at a
//! measured cost in fidelity; this is the middle option.
//!
//! **FP8**: E4M3 weights with one FP32 scale per 128x128 block (scale =
//! block absmax / 448), BF16 activations, FP32 accumulation — the weight
//! format of Qwen's FP8 releases that do convert these projections
//! (Qwen3-Next-80B-A3B-FP8, Qwen3.6-35B-A3B-FP8: `weight_block_size [128,
//! 128]`), quantized at load from this rank's BF16 shard by the block
//! quantizer LongCat's experts use, and read by the existing W8A16 decode
//! kernels (`w8a16_gemv`, and `w8a16_gemv_batch4` for verify and small
//! batches, which streams the weight once for every row and is bit-identical
//! per row to `w8a16_gemv`). Measured on synthetic weights of the real shapes
//! (`scripts/dev/qwen4exp_gdn_fp8_bench.cu`): 1.8-2.1x the BF16 GEMVs, output
//! error vs the BF16 GEMV rel-L2 2.6% (cos 0.99966) against NVFP4's 9.3%
//! (cos 0.9956), and each TP2 rank's bytes and scales identical to the
//! matching slice of the TP1 quantization.
//!
//! `=1` keeps the BF16 copies for prefill (cuBLASLt runs those GEMMs ~2.3x
//! faster than `w8a16_gemm_pipelined` does FP8) and costs 1 byte a weight on
//! top: +28.9 MB a layer per TP2 rank.
//! `=full` frees the BF16 copies (-57.7 MB a layer at TP1, where there is no
//! headroom for `=1`) and prefill reads the FP8 copy too.
//!
//! Why BF16 is the default — measured on the NVFP4 pack, both arms, same
//! prompt, util 0.85 / 16K / bf16 KV:
//!
//! ```text
//!                      requantized NVFP4      BF16 (default)
//! layer construction   7.43 GB / 154.8 MB/L   1.39 GB / 28.9 MB/L
//! attn/GDN arms        7.34 GB                1.32 GB
//! pre-KV               95.8 GB                90.0 GB
//! KV budget            3.9 GB / 172144 tok    9.7 GB / 424464 tok
//! decode               2.207 tok/s            2.188 tok/s
//! ```
//!
//! 6.04 GB back for ~1% of decode. GDN weight bandwidth is simply not
//! what bounds decode here at C=1 — something else dominates — so the
//! usual w4a16-is-faster argument does not apply yet. Revisit if that
//! changes.
//!
//! And it is not only a memory lever: ONLY the routed experts are
//! quantized in this checkpoint. The GDN projections ship BF16, so
//! requantizing them was a lossy round trip we chose, on 36 of 48
//! layers. `=0` opts back into it for A/B. `ATLAS_QWEN4EXP_FP8_GDN` is
//! the FP8 middle option (`gdn_fp8`).

use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::tp_shard::TpGdnDims;
use crate::weight_loader::qwen35::load_layers::linear_attn_arms::dense_bf16_layer;
use crate::weight_map::{DenseWeight, Nvfp4Variant, free_loader_source};

/// The format of the GDN `in_proj_qkvz` / `out_proj` weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnProjections {
    /// BF16 as shipped. The default, for two reasons. MEMORY: requantizing
    /// is where this model's build spent 7.34 GB (152.8 MB a layer,
    /// measured; the MoE costs nothing, its experts upload as shipped).
    /// PRECISION: only the routed experts ship quantized, so BF16 -> NVFP4
    /// would be a lossy round trip chosen, not forced, on 36 of 48 layers.
    /// (FP4-Local's MXFP8 GDN projections bypass this enum: `gdn_mx8`.)
    Bf16,
    /// Requantized to NVFP4 (`ATLAS_QWEN4EXP_BF16_GDN=0`).
    Nvfp4,
    /// A block-scaled FP8 copy for decode, BF16 kept for prefill
    /// (`ATLAS_QWEN4EXP_FP8_GDN=1`).
    Fp8Decode,
    /// Block-scaled FP8 for decode and prefill, BF16 freed (`=full`).
    Fp8Full,
}

impl GdnProjections {
    pub(crate) fn from_env() -> Result<Self> {
        let var = |name| std::env::var(name).ok();
        Self::parse(
            var("ATLAS_QWEN4EXP_BF16_GDN").as_deref(),
            var("ATLAS_QWEN4EXP_FP8_GDN").as_deref(),
            Self::platform_default(),
        )
    }

    /// What an unset `ATLAS_QWEN4EXP_FP8_GDN` means. On gfx1151 it is the
    /// decode-only FP8 copy: Flash-Next decode +12..21% with MTP and +17%
    /// plain (framework f88), ppl dense/above-bound ratio 1.0000 and needles
    /// 12/12 (f91), agentic-webserver 47/50 vs 48/50 BF16 (f93/f92). It costs
    /// the FP8 copy's memory on top of the BF16 kept for prefill. `=0` opts
    /// out. Elsewhere BF16 stays the default (unmeasured there).
    fn platform_default() -> Self {
        if cfg!(atlas_hip) {
            Self::Fp8Decode
        } else {
            Self::Bf16
        }
    }

    fn parse(bf16_gdn: Option<&str>, fp8_gdn: Option<&str>, unset: Self) -> Result<Self> {
        // Unset defers to `unset` unless the NVFP4 requantization was asked for.
        let unset = if bf16_gdn == Some("0") {
            None
        } else {
            Some(unset)
        }
        .filter(|u| matches!(u, Self::Fp8Decode | Self::Fp8Full));
        let fp8 = match fp8_gdn {
            None => unset,
            Some("" | "0") => None,
            Some("1") => Some(Self::Fp8Decode),
            Some("full") => Some(Self::Fp8Full),
            Some(v) => bail!("ATLAS_QWEN4EXP_FP8_GDN={v}: expected 0, 1 or full"),
        };
        // `!= "0"` is the shipped reading of ATLAS_QWEN4EXP_BF16_GDN.
        let nvfp4 = bf16_gdn == Some("0");
        Ok(match (fp8, nvfp4) {
            (Some(_), true) => bail!(
                "ATLAS_QWEN4EXP_FP8_GDN and ATLAS_QWEN4EXP_BF16_GDN=0 each pick the GDN \
                 projection format (FP8 and NVFP4); set one of them"
            ),
            (Some(fp8), false) => fp8,
            (None, true) => Self::Nvfp4,
            (None, false) => Self::Bf16,
        })
    }

    /// Startup-parity word for `ATLAS_QWEN4EXP_FP8_GDN`: 0 off, 1 `=1`, 2 `=full`.
    pub(crate) fn fp8_word(self) -> u64 {
        match self {
            Self::Bf16 | Self::Nvfp4 => 0,
            Self::Fp8Decode => 1,
            Self::Fp8Full => 2,
        }
    }

    pub(super) fn describe(self) -> &'static str {
        match self {
            Self::Bf16 => {
                "BF16 as shipped (no runtime NVFP4 requantization), or native MXFP8 where \
                 the checkpoint ships MXFP8"
            }
            Self::Nvfp4 => "requantized to NVFP4 (ATLAS_QWEN4EXP_BF16_GDN=0)",
            Self::Fp8Decode => {
                "BF16 + a 128x128 block-scaled FP8 copy for decode (ATLAS_QWEN4EXP_FP8_GDN=1)"
            }
            Self::Fp8Full => {
                "128x128 block-scaled FP8, decode and prefill (ATLAS_QWEN4EXP_FP8_GDN=full)"
            }
        }
    }
}

/// The BF16 GDN build plus its FP8 copy; see the module docs.
#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    keep_bf16_prefill: bool,
) -> Result<Box<dyn TransformerLayer>> {
    // A rank's block scales are the TP=1 ones only if its slice boundaries sit
    // on the 128 grid: QKVZ shards rows per [Q|K|V|Z] segment, out_proj
    // columns. Checked, not assumed.
    const BS: usize = 128;
    let h = config.hidden_size;
    let dims = TpGdnDims::from_config(config);
    ensure!(
        h.is_multiple_of(BS)
            && dims.local_key_dim().is_multiple_of(BS)
            && dims.local_value_dim().is_multiple_of(BS),
        "ATLAS_QWEN4EXP_FP8_GDN L{layer_idx}: hidden {h}, local key dim {} and local value \
         dim {} must be multiples of {BS} for TP-exact block scales",
        dims.local_key_dim(),
        dims.local_value_dim(),
    );
    // That lever forces the QKVZ prefill GEMM onto the BF16 copy `=full` frees.
    ensure!(
        keep_bf16_prefill || std::env::var("ATLAS_GDN_BF16_WEIGHTS").as_deref() != Ok("1"),
        "ATLAS_QWEN4EXP_FP8_GDN=full frees the BF16 GDN weights ATLAS_GDN_BF16_WEIGHTS=1 \
         reads; use ATLAS_QWEN4EXP_FP8_GDN=1"
    );
    let mut layer = dense_bf16_layer(
        layer_idx,
        store,
        lp,
        gpu,
        variant,
        config,
        h,
        input_norm,
        post_attn_norm,
        ffn,
    )?;
    let out_key = format!("{lp}.linear_attn.out_proj.weight");
    let quantize_k = gpu.kernel(
        "quantize_bf16_to_fp8_blockscaled",
        "quantize_bf16_to_fp8_blockscaled",
    )?;
    let stream = gpu.default_stream();
    if let Some((qkvz, out_proj)) =
        layer.quantize_dense_gdn_to_fp8(gpu, config, quantize_k, stream, keep_bf16_prefill)?
    {
        // QKVZ is a loader-made buffer (`gpu_concat_rows`, or its shard);
        // out_proj is the store's own tensor at TP=1.
        gpu.free(qkvz.weight)?;
        free_loader_source(store, gpu, &out_key, out_proj)?;
    }
    Ok(Box::new(layer))
}

#[cfg(test)]
mod tests {
    use super::GdnProjections::{self, *};

    #[test]
    fn unset_is_the_shipped_bf16_and_bf16_gdn_keeps_its_reading() {
        assert_eq!(GdnProjections::parse(None, None, Bf16).unwrap(), Bf16);
        assert_eq!(
            GdnProjections::parse(Some("1"), Some("0"), Bf16).unwrap(),
            Bf16
        );
        assert_eq!(GdnProjections::parse(Some("0"), None, Bf16).unwrap(), Nvfp4);
        assert_eq!(
            GdnProjections::parse(Some("0"), Some(""), Bf16).unwrap(),
            Nvfp4
        );
    }

    #[test]
    fn hip_default_is_fp8_decode_and_both_opt_outs_still_work() {
        let p = |b, f| GdnProjections::parse(b, f, Fp8Decode).unwrap();
        assert_eq!(p(None, None), Fp8Decode);
        assert_eq!(p(Some("1"), None), Fp8Decode);
        assert_eq!(p(None, Some("0")), Bf16);
        assert_eq!(p(None, Some("full")), Fp8Full);
        // Asking for the NVFP4 requantization is not a conflict with a default.
        assert_eq!(p(Some("0"), None), Nvfp4);
    }

    #[test]
    fn fp8_values_and_parity_words() {
        let one = GdnProjections::parse(None, Some("1"), Bf16).unwrap();
        let full = GdnProjections::parse(Some("1"), Some("full"), Bf16).unwrap();
        assert_eq!(one, Fp8Decode);
        assert_eq!(full, Fp8Full);
        assert_eq!(
            [Bf16, Nvfp4, one, full].map(GdnProjections::fp8_word),
            [0, 0, 1, 2]
        );
    }

    #[test]
    fn refuses_a_stray_value_and_the_nvfp4_combination() {
        let e = GdnProjections::parse(None, Some("true"), Bf16).unwrap_err();
        assert!(e.to_string().contains("expected 0, 1 or full"), "{e}");
        let e = GdnProjections::parse(Some("0"), Some("1"), Bf16).unwrap_err();
        assert!(e.to_string().contains("set one of them"), "{e}");
    }
}
