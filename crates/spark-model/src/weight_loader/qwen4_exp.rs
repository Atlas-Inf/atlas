// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3.8-Flash-Next` (`qwen4_exp`) weight loader. Port tracked in Avarok
//! #753.
//!
//! **This serves.** Every mechanism the module doc below once listed as
//! missing has shipped; the file is annotated with what each one is validated
//! against, because the failure mode for all of them is fluent-and-wrong.
//!
//! * **mHC low-rank highway** — all 48 layers.
//!   `ops/hyper_connection_lowrank_tests.rs` runs the four entry points on the
//!   GPU against a golden taken from the real `Qwen4ExpTextGatedResidual` on
//!   real checkpoint weights: cos 0.999998, `hc_post` bit-exact, and the two
//!   defects the harness exists to catch (a global instead of grouped RMS, and
//!   the dropped offset-from-1) fail it by three orders of magnitude.
//! * **PLE n-gram injection** at model layer 1 — wired, with the ~320M-row
//!   table read by row off NVMe (`NgramRowCache::open_segmented`; the 128
//!   shards are NOT contiguous, so one base offset would read wrong-but-valid
//!   rows silently). `ATLAS_QWEN4EXP_NO_PLE=1` now DISABLES a mechanism that
//!   is present, for bisecting the mHC spine, and says so loudly.
//! * **The QSA indexer** — decode-side selection plus per-query prefill
//!   selection, both parity-gated at T=2200 where selection actively prunes.
//!   Provably inert at or below `indexer_budget`, so short contexts are exact
//!   rather than approximate.
//! * **Batched / multi-sequence decode** — honoured. Per-sequence PLE and QSA
//!   state ride the layer states, and the earlier clamp-to-1 is lifted.
//!
//! * **MTP** — honoured, through `mtp::load_qwen4exp_mtp_module` rather than
//!   the family `load_mtp_weights` below, which still returns `None` because
//!   this block is not the `MtpWeights` shape. It loads, proposes and verifies
//!   on the mHC highway; it is simply slower than plain decode on this model
//!   (see `kernels/gb10/qwen3.8-flash-next/MODEL.toml` for the numbers).
//!
//! Still refused: the **stacked expert layout**, which neither published
//! release uses.
//!
//! WHY THIS IS MOSTLY qwen35's LOADER. Qwen3.8-Flash-Next and Qwen3.6-35B-A3B
//! share far more than the version numbers suggest: 3:1 GDN/full-attention
//! interleave, MoE with a shared expert, gated attention, mRoPE, a ViT tower,
//! vocab 248320, rope_theta 1e7, head_dim 256, partial rotary 0.25, and the
//! same GDN key geometry. Critically, `load_ssm_qwen35` already reads
//! `in_proj_qkv` and `in_proj_z` as SEPARATE tensors and concatenates them —
//! which is exactly this model's layout, not a coincidence to be re-derived.
//! So the GDN and full-attention arms are called directly, with
//! `config.weight_prefix = "model.language_model"` making
//! `config.layer_prefix(i)` yield the real keys.
//!
//! WHAT IS GENUINELY DIFFERENT, and why each needs care:
//!
//! 1. **There are no per-layer norms.** No `input_layernorm`, no
//!    `post_attention_layernorm`, no final `model.norm`. Normalization lives
//!    inside the hyper-connection blocks as `hc_norm [hc_mult*hidden]`, and
//!    the model-level `hyper_connection_mixer` — which collapses the streams
//!    back to one before `lm_head` — carries the final norm. A loader that
//!    "helpfully" defaults these would be inventing weights.
//! 2. **mHC is 4 residual streams**, mixed low-rank (rank 320). Atlas's mHC
//!    plumbing is DeepSeek-V4's, whose mixer is Sinkhorn-normalized — same
//!    stream layout, different math.
//! 3. **A QSA indexer** on the 12 full-attention layers.
//! 4. **PLE n-gram injection** at one layer, off a ~320M-row table served
//!    from NVMe rather than resident.

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

// `attach`, not `aux`: `aux` is a RESERVED DEVICE NAME on Windows (with con,
// prn, nul, com1-9, lpt1-9), and the reservation applies even with an
// extension — so `aux.rs` cannot be checked out at all there. The Windows
// release build failed with `error: invalid path` from git itself, before any
// compiler ran. `attach` is also the better name: the module is the three
// per-layer attach helpers.
mod attach;
mod ffn;
mod hc;
mod mtp;
mod ple;
mod probe;

pub use mtp::{Qwen4ExpMtpModule, load_qwen4exp_mtp_module};
pub use probe::audit_namespace;

#[cfg(all(test, feature = "cuda"))]
mod shard_layout;
#[cfg(all(test, feature = "cuda"))]
pub use shard_layout::ple_shard_layout;

pub struct Qwen4ExpWeightLoader;

impl ModelWeightLoader for Qwen4ExpWeightLoader {
    fn supports_tp(&self) -> bool {
        // Not attempted. mHC would need the stream buffer sharded alongside
        // every projection, and the PLE row cache is a single-device arena.
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let report = audit_namespace(store, config);
        report.log();
        report.ensure_loadable()?;

        let h = config.hidden_size;
        let variant = crate::weight_map::detect_nvfp4_variant(store, config);
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        tracing::info!(
            "Qwen3.8-Flash-Next: {} layers ({} GDN + {} full attention), \
             {} experts top-{}, hc {} streams x rank {}, indexer budget {}, \
             PLE at {:?}; NVFP4 variant {:?}",
            config.num_hidden_layers,
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            config.num_experts,
            config.num_experts_per_tok,
            config.hc_mult,
            config.hc_lowrank,
            config.index_topk,
            config.ple_layer_ids,
            variant,
        );

        // The model-level mixer collapses the streams before `lm_head` and
        // carries the FINAL NORM (this checkpoint has no `model.norm.weight`).
        // Replicated onto every layer; only the last one consumes it.
        // `input_mix_weight_up` is transposed IN PLACE as each site loads —
        // see `hc::UpTranspose`. One staging buffer serves every site, so it
        // has to outlive the layer loop below.
        let mut up_tr = if config.hc_mult > 0 {
            Some(hc::UpTranspose::new(gpu, config)?)
        } else {
            None
        };
        let hc_head = match up_tr.as_mut() {
            Some(tr) => Some(hc::load_head(store, config, tr)?),
            None => None,
        };

        // PLE scratch is sized once, for the largest prefill CHUNK a pass can
        // present — not the model's context.
        //
        // Deliberately NOT `config.max_position_embeddings`: `--max-seq-len`
        // is never written back into it, so on this model that field is the
        // architectural 262144 and any clamp of it over-allocates. The six
        // buffers total `tokens * 10240 * 14` bytes, which at 8192 is 1.26 GB
        // — enough to push a 94.6 GB resident model past the util pledge on a
        // box with 2.7 GB of headroom, which is exactly what it did.
        //
        // 2048 covers the chunk sizes this model runs at; a larger chunk gets
        // the layer's refusal, which names this variable, rather than a
        // silent overrun.
        // Sized by `max_batch_tokens` — the widest any single forward can be —
        // NOT by a standalone constant. The old 2048 default claimed to "cover
        // the chunk sizes this model runs at" and did not: prefill chunks are
        // capped, but a fused mixed step sums a padded decode batch with a
        // prefill slice and is bounded only by `max_batch_tokens` (8196 at the
        // default serve config). Every prompt past 2048 tokens died with
        //
        //   Prefill chunk layer 1 failed: PLE: 3024 tokens exceeds the 2048
        //   this layer was sized for
        //
        // and the API turned that into a 500. On a BFCL subset draw that was
        // 51 of 334 samples — 15% of the benchmark scoring zero for a reason
        // unrelated to the model, concentrated in the subsets with the largest
        // tool schemas (`live_multiple` carries a median 3.2 KB of tool JSON,
        // up to 22 KB, and scored 30% against `live_simple`'s 93%).
        //
        // Capping the prefill chunk instead was TRIED and does nothing: the
        // failures are batched-forward totals, not single chunks, so they
        // survive a 2048 chunk cap unchanged (measured: 51 before, 51 after).
        //
        // The cost is real — tokens*10240*14 bytes, so 1.18 GB at 8196 against
        // 293 MB at 2048, out of the KV budget. `ATLAS_PLE_MAX_TOKENS` still
        // overrides for anyone who would rather have the KV depth, and the
        // layer's refusal still names it.
        let ple_floor = config.max_batch_tokens.max(2048);
        let max_ple_tokens: usize = match std::env::var("ATLAS_PLE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
        {
            Some(n) => {
                // Below the forward width: honouring it meant a guaranteed 500.
                if n < ple_floor {
                    tracing::warn!(
                        "ATLAS_PLE_MAX_TOKENS={n} is below the {ple_floor}-token forward \
                         width — clamped UP; lower --max-num-batched-tokens instead."
                    );
                }
                n.max(ple_floor)
            }
            None => ple_floor,
        };
        // With PLE disabled for bisection, skip the 21 MB arena and the
        // 128-shard open entirely rather than building what we will not run.
        let ple_off = std::env::var("ATLAS_QWEN4EXP_NO_PLE").as_deref() == Ok("1");
        // GDN projections stay BF16 by DEFAULT on this checkpoint. Measured,
        // both arms, same prompt, util 0.85 / 16K / bf16 KV:
        //
        //                      requantized NVFP4      BF16 (default)
        //   layer construction   7.43 GB / 154.8 MB/L   1.39 GB / 28.9 MB/L
        //   attn/GDN arms        7.34 GB                1.32 GB
        //   pre-KV               95.8 GB                90.0 GB
        //   KV budget            3.9 GB / 172144 tok    9.7 GB / 424464 tok
        //   decode               2.207 tok/s            2.188 tok/s
        //
        // 6.04 GB back for ~1% of decode. GDN weight bandwidth is simply not
        // what bounds decode here at C=1 — something else dominates — so the
        // usual w4a16-is-faster argument does not apply yet. Revisit if that
        // changes.
        //
        // And it is not only a memory lever: ONLY the routed experts are
        // quantized in this checkpoint. The GDN projections ship BF16, so
        // requantizing them was a lossy round trip we chose, on 36 of 48
        // layers. `=0` opts back into it for A/B.
        let bf16_gdn = std::env::var("ATLAS_QWEN4EXP_BF16_GDN").as_deref() != Ok("0");
        tracing::info!(
            "GDN projections: {} on the {} linear-attention layers",
            if bf16_gdn {
                "BF16 as shipped (no runtime NVFP4 requantization)"
            } else {
                "requantized to NVFP4 (ATLAS_QWEN4EXP_BF16_GDN=0)"
            },
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
        );

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        // Per-arm memory attribution. Layer construction costs 7.41 GB on this
        // model (154.5 MB/layer, measured) on top of the 85.2 GB of uploaded
        // shards, and nothing said which arm spent it. Summed here and logged
        // once, so the answer is read rather than guessed.
        let (mut moe_bytes, mut arm_bytes, mut hc_bytes) = (0u64, 0u64, 0u64);
        let free_now = |g: &dyn GpuBackend| g.free_memory().unwrap_or(0) as u64;

        for i in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(i);
            let f0 = free_now(gpu);
            let ffn = ffn::build_moe(store, &lp, config, gpu, variant)?;
            let f1 = free_now(gpu);
            moe_bytes += f0.saturating_sub(f1);

            // Norm placeholders — see module docs. This model keeps its
            // normalization inside the hyper-connection blocks, so there is
            // no per-layer norm tensor to load. Ones-filled buffers keep the
            // shared arms' shape contract without inventing a scale, and they
            // are unreachable at runtime because the mHC forward refuses
            // before any layer executes.
            let input_norm = ones_norm(h, gpu)?;
            let post_attn_norm = ones_norm(h, gpu)?;

            let layer = match config.layer_types[i] {
                LayerType::LinearAttention if bf16_gdn => {
                    // Keep the GDN projections BF16 instead of requantizing
                    // them to NVFP4 at load.
                    //
                    // Two reasons, and the second is the interesting one.
                    // (1) MEMORY: the requantization is where this model's
                    // build spends its 7.34 GB (152.8 MB/layer, measured —
                    // the MoE costs zero because its experts ship NVFP4 and
                    // upload straight through).
                    // (2) PRECISION: these tensors ship as BF16 in this
                    // checkpoint. Only the routed experts are quantized. So
                    // BF16 -> NVFP4 here is a lossy round trip we chose, not
                    // one the checkpoint forced, and it lands on the GDN
                    // projections of 36 of 48 layers.
                    crate::weight_loader::qwen35::load_layers::linear_attn_arms::build_linear_attention_dense_bf16(
                        i, store, &lp, gpu, variant, config, h,
                        input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: GDN layer {i} (BF16)"))?
                }
                LayerType::LinearAttention => {
                    crate::weight_loader::qwen35::load_layers::linear_attn_arms::build_linear_attention_nvfp4(
                        store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: GDN layer {i}"))?
                }
                LayerType::FullAttention => {
                    let kv_dtype = layer_kv_dtypes
                        .get(attn_idx)
                        .copied()
                        .unwrap_or(KvCacheDtype::Bf16);
                    let l = crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4(
                        i, store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        kv_dtype, attn_idx, input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: full-attention layer {i}"))?;
                    attn_idx += 1;
                    l
                }
                other => anyhow::bail!(
                    "qwen4_exp layer {i} has type {other:?}; this architecture is \
                     only linear_attention / full_attention"
                ),
            };
            let f2 = free_now(gpu);
            arm_bytes += f1.saturating_sub(f2);

            // mHC: two sites per layer wrapping attention and the MoE. The
            // residual this model carries is `hc_mult * hidden` wide, so
            // without these the layer would run on a stream it never mixed.
            let mut layer = layer;
            if let Some(tr) = up_tr.as_mut() {
                let (attn, ffn) = hc::load_layer_sites(store, &lp, config, tr)?;
                attach::attach_hc(&mut layer, i, attn, ffn, hc_head.clone(), config)?;
            }
            attach::attach_qsa(&mut layer, i, &lp, store, config, gpu)?;
            // PLE lands on exactly one layer, which on this checkpoint is a
            // GDN one. `load` returns None for every other layer.
            let ple_layer = if ple_off {
                None
            } else {
                ple::load(store, config, i, max_ple_tokens, gpu)?
            };
            if let Some(p) = ple_layer {
                attach::attach_ple(&mut layer, i, p)?;
            }
            layers.push(layer);
            hc_bytes += f2.saturating_sub(free_now(gpu));
        }
        if let Some(tr) = up_tr.take() {
            tr.finish()?;
        }
        tracing::info!(
            "qwen4_exp layer construction: MoE {:.2} GB ({:.1} MB/layer), \
             attn/GDN arms {:.2} GB ({:.1} MB/layer), mHC+PLE {:.2} GB",
            moe_bytes as f64 / 1e9,
            moe_bytes as f64 / 1e6 / config.num_hidden_layers as f64,
            arm_bytes as f64 / 1e9,
            arm_bytes as f64 / 1e6 / config.num_hidden_layers as f64,
            hc_bytes as f64 / 1e9,
        );

        // PLE is wired (PLAN.md phase D) and validated against the reference
        // in `ops/ple_tests.rs`. The escape hatch stays, inverted: it now
        // DISABLES a mechanism that is present, for bisecting, and says so.
        if !config.ple_layer_ids.is_empty()
            && std::env::var("ATLAS_QWEN4EXP_NO_PLE").as_deref() == Ok("1")
        {
            tracing::warn!(
                "ATLAS_QWEN4EXP_NO_PLE=1: PLE n-gram injection at model layer {} \
                 is DISABLED. Output is wrong by construction — this arm exists \
                 to bisect the mHC spine, nothing else.",
                config.ple_layer_ids[0].saturating_sub(1),
            );
        }
        tracing::info!(
            "Qwen3.8-Flash-Next loaded {} layers with the mHC highway live on \
             all of them ({} GDN + {} full-attention).",
            layers.len(),
            layers.len()
                - config
                    .layer_types
                    .iter()
                    .filter(|t| **t == LayerType::FullAttention)
                    .count(),
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
        );
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: embedding")
    }

    /// **This model has no final norm tensor.**
    ///
    /// There is no `model.norm.weight` anywhere in the checkpoint. The
    /// model-level `hyper_connection_mixer` — which collapses the `hc_mult`
    /// residual streams back to a single hidden state before `lm_head` —
    /// carries `hc_norm [hc_mult*hidden]`, and that IS the final
    /// normalization. It is the wrong width to stand in here (10240 against
    /// 2560), and applying it as though it were a plain final norm would be
    /// inventing math.
    ///
    /// A ones-filled buffer keeps the shape contract so the footprint can be
    /// measured at load. It is unreachable at inference because the mHC
    /// forward refuses first; if that ever stops being true, this is the
    /// first thing to fix.
    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        attach::final_norm_placeholder(store, config, gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        anyhow::ensure!(
            config.tie_word_embeddings,
            "qwen4_exp: no lm_head.weight and tie_word_embeddings is false"
        );
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: tied lm_head")
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        // The ViT tower IS the Qwen3-VL family shape the qwen35 loader
        // already reads: 27 blocks under `model.visual.*`, patch 16,
        // spatial-merge 2, plain BF16 weights (no quant tensors under
        // `visual` in this checkpoint), empty deepstack list. The
        // qwen3.8-flash-next kernel target ships its own vision_encoder.cu
        // shadow, so kernels resolve per-target as usual.
        crate::weight_loader::qwen35::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Still `None`, and no longer because MTP is unsupported. The MTP
        // block is effectively a second model: its own 512-expert MoE, its
        // own hyper-connection mixer, its own QSA indexer, and
        // `fc_embedding`/`fc_hidden` where Atlas's `MtpWeights` wants a fused
        // `eh_proj`. It never fit this return type, so #753 item I gave it
        // its own loader — `mtp::load_qwen4exp_mtp_module`, called from
        // `factory::build` — and declining here is what routes callers to it.
        Ok(None)
    }
}

/// A ones-filled `[n]` BF16 norm scale.
///
/// BF16 1.0 is `0x3F80`, so the buffer cannot be produced with `memset`.
fn ones_norm(n: usize, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let host: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3Fu8], n).flatten().collect();
    let ptr = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// `model.language_model` for the multimodal layout, `model` otherwise.
fn embed_prefix(config: &ModelConfig) -> String {
    if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    }
}

/// The model-level hyper-connection mixer that collapses the residual streams.
fn mixer_prefix(config: &ModelConfig) -> String {
    format!("{}.hyper_connection_mixer", embed_prefix(config))
}
