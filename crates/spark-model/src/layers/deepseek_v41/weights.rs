// SPDX-License-Identifier: AGPL-3.0-only

//! One DeepSeek-V4.1 block's weights, as they sit on the GPU, and how they
//! are found in a checkpoint of the 1017 release's layout.
//!
//! FP8 projections keep their bytes; their 32 x 32 block scales are repeated
//! into the per-row layout the MXFP8 tiers read. Routed experts keep their
//! packed FP4 bytes and E8M0 scales. Nothing is requantized.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::hc::HcSite;
use crate::weight_map::{Mxfp8Weight, fp8_block_scales_to_rows};

/// An FP8 projection `[n, k]`.
#[derive(Clone, Copy)]
pub struct Fp8Proj {
    pub w: Mxfp8Weight,
    pub n: usize,
    pub k: usize,
}

/// A routed-expert projection: packed FP4 `[n, k / 2]` bytes and E8M0 scales
/// `[n, k / 32]`.
#[derive(Clone, Copy)]
pub struct Fp4Proj {
    pub w4: DevicePtr,
    pub scales: DevicePtr,
}

#[derive(Clone, Copy)]
pub struct ExpertWeights {
    pub w1: Fp4Proj,
    pub w3: Fp4Proj,
    pub w2: Fp4Proj,
}

pub struct AttentionWeights {
    pub norm: DevicePtr,
    pub q_norm: DevicePtr,
    pub kv_norm: DevicePtr,
    pub wq_a: Fp8Proj,
    pub wq_b: Fp8Proj,
    pub wkv: Fp8Proj,
    pub wo_a: Fp8Proj,
    pub wo_b: Fp8Proj,
    /// `[heads]` FP32.
    pub sink: DevicePtr,
}

/// A source layer's compressor: a BF16 `[512, hidden]` key projection and
/// the norm of the key. Above ratio 1 a gate projection of the same shape
/// pools each group's tokens into one key.
pub struct CompressorWeights {
    pub ratio: usize,
    pub wkv: DevicePtr,
    pub wgate: Option<DevicePtr>,
    pub norm: DevicePtr,
}

/// An index source's indexer: the projection of its query heads and the
/// weight each head's score is given.
pub struct IndexerWeights {
    pub wq_b: Fp8Proj,
    /// `[index heads, hidden]` BF16.
    pub head_weights: DevicePtr,
    /// On a layer that also produces the index keys.
    pub key: Option<IndexKeyWeights>,
}

/// What turns a compressed key, before its rotation, into its index key:
/// a BF16 `[128, 512]` projection and the norm after it.
pub struct IndexKeyWeights {
    pub wk: DevicePtr,
    pub norm: DevicePtr,
}

pub struct MoeWeights {
    pub norm: DevicePtr,
    /// `[experts, hidden]` BF16.
    pub gate: DevicePtr,
    /// `[experts]` FP32.
    pub bias: DevicePtr,
    pub experts: Vec<ExpertWeights>,
    pub shared_w1: Fp8Proj,
    pub shared_w3: Fp8Proj,
    pub shared_w2: Fp8Proj,
}

pub struct BlockWeights {
    pub hc_attn: HcSite,
    pub hc_ffn: HcSite,
    pub attn: AttentionWeights,
    pub compressor: Option<CompressorWeights>,
    pub indexer: Option<IndexerWeights>,
    pub moe: MoeWeights,
}

pub(super) fn tensor(store: &WeightStore, name: &str, dtype: WeightDtype) -> Result<DevicePtr> {
    let t = store.get(name)?;
    ensure!(
        t.dtype == dtype,
        "{name}: {:?}, expected {dtype:?}",
        t.dtype
    );
    Ok(t.ptr)
}

pub(super) fn fp8(store: &WeightStore, gpu: &dyn GpuBackend, prefix: &str) -> Result<Fp8Proj> {
    let weight = store.get(&format!("{prefix}.weight"))?;
    let scale = store.get(&format!("{prefix}.scale"))?;
    ensure!(
        weight.dtype == WeightDtype::FP8E4M3
            && scale.dtype == WeightDtype::FP8E8M0
            && weight.shape.len() == 2,
        "{prefix}: {:?} {:?} with {:?} scales is not block-scaled FP8",
        weight.dtype,
        weight.shape,
        scale.dtype
    );
    let (n, k) = (weight.shape[0], weight.shape[1]);
    let mut grid = vec![0u8; scale.num_elements()];
    gpu.copy_d2h(scale.ptr, &mut grid)?;
    let rows = fp8_block_scales_to_rows(&grid, n, k).with_context(|| prefix.to_string())?;
    let scales = gpu.alloc(rows.len())?;
    gpu.copy_h2d(&rows, scales)?;
    Ok(Fp8Proj {
        w: Mxfp8Weight {
            data: weight.ptr,
            scales,
            tiled: false,
        },
        n,
        k,
    })
}

fn fp4(store: &WeightStore, prefix: &str) -> Result<Fp4Proj> {
    Ok(Fp4Proj {
        w4: tensor(store, &format!("{prefix}.weight"), WeightDtype::UInt8)?,
        scales: tensor(store, &format!("{prefix}.scale"), WeightDtype::FP8E8M0)?,
    })
}

fn hc_site(store: &WeightStore, layer: &str, site: &str) -> Result<HcSite> {
    let part = |name: &str| {
        tensor(
            store,
            &format!("{layer}.hc_{site}_{name}"),
            WeightDtype::FP32,
        )
    };
    Ok(HcSite {
        fn_: part("fn")?,
        scale: part("scale")?,
        base: part("base")?,
    })
}

impl BlockWeights {
    /// Layer `index` from `store`. `compress_ratio` is the layer's ratio when
    /// it produces compressed keys itself, 0 otherwise; `index_source` says
    /// it runs an indexer. `experts` are the routed experts this rank holds.
    pub fn load(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        index: usize,
        compress_ratio: usize,
        index_source: bool,
        experts: std::ops::Range<usize>,
    ) -> Result<Self> {
        let layer = format!("layers.{index}");
        let bf16 = |name: &str| tensor(store, &format!("{layer}.{name}"), WeightDtype::BF16);
        let attn = AttentionWeights {
            norm: bf16("attn_norm.weight")?,
            q_norm: bf16("attn.q_norm.weight")?,
            kv_norm: bf16("attn.kv_norm.weight")?,
            wq_a: fp8(store, gpu, &format!("{layer}.attn.wq_a"))?,
            wq_b: fp8(store, gpu, &format!("{layer}.attn.wq_b"))?,
            wkv: fp8(store, gpu, &format!("{layer}.attn.wkv"))?,
            wo_a: fp8(store, gpu, &format!("{layer}.attn.wo_a"))?,
            wo_b: fp8(store, gpu, &format!("{layer}.attn.wo_b"))?,
            sink: tensor(store, &format!("{layer}.attn.attn_sink"), WeightDtype::FP32)?,
        };
        let compressor = match compress_ratio {
            0 => None,
            ratio => Some(CompressorWeights {
                ratio,
                wkv: bf16("attn.compressor.wkv.weight")?,
                wgate: match ratio {
                    1 => None,
                    _ => Some(bf16("attn.compressor.wgate.weight")?),
                },
                norm: bf16("attn.compressor.norm.weight")?,
            }),
        };
        let indexer = match index_source {
            false => None,
            true => Some(IndexerWeights {
                wq_b: fp8(store, gpu, &format!("{layer}.attn.indexer.wq_b"))?,
                head_weights: bf16("attn.indexer.weights_proj.weight")?,
                key: match compress_ratio {
                    0 => None,
                    _ => Some(IndexKeyWeights {
                        wk: bf16("attn.indexer.wk.weight")?,
                        norm: bf16("attn.indexer.k_norm.weight")?,
                    }),
                },
            }),
        };
        let moe = MoeWeights {
            norm: bf16("ffn_norm.weight")?,
            gate: bf16("ffn.gate.weight")?,
            bias: tensor(store, &format!("{layer}.ffn.gate.bias"), WeightDtype::FP32)?,
            experts: experts
                .map(|e| {
                    let expert = format!("{layer}.ffn.experts.{e}");
                    Ok(ExpertWeights {
                        w1: fp4(store, &format!("{expert}.w1"))?,
                        w3: fp4(store, &format!("{expert}.w3"))?,
                        w2: fp4(store, &format!("{expert}.w2"))?,
                    })
                })
                .collect::<Result<_>>()?,
            shared_w1: fp8(store, gpu, &format!("{layer}.ffn.shared_experts.w1"))?,
            shared_w3: fp8(store, gpu, &format!("{layer}.ffn.shared_experts.w3"))?,
            shared_w2: fp8(store, gpu, &format!("{layer}.ffn.shared_experts.w2"))?,
        };
        Ok(Self {
            hc_attn: hc_site(store, &layer, "attn")?,
            hc_ffn: hc_site(store, &layer, "ffn")?,
            attn,
            compressor,
            indexer,
            moe,
        })
    }
}
