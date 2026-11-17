// SPDX-License-Identifier: AGPL-3.0-only

//! One DeepSeek-V4.1 block's weights, as they sit on the GPU, and how they
//! are found in a checkpoint of the 1017 release's layout.
//!
//! FP8 projections keep their bytes; their 32 x 32 block scales are repeated
//! into the per-row layout the MXFP8 tiers read. Routed experts keep their
//! packed FP4 bytes and E8M0 scales. Nothing is requantized.
//!
//! The wide projections are held in four parts (`parts.rs`), and a rank
//! loads the parts it computes: a quarter of every routed expert on each of
//! four Sparks. The routed experts are most of the model, so the loader
//! leaves them on disk and the parts are read from there.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::hc::HcSite;
use super::parts::{Held, PARTS, PartKernels};
use super::shape::Shape;
use super::weights_experts::{expert_slab, release_experts};
use crate::weight_map::{Mxfp8Weight, fp8_block_scales_to_rows};

/// An FP8 projection `[n, k]`.
#[derive(Clone, Copy)]
pub struct Fp8Proj {
    pub w: Mxfp8Weight,
    pub n: usize,
    pub k: usize,
}

/// One part of a projection: its values, and an E8M0 scale per row and 32
/// columns.
#[derive(Clone, Copy)]
pub struct Slice {
    pub data: DevicePtr,
    pub scales: DevicePtr,
}

impl Fp8Proj {
    /// The projection from row `first` on.
    pub fn rows_from(&self, first: usize) -> Slice {
        Slice {
            data: self.w.data.offset(first * self.k),
            scales: self.w.scales.offset(first * self.k / 32),
        }
    }
}

/// A projection in four parts, of which a rank holds some.
pub type Parted = [Option<Slice>; PARTS];

/// One part of every routed expert of a layer, in one allocation: expert
/// `e` at `base + e * stride`, as the part's rows of w1, of w3 and of w2,
/// each followed by its scales.
#[derive(Clone, Copy)]
pub struct ExpertSlab {
    pub base: DevicePtr,
    /// Bytes of one projection's part, and of its scales.
    pub(super) data: usize,
    pub(super) scales: usize,
}

impl ExpertSlab {
    /// Bytes from one expert to the next.
    pub fn stride(&self) -> usize {
        3 * (self.data + self.scales)
    }

    /// Bytes of one projection's part, and of the scales after it.
    pub fn layout(&self) -> (usize, usize) {
        (self.data, self.scales)
    }

    pub(super) fn slice(&self, expert: usize, projection: usize) -> Slice {
        let at = expert * self.stride() + projection * (self.data + self.scales);
        Slice {
            data: self.base.offset(at),
            scales: self.base.offset(at + self.data),
        }
    }

    /// `[inter / 4, hidden]` FP4: the part's rows of the gate projection.
    pub fn w1(&self, expert: usize) -> Slice {
        self.slice(expert, 0)
    }

    /// The part's rows of the up projection.
    pub fn w3(&self, expert: usize) -> Slice {
        self.slice(expert, 1)
    }

    /// `[hidden / 4, inter]` FP4: the part's rows of the down projection.
    pub fn w2(&self, expert: usize) -> Slice {
        self.slice(expert, 2)
    }
}

pub struct AttentionWeights {
    pub norm: DevicePtr,
    pub q_norm: DevicePtr,
    pub kv_norm: DevicePtr,
    pub wq_a: Fp8Proj,
    /// A part is a quarter of the heads' rows, and a rank has the rows of
    /// the parts it holds ([`fp8_held`]).
    pub wq_b: Fp8Proj,
    pub wkv: Fp8Proj,
    /// The same: a part is a quarter of the groups.
    pub wo_a: Fp8Proj,
    /// The same: a part is a quarter of its rows.
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
    /// Every routed expert, a part at a time.
    pub experts: [Option<ExpertSlab>; PARTS],
    /// The shared expert: a part is a quarter of each projection's rows,
    /// and a rank has the rows of the parts it holds.
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

/// [`fp8`] of a projection whose rows this rank computes a part of: the rows
/// of the parts it holds are kept, and the store's copy of the whole let go,
/// which on one of four ranks is three quarters of it. The projection is
/// still addressed as a whole (`rows_from`, [`row_part`]); the rows of the
/// other parts are not there to read.
pub(super) fn fp8_held(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    prefix: &str,
    held: Held,
) -> Result<Fp8Proj> {
    let whole = fp8(store, gpu, prefix)?;
    if held.count == PARTS {
        return Ok(whole);
    }
    let (n, k) = (whole.n, whole.k);
    ensure!(
        n.is_multiple_of(PARTS),
        "{prefix}: {n} rows do not split in {PARTS} parts"
    );
    let (first, rows) = (held.first * n / PARTS, held.count * n / PARTS);
    let keep = |from: DevicePtr, width: usize| -> Result<DevicePtr> {
        let (kept, before) = (gpu.alloc(rows * width)?, (first * width) as u64);
        gpu.copy_d2d(from.offset(first * width), kept, rows * width)?;
        ensure!(kept.0 >= before, "{prefix}: a part below its own start");
        Ok(DevicePtr(kept.0 - before))
    };
    let (data, scales) = (keep(whole.w.data, k)?, keep(whole.w.scales, k / 32)?);
    store.reclaim(gpu, &format!("{prefix}.weight"))?;
    gpu.free(whole.w.scales)?;
    Ok(Fp8Proj {
        w: Mxfp8Weight {
            data,
            scales,
            tiled: false,
        },
        n,
        k,
    })
}

/// Part `part` of the rows of a whole FP8 projection: a quarter of its
/// output.
pub fn row_part(whole: &Fp8Proj, part: usize) -> Slice {
    whole.rows_from(part * whole.n / PARTS)
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
    /// Free the parts this made for itself: the routed experts. The rest
    /// belongs to the store it was loaded from.
    pub fn free_parts(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for slab in self.moe.experts.iter().flatten() {
            gpu.free(slab.base)?;
        }
        Ok(())
    }

    /// Layer `index` from `store`. `compress_ratio` is the layer's ratio when
    /// it produces compressed keys itself, 0 otherwise; `index_source` says
    /// it runs an indexer. `held` are the parts this rank computes, and
    /// `experts` says whether to load the routed experts at all (a test of
    /// the attention alone leaves them out).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        kernels: &PartKernels,
        shape: &Shape,
        index: usize,
        compress_ratio: usize,
        index_source: bool,
        held: Held,
        experts: bool,
        stream: u64,
    ) -> Result<Self> {
        let layer = format!("layers.{index}");
        Self::load_named(
            store,
            gpu,
            kernels,
            shape,
            &layer,
            compress_ratio,
            index_source,
            held,
            experts,
            stream,
        )
    }

    /// [`Self::load`] of the block whose tensors are named `layer.*`: a
    /// decoder layer, or a draft stage (`mtp.N`), which is a block with
    /// fewer experts and no compressed keys.
    #[allow(clippy::too_many_arguments)]
    pub fn load_named(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        kernels: &PartKernels,
        shape: &Shape,
        layer: &str,
        compress_ratio: usize,
        index_source: bool,
        held: Held,
        experts: bool,
        stream: u64,
    ) -> Result<Self> {
        let bf16 = |name: &str| tensor(store, &format!("{layer}.{name}"), WeightDtype::BF16);
        let attn = AttentionWeights {
            norm: bf16("attn_norm.weight")?,
            q_norm: bf16("attn.q_norm.weight")?,
            kv_norm: bf16("attn.kv_norm.weight")?,
            wq_a: fp8(store, gpu, &format!("{layer}.attn.wq_a"))?,
            wq_b: fp8_held(store, gpu, &format!("{layer}.attn.wq_b"), held)?,
            wkv: fp8(store, gpu, &format!("{layer}.attn.wkv"))?,
            wo_a: fp8_held(store, gpu, &format!("{layer}.attn.wo_a"), held)?,
            wo_b: fp8_held(store, gpu, &format!("{layer}.attn.wo_b"), held)?,
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
            experts: {
                let mut slabs = [None; PARTS];
                for part in held.range().filter(|_| experts) {
                    slabs[part] = Some(expert_slab(
                        store, gpu, kernels, shape, layer, part, stream,
                    )?);
                }
                // The slabs hold this rank's part of every expert, and
                // nothing else reads them: a store that had them whole on
                // the device (a rank alone, or a draft stage's) lets them go.
                if experts {
                    release_experts(store, gpu, layer, shape.experts)?;
                }
                slabs
            },
            shared_w1: fp8_held(store, gpu, &format!("{layer}.ffn.shared_experts.w1"), held)?,
            shared_w3: fp8_held(store, gpu, &format!("{layer}.ffn.shared_experts.w3"), held)?,
            shared_w2: fp8_held(store, gpu, &format!("{layer}.ffn.shared_experts.w2"), held)?,
        };
        Ok(Self {
            hc_attn: hc_site(store, layer, "attn")?,
            hc_ffn: hc_site(store, layer, "ffn")?,
            attn,
            compressor,
            indexer,
            moe,
        })
    }
}
