// SPDX-License-Identifier: AGPL-3.0-only

//! Eager Hy4 text backend with transactional single-sequence verification.

use crate::layers::hy4::{
    Hy4FfnScratch,
    attention::{AttentionCache, Hy4Attention, memory::Owned},
    engram::{EngramScratch, Hy4Engram},
    ffn::Hy4Ffn,
    hc::IndependentHc,
};
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    weights::WeightStore,
};
use std::{collections::HashMap, sync::Arc};

mod batch;
mod benchmark;
mod collectives;
mod compute;
mod draft;
mod model_trait;
mod rollback_benchmark;
mod speculation;
#[cfg(test)]
mod tests;
mod unsupported;
mod vocab;
mod wire;

struct Shared {
    gpu: Box<dyn GpuBackend>,
    store: WeightStore,
    config: ModelConfig,
    comm: Option<Arc<dyn CommBackend>>,
    max_seq: usize,
    page_size: usize,
}
impl Drop for Shared {
    fn drop(&mut self) {
        use atlas_core::scope::ModelResource;
        if let Err(e) = self.gpu.synchronize(self.gpu.default_stream()) {
            tracing::error!("Hy4 teardown synchronization: {e:#}");
        }
        if let Err(e) = self.store.release(self.gpu.as_ref()) {
            tracing::error!("Hy4 checkpoint cleanup: {e:#}");
        }
    }
}
type LayerOwner = (Arc<Shared>, usize);
struct LayerBlocks<'a> {
    attn: Hy4Attention<'a>,
    hc_attn: IndependentHc<'a>,
    hc_mlp: IndependentHc<'a>,
    ffn: Hy4Ffn<'a>,
    engram: Option<Hy4Engram<'a>>,
    input_norm: DevicePtr,
    mlp_norm: DevicePtr,
}
self_cell::self_cell! {
    struct LayerCell {owner:LayerOwner,#[covariant] dependent:LayerBlocks,}
}
struct RequestOwner {
    shared: Arc<Shared>,
    layers: Vec<Arc<LayerCell>>,
}
struct RequestLayer<'a> {
    attn: AttentionCache<'a, 'a>,
    ffn: Hy4FfnScratch<'a>,
    engram: Option<EngramScratch<'a>>,
}
struct RequestData<'a> {
    layers: Vec<RequestLayer<'a>>,
    history: Vec<u32>,
    poisoned: bool,
    checkpoint: Option<speculation::Checkpoint<'a>>,
    draft: Option<draft::State<'a>>,
}
self_cell::self_cell! {
    struct RequestCell {owner:RequestOwner,#[covariant] dependent:RequestData,}
}
struct Runtime<'a> {
    drafter: Option<Mutex<draft::DraftCell>>,
    layers: Vec<Arc<LayerCell>>,
    requests: Mutex<HashMap<usize, RequestCell>>,
    head: IndependentHc<'a>,
    mem: Owned<'a>,
    highway: DevicePtr,
    collapsed: DevicePtr,
    normed: DevicePtr,
    output: DevicePtr,
    post: DevicePtr,
    logits: DevicePtr,
    batch_logits: DevicePtr,
    command: DevicePtr,
    embed: DevicePtr,
    final_norm: DevicePtr,
    vocab: vocab::VocabHead,
    kernels: [KernelHandle; 5],
    max_batch: usize,
    multi: batch::Workspace<'a>,
    prefill: batch::Workspace<'a>,
    verify_logits: std::sync::atomic::AtomicBool,
    head_rows: std::sync::atomic::AtomicUsize,
    benchmark: bool,
    benchmark_done: std::sync::atomic::AtomicBool,
}
self_cell::self_cell! {
    struct ModelCell {owner:Arc<Shared>,#[covariant] dependent:Runtime,}
}
pub(crate) struct Hy4Model {
    cell: Option<ModelCell>,
}
impl Drop for Hy4Model {
    fn drop(&mut self) {
        if let Some(cell) = &self.cell {
            let s = cell.borrow_owner();
            if let Err(e) = s.gpu.synchronize(s.gpu.default_stream()) {
                tracing::error!("Hy4 model drain: {e:#}");
            }
        }
    }
}
impl Hy4Model {
    pub fn new(
        config: ModelConfig,
        store: WeightStore,
        gpu: Box<dyn GpuBackend>,
        comm: Option<Arc<dyn CommBackend>>,
        max_seq: usize,
        page_size: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let timing = crate::layers::hy4::timing::enabled();
        tracing::info!("HY4 CUDA-event timing enabled={timing}");
        ensure!(
            max_seq > 0
                && max_seq <= config.max_position_embeddings
                && page_size > 0
                && max_batch > 0
                && max_batch <= u32::MAX as usize,
            "Hy4 capacity out of range"
        );
        ensure!(
            config.hc_mult == 4
                && matches!(config.tp_world_size, 1 | 2)
                && config.tp_world_size == config.ep_world_size
                && config.tp_rank == config.ep_rank
                && !config.expert_tp,
            "Hy4 requires coincident TP/EP 1 or 2, whole experts"
        );
        if config.tp_world_size > 1 {
            let c = comm
                .as_deref()
                .context("Hy4 TP2/EP2 communicator missing")?;
            ensure!(
                c.rank() == config.tp_rank && c.world_size() == config.tp_world_size,
                "Hy4 communicator rank/world mismatch"
            );
        }
        collectives::register(gpu.as_ref(), comm.as_deref())?;
        let owner = Arc::new(Shared {
            gpu,
            store,
            config,
            comm,
            max_seq,
            page_size,
        });
        ModelCell::try_new(owner, |s| {
            let gpu = s.gpu.as_ref();
            let c = &s.config;
            let store = &s.store;
            let mut layers = Vec::new();
            for i in 0..c.num_hidden_layers {
                layers.push(Arc::new(LayerCell::try_new((s.clone(), i), |(s, i)| {
                    let c = &s.config;
                    let gpu = s.gpu.as_ref();
                    let store = &s.store;
                    let p = c.layer_prefix(*i);
                    let has_engram = c
                        .hy4
                        .as_ref()
                        .context("Hy4 metadata missing")?
                        .engram_layer_ids
                        .contains(&(i + 1));
                    Ok::<_, anyhow::Error>(LayerBlocks {
                        hc_attn: IndependentHc::load(
                            store,
                            &format!("{p}.hc_attn_layer.hc_pre"),
                            false,
                            c,
                            gpu,
                        )?,
                        hc_mlp: IndependentHc::load(
                            store,
                            &format!("{p}.hc_mlp_layer.hc_pre"),
                            false,
                            c,
                            gpu,
                        )?,
                        attn: Hy4Attention::load(store, *i, c, gpu)?,
                        ffn: Hy4Ffn::load(store, *i, c, gpu)?,
                        engram: if has_engram {
                            Some(Hy4Engram::load(c, store, *i, gpu)?)
                        } else {
                            None
                        },
                        input_norm: weight(
                            store,
                            &format!("{p}.input_layernorm.weight"),
                            &[c.hidden_size],
                            false,
                        )?,
                        mlp_norm: weight(
                            store,
                            &format!("{p}.post_attention_layernorm.weight"),
                            &[c.hidden_size],
                            false,
                        )?,
                    })
                })?));
            }
            let mut frozen_map: Option<Vec<u32>> = None;
            for layer in &layers {
                if let Some(engram) = &layer.borrow_dependent().engram {
                    if let Some(first) = &frozen_map {
                        ensure!(
                            first == engram.token_map(),
                            "Hy4 Engram layers must share their frozen token map"
                        );
                    } else {
                        frozen_map = Some(engram.token_map().to_vec());
                    }
                }
            }
            let mut mem = Owned::new(gpu);
            let vocab = vocab::VocabHead::load(c, store, gpu, &mut mem)?;
            let h = c.hidden_size;
            let highway = mem.alloc(4 * h * 2)?;
            let collapsed = mem.alloc(h * 2)?;
            let normed = mem.alloc(h * 2)?;
            let output = mem.alloc(h * 2)?;
            let post = mem.alloc(4 * 4)?;
            let logits = mem.alloc(c.vocab_size * 4)?;
            let batch_logits = mem.alloc(c.vocab_size * 4 * max_batch)?;
            let command = mem.alloc(4 * 4)?;
            Ok(Runtime {
                drafter: None,
                layers,
                requests: Mutex::new(HashMap::new()),
                head: IndependentHc::load(store, "model.hc_head", true, c, gpu)?,
                embed: weight(
                    store,
                    "model.embed_tokens.weight",
                    &[c.vocab_size, h],
                    false,
                )?,
                final_norm: weight(store, "model.norm.weight", &[h], false)?,
                vocab,
                kernels: [
                    gpu.kernel("hy4_model", "hy4_model_embed")?,
                    gpu.kernel("hy4_model", "hy4_model_norm")?,
                    gpu.kernel("hy4_model", "hy4_model_head")?,
                    gpu.kernel("hy4_model", "hy4_model_argmax")?,
                    gpu.kernel("hy4_model", "hy4_model_argmax_batch")?,
                ],
                mem,
                highway,
                collapsed,
                normed,
                output,
                post,
                logits,
                batch_logits,
                command,
                max_batch,
                multi: batch::Workspace::new(
                    s,
                    crate::layers::hy4::batch::MAX_ROWS.min(s.max_seq),
                )?,
                prefill: batch::Workspace::new(
                    s,
                    crate::layers::hy4::batch::PREFILL_ROWS.min(s.max_seq),
                )?,
                verify_logits: std::sync::atomic::AtomicBool::new(false),
                head_rows: std::sync::atomic::AtomicUsize::new(0),
                benchmark: benchmark::enabled()?,
                benchmark_done: std::sync::atomic::AtomicBool::new(false),
            })
        })
        .map(|cell| Self { cell: Some(cell) })
    }
    fn borrow_owner(&self) -> &Arc<Shared> {
        self.cell
            .as_ref()
            .expect("Hy4 model already torn down")
            .borrow_owner()
    }
    fn borrow_dependent(&self) -> &Runtime<'_> {
        self.cell
            .as_ref()
            .expect("Hy4 model already torn down")
            .borrow_dependent()
    }
}
fn weight(store: &WeightStore, name: &str, shape: &[usize], fp32: bool) -> Result<DevicePtr> {
    use spark_runtime::weights::WeightDtype;
    let t = store.get(name)?;
    ensure!(
        t.shape == shape
            && t.dtype
                == if fp32 {
                    WeightDtype::FP32
                } else {
                    WeightDtype::BF16
                },
        "Hy4 {name}: wrong shape/dtype {:?} {:?}",
        t.shape,
        t.dtype
    );
    Ok(t.ptr)
}
impl Runtime<'_> {
    fn allocate(&self, s: &Arc<Shared>, slot: usize) -> Result<()> {
        let mut requests = self.requests.lock();
        ensure!(
            !requests.contains_key(&slot) && requests.len() < self.max_batch,
            "Hy4 slot occupied/capacity exceeded"
        );
        let cell = RequestCell::try_new(
            RequestOwner {
                shared: s.clone(),
                layers: self.layers.clone(),
            },
            |owner| {
                let mut layers = Vec::new();
                for cell in &owner.layers {
                    let b = cell.borrow_dependent();
                    layers.push(RequestLayer {
                        attn: AttentionCache::new(
                            &b.attn,
                            owner.shared.max_seq,
                            owner.shared.page_size,
                        )?,
                        ffn: Hy4FfnScratch::new(
                            &b.ffn,
                            crate::layers::hy4::batch::PREFILL_ROWS.min(s.max_seq),
                        )?,
                        engram: b.engram.as_ref().map(Hy4Engram::scratch).transpose()?,
                    });
                }
                Ok::<_, anyhow::Error>(RequestData {
                    draft: if self.drafter.is_some() {
                        Some(draft::State::new(
                            owner.shared.gpu.as_ref(),
                            owner.shared.config.hidden_size,
                        )?)
                    } else {
                        None
                    },
                    layers,
                    history: Vec::new(),
                    poisoned: false,
                    checkpoint: None,
                })
            },
        )?;
        requests.insert(slot, cell);
        Ok(())
    }
}
