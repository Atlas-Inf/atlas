// SPDX-License-Identifier: AGPL-3.0-only

use super::{attention::memory::Owned, weights::tensor};
use anyhow::{Context, Result, ensure};
use atlas_core::{
    config::ModelConfig,
    hy4_reference::engram::{HashLayout, head_range, multipliers},
};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    weights::{WeightDtype, WeightStore},
};

mod batch;
mod forward;
#[cfg(test)]
mod tests;
pub struct Hy4Engram<'a> {
    mem: Owned<'a>,
    kernels: [KernelHandle; 9],
    layout: HashLayout,
    topology: [usize; 2],
    heads: std::ops::Range<usize>,
    dim: usize,
    h: usize,
    rows: std::ops::Range<usize>,
    table: DevicePtr,
    scales: DevicePtr,
    lookup_norm: DevicePtr,
    map: DevicePtr,
    multipliers: DevicePtr,
    sizes: DevicePtr,
    offsets: DevicePtr,
    projections: [DevicePtr; 5],
    biases: [DevicePtr; 5],
    key_norm: DevicePtr,
    query_norm: DevicePtr,
    gamma: DevicePtr,
    hc_scales: DevicePtr,
}
pub struct EngramScratch<'a> {
    mem: Owned<'a>,
    hashes: DevicePtr,
    lookup: DevicePtr,
    keys: DevicePtr,
    norm_keys: DevicePtr,
    queries: DevicePtr,
    value: DevicePtr,
    signature: [usize; 3],
    batch: Option<batch::BatchScratch<'a>>,
}
impl<'a> Hy4Engram<'a> {
    pub(crate) fn token_map(&self) -> &[u32] {
        &self.layout.map
    }
    pub fn load(
        config: &ModelConfig,
        store: &WeightStore,
        layer: usize,
        gpu: &'a dyn GpuBackend,
    ) -> Result<Self> {
        let cfg = config.hy4.as_ref().context("Hy4 Engram config missing")?;
        ensure!(
            cfg.engram_layer_ids.contains(&(layer + 1))
                && cfg.engram_fp8_embedding
                && cfg.engram_max_ngram_size == 3
                && cfg.engram_n_head_per_ngram == 8
                && cfg.engram_n_embed_per_ngram == 768
                && config.hidden_size == 3072,
            "Hy4 Engram unsupported geometry/layer"
        );
        let sizes: Vec<usize> = cfg
            .engram_layer_head_vocab_sizes
            .get(&(layer + 1))
            .context("Hy4 Engram prime groups missing")?
            .iter()
            .flatten()
            .copied()
            .collect();
        let heads = head_range(&sizes, config.tp_rank, config.tp_world_size)?;
        let base = format!("model.layers.{layer}.engram_layer");
        let t = store.get(&format!("{base}.lookup_table"))?;
        ensure!(
            t.dtype == WeightDtype::Int32
                && t.shape.len() == 1
                && t.shape[0] <= config.vocab_size
                && t.shape[0] > cfg.engram_pad_id as usize,
            "Hy4 Engram token map geometry"
        );
        let mut bytes = vec![0; t.shape[0] * 4];
        gpu.copy_d2h(t.ptr, &mut bytes)?;
        let mut map = Vec::with_capacity(config.vocab_size);
        for b in bytes.chunks_exact(4) {
            let v = i32::from_le_bytes(b.try_into().unwrap());
            ensure!(v >= 0, "Hy4 Engram negative token remap");
            map.push(v as u32);
        }
        let compressed = *map.iter().max().context("Hy4 empty token map")? as usize + 1;
        map.extend((map.len()..config.vocab_size).map(|v| v as u32));
        let layout = HashLayout {
            map,
            pad: cfg.engram_pad_id,
            heads_per_order: cfg.engram_n_head_per_ngram,
            multipliers: multipliers(cfg.engram_seed, layer + 1, compressed, 3)?,
            sizes,
        };
        layout.validate()?;
        let mut offsets = Vec::new();
        let mut n = 0usize;
        for &s in &layout.sizes {
            offsets.push(n as i32);
            n += s;
        }
        let rows = layout.sizes[..heads.start].iter().sum::<usize>()
            ..layout.sizes[..heads.end].iter().sum::<usize>();
        let dim = cfg.engram_n_embed_per_ngram / cfg.engram_n_head_per_ngram;
        let mut mem = Owned::new(gpu);
        let table = table_slice(
            &mut mem,
            store,
            &format!("{base}.multi_head_embedding.embedding.weight"),
            &[n, dim],
            WeightDtype::FP8E4M3,
            rows.start * dim,
            rows.len() * dim,
        )?;
        let scales = table_slice(
            &mut mem,
            store,
            &format!("{base}.multi_head_embedding.embedding.weight_scale_inv"),
            &[n],
            WeightDtype::BF16,
            rows.start * 2,
            rows.len() * 2,
        )?;
        let lookup_norm = join(
            &mut mem,
            store,
            heads
                .clone()
                .map(|i| format!("{base}.lookup_per_head_norms.{i}.weight")),
            dim,
        )?;
        let key_norm = join(
            &mut mem,
            store,
            (0..4).map(|i| format!("{base}.norm_key.{i}.weight")),
            config.hidden_size,
        )?;
        let query_norm = join(
            &mut mem,
            store,
            (0..4).map(|i| format!("{base}.norm_query.{i}.weight")),
            config.hidden_size,
        )?;
        let hc_scales = join(
            &mut mem,
            store,
            (0..4).map(|i| format!("{base}.unified_per_hc_scales.{i}")),
            config.hidden_size,
        )?;
        let gamma = tensor(
            store,
            &format!("{base}.unified_pre_gamma"),
            &[config.hidden_size],
            WeightDtype::BF16,
        )?;
        let mut projections = [DevicePtr::NULL; 5];
        let mut biases = projections;
        for i in 0..5 {
            let stem = if i == 4 {
                format!("{base}.value_proj")
            } else {
                format!("{base}.key_projs.{i}")
            };
            projections[i] = tensor(
                store,
                &format!("{stem}.weight"),
                &[config.hidden_size, layout.sizes.len() * dim],
                WeightDtype::BF16,
            )?;
            biases[i] = tensor(
                store,
                &format!("{stem}.bias"),
                &[config.hidden_size],
                WeightDtype::BF16,
            )?;
        }
        let map = mem.upload(
            &layout
                .map
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let multipliers = mem.upload(
            &layout
                .multipliers
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let sizes = mem.upload(
            &layout
                .sizes
                .iter()
                .flat_map(|v| (*v as i32).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let offsets = mem.upload(
            &offsets
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let names = [
            "hy4_engram_hash",
            "hy4_engram_lookup",
            "hy4_engram_project",
            "hy4_engram_norm",
            "hy4_engram_inject",
            "hy4_engram_hash_batch",
            "hy4_engram_lookup_batch",
            "hy4_engram_norm_batch",
            "hy4_engram_inject_batch",
        ];
        let mut kernels = [KernelHandle(0); 9];
        for (i, name) in names.iter().enumerate() {
            kernels[i] = gpu.kernel("hy4_engram", name)?;
        }
        Ok(Self {
            mem,
            kernels,
            layout,
            topology: [config.tp_rank, config.tp_world_size],
            heads,
            dim,
            h: config.hidden_size,
            rows,
            table,
            scales,
            lookup_norm,
            map,
            multipliers,
            sizes,
            offsets,
            projections,
            biases,
            key_norm,
            query_norm,
            gamma,
            hc_scales,
        })
    }
    pub fn scratch(&self) -> Result<EngramScratch<'a>> {
        let mut mem = Owned::new(self.mem.gpu);
        let hashes = mem.alloc(self.layout.sizes.len() * 4)?;
        let lookup = mem.alloc(self.heads.len() * self.dim * 2)?;
        let keys = mem.alloc(4 * self.h * 2)?;
        let norm_keys = mem.alloc(4 * self.h * 2)?;
        let queries = mem.alloc(4 * self.h * 2)?;
        let value = mem.alloc(self.h * 2)?;
        Ok(EngramScratch {
            mem,
            hashes,
            lookup,
            keys,
            norm_keys,
            queries,
            value,
            signature: [self.h, self.heads.start, self.heads.end],
            batch: None,
        })
    }
}

fn join(
    mem: &mut Owned<'_>,
    store: &WeightStore,
    names: impl Iterator<Item = String>,
    width: usize,
) -> Result<DevicePtr> {
    let mut bytes = Vec::new();
    for name in names {
        let ptr = tensor(store, &name, &[width], WeightDtype::BF16)?;
        let start = bytes.len();
        bytes.resize(start + width * 2, 0);
        mem.gpu.copy_d2h(ptr, &mut bytes[start..])?;
    }
    mem.upload(&bytes)
}

fn table_slice(
    mem: &mut Owned<'_>,
    store: &WeightStore,
    name: &str,
    shape: &[usize],
    dtype: WeightDtype,
    start: usize,
    bytes: usize,
) -> Result<DevicePtr> {
    if let Some(d) = store.deferred(name) {
        ensure!(
            d.shape == shape && d.dtype == dtype,
            "Hy4 Engram deferred table contract: {name}"
        );
        let ptr = mem.alloc(bytes)?;
        #[cfg(unix)]
        for offset in (0..bytes).step_by(64 * 1024 * 1024) {
            let len = (bytes - offset).min(64 * 1024 * 1024);
            spark_runtime::fast_weights::with_deferred_bytes(d, start + offset, len, |b| {
                mem.gpu.copy_h2d(b, ptr.offset(offset))
            })?;
        }
        #[cfg(not(unix))]
        anyhow::bail!("Hy4 Engram deferred loading requires Unix");
        Ok(ptr)
    } else {
        Ok(tensor(store, name, shape, dtype)?.offset(start))
    }
}
