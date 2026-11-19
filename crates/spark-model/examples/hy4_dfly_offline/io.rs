// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

#[derive(Deserialize)]
pub struct Reference {
    pub records: Vec<Case>,
}
#[derive(Deserialize)]
pub struct Case {
    pub case: String,
    pub anchor: usize,
    pub count: usize,
    pub bonus: u32,
    pub reference: Vec<u32>,
}
#[derive(Deserialize)]
struct Index {
    weight_map: HashMap<String, String>,
}
#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct CheckpointIo {
    root: PathBuf,
    index: Index,
}
impl CheckpointIo {
    pub fn open(root: &Path) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            index: serde_json::from_reader(File::open(root.join("model.safetensors.index.json"))?)?,
        })
    }
    pub fn auxiliary_skips(&self) -> Vec<String> {
        let mut prefixes = BTreeSet::new();
        for name in self
            .index
            .weight_map
            .keys()
            .filter(|n| !n.starts_with("model.dfly_layers."))
        {
            let parts: Vec<_> = name.split('.').collect();
            prefixes.insert(if parts.len() > 2 {
                format!("{}.{}.", parts[0], parts[1])
            } else {
                name.clone()
            });
        }
        prefixes.into_iter().collect()
    }
    pub fn rows(
        &self,
        name: &str,
        shape: [usize; 2],
        start: usize,
        rows: usize,
        buffers: &mut Buffers<'_>,
    ) -> Result<DevicePtr> {
        let out = buffers.alloc(rows * shape[1] * 2)?;
        self.copy_rows(name, shape, start, rows, |offset, bytes| {
            buffers.gpu.copy_h2d(bytes, out.offset(offset))
        })?;
        Ok(out)
    }
    pub fn copy_rows(
        &self,
        name: &str,
        shape: [usize; 2],
        start: usize,
        rows: usize,
        mut copy: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            rows > 0 && start.checked_add(rows).is_some_and(|n| n <= shape[0]),
            "invalid checkpoint row slice"
        );
        let (mut file, header_len, tensor) = self.header(name)?;
        ensure!(
            tensor.shape == shape && matches!(tensor.dtype.as_str(), "BF16" | "F32"),
            "unexpected tensor shape/dtype"
        );
        let stride = if tensor.dtype == "F32" { 4 } else { 2 };
        ensure!(
            tensor.data_offsets[1].checked_sub(tensor.data_offsets[0])
                == Some(shape[0] * shape[1] * stride),
            "invalid tensor byte span"
        );
        ensure!(
            8 + header_len + tensor.data_offsets[1] as u64 <= file.metadata()?.len(),
            "truncated tensor"
        );
        file.seek(SeekFrom::Start(
            8 + header_len + ((tensor.data_offsets[0] + start * shape[1] * stride) as u64),
        ))?;
        let elements = rows * shape[1];
        let mut offset = 0;
        while offset < elements {
            let count = (elements - offset).min(1024 * 1024);
            let mut raw = vec![0u8; count * stride];
            file.read_exact(&mut raw)?;
            let converted;
            let source = if stride == 4 {
                converted = raw
                    .chunks_exact(4)
                    .flat_map(|b| {
                        half::bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap()))
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                &converted
            } else {
                &raw
            };
            copy(offset * 2, source)?;
            offset += count;
        }
        Ok(())
    }
    fn header(&self, name: &str) -> Result<(File, u64, Tensor)> {
        let filename = self
            .index
            .weight_map
            .get(name)
            .context("checkpoint tensor missing")?;
        ensure!(
            !Path::new(filename).is_absolute()
                && Path::new(filename)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "invalid shard path"
        );
        let mut file = File::open(self.root.join(filename))?;
        let mut bytes = [0u8; 8];
        file.read_exact(&mut bytes)?;
        let header_len = u64::from_le_bytes(bytes);
        ensure!(header_len <= 64 * 1024 * 1024, "oversized tensor header");
        let mut header = vec![0; header_len as usize];
        file.read_exact(&mut header)?;
        let header: serde_json::Value = serde_json::from_slice(&header)?;
        let tensor =
            serde_json::from_value(header.get(name).context("missing tensor header")?.clone())?;
        Ok((file, header_len, tensor))
    }
    pub fn peak_bytes(&self, world: usize, vocab: usize) -> Result<usize> {
        let mut raw = 0usize;
        let mut local = 0usize;
        let mut largest = 0usize;
        for name in self
            .index
            .weight_map
            .keys()
            .filter(|n| n.starts_with("model.dfly_layers."))
        {
            let (_, _, tensor) = self.header(name)?;
            ensure!(tensor.dtype == "BF16", "expected BF16 auxiliary weight");
            let bytes = tensor
                .shape
                .iter()
                .try_fold(2usize, |n, d| n.checked_mul(*d))
                .context("auxiliary size overflow")?;
            raw = raw.checked_add(bytes).context("auxiliary sum overflow")?;
            largest = largest.max(bytes);
            let sharded = [
                ".q_b_proj.",
                ".linear_gate.",
                ".o_proj.",
                ".kv_b_proj.",
                ".learnable_sink_param",
                ".mlp.",
            ]
            .iter()
            .any(|part| name.contains(part));
            local = local
                .checked_add(bytes / if sharded { world } else { 1 })
                .context("local sum overflow")?;
        }
        let shared = vocab / world * 3072 * 2;
        let scratch = 64 * 1024 * 1024;
        let runtime_margin = 1024 * 1024 * 1024;
        let peak = (raw + largest).max(local + shared + scratch) + runtime_margin;
        println!(
            "DFLY_MEMORY_PLAN auxiliary_raw={raw} auxiliary_local={local} shared={shared} scratch_bound={scratch} runtime_margin={runtime_margin} peak_bound={peak}"
        );
        Ok(peak)
    }
}

pub struct Buffers<'a> {
    pub gpu: &'a dyn GpuBackend,
    reserve: usize,
    allocations: Vec<DevicePtr>,
}
impl<'a> Buffers<'a> {
    pub fn new(gpu: &'a dyn GpuBackend, reserve: usize) -> Self {
        Self {
            gpu,
            reserve,
            allocations: Vec::new(),
        }
    }
    pub fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        guard(self.gpu, self.reserve, bytes)?;
        let ptr = self.gpu.alloc(bytes)?;
        self.allocations.push(ptr);
        Ok(ptr)
    }
    pub fn upload(&mut self, bytes: &[u8]) -> Result<DevicePtr> {
        let ptr = self.alloc(bytes.len())?;
        self.gpu.copy_h2d(bytes, ptr)?;
        Ok(ptr)
    }
}
impl Drop for Buffers<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.synchronize(self.gpu.default_stream());
        for ptr in self.allocations.drain(..).rev() {
            let _ = self.gpu.free(ptr);
        }
    }
}
pub fn guard(gpu: &dyn GpuBackend, reserve: usize, needed: usize) -> Result<()> {
    let free = gpu.free_memory()?;
    ensure!(
        needed.checked_add(reserve).is_some_and(|n| n <= free),
        "OOM guard: {needed} B needed + {reserve} B reserve > {free} B free"
    );
    println!("DFLY_OOM_GUARD free={free} reserve={reserve} needed={needed}");
    Ok(())
}
pub fn fixture(path: &Path) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}
pub fn reference(path: &Path) -> Result<Reference> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}
