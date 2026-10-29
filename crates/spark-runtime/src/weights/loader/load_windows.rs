// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Windows-only streaming shard load (Strix Halo UMA). Parses the safetensors
/// header only, then reads each tensor's byte range with seek+read into a
/// transient host buffer that is dropped after upload. Avoids `memmap2` on the
/// whole shard — a file-backed mapping keeps its pages resident in the unified
/// GPU aperture (counted used by `hipMemGetInfo`) and the driver cannot reclaim
/// them for `cuMemAlloc`, so a 20GB checkpoint double-books ~40GB. Host
/// residency here caps at the largest single tensor instead.
#[cfg(target_os = "windows")]
pub(super) fn stream_shard_windows(
    file: &mut std::fs::File,
    shard_path: &Path,
    tensor_names: &[String],
    skip_fn: &dyn Fn(&str) -> bool,
    gpu: &dyn GpuBackend,
    weights: &mut HashMap<String, WeightTensor>,
    deferred: &mut HashMap<String, crate::weights::DeferredTensor>,
    skipped: &mut usize,
    offload_logged: &mut bool,
    pack_slab: bool,
    csf: &mut CsfAssembler,
) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    // Safetensors layout: [u64 header_len][header JSON][tensor data]. Parse the
    // header JSON directly — `SafeTensors::read_metadata` validates the whole
    // file length, so it can't parse a header-only buffer. `TensorInfo` is
    // Deserialize; the reserved `__metadata__` entry is dropped.
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .context("Failed to read safetensors header length")?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_json = vec![0u8; header_len];
    file.read_exact(&mut header_json)
        .context("Failed to read safetensors header")?;
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_json).context("Failed to parse safetensors header JSON")?;
    let data_base = (8 + header_len) as u64;

    let mut infos: HashMap<String, safetensors::tensor::TensorInfo> = HashMap::new();
    for (k, v) in header {
        if k == "__metadata__" {
            continue;
        }
        let info: safetensors::tensor::TensorInfo = serde_json::from_value(v)
            .with_context(|| format!("Failed to parse safetensors TensorInfo for {k}"))?;
        infos.insert(k, info);
    }

    let plan = super::super::slab_plan::plan_shard(tensor_names, &infos, skip_fn, data_base)?;
    for d in plan.deferred {
        deferred.insert(
            d.name,
            crate::weights::DeferredTensor {
                path: shard_path.to_path_buf(),
                offset: d.offset,
                shape: d.shape,
                dtype: d.dtype,
            },
        );
    }
    *skipped += plan.skipped;

    // Pack mode (opt-in): one alloc per shard, tensors carved at 4096-byte
    // interior offsets — same contract as the fast loader's pack_slab (per-
    // tensor cuMemAlloc rounds >=1 MB to 2 MB pages, ~0.6 GB waste/shard on
    // NVFP4 MoE). If the slab alloc fails, fall back to per-tensor allocs.
    let slab = if pack_slab {
        let (offsets, total) = crate::weights::slab_layout(plan.kept.iter().map(|t| {
            if csf_part(&t.name).is_some() {
                0
            } else {
                t.len
            }
        }));
        if total == 0 {
            None
        } else {
            match gpu.alloc(total) {
                Ok(base) => {
                    gpu.register_slab(base, total);
                    Some((base, offsets))
                }
                Err(_) => None,
            }
        }
    } else {
        None
    };

    // Read each tensor into a reusable page-locked staging buffer, then H2D.
    // A pinned source makes `cuMemcpyHtoDAsync` a true DMA; a pageable source
    // would route through the driver's internal staging pool, which on the
    // Windows UMA driver grows to ~the copied size and is not released — that
    // was the second ~20GB phantom (a file-backed mmap had the same effect).
    let pinned = gpu.alloc_host_pinned(plan.max_len.max(1))?;
    let staging = unsafe { std::slice::from_raw_parts_mut(pinned, plan.max_len) };
    let result = (|| -> Result<()> {
        for (i, t) in plan.kept.iter().enumerate() {
            file.seek(SeekFrom::Start(data_base + t.begin))?;
            file.read_exact(&mut staging[..t.len])
                .with_context(|| format!("Failed to read tensor {}", t.name))?;
            if csf_part(&t.name).is_some() {
                upload_or_assemble(
                    &t.name,
                    &staging[..t.len],
                    t.dtype,
                    t.shape.clone(),
                    gpu,
                    weights,
                    offload_logged,
                    csf,
                )?;
                continue;
            }
            csf.note_plain(&t.name)?;
            let (data, dtype) = stage_upload_bytes(&t.name, &staging[..t.len], t.dtype)?;
            debug_assert_eq!(data.len(), t.len);
            let ptr = match &slab {
                Some((base, offsets)) => {
                    gpu.copy_h2d(&data, base.offset(offsets[i]))?;
                    base.offset(offsets[i])
                }
                None => {
                    // Per-tensor path — identical to the non-slab arm.
                    upload_one(
                        &t.name,
                        &data,
                        t.dtype,
                        t.shape.clone(),
                        gpu,
                        weights,
                        offload_logged,
                    )?;
                    continue;
                }
            };
            weights.insert(
                t.name.clone(),
                WeightTensor {
                    ptr,
                    shape: t.shape.clone(),
                    dtype,
                },
            );
        }
        Ok(())
    })();
    gpu.free_host_pinned(pinned, plan.max_len.max(1))?;
    result
}
