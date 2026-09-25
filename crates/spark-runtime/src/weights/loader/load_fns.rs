// SPDX-License-Identifier: AGPL-3.0-only
//
// Sharded + single safetensors loaders. Split out of `loader.rs` to keep
// the parent under the 500-line cap.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

use super::super::{WeightDtype, WeightTensor, evict_page_cache, f16_to_bf16_bytes};
use super::{SafetensorsIndex, check_oom_guard, estimate_has_fp8, estimate_load_bytes};
use crate::gpu::{DevicePtr, GpuBackend};

/// Upload one tensor's raw on-disk bytes to the store: F16→BF16 convert if
/// needed, GPU-alloc (managed/UVM fallback on OOM), copy, insert. Shared by the
/// mmap path (whole-file view) and the Windows streaming path (per-tensor read).
fn upload_one(
    name: &str,
    raw: &[u8],
    st_dtype: safetensors::Dtype,
    shape: Vec<usize>,
    gpu: &dyn GpuBackend,
    weights: &mut HashMap<String, WeightTensor>,
    offload_logged: &mut bool,
) -> Result<()> {
    // F16 shards: convert bytes to BF16 before upload (same length,
    // different bit layout). EXL3 scale vectors (`keeps_raw_f16`) keep their
    // raw FP16 bytes. WeightDtype stays closed to store dtypes otherwise.
    let converted: Vec<u8>;
    let (data, dtype): (&[u8], _) = if st_dtype == safetensors::Dtype::F16 {
        if crate::weights::keeps_raw_f16(name) {
            (raw, WeightDtype::FP16)
        } else {
            converted = f16_to_bf16_bytes(raw);
            (&converted, WeightDtype::BF16)
        }
    } else {
        (raw, WeightDtype::from_safetensors(st_dtype)?)
    };

    // ATLAS_WEIGHT_MANAGED=1: upload straight into managed (UVM) memory via CPU
    // memcpy, skipping `gpu.alloc`+`copy_h2d`. Diagnostic for the ~2x store
    // phantom on Windows UMA — if managed upload reports ~on-disk size instead
    // of 2x, the doubling is in the pageable->device copy path (driver staging
    // bounce), not the device allocation itself.
    let force_managed = {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("ATLAS_WEIGHT_MANAGED").as_deref() == Ok("1"))
    };

    // Try GPU alloc first; if OOM, fall back to managed (UVM) memory.
    // On GB10 unified memory, managed alloc uses Linux swap for overflow.
    let managed_upload = |gpu: &dyn GpuBackend| -> Result<DevicePtr> {
        let p = gpu.alloc_managed(data.len())?;
        // Use CPU memcpy (not GPU copy_h2d) to avoid GPU page faults.
        // Managed memory is CPU-accessible, so memcpy writes directly to
        // CPU pages. The GPU will page-fault on first access during kernels.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), p.0 as *mut u8, data.len());
        }
        Ok(p)
    };
    let ptr = if force_managed {
        if !*offload_logged {
            tracing::warn!("ATLAS_WEIGHT_MANAGED=1 — uploading weights via managed (UVM) memory");
            *offload_logged = true;
        }
        managed_upload(gpu)?
    } else {
        match gpu.alloc(data.len()) {
            Ok(p) => {
                gpu.copy_h2d(data, p)?;
                p
            }
            Err(_) => {
                if !*offload_logged {
                    tracing::warn!(
                        "GPU alloc failed for {} ({} bytes) — switching to managed (UVM) memory. \
                         Weights will be paged via Linux swap (slower but avoids OOM).",
                        name,
                        data.len()
                    );
                    *offload_logged = true;
                }
                managed_upload(gpu)?
            }
        }
    };

    weights.insert(name.to_string(), WeightTensor { ptr, shape, dtype });
    Ok(())
}

pub(super) fn load_sharded(
    model_dir: &Path,
    index_path: &Path,
    gpu: &dyn GpuBackend,
    oom_reserve_bytes: usize,
    skip_fn: &dyn Fn(&str) -> bool,
    peak_multiplier_override: Option<f64>,
    deferred: &mut HashMap<String, crate::weights::DeferredTensor>,
) -> Result<HashMap<String, WeightTensor>> {
    let index_json = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read {}", index_path.display()))?;
    let mut index: SafetensorsIndex = serde_json::from_str(&index_json)?;
    // EXL3 tensors in unindexed side files join the map (no-op otherwise), so
    // they are grouped into `shard_to_tensors` and estimated below.
    let _merged = crate::weights::merge_exl3_side_files(model_dir, &mut index.weight_map)?;

    let mut offload_logged = false; // Track if we've logged the managed memory fallback

    // Group tensors by shard to minimize mmap overhead
    let mut shard_to_tensors: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, shard_name) in &index.weight_map {
        shard_to_tensors
            .entry(shard_name.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    // Pre-flight: estimate bytes from index with model-building overhead.
    let shard_files: Vec<std::path::PathBuf> =
        shard_to_tensors.keys().map(|s| model_dir.join(s)).collect();
    // The n-gram tables are deferred below, never uploaded, so they must not
    // count toward the peak — see the note in `fast_weights`.
    let preflight_skip = |name: &str| skip_fn(name) || crate::weights::is_ngram_table(name);
    let estimated = estimate_load_bytes(&shard_files, &preflight_skip)?;
    let has_fp8 = estimate_has_fp8(&shard_files, &preflight_skip)?;
    let overhead_multiplier: f64 =
        peak_multiplier_override.unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
    let peak_estimated = (estimated as f64 * overhead_multiplier) as usize;
    let free = gpu.free_memory()?;
    let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
    let est_gb = estimated as f64 / (1024.0 * 1024.0 * 1024.0);
    let peak_gb = peak_estimated as f64 / (1024.0 * 1024.0 * 1024.0);
    let reserve_gb = oom_reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    tracing::info!(
        "Pre-flight estimate: {:.2} GB on-disk weights, {:.1}x overhead = {:.2} GB peak, \
         {:.2} GB free, {:.1} GB reserve (FP8: {})",
        est_gb,
        overhead_multiplier,
        peak_gb,
        free_gb,
        reserve_gb,
        has_fp8,
    );
    if peak_estimated + oom_reserve_bytes > free {
        bail!(
            "OOM pre-flight: model peak memory ({:.2} GB = {:.2} GB weights × {:.1}x \
             model-building overhead) + {:.1} GB reserve = {:.2} GB, \
             but only {:.2} GB GPU memory is available. \
             This model is too large. Use a smaller quantization (NVFP4 instead of FP8) \
             or add more GPUs for expert parallelism.",
            peak_gb,
            est_gb,
            overhead_multiplier,
            reserve_gb,
            peak_gb + reserve_gb,
            free_gb,
        );
    }

    let mut weights = HashMap::new();
    let mut skipped = 0usize;
    let total_shards = shard_to_tensors.len();
    let initial_free = free;

    for (i, (shard_name, tensor_names)) in shard_to_tensors.iter().enumerate() {
        let shard_path = model_dir.join(shard_name);
        tracing::info!(
            "Loading shard {}/{}: {} ({} tensors)",
            i + 1,
            total_shards,
            shard_name,
            tensor_names.len()
        );

        // `mut` is only needed on Windows, where the streaming reader seeks.
        #[cfg_attr(not(target_os = "windows"), allow(unused_mut))]
        let mut file = std::fs::File::open(&shard_path)
            .with_context(|| format!("Failed to open {}", shard_path.display()))?;

        #[cfg(target_os = "windows")]
        {
            // Windows UMA (Strix Halo gfx1151): mmap'ing the shard leaves its
            // file-cache pages resident in the GPU aperture — `hipMemGetInfo`
            // counts them as used and the driver cannot reclaim them for new
            // allocs, so a 20GB checkpoint double-books ~40GB. Stream each
            // tensor via seek+read into a transient buffer instead: host
            // residency caps at the largest single tensor, not the file.
            stream_shard_windows(
                &mut file,
                &shard_path,
                tensor_names,
                skip_fn,
                gpu,
                &mut weights,
                deferred,
                &mut skipped,
                &mut offload_logged,
            )?;
        }
        #[cfg(not(target_os = "windows"))]
        {
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

            for name in tensor_names {
                if skip_fn(name) {
                    skipped += 1;
                    continue;
                }
                // n-gram TABLES are deferred, not uploaded (see `is_ngram_table`).
                // The absolute file offset comes from the pointer delta into the
                // mmap, which is exact and needs no header re-parse.
                if crate::weights::is_ngram_table(name) {
                    let view = tensors.tensor(name)?;
                    let off = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
                    deferred.insert(
                        name.to_string(),
                        crate::weights::DeferredTensor {
                            path: shard_path.clone(),
                            offset: off as u64,
                            shape: view.shape().to_vec(),
                            dtype: WeightDtype::from_safetensors(view.dtype())?,
                        },
                    );
                    skipped += 1;
                    continue;
                }
                let view = tensors.tensor(name)?;
                upload_one(
                    name,
                    view.data(),
                    view.dtype(),
                    view.shape().to_vec(),
                    gpu,
                    &mut weights,
                    &mut offload_logged,
                )?;
            }

            drop(tensors);
            drop(mmap);
            evict_page_cache(&file);
        }

        // OOM guard: check free memory after each shard
        let free_now = gpu.free_memory()?;
        let used = initial_free.saturating_sub(free_now);
        tracing::info!(
            "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
            i + 1,
            total_shards,
            used as f64 / (1024.0 * 1024.0 * 1024.0),
            free_now as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        if !offload_logged {
            check_oom_guard(
                gpu,
                oom_reserve_bytes,
                &format!("weight loading (shard {}/{})", i + 1, total_shards),
            )?;
        }
    }

    if skipped > 0 {
        tracing::info!("EP: skipped {} remote expert tensors", skipped);
    }
    tracing::info!("Loaded {} weight tensors", weights.len());
    Ok(weights)
}

/// Windows-only streaming shard load (Strix Halo UMA). Parses the safetensors
/// header only, then reads each tensor's byte range with seek+read into a
/// transient host buffer that is dropped after upload. Avoids `memmap2` on the
/// whole shard — a file-backed mapping keeps its pages resident in the unified
/// GPU aperture (counted used by `hipMemGetInfo`) and the driver cannot reclaim
/// them for `cuMemAlloc`, so a 20GB checkpoint double-books ~40GB. Host
/// residency here caps at the largest single tensor instead.
#[cfg(target_os = "windows")]
fn stream_shard_windows(
    file: &mut std::fs::File,
    shard_path: &Path,
    tensor_names: &[String],
    skip_fn: &dyn Fn(&str) -> bool,
    gpu: &dyn GpuBackend,
    weights: &mut HashMap<String, WeightTensor>,
    deferred: &mut HashMap<String, crate::weights::DeferredTensor>,
    skipped: &mut usize,
    offload_logged: &mut bool,
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

    // Read each tensor into a reusable page-locked staging buffer, then H2D.
    // A pinned source makes `cuMemcpyHtoDAsync` a true DMA; a pageable source
    // would route through the driver's internal staging pool, which on the
    // Windows UMA driver grows to ~the copied size and is not released — that
    // was the second ~20GB phantom (a file-backed mmap had the same effect).
    let max_len = tensor_names
        .iter()
        .filter(|n| !skip_fn(n) && !crate::weights::is_ngram_table(n))
        .filter_map(|n| infos.get(n.as_str()))
        .map(|i| i.data_offsets.1.saturating_sub(i.data_offsets.0))
        .max()
        .unwrap_or(0);
    let pinned = gpu.alloc_host_pinned(max_len.max(1))?;
    let staging = unsafe { std::slice::from_raw_parts_mut(pinned, max_len) };
    let result = (|| -> Result<()> {
        for name in tensor_names {
            if skip_fn(name) {
                *skipped += 1;
                continue;
            }
            let info = infos
                .get(name.as_str())
                .with_context(|| format!("tensor {name} missing from shard metadata"))?;
            let (begin, end) = info.data_offsets;
            // n-gram TABLES are deferred, not uploaded — same contract as the
            // mmap path. data_offsets are relative to the data section start.
            if crate::weights::is_ngram_table(name) {
                deferred.insert(
                    name.to_string(),
                    crate::weights::DeferredTensor {
                        path: shard_path.to_path_buf(),
                        offset: data_base + begin as u64,
                        shape: info.shape.clone(),
                        dtype: WeightDtype::from_safetensors(info.dtype)?,
                    },
                );
                *skipped += 1;
                continue;
            }
            let len = end.saturating_sub(begin);
            file.seek(SeekFrom::Start(data_base + begin as u64))?;
            file.read_exact(&mut staging[..len])
                .with_context(|| format!("Failed to read tensor {name}"))?;
            upload_one(
                name,
                &staging[..len],
                info.dtype,
                info.shape.clone(),
                gpu,
                weights,
                offload_logged,
            )?;
        }
        Ok(())
    })();
    gpu.free_host_pinned(pinned, max_len.max(1))?;
    result
}

pub(super) fn load_single(
    path: &Path,
    gpu: &dyn GpuBackend,
    oom_reserve_bytes: usize,
    skip_fn: &dyn Fn(&str) -> bool,
) -> Result<HashMap<String, WeightTensor>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

    let mut weights = HashMap::new();
    for (name, view) in tensors.tensors() {
        if skip_fn(&name) {
            continue;
        }
        let shape: Vec<usize> = view.shape().to_vec();
        // F16: convert to BF16 at load — see load_sharded above. EXL3 scale
        // vectors (`keeps_raw_f16`) keep their raw FP16 bytes.
        let converted: Vec<u8>;
        let (data, dtype): (&[u8], _) = if view.dtype() == safetensors::Dtype::F16 {
            if crate::weights::keeps_raw_f16(&name) {
                (view.data(), WeightDtype::FP16)
            } else {
                converted = f16_to_bf16_bytes(view.data());
                (&converted, WeightDtype::BF16)
            }
        } else {
            (view.data(), WeightDtype::from_safetensors(view.dtype())?)
        };

        let ptr = gpu.alloc(data.len())?;
        gpu.copy_h2d(data, ptr)?;

        weights.insert(name, WeightTensor { ptr, shape, dtype });
    }

    // Drop mmap before evicting page cache.
    drop(tensors);
    drop(mmap);
    evict_page_cache(&file);

    // OOM guard after single-file load
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    check_oom_guard(
        gpu,
        oom_reserve_bytes,
        &format!("weight loading ({file_name})"),
    )?;

    Ok(weights)
}
