// SPDX-License-Identifier: AGPL-3.0-only

//! Fast safetensors loader (InstantTensor-style) — pure Rust.
//!
//! Two wins over the mmap-based loader in [`crate::weights`]:
//!
//! 1. **`O_DIRECT`** reads. Bypasses the OS page cache, so the bytes never
//!    compete with GPU allocations on GB10 unified memory. The mmap path
//!    already works around this with `POSIX_FADV_DONTNEED` post-load; here
//!    we avoid the pollution in the first place.
//! 2. **Pipelined read/copy**. One background reader thread fetches the
//!    next tensor while the main thread does `copy_h2d` for the current
//!    one. Overlaps disk I/O with the host→device memcpy.
//!
//! Behavioural parity with [`crate::weights::SafetensorsLoader`] is
//! preserved — same EP filtering, same OOM pre-flight, same UVM fallback
//! on GPU allocation failure, same extra-weights handling.

use crate::gpu::GpuBackend;
use crate::weights::{
    WeightLoader, WeightStore, WeightTensor, check_oom_guard, estimate_has_fp8, estimate_load_bytes,
};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod direct_io;
mod header;

use header::resolve_shards;

/// Pure-Rust InstantTensor-style loader. Same public shape as
/// [`crate::weights::SafetensorsLoader`].
pub struct FastSafetensorsLoader {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
    pub peak_memory_multiplier: Option<f64>,
    /// Skip the W4A4 `*.input_scale` activation scales at load.
    ///
    /// ModelOpt NVFP4 checkpoints ship one 0-dim F32 scalar per quantized
    /// projection. On a 512-expert model that is ~74k four-byte allocations,
    /// each taking a full allocation granule — GBs of padding for values
    /// Atlas never reads, because it serves w4a16 (BF16 activations) and the
    /// NVFP4 loader already treats the key as optional.
    ///
    /// OPT-IN: `step3p7` reads this key on its own path, so it must stay off
    /// unless the model's loader is known not to need it.
    pub skip_activation_scales: bool,
    /// Skip `mtp.*` tensors at load.
    ///
    /// For models whose loader deliberately does not build an MTP head,
    /// uploading its weights is pure waste — on Qwen3.8-Flash-Next that is a
    /// 1.49 GB expert shard plus the MTP backbone, held resident while the KV
    /// cache goes without.
    ///
    /// OPT-IN: a model that DOES build an MTP head must keep them, so this is
    /// set only where `load_mtp_weights` is known to return `None`.
    pub skip_mtp: bool,
    /// Skip any tensor whose name starts with one of these prefixes — dead
    /// namespaces the model's loader never reads (see `SafetensorsLoader::
    /// skip_prefixes`). Same opt-in contract.
    pub skip_prefixes: Vec<String>,
    /// Pack every kept tensor in a shard into ONE `gpu.alloc` per shard,
    /// each tensor at a 4096-byte-aligned interior offset. Per-tensor
    /// cuMemAlloc rounds ≥1 MB allocations to 2 MB pages: on NVFP4 MoE
    /// checkpoints (~1024 packed+scale+scalar tensors per ~1.6 GB shard)
    /// that waste is ~0.6 GB/shard — ~24 GB across 40 MoE shards on
    /// Nemotron 3.5 Super VL, i.e. the whole KV budget.
    ///
    /// OPT-IN only, via the weight loader's `slab_pack_weights` policy —
    /// callers must have audited that no `gpu.free` reachable on their
    /// load path frees a store-owned tensor pointer (interior frees are
    /// also harmless: the CUDA backend turns them into no-ops once the
    /// slab is registered). Single-GPU only.
    pub pack_slab: bool,
    /// When true (default), attempt `O_DIRECT`; fall back to buffered reads if
    /// the filesystem rejects it (tmpfs, overlayfs, some FUSE backends).
    pub try_direct_io: bool,
    /// Per-shard heuristic cap: if a shard's tensor count exceeds this,
    /// we skip `O_DIRECT` for that shard and fall back to buffered +
    /// pipelined reads even when [`Self::try_direct_io`] is `true`.
    ///
    /// Motivation: `O_DIRECT`'s 4 KiB-aligned per-tensor `pread` has a
    /// fixed syscall + copy overhead that kernel readahead amortises for
    /// free on the buffered path. Benchmarks on GB10 showed buffered wins
    /// above ~5k tensors/shard; O_DIRECT wins below. Set to [`usize::MAX`]
    /// to disable.
    pub direct_io_tensor_cap: usize,
    /// When true, advise the kernel to read a whole buffered shard
    /// sequentially before the per-tensor copy loop starts. This helps NFS
    /// mounts where many small tensor reads defeat normal readahead.
    pub prefetch_shards: bool,
    /// Name substrings marking tensors that are **read from disk at use time
    /// and never made resident**. They are excluded from the load AND from the
    /// pre-flight estimate, since counting bytes we will not allocate refuses
    /// models that would in fact fit.
    ///
    /// This exists for embedding tables that are gathered by row rather than
    /// multiplied. Qwen3.8-Flash-Next's n-gram table is 51.2 B parameters —
    /// 41% of that checkpoint — and one token touches ~2.5 KB of it. Loading
    /// it puts the pre-flight at 163 GB on a 119 GB box; skipping it puts the
    /// same model at ~96 GB.
    ///
    /// A tensor named here MUST have a reader that goes to disk
    /// (`atlas_core::ngram_table` is the one that does). Nothing checks that
    /// from here — a name listed without a reader is simply absent from the
    /// store, and the model loader will fail on it by name.
    pub demand_paged_patterns: Vec<String>,
}

/// Default tensor-count cap for per-shard `O_DIRECT`. Above this, the fast
/// loader uses buffered reads even when `try_direct_io = true`. See the
/// field doc on [`FastSafetensorsLoader::direct_io_tensor_cap`].
pub const DEFAULT_DIRECT_IO_TENSOR_CAP: usize = 5000;

impl Default for FastSafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[path = "skip.rs"]
mod skip;

impl FastSafetensorsLoader {
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_prefixes: Vec::new(),
            pack_slab: false,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            demand_paged_patterns: Vec::new(),
        }
    }

    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_prefixes: Vec::new(),
            pack_slab: false,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            demand_paged_patterns: Vec::new(),
        }
    }
}

impl WeightLoader for FastSafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Resolve shard list (sharded index, single file, or unindexed shards).
        let (shard_files, tensor_to_shard): (Vec<PathBuf>, Option<HashMap<String, String>>) =
            resolve_shards(model_dir)?;

        // Pre-flight OOM estimate (identical to SafetensorsLoader).
        //
        // The n-gram tables are DEFERRED further down — they are never
        // uploaded, so counting them here refuses a model that fits. On
        // LongCat-Flash-Lite they are 62.8 of the checkpoint's 138 GB, which
        // is the difference between a 167 GB "peak" and a 98 GB one.
        let preflight_skip = |name: &str| skip_fn(name) || crate::weights::is_ngram_table(name);
        {
            let estimated = estimate_load_bytes(&shard_files, &preflight_skip)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &preflight_skip)?;
            let mult = self
                .peak_memory_multiplier
                .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let peak = (estimated as f64 * mult) as usize;
            let free = gpu.free_memory()?;
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Fast-load pre-flight: {:.2} GB on-disk, {:.1}x overhead = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                gib(estimated),
                mult,
                gib(peak),
                gib(free),
                gib(oom_reserve_bytes),
                has_fp8,
            );
            crate::progress::preflight(gib(estimated), gib(free));
            if peak + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight: peak {:.2} GB + {:.2} GB reserve exceeds {:.2} GB free. \
                     Use a smaller quantization or add more GPUs for EP.",
                    gib(peak),
                    gib(oom_reserve_bytes),
                    gib(free),
                );
            }
        }

        // Load each shard. Loaded tensors filtered by EP rules upstream.
        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        // Locations of tensors deliberately NOT uploaded (the n-gram tables).
        let mut deferred: HashMap<String, crate::weights::DeferredTensor> = HashMap::new();
        let total_shards = shard_files.len();
        let initial_free = gpu.free_memory()?;
        let mut offload_logged = false;

        for (i, shard_path) in shard_files.iter().enumerate() {
            // When an index is present, only load the tensors it routes here;
            // otherwise load everything in the shard. `None` means "load all".
            let shard_name = shard_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let tensor_filter: Option<Vec<String>> = tensor_to_shard.as_ref().map(|map| {
                map.iter()
                    .filter(|(_, s)| *s == shard_name)
                    .map(|(t, _)| t.clone())
                    .collect()
            });

            tracing::info!(
                "Fast-loading shard {}/{}: {}{}",
                i + 1,
                total_shards,
                shard_name,
                tensor_filter
                    .as_ref()
                    .map(|v| format!(" ({} tensors)", v.len()))
                    .unwrap_or_default(),
            );
            crate::progress::shard_start(i + 1, total_shards, shard_name);

            shard::load_shard_fast(
                shard_path,
                tensor_filter.as_deref(),
                gpu,
                &skip_fn,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                self.prefetch_shards,
                self.pack_slab,
                &mut weights,
                &mut deferred,
                &mut offload_logged,
            )?;

            let free_now = gpu.free_memory().unwrap_or(0);
            let used = initial_free.saturating_sub(free_now);
            tracing::info!(
                "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            crate::progress::shard_done(
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if !offload_logged {
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("fast weight loading (shard {}/{})", i + 1, total_shards),
                )?;
            }
        }

        // Extra weights (e.g. MTP grafted from another quantization).
        let no_skip = |_: &str| false;
        let extra = model_dir.join("extra_weights.safetensors");
        if extra.exists() {
            tracing::info!("Fast-loading extra_weights.safetensors");
            let mut extra_offload = false;
            shard::load_shard_fast(
                &extra,
                None,
                gpu,
                &no_skip,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                self.prefetch_shards,
                self.pack_slab,
                &mut weights,
                &mut deferred,
                &mut extra_offload,
            )?;
        }

        tracing::info!("Fast-loaded {} weight tensors", weights.len());
        let mut store = WeightStore::from_map(weights);
        for (name, d) in deferred {
            store.defer(name, d);
        }
        Ok(store)
    }
}

mod shard;
