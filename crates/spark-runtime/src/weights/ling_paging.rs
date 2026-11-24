// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded, read-only paging views for the original Ling FP8 checkpoint.
//!
//! This module owns host-side `MAP_PRIVATE` mappings of the shards named by
//! [`super::DeferredTensor`]. It intentionally exposes immutable byte slices
//! only. No CUDA pointer or lifetime promise is exposed here. A future GPU
//! consumer must retain [`LingPage`] as its explicit mapping owner until the
//! GPU work has completed. The checkpoint files must not be modified or
//! truncated while any returned page is alive, as required by `memmap2`.

use super::{DeferredTensor, WeightDtype};
use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The on-disk byte kind accepted by the Ling paging reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LingPageKind {
    /// Original Ling expert weights, stored as IEEE E4M3 bytes.
    E4M3Weight,
    PackedByte,
    E8M0Scale,
    /// Block scales stored as BF16 bytes.
    Bf16Scale,
    /// Block scales stored as F32 bytes.
    F32Scale,
}

impl LingPageKind {
    fn from_dtype(dtype: WeightDtype) -> Result<Self> {
        match dtype {
            WeightDtype::FP8E4M3 => Ok(Self::E4M3Weight),
            WeightDtype::UInt8 => Ok(Self::PackedByte),
            WeightDtype::FP8E8M0 => Ok(Self::E8M0Scale),
            WeightDtype::BF16 => Ok(Self::Bf16Scale),
            WeightDtype::FP32 => Ok(Self::F32Scale),
            other => {
                bail!("Ling paging accepts only FP8 E4M3 weights or BF16/F32 scales, got {other:?}")
            }
        }
    }

    /// Bytes per element in the original file representation.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::E4M3Weight | Self::PackedByte | Self::E8M0Scale => 1,
            Self::Bf16Scale => 2,
            Self::F32Scale => 4,
        }
    }
}

/// An immutable host view over one validated tensor range.
///
/// The `Arc<Mmap>` is deliberately retained by each page. This makes the
/// mapping lifetime explicit for callers that later hand `bytes()` to a GPU
/// operation. This type does not expose a raw pointer or claim that a CUDA
/// operation may outlive the page. The source shard must remain unchanged
/// while this page is alive.
#[derive(Clone)]
pub struct LingPage {
    mapping: Arc<Mmap>,
    range: Range<usize>,
    path: Arc<PathBuf>,
    kind: LingPageKind,
}

impl LingPage {
    /// Immutable bytes exactly covering the deferred tensor.
    pub fn bytes(&self) -> &[u8] {
        &self.mapping[self.range.clone()]
    }

    /// Raw address of [`Self::bytes`], valid while this page is retained.
    ///
    /// This is intentionally a host address.  A GPU backend that advertises
    /// pageable host-page-table access may turn it into a device address for
    /// read-only kernels; other backends must reject that operation and use
    /// the ordered staging fallback instead.
    pub fn host_ptr(&self) -> *mut u8 {
        self.bytes().as_ptr() as *mut u8
    }

    /// The validated on-disk representation of this page.
    pub fn kind(&self) -> LingPageKind {
        self.kind
    }

    /// The source shard. The path is retained for diagnostics and ownership
    /// accounting, not as a promise that the file remains externally mutable.
    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    /// Number of bytes in the page.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Whether the page contains no bytes. Valid Ling tensors are non-empty,
    /// so this is mainly useful to generic callers.
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}

/// A bounded cache of read-only shard mappings for original Ling weights.
///
/// A shard is mapped at most once for the lifetime of the reader. The bound is
/// on the number of whole-shard virtual mappings, rather than on resident
/// pages. Linux still faults only the requested ranges into memory, while the
/// explicit bound prevents an accidental unbounded mapping cache.
///
/// The caller must keep the checkpoint shards immutable for the lifetime of
/// this reader and all [`LingPage`] values it returns.
pub struct LingPagingReader {
    max_mapped_shards: usize,
    mappings: HashMap<PathBuf, Arc<Mmap>>,
}

impl LingPagingReader {
    /// Create a reader with a hard cap on distinct shard mappings.
    pub fn new(max_mapped_shards: usize) -> Result<Self> {
        if max_mapped_shards == 0 {
            bail!("Ling paging requires a non-zero shard mapping bound")
        }
        Ok(Self {
            max_mapped_shards,
            mappings: HashMap::new(),
        })
    }

    /// Number of distinct shards mapped so far.
    pub fn mapped_shards(&self) -> usize {
        self.mappings.len()
    }

    /// Maximum number of distinct shard mappings this reader may retain.
    pub fn max_mapped_shards(&self) -> usize {
        self.max_mapped_shards
    }

    /// Validate and expose one deferred Ling tensor as immutable bytes.
    pub fn page(&mut self, tensor: &DeferredTensor) -> Result<LingPage> {
        let kind = LingPageKind::from_dtype(tensor.dtype)?;
        let path = std::fs::canonicalize(&tensor.path)
            .with_context(|| format!("canonicalize Ling shard {}", tensor.path.display()))?;
        let mapping = if let Some(mapping) = self.mappings.get(&path) {
            Arc::clone(mapping)
        } else {
            if self.mappings.len() >= self.max_mapped_shards {
                bail!(
                    "Ling paging shard mapping bound {} reached at {}",
                    self.max_mapped_shards,
                    path.display()
                );
            }
            let mapping = Arc::new(map_private_read_only(&path)?);
            // Validate before caching a newly touched shard. An invalid
            // request must not consume one of the bounded mapping slots.
            let range = checked_range(&mapping, tensor, kind)?;
            self.mappings.insert(path.clone(), Arc::clone(&mapping));
            return Ok(LingPage {
                mapping,
                range,
                path: Arc::new(path),
                kind,
            });
        };
        let range = checked_range(&mapping, tensor, kind)?;
        Ok(LingPage {
            mapping,
            range,
            path: Arc::new(path),
            kind,
        })
    }

    /// Read an original E4M3 expert weight, rejecting scale or other dtypes.
    pub fn e4m3_weight(&mut self, tensor: &DeferredTensor) -> Result<LingPage> {
        let page = self.page(tensor)?;
        if page.kind != LingPageKind::E4M3Weight {
            bail!("Ling tensor is not an original E4M3 expert weight")
        }
        Ok(page)
    }

    /// Read a BF16 block scale, rejecting weights and F32 scales.
    pub fn bf16_scale(&mut self, tensor: &DeferredTensor) -> Result<LingPage> {
        let page = self.page(tensor)?;
        if page.kind != LingPageKind::Bf16Scale {
            bail!("Ling tensor is not a BF16 block scale")
        }
        Ok(page)
    }

    /// Read an F32 block scale, rejecting weights and BF16 scales.
    pub fn f32_scale(&mut self, tensor: &DeferredTensor) -> Result<LingPage> {
        let page = self.page(tensor)?;
        if page.kind != LingPageKind::F32Scale {
            bail!("Ling tensor is not an F32 block scale")
        }
        Ok(page)
    }
}

fn map_private_read_only(path: &Path) -> Result<Mmap> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open Ling shard {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("stat Ling shard {}", path.display()))?
        .len();
    if file_len == 0 {
        bail!("Ling shard {} is empty", path.display())
    }
    let map_len = usize::try_from(file_len).map_err(|_| {
        anyhow::anyhow!(
            "Ling shard {} is too large for this process",
            path.display()
        )
    })?;
    // SAFETY: `LingPagingReader`'s public contract requires the checkpoint
    // shards to remain unchanged and untruncated for the lifetime of every
    // returned page. `file` is opened read-only and remains alive for the
    // mapping call. `map_copy_read_only` creates a private, read-only mapping;
    // the returned Mmap owns that mapping after this function returns.
    unsafe {
        MmapOptions::new()
            .len(map_len)
            .map_copy_read_only(&file)
            .with_context(|| format!("MAP_PRIVATE Ling shard {}", path.display()))
    }
}

fn checked_range(
    mapping: &Mmap,
    tensor: &DeferredTensor,
    kind: LingPageKind,
) -> Result<Range<usize>> {
    let elements = tensor.shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| anyhow::anyhow!("Ling tensor shape overflows usize"))
    })?;
    let bytes = elements
        .checked_mul(kind.element_bytes())
        .ok_or_else(|| anyhow::anyhow!("Ling tensor byte length overflows usize"))?;
    let start = usize::try_from(tensor.offset)
        .map_err(|_| anyhow::anyhow!("Ling tensor offset does not fit usize"))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| anyhow::anyhow!("Ling tensor range overflows usize"))?;
    if start > mapping.len() || end > mapping.len() {
        bail!(
            "Ling tensor range {}..{} exceeds shard mapping length {}",
            start,
            end,
            mapping.len()
        )
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::{LingPageKind, LingPagingReader};
    use crate::weights::{DeferredTensor, WeightDtype};
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    /// Small dependency-free temporary directory used by the synthetic shard
    /// tests. The production crate intentionally has no `tempfile` dependency.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("atlas-ling-paging-{}-{serial}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn shard(path: &std::path::Path, bytes: &[u8]) {
        let mut file = File::create(path).unwrap();
        // Synthetic safetensors-like prefix: an 8-byte header length followed
        // by a small header payload. DeferredTensor.offset is absolute, just as
        // the production loader records it.
        file.write_all(&8u64.to_le_bytes()).unwrap();
        file.write_all(b"{}      ").unwrap();
        file.write_all(bytes).unwrap();
    }

    fn tensor(path: &Path, offset: u64, shape: Vec<usize>, dtype: WeightDtype) -> DeferredTensor {
        DeferredTensor {
            path: path.to_path_buf(),
            offset,
            shape,
            dtype,
        }
    }

    #[test]
    fn maps_a_shard_once_and_returns_the_exact_immutable_range() {
        let dir = TempDir::new();
        let path = dir.path().join("model-00001.safetensors");
        shard(&path, &[0x11, 0x22, 0x33, 0x44]);
        let t = tensor(&path, 16, vec![4], WeightDtype::FP8E4M3);
        let mut reader = LingPagingReader::new(1).unwrap();

        let first = reader.page(&t).unwrap();
        assert_eq!(first.kind(), LingPageKind::E4M3Weight);
        assert_eq!(first.bytes(), &[0x11, 0x22, 0x33, 0x44]);
        assert_eq!(reader.mapped_shards(), 1);

        let second = reader.e4m3_weight(&t).unwrap();
        assert_eq!(second.bytes(), first.bytes());
        assert_eq!(reader.mapped_shards(), 1);
    }

    #[test]
    fn validates_bf16_and_f32_scale_ranges_without_conversion() {
        let dir = TempDir::new();
        let path = dir.path().join("model-00001.safetensors");
        let raw = [0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f];
        shard(&path, &raw);
        let mut reader = LingPagingReader::new(1).unwrap();

        let bf16 = reader
            .bf16_scale(&tensor(&path, 16, vec![1], WeightDtype::BF16))
            .unwrap();
        assert_eq!(bf16.kind(), LingPageKind::Bf16Scale);
        assert_eq!(bf16.bytes(), &raw[..2]);
        let f32 = reader
            .f32_scale(&tensor(&path, 18, vec![1], WeightDtype::FP32))
            .unwrap();
        assert_eq!(f32.kind(), LingPageKind::F32Scale);
        assert_eq!(f32.bytes(), &raw[2..6]);
    }

    #[test]
    fn rejects_unsupported_dtype_and_out_of_bounds_ranges() {
        let dir = TempDir::new();
        let path = dir.path().join("model-00001.safetensors");
        shard(&path, &[0; 4]);
        let mut reader = LingPagingReader::new(1).unwrap();

        assert!(
            reader
                .page(&tensor(&path, 16, vec![4], WeightDtype::Int32))
                .is_err()
        );
        assert!(
            reader
                .page(&tensor(&path, 16, vec![5], WeightDtype::FP8E4M3))
                .is_err()
        );
        assert_eq!(reader.mapped_shards(), 0);
    }

    #[test]
    fn enforces_the_shard_mapping_bound() {
        let dir = TempDir::new();
        let first = dir.path().join("model-00001.safetensors");
        let second = dir.path().join("model-00002.safetensors");
        shard(&first, &[1]);
        shard(&second, &[2]);
        let mut reader = LingPagingReader::new(1).unwrap();
        reader
            .page(&tensor(&first, 16, vec![1], WeightDtype::FP8E4M3))
            .unwrap();
        assert!(
            reader
                .page(&tensor(&second, 16, vec![1], WeightDtype::FP8E4M3))
                .is_err()
        );
        assert_eq!(reader.mapped_shards(), 1);
    }

    #[test]
    fn rejects_a_zero_mapping_bound() {
        assert!(LingPagingReader::new(0).is_err());
    }
}
