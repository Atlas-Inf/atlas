// SPDX-License-Identifier: AGPL-3.0-only

//! Size an image from its header before decoding it.
//!
//! A few-KB PNG may declare 16384x16384; decoded and patched it is hundreds of
//! MiB of f32. Admission must know an item's vision tokens BEFORE that, so it
//! can refuse a request over its limits without allocating anything, and
//! charge the in-flight budget for exactly what the decode will hold.
//! [`probe_image`] reads only the header (format and dimensions) and fixes the
//! geometry with the same function the decode uses (`still_target`);
//! [`ProbedImage::preprocess`] then decodes and checks the two agree.

use super::*;

/// An image whose geometry is known and whose pixels are not yet decoded.
pub struct ProbedImage {
    pub(super) bytes: Vec<u8>,
    pub(super) fmt: image::ImageFormat,
    pub(super) max_pixels: Option<usize>,
    /// Target (H, W) in pixels of the oriented image, as the decode sizes it.
    pub(super) target: (u32, u32),
    /// The decoded buffer's bytes (native color type), from the header.
    pub(super) decoded_bytes: u64,
    /// Patch grid the decode will produce.
    pub grid_h: usize,
    pub grid_w: usize,
}

impl std::fmt::Debug for ProbedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ProbedImage {{ {:?}, {} bytes, grid {}x{} }}",
            self.fmt,
            self.bytes.len(),
            self.grid_h,
            self.grid_w
        )
    }
}

impl ProbedImage {
    /// Vision tokens (merged rows) the image will occupy.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        image_pad_count(self.grid_h, self.grid_w, spatial_merge_size)
    }

    /// Patches the encoder will see.
    pub fn patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Decode and patch: `(pixels, grid_h, grid_w)` as
    /// [`preprocess_image_with_max_pixels`] returns them.
    pub fn preprocess(self, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
        preprocess_probed(self, vcfg)
    }
}

/// Read `data_uri`'s header and fix the geometry its decode will have.
/// Allocates only the compressed bytes. The EXIF orientation is read too: the
/// decode turns the image upright before sizing it, and GLM's sizing is not
/// symmetric in its sides.
pub fn probe_image(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<ProbedImage> {
    // Before anything divides by them.
    validate_geometry(vcfg)?;
    let (_mime, bytes) = decode_data_uri_bytes(data_uri)?;
    let fmt = sniff_format(&bytes)?;
    let mut decoder = limited_reader(&bytes, fmt)
        .into_decoder()
        .context("image decode failed")?;
    let (w, h) = decoder.dimensions();
    ensure!(
        w > 0 && h > 0 && w <= DECODE_MAX_SIDE && h <= DECODE_MAX_SIDE,
        "image dimensions {w}x{h} are outside 1..={DECODE_MAX_SIDE}"
    );
    let decoded_bytes = decoder.total_bytes();
    check_decode_size(decoded_bytes)?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    drop(decoder);
    let (w, h) = if orientation_swaps_sides(orientation) {
        (h, w)
    } else {
        (w, h)
    };
    let (th, tw) = still_target(vcfg, h, w, max_pixels)?;
    Ok(ProbedImage {
        bytes,
        fmt,
        max_pixels,
        target: (th, tw),
        decoded_bytes,
        grid_h: th as usize / vcfg.patch_size,
        grid_w: tw as usize / vcfg.patch_size,
    })
}

/// A counting gate: at most `limit()` holders at once; acquiring waits.
pub(super) struct Gate {
    busy: std::sync::Mutex<usize>,
    free: std::sync::Condvar,
    limit: fn() -> usize,
}

/// One holder of a [`Gate`], released on drop.
pub(super) struct GateSlot(&'static Gate);

impl Gate {
    pub(super) const fn new(limit: fn() -> usize) -> Self {
        Self {
            busy: std::sync::Mutex::new(0),
            free: std::sync::Condvar::new(),
            limit,
        }
    }

    pub(super) fn acquire(&'static self) -> GateSlot {
        let limit = (self.limit)();
        let mut busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        while *busy >= limit {
            busy = self.free.wait(busy).unwrap_or_else(|e| e.into_inner());
        }
        *busy += 1;
        GateSlot(self)
    }
}

impl Drop for GateSlot {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        self.0.free.notify_one();
    }
}

/// Decodes running at once, server-wide (the request path runs on a blocking
/// pool of hundreds of threads): min(cores, 8), and of those at most 2 past
/// [`DECODE_LARGE`]. Each holds its decoded buffer, an RGB8 copy of at most
/// the same size, and its f32 patches (at most 96 MiB: 16384 patches x
/// 6 KiB). Worst-case transient host memory for decodes: decoded buffers
/// 2 x 640 + 6 x 192 MiB = 2.4 GiB, twice that with the RGB8 copies, plus
/// 8 x 96 MiB of patches: about 5.5 GiB (an EXIF rotation adds one more copy
/// of the image being rotated).
static DECODES: Gate = Gate::new(|| {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8)
});
static LARGE_DECODES: Gate = Gate::new(|| 2);

/// The slots one image's decode and patching holds, released on drop.
pub(super) struct DecodeSlot {
    _large: Option<GateSlot>,
    _any: GateSlot,
}

/// Take a decode slot, and the large-decode gate first for an image whose
/// decoded buffer exceeds [`DECODE_LARGE`] (a large decode never holds a
/// small slot while it waits for the large gate).
pub(super) fn decode_slot(decoded_bytes: u64) -> DecodeSlot {
    let large = (decoded_bytes > DECODE_LARGE).then(|| LARGE_DECODES.acquire());
    DecodeSlot {
        _large: large,
        _any: DECODES.acquire(),
    }
}

/// Run a decode the probe cannot size (a video, before admission) as a large
/// decode: under both gates, so it does not add to the worst case above.
pub fn decode_gated<T>(decode: impl FnOnce() -> T) -> T {
    let _slot = decode_slot(u64::MAX);
    decode()
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
