// SPDX-License-Identifier: AGPL-3.0-only

//! [`CsfAssembler`]: gathers the components of compressed NVFP4 scales, in
//! either encoding, as a loader streams tensors past it, and decodes each
//! scale once its set is complete. Split out of `nvfp4_csf.rs` (500-line cap).

use anyhow::{Context, Result, bail, ensure};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use super::super::nvfp4_csf32::{Csf32Parts, GeometryCache, shape_of};
use super::{CsfEncoding, csf_part, decode_nvfp4_csf, geometry};

/// Refuse a decoded `[rows, cols]` scale that is not the `[n, k/16]` E4M3
/// matrix its packed NVFP4 weight `[n, k/2]` needs.
fn check_against_weight(scale: &str, got: [usize; 2], weight: Option<&[usize]>) -> Result<()> {
    let wname = scale
        .strip_suffix("_scale")
        .filter(|w| w.ends_with(".weight"))
        .with_context(|| format!("CSF scale {scale} is not a `.weight_scale`"))?;
    let Some(w) = weight else {
        bail!("CSF scale {scale} decoded, but there is no `{wname}` for it to scale");
    };
    let want = match *w {
        [n, k_packed] => k_packed
            .checked_mul(2)
            .filter(|k| k % 16 == 0)
            .map(|k| [n, k / 16]),
        _ => None,
    };
    ensure!(
        want == Some(got),
        "CSF scale {scale} decoded to {got:?}, but its weight `{wname}` {w:?} needs \
         {want:?} ([n, k/2] packed NVFP4 -> [n, k/16])"
    );
    Ok(())
}

/// One decoded `weight_scale`, ready to upload as FP8 E4M3.
pub struct DecodedScale {
    pub name: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

struct Part {
    shape: Vec<usize>,
    /// Empty for a deferred component (only its shape is known here).
    bytes: Vec<u8>,
}

/// The components of one scale gathered so far, indexed by [`CsfPart::slot`].
struct Pending {
    encoding: CsfEncoding,
    /// The loader deferred this scale's components (expert TP): they stay on
    /// disk and are decoded when the rank's slice is read.
    deferred: bool,
    parts: [Option<Part>; 5],
}

/// Decode one complete set: `parts` holds each component's header shape and
/// raw bytes, in slot order. Shared by the assembler and the deferred-slice
/// reader (`fast_weights::with_deferred_slice`).
pub fn decode_parts(
    encoding: CsfEncoding,
    parts: &[(&[usize], &[u8])],
    geometries: &mut GeometryCache,
) -> Result<(Vec<u8>, [usize; 2])> {
    ensure!(
        parts.len() == encoding.part_count(),
        "{} scale needs {} components, got {}",
        encoding.label(),
        encoding.part_count(),
        parts.len()
    );
    match encoding {
        CsfEncoding::Slab => decode_nvfp4_csf(parts[0].1, parts[0].0, parts[1].1),
        CsfEncoding::Blocks32 => geometries.decode(&Csf32Parts {
            bases: parts[0].1,
            codes: parts[1].1,
            page_bases: parts[2].1,
            offsets: parts[3].1,
            exceptions: parts[4].1,
        }),
    }
}

/// The decoded `[rows, cols]` of a set from its components' header shapes,
/// for a deferred set whose bytes are not read at load.
fn decoded_shape(encoding: CsfEncoding, shapes: &[&[usize]]) -> Result<[usize; 2]> {
    match encoding {
        CsfEncoding::Slab => geometry(shapes[0]).map(|(rows, cols, _)| [rows, cols]),
        CsfEncoding::Blocks32 => shape_of(shapes[0][0], shapes[1][0]),
    }
}

/// Gathers the components of compressed scales as a loader streams tensors
/// past it, in any order and across shards, and decodes each scale once all
/// of its components (two for the slab encoding, five for `csf_32x2_v1`) have
/// arrived.
///
/// The disk loaders (`SafetensorsLoader`, `FastSafetensorsLoader`) route CSF
/// parts here instead of uploading them, so the store only ever sees the
/// decoded `weight_scale`, and the U16/U32 component streams (dtypes the
/// store does not carry) never reach it. Under expert TP the fast loader
/// instead [defers](Self::defer) a scale's components, which are checked here
/// by shape and decoded per slice later. The RDMA weight tier, which copies
/// tensors as they are, refuses CSF parts.
///
/// It is also the one place that knows which scales came from components, so
/// the loaders report every plain tensor too ([`Self::note_plain`]): a scale
/// both plain and CSF-compressed is refused whichever arrives first.
#[derive(Default)]
pub struct CsfAssembler {
    pending: HashMap<String, Pending>,
    /// Every scale decoded (or, deferred, fully gathered) so far, with its
    /// `[rows, cols]`.
    decoded: HashMap<String, [usize; 2]>,
    /// How many of `decoded` were deferred rather than decoded.
    deferred: usize,
    /// Plain `.weight_scale` tensors the loader uploaded as they were.
    plain_scales: HashSet<String>,
    /// `csf_32x2_v1` block order per scale shape.
    geometries: GeometryCache,
    bytes_in: usize,
    bytes_out: usize,
    time: Duration,
}

impl CsfAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hand over one component. Returns the decoded scale when this completes
    /// its set, `None` while a component is still to come.
    pub fn offer(
        &mut self,
        name: &str,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<Option<DecodedScale>> {
        self.add(name, shape, bytes.len(), Some(bytes))
    }

    /// Record a component the loader deferred (left on disk, `byte_len`
    /// bytes): it is validated like an offered one, and a complete deferred
    /// set is checked against its weight in [`Self::finish`], but nothing is
    /// read or decoded here.
    pub fn defer(&mut self, name: &str, shape: &[usize], byte_len: usize) -> Result<()> {
        self.add(name, shape, byte_len, None).map(drop)
    }

    fn add(
        &mut self,
        name: &str,
        shape: &[usize],
        byte_len: usize,
        bytes: Option<&[u8]>,
    ) -> Result<Option<DecodedScale>> {
        let (base, part) =
            csf_part(name).with_context(|| format!("{name} is not a CSF stream tensor"))?;
        ensure!(
            !self.decoded.contains_key(base),
            "{base}: compressed scale already completed"
        );
        let elem = part.elem_bytes();
        // The byte length against the header shape pins the dtype (U8 vs U16
        // vs U32) without trusting the dtype string: a stream with another
        // width cannot satisfy both.
        let want = shape
            .iter()
            .try_fold(elem, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: shape {shape:?} overflows"))?;
        ensure!(
            want == byte_len,
            "{name}: {byte_len} bytes for shape {shape:?} — expected {want} ({})",
            ["", "U8", "U16", "", "U32"][elem],
        );
        let encoding = part.encoding();
        ensure!(
            encoding == CsfEncoding::Slab || shape.len() == 1,
            "{name}: csf_32x2_v1 components are 1-D, got shape {shape:?}"
        );
        ensure!(
            !self.plain_scales.contains(base),
            "{base} is in the checkpoint both plain and CSF-compressed"
        );
        let deferred = bytes.is_none();
        let slot = self.pending.entry(base.to_string()).or_insert(Pending {
            encoding,
            deferred,
            parts: Default::default(),
        });
        ensure!(
            slot.encoding == encoding,
            "{base} has components of both the {} and the {} encodings",
            slot.encoding.label(),
            encoding.label()
        );
        ensure!(
            slot.deferred == deferred,
            "{base} has components both deferred and read — a skip/defer filter split its set"
        );
        let held = &mut slot.parts[part.slot()];
        ensure!(held.is_none(), "{name} appears twice");
        *held = Some(Part {
            shape: shape.to_vec(),
            bytes: bytes.map(<[u8]>::to_vec).unwrap_or_default(),
        });
        let n = encoding.part_count();
        if slot.parts[..n].iter().any(Option::is_none) {
            return Ok(None);
        }
        let Some(Pending { parts, .. }) = self.pending.remove(base) else {
            unreachable!("the set was just checked complete")
        };
        let p: Vec<Part> = parts.into_iter().take(n).flatten().collect();
        if deferred {
            let shapes: Vec<&[usize]> = p.iter().map(|x| x.shape.as_slice()).collect();
            let got = decoded_shape(encoding, &shapes)
                .with_context(|| format!("deferred {} scale {base}", encoding.label()))?;
            self.decoded.insert(base.to_string(), got);
            self.deferred += 1;
            return Ok(None);
        }
        let t = Instant::now();
        let views: Vec<(&[usize], &[u8])> = p
            .iter()
            .map(|x| (x.shape.as_slice(), x.bytes.as_slice()))
            .collect();
        let (out, [rows, cols]) = decode_parts(encoding, &views, &mut self.geometries)
            .with_context(|| format!("decoding {} scale {base}", encoding.label()))?;
        self.time += t.elapsed();
        self.decoded.insert(base.to_string(), [rows, cols]);
        self.bytes_in += p.iter().map(|x| x.bytes.len()).sum::<usize>();
        self.bytes_out += out.len();
        Ok(Some(DecodedScale {
            name: base.to_string(),
            shape: vec![rows, cols],
            bytes: out,
        }))
    }

    /// Report a tensor the loader uploaded as it was (not a CSF half).
    /// Refuses a scale the assembler already decoded from a pair, and
    /// remembers a plain `weight_scale` so a later pair for it is refused.
    pub fn note_plain(&mut self, name: &str) -> Result<()> {
        ensure!(
            !self.decoded.contains_key(name),
            "{name} is in the checkpoint both plain and CSF-compressed"
        );
        if name.ends_with(".weight_scale") {
            self.plain_scales.insert(name.to_string());
        }
        Ok(())
    }

    /// Refuse a load that left a half without its partner, or decoded (or
    /// deferred) a scale that is not the `[n, k/16]` its packed weight
    /// `[n, k/2]` needs
    /// (`weight_shape` looks a loaded or deferred tensor's shape up by name),
    /// and log what the decode cost. A lone half means a scale the model will
    /// look for and not find — or worse, find stale — so it is an error, not a
    /// warning.
    ///
    /// The shape check runs here, at the end of the load and before any store
    /// exists, rather than before each upload: a pair and its weight can sit
    /// in different shards, in either order.
    pub fn finish(self, weight_shape: impl Fn(&str) -> Option<Vec<usize>>) -> Result<()> {
        if !self.pending.is_empty() {
            let mut names: Vec<_> = self.pending.keys().take(4).cloned().collect();
            names.sort();
            bail!(
                "{} CSF scale(s) with components still missing (without a partner), e.g. \
                 {names:?} — the checkpoint is incomplete or a skip filter split a set",
                self.pending.len()
            );
        }
        for (name, &got) in &self.decoded {
            let w = name.strip_suffix("_scale").and_then(&weight_shape);
            check_against_weight(name, got, w.as_deref())?;
        }
        if self.deferred > 0 {
            tracing::info!(
                "Deferred {} CSF NVFP4 scale(s) to be decoded per expert-TP slice",
                self.deferred
            );
        }
        if self.decoded.len() > self.deferred {
            let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
            tracing::info!(
                "Decoded {} CSF NVFP4 scale matrices: {:.2} GiB compressed -> {:.2} GiB E4M3 \
                 in {:.2} s on the load thread",
                self.decoded.len() - self.deferred,
                gib(self.bytes_in),
                gib(self.bytes_out),
                self.time.as_secs_f64(),
            );
        }
        Ok(())
    }
}
