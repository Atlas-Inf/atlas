// SPDX-License-Identifier: AGPL-3.0-only

//! [`CsfAssembler`]: gathers CSF-compressed NVFP4 scale pairs as the fast
//! loader streams tensors past it, and decodes each scale once both streams
//! have arrived (`atlas_core::modelopt_codec`). The store only ever sees the
//! decoded `weight_scale`.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::modelopt_codec::{csf_scale_shape, decode_csf_scales};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

const FIXED: &str = ".nvfp4_csf_fixed";
const EXCEPTIONS: &str = ".nvfp4_csf_exceptions";

/// One decoded `weight_scale`, ready to upload as FP8 E4M3.
pub(super) struct DecodedScale {
    pub(super) name: String,
    pub(super) shape: Vec<usize>,
    pub(super) bytes: Vec<u8>,
}

/// The streams of one scale gathered so far.
#[derive(Default)]
struct Pending {
    /// `(fixed_rows, fixed_cols, bytes)`.
    fixed: Option<(usize, usize, Vec<u8>)>,
    exceptions: Option<Vec<u8>>,
}

/// A pair can arrive in either order and across shards, so a stream is held
/// here until its partner comes.
#[derive(Default)]
pub(super) struct CsfAssembler {
    pending: HashMap<String, Pending>,
    /// Every scale decoded so far.
    decoded: HashSet<String>,
    /// Plain `.weight_scale` tensors the loader kept as they were.
    plain_scales: HashSet<String>,
    bytes_in: usize,
    bytes_out: usize,
    time: Duration,
}

impl CsfAssembler {
    /// Hand over one stream. Returns the decoded scale when this completes
    /// its pair, `None` while its partner is still to come.
    pub(super) fn offer(
        &mut self,
        name: &str,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<Option<DecodedScale>> {
        let (base, is_fixed) = match (name.strip_suffix(FIXED), name.strip_suffix(EXCEPTIONS)) {
            (Some(base), _) => (base, true),
            (_, Some(base)) => (base, false),
            _ => bail!("{name} is not a CSF stream"),
        };
        ensure!(
            !self.plain_scales.contains(base),
            "{base} is in the checkpoint both plain and CSF-compressed"
        );
        ensure!(!self.decoded.contains(base), "{name} appears twice");
        let slot = self.pending.entry(base.to_string()).or_default();
        if is_fixed {
            let &[rows, cols] = shape else {
                bail!("{name} is {shape:?}, expected 2-D");
            };
            ensure!(slot.fixed.is_none(), "{name} appears twice");
            slot.fixed = Some((rows, cols, bytes.to_vec()));
        } else {
            ensure!(slot.exceptions.is_none(), "{name} appears twice");
            slot.exceptions = Some(bytes.to_vec());
        }
        if slot.fixed.is_none() || slot.exceptions.is_none() {
            return Ok(None);
        }
        let Some(Pending {
            fixed: Some((fr, fc, fixed)),
            exceptions: Some(exceptions),
        }) = self.pending.remove(base)
        else {
            unreachable!("the pair was just checked complete")
        };
        let t = Instant::now();
        let (rows, cols) = csf_scale_shape(fr, fc).with_context(|| format!("CSF scale {base}"))?;
        let out = decode_csf_scales(&fixed, fr, fc, &exceptions)
            .with_context(|| format!("decoding CSF scale {base}"))?;
        self.time += t.elapsed();
        self.decoded.insert(base.to_string());
        self.bytes_in += fixed.len() + exceptions.len();
        self.bytes_out += out.len();
        Ok(Some(DecodedScale {
            name: base.to_string(),
            shape: vec![rows, cols],
            bytes: out,
        }))
    }

    /// Report a tensor the loader kept as it was. Refuses a scale already
    /// decoded from a pair, and remembers a plain `weight_scale` so a later
    /// pair for it is refused.
    pub(super) fn note_plain(&mut self, name: &str) -> Result<()> {
        ensure!(
            !self.decoded.contains(name) && !self.pending.contains_key(name),
            "{name} is in the checkpoint both plain and CSF-compressed"
        );
        if name.ends_with(".weight_scale") {
            self.plain_scales.insert(name.to_string());
        }
        Ok(())
    }

    /// Refuse a load that left a stream without its partner: the model would
    /// look for that scale and not find it. Logs what the decode cost.
    pub(super) fn finish(self) -> Result<()> {
        if !self.pending.is_empty() {
            let mut names: Vec<_> = self.pending.keys().take(4).cloned().collect();
            names.sort();
            bail!(
                "{} CSF scale(s) without a partner stream, e.g. {names:?}: the checkpoint is \
                 incomplete or a skip filter split a pair",
                self.pending.len()
            );
        }
        if !self.decoded.is_empty() {
            let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
            tracing::info!(
                "Decoded {} CSF NVFP4 scales: {:.2} GiB compressed -> {:.2} GiB E4M3 in {:.2} s \
                 on the load thread",
                self.decoded.len(),
                gib(self.bytes_in),
                gib(self.bytes_out),
                self.time.as_secs_f64(),
            );
        }
        Ok(())
    }
}
