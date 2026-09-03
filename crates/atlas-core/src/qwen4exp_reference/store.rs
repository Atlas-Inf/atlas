// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint reader for the CPU reference forward.
//!
//! Reads any tensor of a qwen4_exp checkpoint back as `f32`, dequantizing
//! NVFP4 (packed nibbles + an FP8 block scale along the input dim + a
//! per-tensor f32) and FP8 on the way. Slow and single-threaded on purpose:
//! its job is to be OBVIOUSLY right so a GPU layer has something to disagree
//! with.
//!
//! Lifted out of `examples/qwen4exp_forward.rs` so the reference forward and
//! its highway taps can live in the library, where tests can reach them.

use anyhow::{Context, Result};

use crate::weight_manifest::TensorLocation;
use std::collections::BTreeMap;

/// A checkpoint whose tensors are read back as `f32`, with a small cache.
pub struct RefStore {
    located: BTreeMap<String, TensorLocation>,
    cache: std::cell::RefCell<BTreeMap<String, std::rc::Rc<Vec<f32>>>>,
}

impl RefStore {
    /// Wrap an already-located checkpoint.
    pub fn new(located: BTreeMap<String, TensorLocation>) -> Self {
        Self {
            located,
            cache: std::cell::RefCell::new(BTreeMap::new()),
        }
    }

    /// Whether a tensor name is present.
    pub fn has(&self, name: &str) -> bool {
        self.located.contains_key(name)
    }
}

impl RefStore {
    /// Raw bytes of a tensor plus its location.
    pub fn raw(&self, name: &str) -> Result<(Vec<u8>, &TensorLocation)> {
        let loc = self
            .located
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        let file = std::fs::File::open(&loc.path)?;
        let mut bytes = vec![0u8; loc.span.len as usize];
        // Positional read shared with the PLE table reader; its unix/windows cfg
        // pair is what keeps the Windows release leg building.
        crate::ngram_table::read_exact_at(&file, &mut bytes, loc.span.abs_offset)?;
        Ok((bytes, loc))
    }

    /// Experts are the bulk of the model and are read once per token; caching
    /// the dequantized form would need hundreds of GB.
    fn cacheable(name: &str) -> bool {
        !name.contains(".experts.")
    }

    /// A tensor as `f32`, dequantized and cached where affordable.
    pub fn get(&self, name: &str) -> Result<std::rc::Rc<Vec<f32>>> {
        if let Some(hit) = self.cache.borrow().get(name) {
            return Ok(hit.clone());
        }
        let (raw, loc) = self.raw(name)?;

        // NVFP4: nibbles packed two per byte, an FP8 block scale along the
        // input dim, and a per-tensor f32 on top of that.
        if loc.dtype == "U8" && self.located.contains_key(&format!("{name}_scale")) {
            let rows = loc.shape[0];
            let cols = loc.shape[1] * 2;
            let scale_loc = &self.located[&format!("{name}_scale")];
            let group = cols / scale_loc.shape[1];
            let (scale, _) = self.raw(&format!("{name}_scale"))?;
            let (g2, _) = self.raw(&format!("{name}_scale_2"))?;
            let global = f32::from_le_bytes([g2[0], g2[1], g2[2], g2[3]]);
            let values = crate::numeric::nvfp4_dequant(&raw, &scale, global, rows, cols, group)
                .map_err(anyhow::Error::msg)?;
            let shared = std::rc::Rc::new(values);
            if Self::cacheable(name) {
                self.cache
                    .borrow_mut()
                    .insert(name.to_string(), shared.clone());
            }
            return Ok(shared);
        }

        let values: Vec<f32> = match loc.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F8_E4M3" => raw
                .iter()
                .map(|b| crate::numeric::fp8_e4m3_to_f32(*b))
                .collect(),
            other => anyhow::bail!("{name}: no f32 path for dtype {other}"),
        };
        let shared = std::rc::Rc::new(values);
        if Self::cacheable(name) {
            self.cache
                .borrow_mut()
                .insert(name.to_string(), shared.clone());
        }
        Ok(shared)
    }
}
