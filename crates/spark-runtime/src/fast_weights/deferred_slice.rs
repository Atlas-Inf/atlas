// SPDX-License-Identifier: AGPL-3.0-only

//! One rank's rectangle of a deferred `[rows, row_bytes]` tensor (expert TP).
//!
//! A plain deferred tensor is read straight from disk: just the rank's rows,
//! or the whole matrix to take its columns. A CSF-compressed `weight_scale`
//! has no on-disk form to slice, so the fast loader defers its components
//! instead (two for the slab encoding, five for `csf_32x2_v1`); here they are
//! read, the one matrix is decoded, and the same rectangle is taken from it.
//! The temporaries live for one call: the compressed parts (~0.3 MiB on GLM
//! FP4-Local), the decoded matrix (0.5 MiB) and, for a column slice, the
//! rank's half — never more than one matrix, never the 17.7 GiB of all of
//! them.

use anyhow::{Context, Result, bail, ensure};
use std::cell::RefCell;

use super::with_deferred_bytes;
use crate::weights::WeightStore;
use crate::weights::nvfp4_csf::{CsfEncoding, decode_parts, parts_of};
use crate::weights::nvfp4_csf32::GeometryCache;

/// Which rectangle of a `[rows, row_bytes]` matrix a rank keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceAxis {
    /// Rows `[rank * rows / world, (rank + 1) * rows / world)`: a
    /// column-parallel projection (gate/up), whose rows are output features.
    Rows,
    /// Bytes `[rank * row_bytes / world, ...)` of every row: a row-parallel
    /// projection (down), whose columns are input features.
    Cols,
}

/// Run `f` over rank `rank` of `world`'s `axis` slice of the deferred tensor
/// `name`, of full shape `[rows, row_bytes]`: a plain deferred tensor, or a
/// CSF-compressed scale whose components were deferred.
#[allow(clippy::too_many_arguments)]
pub fn with_deferred_slice<T>(
    store: &WeightStore,
    name: &str,
    rows: usize,
    row_bytes: usize,
    axis: SliceAxis,
    rank: usize,
    world: usize,
    f: impl FnOnce(&[u8]) -> Result<T>,
) -> Result<T> {
    ensure!(
        world > 0 && rank < world && rows.is_multiple_of(world) && row_bytes.is_multiple_of(world),
        "{name} [{rows}, {row_bytes} bytes] does not split {world} ways (rank {rank})"
    );
    let (start, len) = match axis {
        SliceAxis::Rows => {
            let local = rows / world;
            (rank * local * row_bytes, local * row_bytes)
        }
        SliceAxis::Cols => (0, rows * row_bytes),
    };
    let take = |bytes: &[u8]| match axis {
        SliceAxis::Rows => f(bytes),
        SliceAxis::Cols => f(&column_slice(bytes, row_bytes, rank, row_bytes / world)),
    };
    if let Some(d) = store.deferred(name) {
        ensure!(
            d.shape.len() == 2
                && d.shape[0] == rows
                && d.shape[1] * d.dtype.byte_size() == row_bytes,
            "{name} has shape {:?}, expected [{rows}, {row_bytes} bytes]",
            d.shape
        );
        return with_deferred_bytes(d, start, len, take);
    }
    if let Some(decoded) = decode_deferred_csf_scale(store, name) {
        let (bytes, shape) = decoded?;
        // E4M3: one byte per scale.
        ensure!(
            shape == [rows, row_bytes],
            "{name} decodes to {shape:?}, expected [{rows}, {row_bytes}]"
        );
        return take(&bytes[start..start + len]);
    }
    bail!("{name} was not deferred by the fast loader")
}

thread_local! {
    /// `csf_32x2_v1` block order per shape, kept across a load's slices.
    static GEOMETRIES: RefCell<GeometryCache> = RefCell::default();
}

/// Decode the CSF-compressed scale `name` from its deferred components:
/// `None` when it has none, else the E4M3 bytes and `[rows, cols]`.
pub fn decode_deferred_csf_scale(
    store: &WeightStore,
    name: &str,
) -> Option<Result<(Vec<u8>, [usize; 2])>> {
    let encoding = [CsfEncoding::Slab, CsfEncoding::Blocks32]
        .into_iter()
        .find(|&e| parts_of(e).any(|(s, _)| store.deferred(&format!("{name}{s}")).is_some()))?;
    let read = || -> Result<(Vec<u8>, [usize; 2])> {
        let mut parts = Vec::new();
        for (suffix, part) in parts_of(encoding) {
            let pname = format!("{name}{suffix}");
            let d = store
                .deferred(&pname)
                .with_context(|| format!("{pname} is missing; its set was deferred"))?;
            // The header's byte span was checked against this at load
            // (`CsfAssembler::defer`); `d.dtype` is only a staging container.
            let len = d.shape.iter().product::<usize>() * part.elem_bytes();
            let bytes = with_deferred_bytes(d, 0, len, |b| Ok(b.to_vec()))?;
            parts.push((d.shape.as_slice(), bytes));
        }
        let views: Vec<(&[usize], &[u8])> = parts.iter().map(|(s, b)| (*s, b.as_slice())).collect();
        GEOMETRIES.with(|g| decode_parts(encoding, &views, &mut g.borrow_mut()))
    };
    Some(read().with_context(|| format!("decoding deferred {} scale {name}", encoding.label())))
}

/// Bytes `[rank*local, (rank+1)*local)` of every `row_bytes`-wide row.
fn column_slice(bytes: &[u8], row_bytes: usize, rank: usize, local: usize) -> Vec<u8> {
    bytes
        .chunks_exact(row_bytes)
        .flat_map(|row| &row[rank * local..(rank + 1) * local])
        .copied()
        .collect()
}

#[cfg(test)]
#[path = "deferred_slice_tests.rs"]
mod tests;
