// SPDX-License-Identifier: AGPL-3.0-only

//! Buffer-reusing aux-state collection (see `collect_aux_states_into` in
//! `mod.rs`). Split out for the ≤500 LoC cap.

use anyhow::Result;

/// Refill `out` with `(layer_idx, blob)` pairs for layers `0..n_layers`,
/// reusing each existing entry's `Vec<u8>` capacity when its layer index
/// still lines up at the same position.
///
/// `fill(i, buf)` writes layer `i`'s aux blob into `buf` and returns
/// `true`, or returns `false` when that layer has nothing to carry this
/// save (e.g. the sequence never ran it). Stale entries — a layer that
/// produced a blob last save but not this one — are removed in place, so
/// the result is exactly the set `fill` produced, in layer order, and the
/// surviving entries keep their allocations across calls.
///
/// Layer indices are strictly increasing, so a linear in-place scan with
/// at most one `remove`/`insert` shuffle per entry suffices; entries whose
/// indices no longer match simply stop being eligible for reuse.
pub(super) fn collect_aux_reuse_into(
    out: &mut Vec<(u32, Vec<u8>)>,
    n_layers: usize,
    mut fill: impl FnMut(u32, &mut Vec<u8>) -> Result<bool>,
) -> Result<()> {
    let mut k = 0usize;
    for i in 0..n_layers as u32 {
        let reuse = matches!(out.get(k), Some((idx, _)) if *idx == i);
        if reuse {
            let produced = fill(i, &mut out[k].1)?;
            if produced {
                k += 1;
            } else {
                // Layer i carried aux in the entry we're reusing but has none
                // this save — drop the stale entry so `apply_aux_states`
                // can't replay it.
                out.remove(k);
            }
        } else {
            let mut buf = Vec::new();
            if fill(i, &mut buf)? {
                out.insert(k, (i, buf));
                k += 1;
            }
        }
    }
    out.truncate(k);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes produced per layer for a simulated save; `None` = no aux.
    fn fill_from<'a>(
        blobs: &'a [Option<Vec<u8>>],
    ) -> impl FnMut(u32, &mut Vec<u8>) -> Result<bool> + 'a {
        move |i, buf| match &blobs[i as usize] {
            Some(b) => {
                buf.clear();
                buf.extend_from_slice(b);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    #[test]
    fn reuse_keeps_capacity_and_pointer() {
        let mut out: Vec<(u32, Vec<u8>)> = Vec::new();
        let blobs = vec![None, Some(vec![7u8; 64]), None, Some(vec![9u8; 96])];
        collect_aux_reuse_into(&mut out, 4, fill_from(&blobs)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 1);
        assert_eq!(out[1].0, 3);
        let (cap0, ptr0) = (out[0].1.capacity(), out[0].1.as_ptr());
        let (cap1, ptr1) = (out[1].1.capacity(), out[1].1.as_ptr());

        // Same layers produce again — the buffers must be reused in place.
        let blobs2 = vec![None, Some(vec![1u8; 64]), None, Some(vec![2u8; 96])];
        collect_aux_reuse_into(&mut out, 4, fill_from(&blobs2)).unwrap();
        assert_eq!(out[0].1, vec![1u8; 64]);
        assert_eq!(out[1].1, vec![2u8; 96]);
        assert_eq!((out[0].1.capacity(), out[0].1.as_ptr()), (cap0, ptr0));
        assert_eq!((out[1].1.capacity(), out[1].1.as_ptr()), (cap1, ptr1));
    }

    #[test]
    fn stale_entry_is_dropped_and_gaps_realign() {
        let mut out: Vec<(u32, Vec<u8>)> = Vec::new();
        let blobs = vec![Some(vec![1u8; 8]), Some(vec![2u8; 8]), Some(vec![3u8; 8])];
        collect_aux_reuse_into(&mut out, 3, fill_from(&blobs)).unwrap();
        assert_eq!(out.iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 1, 2]);

        // Layer 1 stops producing; layer 3 (a NEW layer) starts. Order and
        // contents must be exact — a replayed stale blob would corrupt.
        let blobs2 = vec![
            Some(vec![1u8; 8]),
            None,
            Some(vec![3u8; 8]),
            Some(vec![4u8; 8]),
        ];
        collect_aux_reuse_into(&mut out, 4, fill_from(&blobs2)).unwrap();
        assert_eq!(out.iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 2, 3]);
        assert_eq!(out[0].1, vec![1u8; 8]);
        assert_eq!(out[1].1, vec![3u8; 8]);
        assert_eq!(out[2].1, vec![4u8; 8]);
        // Entries 0 and 1 kept their capacity from the first round.
        assert!(out[0].1.capacity() >= 8);
        assert!(out[1].1.capacity() >= 8);
    }

    #[test]
    fn error_propagates() {
        let mut out: Vec<(u32, Vec<u8>)> = Vec::new();
        let r = collect_aux_reuse_into(&mut out, 2, |_i, _buf| Err(anyhow::anyhow!("d2h failed")));
        assert!(r.is_err());
    }
}
