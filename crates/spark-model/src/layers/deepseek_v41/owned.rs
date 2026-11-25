// SPDX-License-Identifier: AGPL-3.0-only

//! A prefill batch's rows shared out between the ranks: which rows a rank
//! keeps the stream of, and how what a rank computes for its own rows, and
//! what it computes for the others', gets to where it is read.

use std::ops::Range;

use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::parts::{Held, PARTS, gather};

/// The rows of a batch whose stream one rank keeps.
///
/// Every rank used to keep every row's stream, and did every row's mixing:
/// four times the work of a prefill's hyper-connections, which are a fifth
/// of it. With the rows shared out, a part keeps `each` consecutive rows of
/// the batch, the last part what is left; a rank mixes its own rows' stream
/// and computes what only its input decides (the low-rank query and the
/// window key, the routing), and the ranks gather those by rows. What every
/// rank computes a part of (the heads, the output projections, the experts)
/// still runs on every row, from the gathered inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owned {
    pub first: usize,
    pub count: usize,
    /// Rows of a part: a batch's rows over the parts, rounded up.
    pub each: usize,
}

impl Owned {
    pub fn of(rows: usize, held: Held) -> Self {
        let each = rows.div_ceil(PARTS);
        let first = (held.first * each).min(rows);
        let end = ((held.first + held.count) * each).min(rows);
        Self {
            first,
            count: end - first,
            each,
        }
    }

    /// The rows of part `part` among a batch's `rows`.
    pub fn of_part(rows: usize, part: usize) -> Range<usize> {
        let each = rows.div_ceil(PARTS);
        (part * each).min(rows)..((part + 1) * each).min(rows)
    }
}

/// [`gather`] of a buffer of rows the ranks own a part each of
/// ([`Owned`]): `base` holds `row_bytes` a row, and room for `PARTS` parts
/// of `own.each` rows.
#[allow(clippy::too_many_arguments)]
pub fn gather_rows(
    gpu: &dyn GpuBackend,
    comm: Option<&dyn CommBackend>,
    held: Held,
    own: Owned,
    base: DevicePtr,
    recv: DevicePtr,
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    let bytes = own.each * row_bytes;
    gather(gpu, comm, held, base, recv, bytes, bytes, stream)
}

/// A sublayer's output to the ranks that keep its rows ([`Owned`]). `base`
/// is the output in parts as [`gather`] takes it, a part `[rows, row_bytes]`:
/// a rank has its parts of every row and needs every part of its own rows,
/// so each sends the others their rows of its part (an all-to-all, a
/// quarter of a gather's traffic) and a rank ends with its own rows of every
/// part where the gather would have left them. The rows it does not keep
/// are not brought. Where the all-to-all does not serve (fewer than four
/// ranks, rows that do not divide by the parts, so that a part's share
/// would run past it), the gather.
#[allow(clippy::too_many_arguments)]
pub fn to_owners(
    gpu: &dyn GpuBackend,
    comm: Option<&dyn CommBackend>,
    held: Held,
    own: Owned,
    base: DevicePtr,
    recv: DevicePtr,
    rows: usize,
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    let part = rows * row_bytes;
    let exact = held.count == 1 && rows.is_multiple_of(PARTS) && own.each * PARTS == rows;
    if let Some(comm) = comm.filter(|_| exact) {
        let plane = spark_comm::AllToAllPlane {
            src: base.offset(held.first * part).0,
            src_stride: (own.each * row_bytes) as i64,
            dst: base.offset(own.first * row_bytes).0,
            dst_stride: part as i64,
            rows: own.each,
            row_bytes,
            pitch: row_bytes,
        };
        if comm.all_to_all_async(&[plane], stream)? {
            return Ok(());
        }
    }
    gather(gpu, comm, held, base, recv, part, part, stream)
}

#[cfg(test)]
mod tests {
    use super::{Held, Owned, PARTS};

    #[test]
    fn a_batchs_rows_are_shared_out_a_part_each() {
        let rank = |r: usize| Held::of_rank(r, PARTS).expect("a rank of four");
        // Rows that divide: a quarter each, in order.
        let of = |rows: usize, r: usize| {
            let own = Owned::of(rows, rank(r));
            (own.first, own.count, own.each)
        };
        assert_eq!(of(8192, 0), (0, 2048, 2048));
        assert_eq!(of(8192, 3), (6144, 2048, 2048));
        // Rows that do not: the parts round up, and the last takes what is left.
        assert_eq!(of(1401, 0), (0, 351, 351));
        assert_eq!(of(1401, 3), (1053, 348, 351));
        // Every row is some part's, once, whatever the count.
        for rows in [1, 2, 3, 5, 1024, 1025, 1401, 7424, 8191] {
            let mut next = 0;
            for part in 0..PARTS {
                let range = Owned::of_part(rows, part);
                let own = Owned::of(rows, rank(part));
                assert_eq!((own.first, own.count), (range.start, range.len()));
                assert_eq!(range.start, next.min(rows), "{rows} rows, part {part}");
                next = range.end;
                assert!(PARTS * own.each >= rows && PARTS * own.each < rows + PARTS);
            }
            assert_eq!(next, rows);
        }
        // A rank that holds two parts keeps two parts' rows.
        let two = Owned::of(1401, Held::of_rank(1, 2).expect("a rank of two"));
        assert_eq!((two.first, two.count), (702, 699));
    }
}
