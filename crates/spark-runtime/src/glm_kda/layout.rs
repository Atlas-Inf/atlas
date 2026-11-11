// SPDX-License-Identifier: AGPL-3.0-only

//! Checked scratch layout for Atlas's fixed H32/D128, chunk16 KDA operator.

pub const HEADS: usize = 32;
pub const DIM: usize = 128;
pub const CHUNK: usize = 16;
pub const MIN_ROWS: usize = 2048;
pub const MAX_ROWS: usize = 4100;

/// Offsets into one caller-owned allocation; shared with the native CUDA unit.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    pub kd: u64,
    pub qd: u64,
    pub kr: u64,
    pub gt: u64,
    pub inv: u64,
    pub mqk: u64,
    pub beta: u64,
    pub bytes: u64,
}

impl ScratchLayout {
    pub fn new(rows: usize) -> Result<Self, &'static str> {
        if !(MIN_ROWS..=MAX_ROWS).contains(&rows) {
            return Err("native KDA requires 2048..=4100 rows");
        }
        let tiles = rows.div_ceil(CHUNK) * HEADS;
        let mut offset = 0u64;
        let mut take = |bytes: usize| {
            let start = offset;
            offset += bytes as u64;
            start
        };
        let kd = take(tiles * CHUNK * DIM * 2);
        let qd = take(tiles * CHUNK * DIM * 2);
        let kr = take(tiles * CHUNK * DIM * 2);
        let gt = take(tiles * DIM * 4);
        let inv = take(tiles * CHUNK * CHUNK * 2);
        let mqk = take(tiles * CHUNK * CHUNK * 2);
        let beta = take(rows * HEADS * 2);
        Ok(Self {
            kd,
            qd,
            kr,
            gt,
            inv,
            mqk,
            beta,
            bytes: offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_rows_fail_before_any_allocation_or_launch() {
        for rows in [0, 1, 2047, 4101, usize::MAX] {
            assert!(ScratchLayout::new(rows).is_err());
        }
    }

    #[test]
    fn exact_full_tile_and_tail_capacity() {
        let full = ScratchLayout::new(4096).unwrap();
        assert_eq!(full.beta, 113_246_208);
        assert_eq!(full.bytes, 113_508_352);
        let tail = ScratchLayout::new(4100).unwrap();
        assert_eq!(tail.beta - full.beta, (HEADS * 13_824) as u64);
        assert_eq!(tail.bytes - tail.beta, (4100 * HEADS * 2) as u64);
        assert_eq!(std::mem::size_of::<ScratchLayout>(), 64);
    }

    #[test]
    fn planes_are_disjoint_aligned_and_cover_every_row() {
        for rows in [2048, 2049, 4095, 4096, 4100] {
            let l = ScratchLayout::new(rows).unwrap();
            let starts = [l.kd, l.qd, l.kr, l.gt, l.inv, l.mqk, l.beta];
            assert!(starts.iter().all(|s| s % 128 == 0));
            assert!(starts.windows(2).all(|s| s[0] < s[1]));
            assert_eq!(l.bytes - l.beta, (rows * HEADS * 2) as u64);
            let tiles = rows.div_ceil(CHUNK) * HEADS;
            assert_eq!(l.beta, (tiles * 13_824) as u64);
        }
    }
}
