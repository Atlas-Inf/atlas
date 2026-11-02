// SPDX-License-Identifier: AGPL-3.0-only

//! Checked live FP32 pools and independent decode rows. No device work.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

/// Borrowed, validated live FP32 pool geometry; capacity excludes the dummy slot.
#[derive(Clone, Copy, Debug)]
pub struct SsmPoolView<'a> {
    h_bases: &'a [DevicePtr],
    conv_bases: &'a [DevicePtr],
    h_bytes: usize,
    conv_bytes: usize,
    capacity: u32,
}

pub(crate) fn checked_span(ptr: DevicePtr, bytes: usize) -> Result<()> {
    ensure!(
        ptr.0 != 0 && ptr.0.is_multiple_of(4),
        "SSM pointer must be nonnull and FP32 aligned"
    );
    ensure!(bytes > 0, "SSM span must be nonempty");
    ptr.0
        .checked_add(u64::try_from(bytes)?)
        .ok_or_else(|| anyhow::anyhow!("SSM address overflow"))?;
    Ok(())
}

impl<'a> SsmPoolView<'a> {
    pub(crate) fn new(
        h_bases: &'a [DevicePtr],
        conv_bases: &'a [DevicePtr],
        h_bytes: usize,
        h_stored_bytes: usize,
        conv_bytes: usize,
        slot_capacity: usize,
    ) -> Result<Self> {
        ensure!(
            !h_bases.is_empty() && h_bases.len() == conv_bases.len(),
            "SSM layer pool counts disagree"
        );
        ensure!(
            h_bytes > 0 && h_bytes.is_multiple_of(4) && h_stored_bytes == h_bytes,
            "indexed SSM requires live FP32 H storage"
        );
        ensure!(
            conv_bytes > 0 && conv_bytes.is_multiple_of(4),
            "invalid FP32 convolution stride"
        );
        ensure!(
            slot_capacity > 0 && slot_capacity <= i32::MAX as usize,
            "invalid live SSM slot capacity"
        );
        let h_span = h_bytes
            .checked_mul(slot_capacity)
            .ok_or_else(|| anyhow::anyhow!("SSM H pool overflow"))?;
        let c_span = conv_bytes
            .checked_mul(slot_capacity)
            .ok_or_else(|| anyhow::anyhow!("SSM conv pool overflow"))?;
        let mut ranges = Vec::new();
        for (bases, bytes) in [(h_bases, h_span), (conv_bases, c_span)] {
            for &base in bases {
                checked_span(base, bytes)?;
                let end = base.0 + bytes as u64;
                ensure!(
                    ranges
                        .iter()
                        .all(|&(start, stop)| end <= start || base.0 >= stop),
                    "live SSM pool allocations overlap"
                );
                ranges.push((base.0, end));
            }
        }
        Ok(Self {
            h_bases,
            conv_bases,
            h_bytes,
            conv_bytes,
            capacity: slot_capacity as u32,
        })
    }

    pub(crate) fn layer_count(self) -> usize {
        self.h_bases.len()
    }

    pub(crate) fn validate_disjoint(self, ptr: DevicePtr, bytes: usize) -> Result<()> {
        checked_span(ptr, bytes)?;
        let end = ptr.0 + bytes as u64;
        for (bases, stride) in [
            (self.h_bases, self.h_bytes),
            (self.conv_bases, self.conv_bytes),
        ] {
            for &base in bases {
                let stop = base.0 + (stride * self.capacity as usize) as u64;
                ensure!(
                    end <= base.0 || ptr.0 >= stop,
                    "SSM metadata overlaps live state"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn row_pointers(
        self,
        ordinal: usize,
        slot: usize,
    ) -> Result<(DevicePtr, DevicePtr)> {
        ensure!(
            ordinal < self.layer_count() && slot < self.capacity as usize,
            "SSM row outside live pool"
        );
        // The constructor checked the complete live extent and address arithmetic.
        Ok((
            self.h_bases[ordinal].offset(slot * self.h_bytes),
            self.conv_bases[ordinal].offset(slot * self.conv_bytes),
        ))
    }
}

/// Fixed device ID address and validated host row order. IDs must be uploaded
/// before use; construction performs no upload and gives no synchronization guarantee.
#[derive(Clone, Copy, Debug)]
pub struct SsmBatchView<'a> {
    pool: SsmPoolView<'a>,
    slots: DevicePtr,
    rows: u32,
}

impl<'a> SsmBatchView<'a> {
    pub(crate) fn new(pool: SsmPoolView<'a>, slots: DevicePtr, host_slots: &[i32]) -> Result<Self> {
        ensure!(
            (1..=8).contains(&host_slots.len()),
            "indexed SSM supports one to eight independent rows"
        );
        pool.validate_disjoint(slots, host_slots.len() * 4)?;
        for (row, &slot) in host_slots.iter().enumerate() {
            ensure!(
                slot >= 0 && (slot as u32) < pool.capacity,
                "SSM slot is padding, dummy, or out of range"
            );
            ensure!(
                !host_slots[..row].contains(&slot),
                "duplicate live SSM slot"
            );
        }
        Ok(Self {
            pool,
            slots,
            rows: host_slots.len() as u32,
        })
    }

    pub fn layer(self, ssm_ordinal: usize) -> Result<SsmBatchLayer> {
        let (h_base, conv_base) = self.pool.row_pointers(ssm_ordinal, 0)?;
        Ok(SsmBatchLayer {
            slots: self.slots,
            rows: self.rows,
            capacity: self.pool.capacity,
            h_base,
            conv_base,
            h_stride: (self.pool.h_bytes / 4) as u64,
            conv_stride: (self.pool.conv_bytes / 4) as u64,
        })
    }
}

/// Launch descriptor for one SSM ordinal. Strides are FP32 elements, not bytes.
#[derive(Clone, Copy, Debug)]
pub struct SsmBatchLayer {
    slots: DevicePtr,
    rows: u32,
    capacity: u32,
    h_base: DevicePtr,
    conv_base: DevicePtr,
    h_stride: u64,
    conv_stride: u64,
}

impl SsmBatchLayer {
    pub fn slots(self) -> DevicePtr {
        self.slots
    }
    pub fn rows(self) -> u32 {
        self.rows
    }
    pub fn slot_capacity(self) -> u32 {
        self.capacity
    }
    pub fn h_base(self) -> DevicePtr {
        self.h_base
    }
    pub fn conv_base(self) -> DevicePtr {
        self.conv_base
    }
    pub fn h_stride_elements(self) -> u64 {
        self.h_stride
    }
    pub fn conv_stride_elements(self) -> u64 {
        self.conv_stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_ranges_reject_aliasing_but_allow_exact_adjacency() {
        // Each H and conv live allocation is 128 bytes.
        for (h, c) in [
            (vec![DevicePtr(0x1000)], vec![DevicePtr(0x1000)]),
            (vec![DevicePtr(0x1000)], vec![DevicePtr(0x107c)]),
            (
                vec![DevicePtr(0x1000), DevicePtr(0x107c)],
                vec![DevicePtr(0x2000), DevicePtr(0x3000)],
            ),
            (
                vec![DevicePtr(0x1000), DevicePtr(0x2000)],
                vec![DevicePtr(0x3000), DevicePtr(0x307c)],
            ),
            (
                vec![DevicePtr(0x1000), DevicePtr(0x2000)],
                vec![DevicePtr(0x3000), DevicePtr(0x107c)],
            ),
        ] {
            assert!(SsmPoolView::new(&h, &c, 16, 16, 16, 8).is_err());
        }
        let h = [DevicePtr(0x1000), DevicePtr(0x1080)];
        let c = [DevicePtr(0x1100), DevicePtr(0x1180)];
        let pool = SsmPoolView::new(&h, &c, 16, 16, 16, 8).unwrap();
        for address in [0x1000, 0x107c, 0x117c, 0x11fc, 0x0ffc] {
            assert!(SsmBatchView::new(pool, DevicePtr(address), &[0, 1]).is_err());
        }
        assert!(SsmBatchView::new(pool, DevicePtr(0x0ff8), &[0, 1]).is_ok());
        assert!(SsmBatchView::new(pool, DevicePtr(0x1200), &[0, 1]).is_ok());
    }

    fn pool<'a>(h: &'a [DevicePtr], c: &'a [DevicePtr]) -> SsmPoolView<'a> {
        SsmPoolView::new(h, c, 64, 64, 32, 8).unwrap()
    }

    #[test]
    fn nonprefix_rows_derive_float_strides_and_exact_layer_bases() {
        let h = [DevicePtr(0x1000), DevicePtr(0x3000)];
        let c = [DevicePtr(0x5000), DevicePtr(0x7000)];
        let view = SsmBatchView::new(pool(&h, &c), DevicePtr(0x9000), &[6, 1, 4, 0]).unwrap();
        let layer = view.layer(1).unwrap();
        assert_eq!(layer.h_base(), h[1]);
        assert_eq!(layer.conv_base(), c[1]);
        assert_eq!(layer.rows(), 4);
        assert_eq!(layer.slot_capacity(), 8);
        assert_eq!(layer.h_stride_elements(), 16);
        assert_eq!(layer.conv_stride_elements(), 8);
        assert_eq!(layer.slots(), DevicePtr(0x9000));
        assert!(view.layer(2).is_err());
    }

    #[test]
    fn malformed_present_pools_and_live_ids_are_errors() {
        let h = [DevicePtr(0x1000)];
        let c = [DevicePtr(0x3000)];
        assert!(
            SsmBatchView::new(pool(&h, &c), DevicePtr(0x9000), &[7, 0, 6, 1, 5, 2, 4, 3]).is_ok()
        );
        for ids in [
            vec![],
            vec![-1],
            vec![8],
            vec![0, 0],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 0],
        ] {
            assert!(SsmBatchView::new(pool(&h, &c), DevicePtr(0x9000), &ids).is_err());
        }
        for ptr in [DevicePtr::NULL, DevicePtr(3), DevicePtr(u64::MAX - 3)] {
            assert!(SsmBatchView::new(pool(&h, &c), ptr, &[0, 1]).is_err());
        }
        for (hb, stored, cb, slots) in [
            (0, 0, 32, 8),
            (64, 32, 32, 8),
            (65, 65, 32, 8),
            (64, 64, 31, 8),
            (64, 64, 32, 0),
            (usize::MAX - 3, usize::MAX - 3, 32, 8),
            (64, 64, 32, usize::MAX),
        ] {
            assert!(SsmPoolView::new(&h, &c, hb, stored, cb, slots).is_err());
        }
        assert!(SsmPoolView::new(&h, &[], 64, 64, 32, 8).is_err());
        assert!(SsmPoolView::new(&[DevicePtr::NULL], &c, 64, 64, 32, 8).is_err());
        assert!(SsmPoolView::new(&[DevicePtr(1)], &c, 64, 64, 32, 8).is_err());
        assert!(SsmPoolView::new(&[DevicePtr(u64::MAX - 3)], &c, 64, 64, 32, 8).is_err());
    }
}
