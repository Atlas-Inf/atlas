// SPDX-License-Identifier: AGPL-3.0-only
//! Checked, allocation-free storage plan for Atlas GLM sparse CUDA dispatch.

pub(crate) const MAX_CONTEXT: usize = crate::speculative::glm_repair_policy::MAX_LONG_CONTEXT;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Span {
    pub ptr: u64,
    pub bytes: usize,
}

/// Atlas-owned operands; never passed across a host ABI boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Operands {
    pub q: u64,
    pub kv: u64,
    pub selected: u64,
    pub table: u64,
    pub qpad: u64,
    pub packed_kv: u64,
    pub metadata: u64,
    pub main_out: u64,
    pub tail_out: u64,
    pub out: u64,
    pub metadata_bytes: u64,
    pub stream: u64,
    pub rows: u32,
    pub seq_start: u32,
    pub physical_blocks: u32,
    pub block_table_count: u32,
}

pub(crate) fn admit(
    rows: usize,
    seq_start: usize,
    graph_capture: bool,
    stream_capturing: bool,
    capture_verify_intermediates: bool,
) -> Option<usize> {
    if graph_capture
        || stream_capturing
        || capture_verify_intermediates
        || !(2048..=4100).contains(&rows)
        || seq_start < 2048
    {
        return None;
    }
    seq_start
        .checked_add(rows)
        .filter(|&end| end <= MAX_CONTEXT)
}

fn mul(a: usize, b: usize) -> Result<usize, &'static str> {
    a.checked_mul(b).ok_or("native sparse byte count overflow")
}

pub(crate) struct Metadata {
    pub offsets: [usize; 8],
    pub bytes: usize,
}

impl Metadata {
    pub fn new(rows: usize) -> Result<Self, &'static str> {
        let ids = mul(rows, 2048 * 4)?;
        let counts = mul(rows, 4)?;
        let lse = mul(rows, 32 * 4)?;
        let sizes = [ids, ids, counts, counts, counts, lse, lse, lse];
        let mut offsets = [0; 8];
        let mut end = 0usize;
        for (i, bytes) in sizes.into_iter().enumerate() {
            offsets[i] = end
                .checked_add(255)
                .ok_or("native sparse alignment overflow")?
                & !255;
            end = offsets[i]
                .checked_add(bytes)
                .ok_or("native sparse metadata overflow")?;
        }
        Ok(Self {
            offsets,
            bytes: end,
        })
    }
}

pub(crate) fn required_bytes(
    rows: usize,
    seq_end: usize,
    physical_blocks: usize,
    block_table_count: usize,
) -> Result<([usize; 10], Metadata), &'static str> {
    let output = mul(rows, 32 * 512 * 2)?;
    let packed_rows = seq_end
        .checked_add(63)
        .ok_or("native sparse history overflow")?
        & !63;
    let metadata = Metadata::new(rows)?;
    Ok((
        [
            output,
            mul(physical_blocks, 16 * 512 * 2)?,
            mul(rows, 2051 * 4)?,
            mul(block_table_count, 4)?,
            mul(rows, 32 * 576 * 2)?,
            mul(packed_rows, 656)?,
            metadata.bytes,
            output,
            output,
            output,
        ],
        metadata,
    ))
}

pub(crate) struct Plan {
    pub operands: Operands,
}

impl Plan {
    pub fn new(
        rows: usize,
        seq_start: usize,
        physical_blocks: usize,
        block_table_count: usize,
        storage: [Span; 10],
        stream: u64,
    ) -> Result<Self, &'static str> {
        let seq_end = admit(rows, seq_start, false, false, false)
            .ok_or("native sparse unqualified prefill shape")?;
        if !(1..=MAX_CONTEXT).contains(&physical_blocks) || block_table_count < seq_end.div_ceil(16)
        {
            return Err("native sparse missing physical cache or logical block table");
        }
        let (required, metadata) =
            required_bytes(rows, seq_end, physical_blocks, block_table_count)?;
        let alignment = [2u64, 2, 4, 4, 256, 256, 256, 256, 256, 256];
        let mut ends = [0u64; 10];
        for i in 0..10 {
            let s = storage[i];
            if s.ptr == 0 || !s.ptr.is_multiple_of(alignment[i]) || s.bytes < required[i] {
                return Err("native sparse null, misaligned, or undersized operand");
            }
            ends[i] = s
                .ptr
                .checked_add(u64::try_from(s.bytes).map_err(|_| "native sparse capacity overflow")?)
                .ok_or("native sparse pointer overflow")?;
            for j in 0..i {
                if ends[i] > storage[j].ptr && ends[j] > s.ptr {
                    return Err("native sparse overlapping live operands or scratch owners");
                }
            }
        }
        // Check every planned metadata subspan start independently.
        for offset in metadata.offsets {
            let ptr = storage[6]
                .ptr
                .checked_add(offset as u64)
                .ok_or("native sparse metadata pointer overflow")?;
            if !ptr.is_multiple_of(256) || ptr >= ends[6] {
                return Err("native sparse invalid metadata subspan");
            }
        }
        Ok(Self {
            operands: Operands {
                q: storage[0].ptr,
                kv: storage[1].ptr,
                selected: storage[2].ptr,
                table: storage[3].ptr,
                qpad: storage[4].ptr,
                packed_kv: storage[5].ptr,
                metadata: storage[6].ptr,
                main_out: storage[7].ptr,
                tail_out: storage[8].ptr,
                out: storage[9].ptr,
                metadata_bytes: metadata.bytes as u64,
                stream,
                rows: u32::try_from(rows).map_err(|_| "native sparse row count overflow")?,
                seq_start: u32::try_from(seq_start)
                    .map_err(|_| "native sparse position overflow")?,
                physical_blocks: u32::try_from(physical_blocks)
                    .map_err(|_| "native sparse block count overflow")?,
                block_table_count: u32::try_from(block_table_count)
                    .map_err(|_| "native sparse table count overflow")?,
            },
        })
    }
}

#[cfg(test)]
#[path = "glm_sparse_native_plan_tests.rs"]
mod tests;
