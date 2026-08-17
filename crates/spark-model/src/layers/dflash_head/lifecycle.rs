// SPDX-License-Identifier: AGPL-3.0-only

use std::fmt;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SequenceGeneration {
    slot: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureStatus {
    Live,
    Retired,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureDescriptor {
    owner: SequenceGeneration,
    absolute_position: usize,
    valid_rows: usize,
    row_capacity: usize,
    row_stride_bytes: usize,
    status: CaptureStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DflashGraphIdentity {
    owner: SequenceGeneration,
    block_table_ptr: u64,
    ctx_ptr: u64,
    markov_ptr: u64,
    lane: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DsparkLifecycleError {
    ZeroGeneration {
        slot: usize,
    },
    ZeroCapacity,
    ZeroStride,
    StrideMismatch {
        expected: usize,
        found: usize,
    },
    ValidRows {
        valid: usize,
        capacity: usize,
    },
    StaleOwner {
        expected: SequenceGeneration,
        found: SequenceGeneration,
    },
    Retired {
        owner: SequenceGeneration,
    },
    PositionRegression {
        current: usize,
        proposed: usize,
    },
    RowOutOfRange {
        row: usize,
        valid_rows: usize,
    },
    OffsetOverflow,
    ZeroPointer {
        field: &'static str,
    },
    InvalidLane {
        lane: usize,
    },
}

impl fmt::Display for DsparkLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroGeneration { slot } => {
                write!(f, "DSpark owner slot {slot} has zero generation")
            }
            Self::ZeroCapacity => f.write_str("DSpark capture row capacity must be nonzero"),
            Self::ZeroStride => f.write_str("DSpark capture row stride must be nonzero"),
            Self::StrideMismatch { expected, found } => write!(
                f,
                "DSpark capture row stride changed from {expected} to {found}"
            ),
            Self::ValidRows { valid, capacity } => write!(
                f,
                "DSpark capture valid_rows={valid} exceeds capacity={capacity}"
            ),
            Self::StaleOwner { expected, found } => write!(
                f,
                "DSpark stale owner slot={} generation={}; live slot={} generation={}",
                expected.slot, expected.generation, found.slot, found.generation
            ),
            Self::Retired { owner } => write!(
                f,
                "DSpark owner slot={} generation={} is retired",
                owner.slot, owner.generation
            ),
            Self::PositionRegression { current, proposed } => {
                write!(f, "DSpark position regression {current}->{proposed}")
            }
            Self::RowOutOfRange { row, valid_rows } => {
                write!(f, "DSpark row {row} is outside valid_rows={valid_rows}")
            }
            Self::OffsetOverflow => f.write_str("DSpark capture row offset overflow"),
            Self::ZeroPointer { field } => {
                write!(f, "DSpark graph identity requires nonzero {field}")
            }
            Self::InvalidLane { lane } => {
                write!(f, "DSpark graph identity rejected lane {lane}")
            }
        }
    }
}

impl std::error::Error for DsparkLifecycleError {}

impl SequenceGeneration {
    pub fn new(slot: usize, generation: u64) -> Result<Self, DsparkLifecycleError> {
        if generation == 0 {
            Err(DsparkLifecycleError::ZeroGeneration { slot })
        } else {
            Ok(Self { slot, generation })
        }
    }

    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl CaptureDescriptor {
    pub fn bind(
        owner: SequenceGeneration,
        absolute_position: usize,
        valid_rows: usize,
        row_capacity: usize,
        row_stride_bytes: usize,
    ) -> Result<Self, DsparkLifecycleError> {
        SequenceGeneration::new(owner.slot, owner.generation)?;
        validate_shape(valid_rows, row_capacity, row_stride_bytes)?;
        Ok(Self {
            owner,
            absolute_position,
            valid_rows,
            row_capacity,
            row_stride_bytes,
            status: CaptureStatus::Live,
        })
    }

    pub fn owner(&self) -> SequenceGeneration {
        self.owner
    }

    pub fn absolute_position(&self) -> usize {
        self.absolute_position
    }

    pub fn valid_rows(&self) -> usize {
        self.valid_rows
    }

    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }

    pub fn row_stride_bytes(&self) -> usize {
        self.row_stride_bytes
    }

    pub fn status(&self) -> CaptureStatus {
        self.status
    }

    pub fn validate_access(
        &self,
        expected: SequenceGeneration,
    ) -> Result<(), DsparkLifecycleError> {
        validate_owner(self.owner, expected)?;
        if self.status == CaptureStatus::Retired {
            return Err(DsparkLifecycleError::Retired { owner: self.owner });
        }
        Ok(())
    }

    pub fn row_range(
        &self,
        expected: SequenceGeneration,
        row: usize,
    ) -> Result<Range<usize>, DsparkLifecycleError> {
        self.validate_access(expected)?;
        if row >= self.valid_rows {
            return Err(DsparkLifecycleError::RowOutOfRange {
                row,
                valid_rows: self.valid_rows,
            });
        }
        let start = row
            .checked_mul(self.row_stride_bytes)
            .ok_or(DsparkLifecycleError::OffsetOverflow)?;
        let end = start
            .checked_add(self.row_stride_bytes)
            .ok_or(DsparkLifecycleError::OffsetOverflow)?;
        Ok(start..end)
    }

    pub fn advance(
        &mut self,
        expected: SequenceGeneration,
        absolute_position: usize,
        valid_rows: usize,
        row_stride_bytes: usize,
    ) -> Result<(), DsparkLifecycleError> {
        self.validate_access(expected)?;
        if absolute_position < self.absolute_position {
            return Err(DsparkLifecycleError::PositionRegression {
                current: self.absolute_position,
                proposed: absolute_position,
            });
        }
        if row_stride_bytes != self.row_stride_bytes {
            return Err(DsparkLifecycleError::StrideMismatch {
                expected: self.row_stride_bytes,
                found: row_stride_bytes,
            });
        }
        validate_shape(valid_rows, self.row_capacity, row_stride_bytes)?;
        self.absolute_position = absolute_position;
        self.valid_rows = valid_rows;
        Ok(())
    }

    pub fn retire(&mut self, expected: SequenceGeneration) -> Result<(), DsparkLifecycleError> {
        validate_owner(self.owner, expected)?;
        self.status = CaptureStatus::Retired;
        self.valid_rows = 0;
        Ok(())
    }
}

impl DflashGraphIdentity {
    pub fn new(
        owner: SequenceGeneration,
        block_table_ptr: u64,
        ctx_ptr: u64,
        markov_ptr: u64,
        lane: usize,
    ) -> Result<Self, DsparkLifecycleError> {
        SequenceGeneration::new(owner.slot, owner.generation)?;
        nonzero_pointer("block_table_ptr", block_table_ptr)?;
        nonzero_pointer("ctx_ptr", ctx_ptr)?;
        nonzero_pointer("markov_ptr", markov_ptr)?;
        if lane == usize::MAX {
            return Err(DsparkLifecycleError::InvalidLane { lane });
        }
        Ok(Self {
            owner,
            block_table_ptr,
            ctx_ptr,
            markov_ptr,
            lane,
        })
    }

    pub fn owner(&self) -> SequenceGeneration {
        self.owner
    }

    pub fn block_table_ptr(&self) -> u64 {
        self.block_table_ptr
    }

    pub fn ctx_ptr(&self) -> u64 {
        self.ctx_ptr
    }

    pub fn markov_ptr(&self) -> u64 {
        self.markov_ptr
    }

    pub fn lane(&self) -> usize {
        self.lane
    }
}

fn nonzero_pointer(field: &'static str, pointer: u64) -> Result<(), DsparkLifecycleError> {
    if pointer == 0 {
        Err(DsparkLifecycleError::ZeroPointer { field })
    } else {
        Ok(())
    }
}

fn validate_owner(
    found: SequenceGeneration,
    expected: SequenceGeneration,
) -> Result<(), DsparkLifecycleError> {
    SequenceGeneration::new(expected.slot, expected.generation)?;
    if found == expected {
        Ok(())
    } else {
        Err(DsparkLifecycleError::StaleOwner { expected, found })
    }
}

fn validate_shape(
    valid_rows: usize,
    row_capacity: usize,
    row_stride_bytes: usize,
) -> Result<(), DsparkLifecycleError> {
    if row_capacity == 0 {
        return Err(DsparkLifecycleError::ZeroCapacity);
    }
    if row_stride_bytes == 0 {
        return Err(DsparkLifecycleError::ZeroStride);
    }
    if valid_rows > row_capacity {
        return Err(DsparkLifecycleError::ValidRows {
            valid: valid_rows,
            capacity: row_capacity,
        });
    }
    Ok(())
}
