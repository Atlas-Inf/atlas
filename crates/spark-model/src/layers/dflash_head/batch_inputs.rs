// SPDX-License-Identifier: AGPL-3.0-only

//! Pure host-side input and ownership contract for Lightning DSpark B×γ rows.
//!
//! The contract freezes only the input identity and layout. It does not allocate
//! batch scratch or dispatch a kernel; the current proposer remains the existing
//! serial/lane implementation until a later phase replaces its compute body.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use spark_runtime::gpu::DevicePtr;

use super::{CaptureDescriptor, CaptureStatus, LIGHTNING_SERVED_GAMMA, SequenceGeneration};

/// One immutable sequence identity entering a Lightning DSpark batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsparkBatchSequence {
    pub owner: SequenceGeneration,
    pub last_token: u32,
    pub absolute_position: usize,
    pub target_hidden: DevicePtr,
}

/// Validated `[sequence][gamma]` input layout for one DSpark propose batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsparkBatchInput {
    gamma: usize,
    capacity: usize,
    total_rows: usize,
    sequences: Vec<DsparkBatchSequence>,
}

/// Typed failures at the DSpark batch-input boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DsparkBatchInputError {
    GammaZero,
    GammaMismatch {
        expected: usize,
        found: usize,
    },
    EmptyBatch,
    LengthMismatch {
        field: &'static str,
        expected: usize,
        found: usize,
    },
    CapacityExceeded {
        capacity: usize,
        batch: usize,
    },
    RowCountOverflow {
        batch: usize,
        gamma: usize,
    },
    OwnerZero {
        sequence: usize,
        owner: SequenceGeneration,
    },
    ExpectedOwnerMismatch {
        sequence: usize,
        descriptor: SequenceGeneration,
        expected: SequenceGeneration,
    },
    DuplicateOwner {
        first: usize,
        second: usize,
        owner: SequenceGeneration,
    },
    MissingLifecycle {
        sequence: usize,
        owner: SequenceGeneration,
    },
    LifecycleOwnerMismatch {
        sequence: usize,
        descriptor: SequenceGeneration,
        lifecycle: SequenceGeneration,
    },
    LifecycleNotLive {
        sequence: usize,
        owner: SequenceGeneration,
        status: CaptureStatus,
    },
    ZeroTargetHidden {
        sequence: usize,
    },
    SequenceOutOfBounds {
        sequence: usize,
        batch: usize,
    },
    QueryOutOfBounds {
        query: usize,
        gamma: usize,
    },
    ZeroDimension {
        field: &'static str,
    },
    ArithmeticOverflow {
        operation: &'static str,
        lhs: usize,
        rhs: usize,
    },
}

impl fmt::Display for DsparkBatchInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GammaZero => formatter.write_str("Lightning DSpark gamma must be nonzero"),
            Self::GammaMismatch { expected, found } => write!(
                formatter,
                "Lightning DSpark gamma mismatch: expected {expected}, found {found}"
            ),
            Self::EmptyBatch => formatter.write_str("Lightning DSpark batch must be nonempty"),
            Self::LengthMismatch {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "Lightning DSpark batch {field} length mismatch: expected {expected}, found {found}"
            ),
            Self::CapacityExceeded { capacity, batch } => write!(
                formatter,
                "Lightning DSpark batch width {batch} exceeds explicit capacity {capacity}"
            ),
            Self::RowCountOverflow { batch, gamma } => write!(
                formatter,
                "Lightning DSpark row count overflow for batch={batch}, gamma={gamma}"
            ),
            Self::OwnerZero { sequence, owner } => write!(
                formatter,
                "Lightning DSpark sequence {sequence} has invalid zero-generation owner {owner:?}"
            ),
            Self::ExpectedOwnerMismatch {
                sequence,
                descriptor,
                expected,
            } => write!(
                formatter,
                "Lightning DSpark sequence {sequence} descriptor owner {descriptor:?} does not match expected {expected:?}"
            ),
            Self::DuplicateOwner {
                first,
                second,
                owner,
            } => write!(
                formatter,
                "Lightning DSpark owner {owner:?} is duplicated at sequences {first} and {second}"
            ),
            Self::MissingLifecycle { sequence, owner } => write!(
                formatter,
                "Lightning DSpark sequence {sequence} owner {owner:?} has no lifecycle descriptor"
            ),
            Self::LifecycleOwnerMismatch {
                sequence,
                descriptor,
                lifecycle,
            } => write!(
                formatter,
                "Lightning DSpark sequence {sequence} descriptor owner {descriptor:?} differs from lifecycle owner {lifecycle:?}"
            ),
            Self::LifecycleNotLive {
                sequence,
                owner,
                status,
            } => write!(
                formatter,
                "Lightning DSpark sequence {sequence} owner {owner:?} lifecycle is {status:?}, not Live"
            ),
            Self::ZeroTargetHidden { sequence } => write!(
                formatter,
                "Lightning DSpark sequence {sequence} has a null target-hidden pointer"
            ),
            Self::SequenceOutOfBounds { sequence, batch } => write!(
                formatter,
                "Lightning DSpark sequence index {sequence} is outside batch width {batch}"
            ),
            Self::QueryOutOfBounds { query, gamma } => write!(
                formatter,
                "Lightning DSpark gamma row {query} is outside gamma {gamma}"
            ),
            Self::ZeroDimension { field } => {
                write!(formatter, "Lightning DSpark {field} must be nonzero")
            }
            Self::ArithmeticOverflow {
                operation,
                lhs,
                rhs,
            } => write!(
                formatter,
                "Lightning DSpark {operation} overflow for {lhs} × {rhs}"
            ),
        }
    }
}

impl std::error::Error for DsparkBatchInputError {}

/// Validate the length-only portion of the production seam.
///
/// This intentionally does not reject an empty batch: the existing `n < 2`
/// proposer behavior may return `Ok(None)` after this structural check. The
/// full contract rejects empty batches when a native-width input is requested.
pub(crate) fn validate_batch_input_lengths(
    batch_len: usize,
    owners_len: usize,
    last_tokens_len: usize,
    target_hiddens_len: usize,
    positions_len: usize,
    states_len: usize,
    expected_owners_len: usize,
) -> Result<(), DsparkBatchInputError> {
    check_len("owners", batch_len, owners_len)?;
    check_len("last_tokens", batch_len, last_tokens_len)?;
    check_len("target_hiddens", batch_len, target_hiddens_len)?;
    check_len("positions", batch_len, positions_len)?;
    check_len("states", batch_len, states_len)?;
    check_len("expected_owners", batch_len, expected_owners_len)
}

impl DsparkBatchInput {
    /// Validate explicit per-sequence identity, ownership, and the `[B, γ]`
    /// row layout. Lifecycle descriptors are copied only for validation; the
    /// resulting contract owns no mutable state or borrowed batch index.
    #[allow(clippy::too_many_arguments)]
    pub fn validate(
        gamma: usize,
        capacity: usize,
        owners: &[SequenceGeneration],
        last_tokens: &[u32],
        positions: &[usize],
        target_hiddens: &[DevicePtr],
        expected_owners: &[SequenceGeneration],
        lifecycles: &[Option<CaptureDescriptor>],
    ) -> Result<Self, DsparkBatchInputError> {
        if gamma == 0 {
            return Err(DsparkBatchInputError::GammaZero);
        }
        if gamma != LIGHTNING_SERVED_GAMMA {
            return Err(DsparkBatchInputError::GammaMismatch {
                expected: LIGHTNING_SERVED_GAMMA,
                found: gamma,
            });
        }
        let batch = expected_owners.len();
        validate_batch_input_lengths(
            batch,
            owners.len(),
            last_tokens.len(),
            target_hiddens.len(),
            positions.len(),
            lifecycles.len(),
            expected_owners.len(),
        )?;
        if batch == 0 {
            return Err(DsparkBatchInputError::EmptyBatch);
        }
        if batch > capacity {
            return Err(DsparkBatchInputError::CapacityExceeded { capacity, batch });
        }
        let total_rows = batch
            .checked_mul(gamma)
            .ok_or(DsparkBatchInputError::RowCountOverflow { batch, gamma })?;

        let mut seen = HashMap::with_capacity(batch);
        let mut sequences = Vec::with_capacity(batch);
        for index in 0..batch {
            let descriptor_owner = owners[index];
            if descriptor_owner.generation() == 0 {
                return Err(DsparkBatchInputError::OwnerZero {
                    sequence: index,
                    owner: descriptor_owner,
                });
            }
            if descriptor_owner != expected_owners[index] {
                return Err(DsparkBatchInputError::ExpectedOwnerMismatch {
                    sequence: index,
                    descriptor: descriptor_owner,
                    expected: expected_owners[index],
                });
            }
            if let Some(first) = seen.insert(descriptor_owner, index) {
                return Err(DsparkBatchInputError::DuplicateOwner {
                    first,
                    second: index,
                    owner: descriptor_owner,
                });
            }
            if target_hiddens[index].0 == 0 {
                return Err(DsparkBatchInputError::ZeroTargetHidden { sequence: index });
            }
            let lifecycle = match lifecycles[index].as_ref() {
                Some(lifecycle) => lifecycle,
                None => {
                    return Err(DsparkBatchInputError::MissingLifecycle {
                        sequence: index,
                        owner: descriptor_owner,
                    });
                }
            };
            if lifecycle.owner() != descriptor_owner {
                return Err(DsparkBatchInputError::LifecycleOwnerMismatch {
                    sequence: index,
                    descriptor: descriptor_owner,
                    lifecycle: lifecycle.owner(),
                });
            }
            if lifecycle.status() != CaptureStatus::Live {
                return Err(DsparkBatchInputError::LifecycleNotLive {
                    sequence: index,
                    owner: descriptor_owner,
                    status: lifecycle.status(),
                });
            }
            sequences.push(DsparkBatchSequence {
                owner: descriptor_owner,
                last_token: last_tokens[index],
                absolute_position: positions[index],
                target_hidden: target_hiddens[index],
            });
        }
        Ok(Self {
            gamma,
            capacity,
            total_rows,
            sequences,
        })
    }

    pub fn gamma(&self) -> usize {
        self.gamma
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn batch_len(&self) -> usize {
        self.sequences.len()
    }

    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    pub fn sequence(&self, sequence: usize) -> &DsparkBatchSequence {
        &self.sequences[sequence]
    }

    /// Return `sequence * gamma + query`, with both indices checked.
    pub fn row_index(&self, sequence: usize, query: usize) -> Result<usize, DsparkBatchInputError> {
        self.check_sequence(sequence)?;
        if query >= self.gamma {
            return Err(DsparkBatchInputError::QueryOutOfBounds {
                query,
                gamma: self.gamma,
            });
        }
        let base = checked_product("row index", sequence, self.gamma)?;
        base.checked_add(query)
            .ok_or(DsparkBatchInputError::ArithmeticOverflow {
                operation: "row index",
                lhs: base,
                rhs: query,
            })
    }

    /// Return the contiguous row range for one sequence.
    pub fn sequence_row_range(
        &self,
        sequence: usize,
    ) -> Result<Range<usize>, DsparkBatchInputError> {
        self.check_sequence(sequence)?;
        let start = checked_product("sequence row range", sequence, self.gamma)?;
        let end =
            start
                .checked_add(self.gamma)
                .ok_or(DsparkBatchInputError::ArithmeticOverflow {
                    operation: "sequence row range",
                    lhs: start,
                    rhs: self.gamma,
                })?;
        Ok(start..end)
    }

    /// Return total bytes for `[batch, gamma, hidden_width]` rows.
    pub fn total_bytes(
        &self,
        hidden_width: usize,
        element_bytes: usize,
    ) -> Result<usize, DsparkBatchInputError> {
        let row_bytes = checked_row_bytes(hidden_width, element_bytes)?;
        checked_product("total bytes", self.total_rows, row_bytes)
    }

    /// Return the byte offset of one checked row in the packed batch.
    pub fn row_byte_offset(
        &self,
        sequence: usize,
        query: usize,
        hidden_width: usize,
        element_bytes: usize,
    ) -> Result<usize, DsparkBatchInputError> {
        let row = self.row_index(sequence, query)?;
        let row_bytes = checked_row_bytes(hidden_width, element_bytes)?;
        checked_product("row byte offset", row, row_bytes)
    }

    /// Return the byte range of one checked row in the packed batch.
    pub fn row_byte_range(
        &self,
        sequence: usize,
        query: usize,
        hidden_width: usize,
        element_bytes: usize,
    ) -> Result<Range<usize>, DsparkBatchInputError> {
        let start = self.row_byte_offset(sequence, query, hidden_width, element_bytes)?;
        let row_bytes = checked_row_bytes(hidden_width, element_bytes)?;
        let end =
            start
                .checked_add(row_bytes)
                .ok_or(DsparkBatchInputError::ArithmeticOverflow {
                    operation: "row byte range",
                    lhs: start,
                    rhs: row_bytes,
                })?;
        Ok(start..end)
    }

    /// Return the byte range covering all rows for one sequence.
    pub fn sequence_byte_range(
        &self,
        sequence: usize,
        hidden_width: usize,
        element_bytes: usize,
    ) -> Result<Range<usize>, DsparkBatchInputError> {
        let rows = self.sequence_row_range(sequence)?;
        let row_bytes = checked_row_bytes(hidden_width, element_bytes)?;
        let start = checked_product("sequence byte range", rows.start, row_bytes)?;
        let row_count = rows.end - rows.start;
        let size = checked_product("sequence byte range", row_count, row_bytes)?;
        let end = start
            .checked_add(size)
            .ok_or(DsparkBatchInputError::ArithmeticOverflow {
                operation: "sequence byte range",
                lhs: start,
                rhs: size,
            })?;
        Ok(start..end)
    }

    fn check_sequence(&self, sequence: usize) -> Result<(), DsparkBatchInputError> {
        if sequence < self.sequences.len() {
            Ok(())
        } else {
            Err(DsparkBatchInputError::SequenceOutOfBounds {
                sequence,
                batch: self.sequences.len(),
            })
        }
    }
}

fn check_len(
    field: &'static str,
    expected: usize,
    found: usize,
) -> Result<(), DsparkBatchInputError> {
    if expected == found {
        Ok(())
    } else {
        Err(DsparkBatchInputError::LengthMismatch {
            field,
            expected,
            found,
        })
    }
}

fn checked_row_bytes(
    hidden_width: usize,
    element_bytes: usize,
) -> Result<usize, DsparkBatchInputError> {
    if hidden_width == 0 {
        return Err(DsparkBatchInputError::ZeroDimension {
            field: "hidden_width",
        });
    }
    if element_bytes == 0 {
        return Err(DsparkBatchInputError::ZeroDimension {
            field: "element_bytes",
        });
    }
    checked_product("row bytes", hidden_width, element_bytes)
}

fn checked_product(
    operation: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize, DsparkBatchInputError> {
    lhs.checked_mul(rhs)
        .ok_or(DsparkBatchInputError::ArithmeticOverflow {
            operation,
            lhs,
            rhs,
        })
}
