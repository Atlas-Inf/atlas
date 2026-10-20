// SPDX-License-Identifier: AGPL-3.0-only
//! Source origin travels with the real dense conversion, not its recycled address.
use super::*;
use crate::weight_map::DenseWeight;

pub(in crate::weight_loader::glm5) struct LoadedDense<'a> {
    pub(in crate::weight_loader::glm5) dense: DenseWeight,
    store: &'a WeightStore,
    log: Option<&'a RetirementLog<'a>>,
    origin: Option<Origin>,
    checkpoint: Option<Identity>,
    bytes: usize,
}

impl<'a> LoadedDense<'a> {
    pub(in crate::weight_loader::glm5) fn load(
        store: &'a WeightStore,
        name: &str,
        gpu: &dyn GpuBackend,
        log: Option<&'a RetirementLog<'a>>,
        keep_f32: bool,
    ) -> Result<Self> {
        // The production loader has no construction log. Capture ownership
        // before conversion: a later allocation may reuse a retired address.
        let checkpoint = if log.is_none() {
            ensure!(
                !store.was_reclaimed(name),
                "checkpoint already reclaimed: {name}"
            );
            let source = Identity::read(store, name)?;
            (source.dtype
                == if keep_f32 {
                    WeightDtype::FP32
                } else {
                    WeightDtype::BF16
                })
            .then_some(source)
        } else {
            None
        };
        let origin = log.map(|log| log.capture(store, name, gpu)).transpose()?;
        let native = origin.as_ref().is_some_and(|origin| {
            origin.identity.dtype
                == if keep_f32 {
                    WeightDtype::FP32
                } else {
                    WeightDtype::BF16
                }
        });
        let bytes = if let Some(origin) = &origin {
            let width = if keep_f32 || origin.identity.dtype == WeightDtype::UInt8 {
                4
            } else {
                2
            };
            origin
                .identity
                .shape
                .iter()
                .try_fold(width, |n: usize, dim| {
                    n.checked_mul(*dim).context("dense output extent overflow")
                })?
        } else {
            0
        };
        let dense = if keep_f32 {
            crate::weight_map::dense_keep_f32(store, name, gpu)?
        } else {
            crate::weight_map::dense_auto(store, name, gpu)?
        };
        if let Some(source) = &checkpoint {
            ensure!(
                source.ptr == dense.weight,
                "native dense source changed address"
            );
        }
        let origin = if native {
            ensure!(
                origin
                    .as_ref()
                    .is_some_and(|origin| origin.identity.ptr == dense.weight),
                "native dense source changed address"
            );
            origin
        } else {
            if let Some(log) = log {
                log.validate_derived(store, dense.weight, bytes, gpu)?;
            }
            None
        };
        Ok(Self {
            dense,
            store,
            log,
            origin,
            checkpoint,
            bytes,
        })
    }

    pub(in crate::weight_loader::glm5) fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        match (self.log, self.origin) {
            (Some(log), Some(origin)) => log.release_origin(self.store, origin, gpu),
            (Some(log), None) => {
                log.release_derived(self.store, self.dense.weight, self.bytes, gpu)
            }
            (None, None) => match self.checkpoint {
                Some(source) => {
                    source.matches(self.store)?;
                    ensure!(
                        source.ptr == self.dense.weight,
                        "checkpoint release changed address"
                    );
                    ensure!(
                        !self.store.was_reclaimed(&source.name),
                        "checkpoint already reclaimed: {}",
                        source.name
                    );
                    // Record the retirement attempt before freeing, including
                    // ambiguous failures. Store cleanup must not retry it;
                    // the backend retains its own failed-free cleanup ledger.
                    self.store.reclaim(gpu, &source.name)
                }
                None => gpu.free(self.dense.weight),
            },
            (None, Some(_)) => unreachable!("origin requires construction log"),
        }
    }

    pub(in crate::weight_loader::glm5) fn replaced(
        self,
        ptr: DevicePtr,
        bytes: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        if ptr == self.dense.weight {
            return Ok(self);
        }
        let store = self.store;
        let log = self.log;
        if let Some(log) = log {
            log.validate_derived(store, ptr, bytes, gpu)?;
        }
        self.release(gpu)?;
        Ok(Self {
            dense: DenseWeight { weight: ptr },
            store,
            log,
            origin: None,
            checkpoint: None,
            bytes,
        })
    }

    pub(in crate::weight_loader::glm5) fn into_dense(self) -> DenseWeight {
        self.dense
    }
}

#[cfg(test)]
#[path = "retirement_dense_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "retirement_reuse_tests.rs"]
mod reuse_tests;
