// SPDX-License-Identifier: AGPL-3.0-only
//! Exact checkpoint ownership retirement during one GLM construction.
//! Private construction entry points are staged until the resident loader exists.
#![cfg_attr(not(test), allow(dead_code))]

use anyhow::{Context, Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct Identity {
    name: String,
    ptr: DevicePtr,
    dtype: WeightDtype,
    shape: Vec<usize>,
    bytes: usize,
}

impl Identity {
    fn read(store: &WeightStore, name: &str) -> Result<Self> {
        let tensor = store.get(name)?;
        let bytes = tensor
            .shape
            .iter()
            .try_fold(tensor.dtype.byte_size(), |n, dim| {
                n.checked_mul(*dim).context("checkpoint extent overflow")
            })?;
        ensure!(
            bytes > 0 && !tensor.ptr.is_null(),
            "invalid checkpoint extent: {name}"
        );
        tensor
            .ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .context("checkpoint span overflow")?;
        Ok(Self {
            name: name.into(),
            ptr: tensor.ptr,
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            bytes,
        })
    }

    fn matches(&self, store: &WeightStore) -> Result<()> {
        let actual = Self::read(store, &self.name)?;
        ensure!(
            self.ptr == actual.ptr
                && self.dtype == actual.dtype
                && self.shape == actual.shape
                && self.bytes == actual.bytes,
            "changed checkpoint identity: {}",
            self.name
        );
        Ok(())
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.ptr.0 < other.ptr.0 + other.bytes as u64
            && other.ptr.0 < self.ptr.0 + self.bytes as u64
    }
}

/// Only actual source capture mints this receipt, before conversion or free.
pub(crate) struct Origin {
    identity: Identity,
    owner: Arc<()>,
}

struct Attempt {
    identity: Identity,
    succeeded: bool,
}
#[derive(Default)]
struct State {
    attempts: HashMap<String, Attempt>,
    failure: Option<String>,
    rebuilt: bool,
}

pub(crate) struct RetirementLog<'s> {
    store: &'s WeightStore,
    gpu: &'s dyn GpuBackend,
    index: Vec<Identity>,
    backend: usize,
    owner: Arc<()>,
    state: Mutex<State>,
}

fn backend(gpu: &dyn GpuBackend) -> usize {
    gpu as *const dyn GpuBackend as *const () as usize
}

impl<'s> RetirementLog<'s> {
    pub(crate) fn new(store: &'s WeightStore, gpu: &'s dyn GpuBackend) -> Result<Self> {
        let mut index = store
            .names()
            .map(|name| Identity::read(store, name))
            .collect::<Result<Vec<_>>>()?;
        index.sort_unstable_by_key(|owner| owner.ptr.0);
        for pair in index.windows(2) {
            ensure!(
                !pair[0].overlaps(&pair[1]),
                "checkpoint {} aliases live owner {}",
                pair[0].name,
                pair[1].name
            );
        }
        Ok(Self {
            store,
            gpu,
            index,
            backend: backend(gpu),
            owner: Arc::new(()),
            state: Mutex::new(State::default()),
        })
    }

    fn check(&self, gpu: &dyn GpuBackend, state: &State) -> Result<()> {
        ensure!(
            backend(self.gpu) == backend(gpu),
            "foreign retirement backend"
        );
        ensure!(!state.rebuilt, "checkpoint ownership already transferred");
        ensure!(
            state.failure.is_none(),
            "failed GLM construction: {:?}",
            state.failure
        );
        Ok(())
    }

    pub(crate) fn capture(
        &self,
        store: &WeightStore,
        name: &str,
        gpu: &dyn GpuBackend,
    ) -> Result<Origin> {
        let state = self.state.lock();
        self.check(gpu, &state)?;
        ensure!(
            std::ptr::eq(store, self.store),
            "foreign construction checkpoint store"
        );
        ensure!(
            !state.attempts.contains_key(name),
            "checkpoint already retired/attempted: {name}"
        );
        let identity = Identity::read(store, name)?;
        Ok(Origin {
            identity,
            owner: self.owner.clone(),
        })
    }

    pub(crate) fn store(&self) -> &WeightStore {
        self.store
    }

    pub(crate) fn capture_exact(
        &self,
        name: &str,
        ptr: DevicePtr,
        dtype: WeightDtype,
        shape: &[usize],
        gpu: &dyn GpuBackend,
    ) -> Result<Origin> {
        let origin = self.capture(self.store, name, gpu)?;
        ensure!(
            origin.identity.ptr == ptr
                && origin.identity.dtype == dtype
                && origin.identity.shape == shape,
            "checkpoint projection identity mismatch: {name}"
        );
        Ok(origin)
    }

    pub(crate) fn is_live(&self, name: &str, gpu: &dyn GpuBackend) -> Result<bool> {
        let state = self.state.lock();
        self.check(gpu, &state)?;
        self.store.get(name)?;
        Ok(!state.attempts.contains_key(name))
    }

    pub(crate) fn disjoint_live(
        &self,
        ptr: DevicePtr,
        bytes: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.validate_derived(self.store, ptr, bytes, gpu)
    }

    pub(crate) fn release_origin(
        &self,
        store: &WeightStore,
        origin: Origin,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.owner, &origin.owner),
            "foreign checkpoint origin"
        );
        // Recheck the exact live map immediately before the irreversible call.
        origin.identity.matches(store)?;
        let current = self.capture(store, &origin.identity.name, gpu)?;
        let mut state = self.state.lock();
        self.check(gpu, &state)?;
        ensure!(
            !state.attempts.contains_key(&current.identity.name),
            "duplicate retirement attempt"
        );
        let name = current.identity.name.clone();
        let ptr = current.identity.ptr;
        state.attempts.insert(
            name.clone(),
            Attempt {
                identity: current.identity,
                succeeded: false,
            },
        );
        // CUDA removes its ledger entry before cuMemFree. A failed call is not retryable.
        if let Err(error) = gpu.free(ptr) {
            let message = format!(
                "free checkpoint {name} at {:#x} failed: {error:#}; allocation ownership unknown, context teardown may be required",
                ptr.0
            );
            state.failure = Some(message.clone());
            anyhow::bail!(message);
        }
        state
            .attempts
            .get_mut(&name)
            .expect("inserted attempt")
            .succeeded = true;
        Ok(())
    }

    pub(crate) fn release_checkpoint(
        &self,
        store: &WeightStore,
        name: &str,
        expected: DevicePtr,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let origin = self.capture(store, name, gpu)?;
        ensure!(
            origin.identity.ptr == expected,
            "checkpoint pointer mismatch: {name}"
        );
        self.release_origin(store, origin, gpu)
    }

    fn validate_derived(
        &self,
        store: &WeightStore,
        ptr: DevicePtr,
        bytes: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let state = self.state.lock();
        self.check(gpu, &state)?;
        ensure!(
            std::ptr::eq(store, self.store),
            "foreign construction checkpoint store"
        );
        ensure!(
            bytes > 0 && !ptr.is_null(),
            "invalid derived allocation extent"
        );
        let end = ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .context("derived extent overflow")?;
        let first = self
            .index
            .partition_point(|owner| owner.ptr.0 + owner.bytes as u64 <= ptr.0);
        for owner in self.index[first..]
            .iter()
            .take_while(|owner| owner.ptr.0 < end)
        {
            ensure!(
                state.attempts.contains_key(&owner.name),
                "derived allocation aliases live checkpoint {}",
                owner.name
            );
        }
        Ok(())
    }

    fn release_derived(
        &self,
        store: &WeightStore,
        ptr: DevicePtr,
        bytes: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.validate_derived(store, ptr, bytes, gpu)?;
        let mut state = self.state.lock();
        self.check(gpu, &state)?;
        if let Err(error) = gpu.free(ptr) {
            let message = format!(
                "free derived allocation {:#x} failed: {error:#}; context teardown may be required",
                ptr.0
            );
            state.failure = Some(message.clone());
            anyhow::bail!(message);
        }
        Ok(())
    }

    /// End the immutable construction borrow before rebuilding the one owned map.
    pub(crate) fn finish(self) -> OwnershipReceipt {
        OwnershipReceipt {
            backend: self.backend,
            index: self.index,
            state: self.state,
        }
    }
}

pub(crate) struct OwnershipReceipt {
    backend: usize,
    index: Vec<Identity>,
    state: Mutex<State>,
}

impl OwnershipReceipt {
    pub(crate) fn rebuild(&self, store: &mut WeightStore, gpu: &dyn GpuBackend) -> Result<()> {
        self.replace_map(store, gpu, false)
    }

    fn replace_map(
        &self,
        store: &mut WeightStore,
        gpu: &dyn GpuBackend,
        cleanup: bool,
    ) -> Result<()> {
        let mut state = self.state.lock();
        ensure!(self.backend == backend(gpu), "foreign retirement backend");
        ensure!(!state.rebuilt, "checkpoint ownership already transferred");
        ensure!(
            cleanup || state.failure.is_none(),
            "failed GLM construction: {:?}",
            state.failure
        );
        ensure!(
            store.len() == self.index.len(),
            "changed checkpoint ownership count"
        );
        for owner in &self.index {
            owner.matches(store)?;
        }
        for attempt in state.attempts.values() {
            attempt.identity.matches(store)?;
            ensure!(
                cleanup || attempt.succeeded,
                "unknown checkpoint free outcome"
            );
        }
        let mut live = HashMap::with_capacity(store.len() - state.attempts.len());
        for name in store.names() {
            if state.attempts.contains_key(name) {
                continue;
            }
            let tensor = store.get(name)?;
            live.insert(
                name.into(),
                WeightTensor {
                    ptr: tensor.ptr,
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                },
            );
        }
        *store = WeightStore::from_map(live); // Host metadata only; WeightStore has no freeing Drop.
        state.rebuilt = true;
        Ok(())
    }

    pub(crate) fn cleanup(
        &self,
        store: &mut WeightStore,
        gpu: &(dyn GpuBackend + 'static),
    ) -> Result<()> {
        use atlas_core::scope::ModelResource;
        self.replace_map(store, gpu, true)?;
        let release = store.release(gpu);
        let state = self.state.lock();
        if let Some(failure) = &state.failure {
            anyhow::bail!("{failure}; remaining checkpoint cleanup: {release:?}");
        }
        release
    }
}

#[cfg(test)]
#[path = "retirement_tests.rs"]
mod tests;

#[path = "retirement_dense.rs"]
mod dense;
pub(super) use dense::LoadedDense;

#[cfg(test)]
#[path = "retirement_loader_tests.rs"]
mod loader_tests;

#[cfg(test)]
#[path = "retirement_adoption_tests.rs"]
mod adoption_tests;
