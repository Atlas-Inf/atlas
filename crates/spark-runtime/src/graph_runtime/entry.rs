// SPDX-License-Identifier: AGPL-3.0-only

use super::{GraphCost, GraphIdentity, GraphKey, GraphRuntimeError};
use crate::gpu::{GpuBackend, GraphHandle};
use parking_lot::Mutex;
use std::sync::Arc;

enum EntryState {
    Active(GraphHandle),
    Evicted(GraphHandle),
    Retiring { handle: GraphHandle, event: u64 },
    Destroyed,
}

pub struct GraphEntry {
    identity: GraphIdentity,
    stream: u64,
    own_cost: GraphCost,
    total_cost: GraphCost,
    dependencies: Vec<Arc<GraphEntry>>,
    backend: Arc<dyn GpuBackend>,
    state: Mutex<EntryState>,
}

#[derive(Clone)]
pub struct GraphLease(pub(crate) Arc<GraphEntry>);

impl GraphLease {
    pub fn identity(&self) -> &GraphIdentity {
        &self.0.identity
    }

    pub fn key(&self) -> &GraphKey {
        &self.0.identity.key
    }

    pub fn stream(&self) -> u64 {
        self.0.stream
    }

    pub fn cost(&self) -> GraphCost {
        self.0.total_cost
    }

    pub fn is_active(&self) -> bool {
        matches!(*self.0.state.lock(), EntryState::Active(_))
    }
}

impl GraphEntry {
    pub(super) fn new(
        identity: GraphIdentity,
        stream: u64,
        cost: GraphCost,
        dependencies: Vec<GraphLease>,
        backend: Arc<dyn GpuBackend>,
        handle: GraphHandle,
    ) -> Arc<Self> {
        let total_cost = dependencies
            .iter()
            .fold(cost, |total, dependency| total.include(dependency.cost()));
        Arc::new(Self {
            identity,
            stream,
            own_cost: cost,
            total_cost,
            dependencies: dependencies.into_iter().map(|lease| lease.0).collect(),
            backend,
            state: Mutex::new(EntryState::Active(handle)),
        })
    }

    pub(super) fn key(&self) -> &GraphKey {
        &self.identity.key
    }

    pub(super) fn total_cost(&self) -> GraphCost {
        self.total_cost
    }

    pub(super) fn own_cost(&self) -> GraphCost {
        self.own_cost
    }

    pub(super) fn dependency_count(&self) -> usize {
        self.dependencies.len()
    }

    pub(super) fn launch(&self, stream: u64) -> Result<(), GraphRuntimeError> {
        if stream != self.stream {
            return Err(GraphRuntimeError::new(
                super::GraphFallbackReason::StreamMismatch,
                format!(
                    "graph captured on stream {} cannot launch on stream {stream}",
                    self.stream
                ),
            ));
        }
        let state = self.state.lock();
        let handle = match *state {
            EntryState::Active(handle) => handle,
            EntryState::Evicted(_) | EntryState::Retiring { .. } => {
                return Err(GraphRuntimeError::new(
                    super::GraphFallbackReason::Retired,
                    "graph was evicted before launch",
                ));
            }
            EntryState::Destroyed => {
                return Err(GraphRuntimeError::new(
                    super::GraphFallbackReason::Retired,
                    "graph was destroyed before launch",
                ));
            }
        };
        self.backend.launch_graph(handle, stream).map_err(|error| {
            GraphRuntimeError::new(
                super::GraphFallbackReason::ReplayLaunchFailed,
                format!("cuGraphLaunch failed: {error:#}"),
            )
        })
    }

    pub(super) fn mark_evicted(&self) -> bool {
        let mut state = self.state.lock();
        if let EntryState::Active(handle) = *state {
            *state = EntryState::Evicted(handle);
            true
        } else {
            false
        }
    }

    pub(super) fn try_fence_retirement(&self) -> anyhow::Result<()> {
        let mut state = self.state.lock();
        let EntryState::Evicted(handle) = *state else {
            return Ok(());
        };
        let event = self.backend.create_event()?;
        if let Err(error) = self.backend.record_event(event, self.stream) {
            let _ = self.backend.destroy_event(event);
            return Err(error);
        }
        *state = EntryState::Retiring { handle, event };
        Ok(())
    }

    pub(super) fn poll_retirement(&self) -> anyhow::Result<bool> {
        let mut state = self.state.lock();
        let EntryState::Retiring { handle, event } = *state else {
            return Ok(matches!(*state, EntryState::Destroyed));
        };
        if !self.backend.event_query(event)? {
            return Ok(false);
        }
        self.backend.destroy_graph(handle)?;
        let _ = self.backend.destroy_event(event);
        *state = EntryState::Destroyed;
        Ok(true)
    }

    pub(super) fn force_destroy_after_sync(&self) -> anyhow::Result<bool> {
        let mut state = self.state.lock();
        let (handle, event) = match *state {
            EntryState::Active(handle) | EntryState::Evicted(handle) => (handle, None),
            EntryState::Retiring { handle, event } => (handle, Some(event)),
            EntryState::Destroyed => return Ok(false),
        };
        self.backend.destroy_graph(handle)?;
        if let Some(event) = event {
            let _ = self.backend.destroy_event(event);
        }
        *state = EntryState::Destroyed;
        Ok(true)
    }
}
