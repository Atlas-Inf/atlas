// SPDX-License-Identifier: AGPL-3.0-only

use super::{GraphEntry, GraphFallbackReason, GraphKey, GraphLease, GraphPhase, GraphPolicies};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

struct PhaseCache {
    entries: HashMap<GraphKey, Arc<GraphEntry>>,
    lru: VecDeque<GraphKey>,
    resident_bytes: u64,
    negative: HashMap<GraphKey, GraphFallbackReason>,
    negative_lru: VecDeque<GraphKey>,
}

impl PhaseCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            resident_bytes: 0,
            negative: HashMap::new(),
            negative_lru: VecDeque::new(),
        }
    }

    fn touch(&mut self, key: &GraphKey) {
        self.lru.retain(|candidate| candidate != key);
        self.lru.push_back(key.clone());
    }

    fn get(&mut self, key: &GraphKey) -> Option<GraphLease> {
        let entry = self.entries.get(key)?.clone();
        self.touch(key);
        Some(GraphLease(entry))
    }

    fn remove(&mut self, key: &GraphKey) -> Option<Arc<GraphEntry>> {
        self.lru.retain(|candidate| candidate != key);
        let entry = self.entries.remove(key)?;
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(entry.total_cost().total_bytes());
        Some(entry)
    }

    fn pop_lru(&mut self) -> Option<Arc<GraphEntry>> {
        while let Some(key) = self.lru.pop_front() {
            if let Some(entry) = self.entries.remove(&key) {
                self.resident_bytes = self
                    .resident_bytes
                    .saturating_sub(entry.total_cost().total_bytes());
                return Some(entry);
            }
        }
        None
    }

    fn block(&mut self, key: GraphKey, reason: GraphFallbackReason, limit: usize) {
        if limit == 0 {
            return;
        }
        self.negative_lru.retain(|candidate| candidate != &key);
        self.negative.insert(key.clone(), reason);
        self.negative_lru.push_back(key);
        while self.negative.len() > limit {
            if let Some(oldest) = self.negative_lru.pop_front() {
                self.negative.remove(&oldest);
            }
        }
    }
}

pub(super) struct GraphCache {
    phases: [PhaseCache; GraphPhase::COUNT],
    policies: GraphPolicies,
    global_lru: VecDeque<GraphKey>,
    max_entries: usize,
    max_bytes: u64,
}

pub(super) struct InsertResult {
    /// The previously resident entry for the same key, replaced by this
    /// insert. Reported separately from `evicted` so metric attribution never
    /// depends on the order entries happen to be pushed.
    pub recaptured: Option<Arc<GraphEntry>>,
    /// Entries displaced to satisfy a phase or global quota.
    pub evicted: Vec<Arc<GraphEntry>>,
}

impl GraphCache {
    pub fn new(policies: GraphPolicies, max_entries: usize, max_bytes: u64) -> Self {
        Self {
            phases: std::array::from_fn(|_| PhaseCache::new()),
            policies,
            global_lru: VecDeque::new(),
            max_entries,
            max_bytes,
        }
    }

    pub fn get(&mut self, key: &GraphKey) -> Option<GraphLease> {
        let lease = self.phases[key.phase.index()].get(key);
        if lease.is_some() {
            self.touch_global(key);
        }
        lease
    }

    pub fn touch(&mut self, key: &GraphKey) {
        if self.phases[key.phase.index()].entries.contains_key(key) {
            self.phases[key.phase.index()].touch(key);
            self.touch_global(key);
        }
    }

    pub fn cost_fits_global_limit(&self, cost: u64) -> bool {
        self.max_entries > 0 && cost <= self.max_bytes
    }

    pub fn negative_reason(&self, key: &GraphKey) -> Option<GraphFallbackReason> {
        self.phases[key.phase.index()].negative.get(key).copied()
    }

    pub fn block(&mut self, key: GraphKey, reason: GraphFallbackReason) {
        let limit = self.policies.phase(key.phase).max_entries;
        self.phases[key.phase.index()].block(key, reason, limit);
    }

    pub fn insert(&mut self, entry: Arc<GraphEntry>) -> InsertResult {
        let key = entry.key().clone();
        let policy = self.policies.phase(key.phase);
        let recaptured = self.remove(&key);
        let mut evicted = Vec::new();
        {
            let phase = &mut self.phases[key.phase.index()];
            phase.negative.remove(&key);
            phase.negative_lru.retain(|candidate| candidate != &key);
            phase.resident_bytes = phase
                .resident_bytes
                .saturating_add(entry.total_cost().total_bytes());
            phase.entries.insert(key.clone(), entry);
            phase.touch(&key);
        }
        self.touch_global(&key);
        while self.phases[key.phase.index()].entries.len() > policy.max_entries
            || self.phases[key.phase.index()].resident_bytes > policy.max_estimated_bytes
        {
            let Some(entry) = self.phases[key.phase.index()].pop_lru() else {
                break;
            };
            self.global_lru.retain(|candidate| candidate != entry.key());
            evicted.push(entry);
        }
        while self.total_entries() > self.max_entries || self.total_bytes() > self.max_bytes {
            let Some(oldest) = self.global_lru.pop_front() else {
                break;
            };
            if let Some(entry) = self.phases[oldest.phase.index()].remove(&oldest) {
                evicted.push(entry);
            }
        }
        InsertResult {
            recaptured,
            evicted,
        }
    }

    pub fn remove(&mut self, key: &GraphKey) -> Option<Arc<GraphEntry>> {
        self.global_lru.retain(|candidate| candidate != key);
        self.phases[key.phase.index()].remove(key)
    }

    pub fn remove_matching(
        &mut self,
        mut predicate: impl FnMut(&GraphKey) -> bool,
    ) -> Vec<Arc<GraphEntry>> {
        let keys: Vec<GraphKey> = self
            .phases
            .iter()
            .flat_map(|phase| phase.entries.keys())
            .filter(|key| predicate(key))
            .cloned()
            .collect();
        keys.into_iter()
            .filter_map(|key| self.remove(&key))
            .collect()
    }

    pub fn leases(&self, phase: GraphPhase) -> Vec<GraphLease> {
        self.phases[phase.index()]
            .entries
            .values()
            .cloned()
            .map(GraphLease)
            .collect()
    }

    fn touch_global(&mut self, key: &GraphKey) {
        self.global_lru.retain(|candidate| candidate != key);
        self.global_lru.push_back(key.clone());
    }

    fn total_entries(&self) -> usize {
        self.phases.iter().map(|phase| phase.entries.len()).sum()
    }

    fn total_bytes(&self) -> u64 {
        self.phases.iter().map(|phase| phase.resident_bytes).sum()
    }

    pub fn drain(&mut self) -> Vec<Arc<GraphEntry>> {
        let mut entries = Vec::new();
        for phase in &mut self.phases {
            entries.extend(phase.entries.drain().map(|(_, entry)| entry));
            phase.lru.clear();
            phase.resident_bytes = 0;
            phase.negative.clear();
            phase.negative_lru.clear();
        }
        self.global_lru.clear();
        entries
    }

    #[cfg(test)]
    pub fn resident_counts(&self) -> [usize; GraphPhase::COUNT] {
        std::array::from_fn(|index| self.phases[index].entries.len())
    }
}
