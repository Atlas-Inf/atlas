// SPDX-License-Identifier: AGPL-3.0-only

//! Debug-only allocation audit for the phased UNIFIED transpose frees
//! (ATLAS_MOE_TRANSPOSE_AUDIT=1). Records which call freed each device
//! pointer and reports when a later call's source experts point at memory
//! an earlier call already freed.

use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

static AUDIT_CALL: AtomicUsize = AtomicUsize::new(0);
static AUDIT_FREED: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();

fn audit_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_TRANSPOSE_AUDIT").as_deref() == Ok("1"))
}

pub(super) fn audit_enter(layer: usize, experts: usize, keep_originals: bool) -> usize {
    let call = AUDIT_CALL.fetch_add(1, Ordering::Relaxed);
    if audit_on() {
        tracing::info!(target: "atlas::moe_audit",
            "call={call} layer={layer:#x} experts={experts} keep_originals={keep_originals}");
    }
    call
}

pub(super) fn audit_layout(call: usize, unified: bool, hybrid: bool) {
    if audit_on() {
        tracing::info!(target: "atlas::moe_audit",
            "call={call} unified_layout={unified} hybrid_layout={hybrid}");
    }
}

pub(super) fn audit_freed2(call: usize, a: DevicePtr, b: DevicePtr) {
    if !audit_on() {
        return;
    }
    let mut f = AUDIT_FREED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    f.insert(a.0, call);
    f.insert(b.0, call);
}

pub(super) fn audit_src(call: usize, tag: &str, src: &[QuantizedWeight]) {
    if !audit_on() {
        return;
    }
    let freed = AUDIT_FREED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    let mut live = 0usize;
    let mut stale = 0usize;
    let mut first: Option<(u64, usize)> = None;
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for w in src {
        if w.is_null() {
            continue;
        }
        live += 1;
        lo = lo.min(w.weight.0);
        hi = hi.max(w.weight.0);
        for p in [w.weight.0, w.weight_scale.0] {
            if let Some(&c) = freed.get(&p) {
                stale += 1;
                if first.is_none() {
                    first = Some((p, c));
                }
            }
        }
    }
    tracing::info!(target: "atlas::moe_audit",
        "call={call} {tag}: live={live} stale={stale} first_stale={first:?} weight=[{lo:#x},{hi:#x}]");
}

pub(super) fn audit_sync(call: usize, phase: &str, gpu: &dyn GpuBackend) {
    if !audit_on() {
        return;
    }
    let r = gpu.synchronize(gpu.default_stream());
    tracing::info!(target: "atlas::moe_audit", "call={call} {phase} sync: {r:?}");
}
