// SPDX-License-Identifier: AGPL-3.0-only
use super::k128w_kernels;
use spark_runtime::gpu::mock::MockGpuBackend;

fn handles(k128w: bool, silu: bool, persist: bool) -> ([u64; 4], u32) {
    let (down, gate_up, ctas) = k128w_kernels(&MockGpuBackend::new(), k128w, silu, persist);
    let ids = [down.grid, down.persist, gate_up.grid, gate_up.persist].map(|h| h.0);
    (ids.map(|h| u64::from(h != 0)), ctas)
}

#[test]
fn k128w_persist_adds_the_twins_and_keeps_the_grid_kernels() {
    // Flag off: the base grid kernels only.
    assert_eq!(handles(true, true, false), ([1, 0, 1, 0], 0));
    // Flag on: both sets, two CTAs per SM (the mock's 48 SMs).
    assert_eq!(handles(true, true, true), ([1, 1, 1, 1], 96));
    assert_eq!(handles(true, false, true), ([1, 1, 0, 0], 96));
    // K128W off: nothing, whatever the flag.
    assert_eq!(handles(false, true, true), ([0, 0, 0, 0], 0));
}
