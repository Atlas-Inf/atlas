// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
#[test]
fn typed_projection_geometry_capacity_and_alias_contract() {
    let w = QuantizedWeight {
        weight: DevicePtr(0x100000),
        weight_scale: DevicePtr(0x600000),
        weight_scale_2: 1.,
        ..QuantizedWeight::null()
    };
    let a = DevicePtr(0x1000);
    let c = DevicePtr(0x800000);
    for p in [
        SharedProjection::Gate,
        SharedProjection::Up,
        SharedProjection::Down,
    ] {
        let (n, k, _) = p.geometry();
        let bytes = 5 * n as usize * 2;
        assert_eq!(checked_output(p, n, k, a, &w, c, c, bytes).unwrap(), bytes);
        assert!(checked_output(p, n, k, a, &w, c, c, bytes - 1).is_err());
        assert!(checked_output(p, n, k, a, &w, c, DevicePtr(c.0 + 2), bytes).is_err());
        assert!(checked_output(p, n + 1, k, a, &w, c, c, bytes).is_err());
        assert!(checked_output(p, n, k, a, &w, a, a, bytes).is_err());
        assert!(checked_output(p, n, k, DevicePtr(a.0 + 2), &w, c, c, bytes).is_err());
        let mut bad = w;
        bad.weight_scale_2 = f32::NAN;
        assert!(checked_output(p, n, k, a, &bad, c, c, bytes).is_err());
        bad = w;
        bad.weight_scale_2_vec = DevicePtr(16);
        assert!(checked_output(p, n, k, a, &bad, c, c, bytes).is_err());
    }
}
