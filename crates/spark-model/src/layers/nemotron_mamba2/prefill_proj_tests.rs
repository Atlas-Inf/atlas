// SPDX-License-Identifier: AGPL-3.0-only

use super::prefill_proj::use_lightning_scalar_in_proj;

#[test]
fn lightning_scalar_projection_is_bounded_to_verify_widths() {
    assert!(use_lightning_scalar_in_proj(true, 1));
    assert!(use_lightning_scalar_in_proj(true, 4));
    assert!(use_lightning_scalar_in_proj(true, 16));
    assert!(!use_lightning_scalar_in_proj(true, 17));
    assert!(!use_lightning_scalar_in_proj(false, 4));
}
