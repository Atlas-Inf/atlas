// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn registered_hy4_fails_before_constructing_a_forward_path() {
    let config = atlas_core::config::parse_config(include_str!(
        "../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    let error = match super::loader_for_config(&config) {
        Ok(_) => panic!("Hy4 must not borrow another family's loader"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "hy_v4_flash (hy4-flash/nvfp4): forward path not yet implemented; configuration and tensor-contract registration only"
    );
}
