// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn registered_hy4_selects_its_own_tp_loader() {
    let config = atlas_core::config::parse_config(include_str!(
        "../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    let loader = super::loader_for_config(&config).unwrap();
    assert!(loader.supports_tp());
    assert!(
        loader
            .dead_namespaces(&config)
            .contains(&"model.dfly_layers.".to_owned())
    );
    assert!(loader.dead_namespaces(&config).contains(&"vit.".to_owned()));
}
