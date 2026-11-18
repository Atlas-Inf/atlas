// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn hy4_registration_is_distinct_and_preserves_generation_sampling() {
    assert_eq!(
        super::resolve_name("hy_v4_flash", 3072, &["atakans/Hy4-flash-NVFP4"]),
        Some("hy4-flash")
    );
    assert_eq!(super::resolve_name("hy_v4_flash", 4096, &[]), None);
    let model: toml::Value = toml::from_str(include_str!(
        "../../../../kernels/gb10/hy4-flash/MODEL.toml"
    ))
    .unwrap();
    let generation: serde_json::Value = serde_json::from_str(include_str!(
        "../../../atlas-core/src/config/fixtures/hy4/generation_config.json"
    ))
    .unwrap();
    for category in ["thinking_text", "thinking_coding", "non_thinking", "tools"] {
        let preset = &model["sampling"][category];
        assert_eq!(
            preset["temperature"].as_float(),
            generation["temperature"].as_f64()
        );
        assert_eq!(preset["top_p"].as_float(), generation["top_p"].as_f64());
        assert_eq!(generation["top_k"], -1);
        assert_eq!(preset["top_k"].as_integer(), Some(0));
    }
    let kernel: toml::Value = toml::from_str(include_str!(
        "../../../../kernels/gb10/hy4-flash/nvfp4/KERNEL.toml"
    ))
    .unwrap();
    assert!(kernel["build"]["extra_nvcc_flags"].is_array());
}
