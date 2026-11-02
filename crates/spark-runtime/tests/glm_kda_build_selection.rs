// SPDX-License-Identifier: AGPL-3.0-only

#[allow(dead_code)]
#[path = "../build_glm_kda.rs"]
mod native_build;

#[test]
fn kda_native_object_is_scoped_to_the_glm_cuda_target() {
    assert!(native_build::selected("gb10", "glm-5.3-flash"));
    assert!(native_build::selected("gb10", "*"));
    for hardware in ["metal", "strix", "strix-hip"] {
        assert!(!native_build::selected(hardware, "glm-5.3-flash"));
        assert!(!native_build::selected(hardware, "*"));
    }
    assert!(!native_build::selected("gb10", "qwen3.6-27b"));
}
