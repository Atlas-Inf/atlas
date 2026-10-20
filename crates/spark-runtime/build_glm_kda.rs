// SPDX-License-Identifier: AGPL-3.0-only

pub fn selected(hardware: &str, model: &str) -> bool {
    hardware == "gb10" && matches!(model, "glm-5.3-flash" | "*")
}

/// Compile the Atlas-owned operator; CUDA/CUTLASS provide device primitives.
pub fn build(cutlass_home: &std::path::Path) {
    use std::{path::PathBuf, process::Command};
    let source_dir = PathBuf::from("cuda/glm_kda");
    println!("cargo:rerun-if-changed={}", source_dir.display());
    // A dependency checkout may change without CUTLASS_HOME changing. Rebuild
    // the object whenever an included header changes in that same directory.
    for directory in ["include", "examples/common", "tools/util/include"] {
        println!(
            "cargo:rerun-if-changed={}",
            cutlass_home.join(directory).display()
        );
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set"));
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let arch = std::env::var("ATLAS_CUDA_ARCH").unwrap_or_else(|_| "sm_121f".into());
    let object = out.join("atlas_glm_kda.o");
    let library = out.join("libatlas_glm_kda.a");
    let status = Command::new(PathBuf::from(cuda).join("bin/nvcc"))
        .args([
            "-c",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcompiler",
            "-fPIC",
        ])
        .arg(format!("-arch={arch}"))
        .arg("-I")
        .arg(cutlass_home.join("include"))
        .arg("-I")
        .arg(cutlass_home.join("examples/common"))
        .arg("-I")
        .arg(cutlass_home.join("tools/util/include"))
        .arg(source_dir.join("launch.cu"))
        .arg("-o")
        .arg(&object)
        .status()
        .expect("failed to spawn nvcc for native GLM KDA");
    assert!(
        status.success(),
        "nvcc failed while compiling native GLM KDA"
    );
    let status = Command::new("ar")
        .arg("crs")
        .arg(&library)
        .arg(&object)
        .status()
        .expect("failed to spawn ar for native GLM KDA");
    assert!(status.success(), "ar failed while archiving native GLM KDA");
    println!("cargo:rustc-cfg=atlas_glm_kda");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=atlas_glm_kda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
