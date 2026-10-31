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
    let nvcc = PathBuf::from(cuda).join("bin/nvcc");
    let includes = ["include", "examples/common", "tools/util/include"]
        .map(|directory| format!("-I{}", cutlass_home.join(directory).display()));
    // Atlas's own operator, then SparkGLM's FP32-recurrence FlashKDA library
    // and its bridge (cuda/glm_kda_fp32, ATLAS_KDA_FLASH_LIBRARY), each with
    // the flags SparkGLM built it with (flash_kda/rebuild.sh, build-native.sh).
    let sparkglm = PathBuf::from("cuda/glm_kda_fp32");
    println!("cargo:rerun-if-changed={}", sparkglm.display());
    let units: [(PathBuf, &str, &[&str]); 3] = [
        (source_dir.join("launch.cu"), "atlas_glm_kda.o", &[]),
        (
            sparkglm.join("kernels.cu"),
            "atlas_sparkglm_kda.o",
            &[
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--use_fast_math",
                "-Icuda/glm_kda_fp32/flashkda",
            ],
        ),
        (
            sparkglm.join("runtime_bridge.cu"),
            "atlas_sparkglm_kda_bridge.o",
            &["--fmad=false"],
        ),
    ];
    let jobs: Vec<_> = units
        .iter()
        .map(|(source, object, flags)| {
            let child = Command::new(&nvcc)
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
                .args(&includes)
                .args(*flags)
                .arg(source)
                .arg("-o")
                .arg(out.join(object))
                .spawn()
                .expect("failed to spawn nvcc for native GLM KDA");
            (source, out.join(object), child)
        })
        .collect();
    let mut objects = Vec::new();
    for (source, object, mut child) in jobs {
        let status = child.wait().expect("nvcc for native GLM KDA");
        assert!(
            status.success(),
            "nvcc failed while compiling {}",
            source.display()
        );
        objects.push(object);
    }
    let library = out.join("libatlas_glm_kda.a");
    let _ = std::fs::remove_file(&library);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&library)
        .args(&objects)
        .status()
        .expect("failed to spawn ar for native GLM KDA");
    assert!(status.success(), "ar failed while archiving native GLM KDA");
    println!("cargo:rustc-cfg=atlas_glm_kda");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=atlas_glm_kda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
