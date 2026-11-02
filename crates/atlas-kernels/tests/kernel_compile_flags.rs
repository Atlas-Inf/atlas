// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../build_kernel_flags.rs"]
mod flags;

fn config(text: &str) -> toml::Value {
    toml::from_str(text).expect("fixture TOML")
}

#[test]
fn source_flags_preserve_order_and_do_not_change_other_kernels() {
    let cfg = config(
        r#"[build.kernel_nvcc_flags]
attention = ["--use_fast_math", "--fmad=true"]
prep = ["--prec-div=true", "--ftz=false"]
"#,
    );
    let by_source = flags::parse(&cfg, "nvidia").unwrap();
    let base = vec!["--fmad=false".to_string()];
    assert_eq!(
        flags::for_source(&base, &by_source, "attention"),
        ["--fmad=false", "--use_fast_math", "--fmad=true"]
    );
    assert_eq!(
        flags::for_source(&base, &by_source, "prep"),
        ["--fmad=false", "--prec-div=true", "--ftz=false"]
    );
    assert_eq!(flags::for_source(&base, &by_source, "ordinary"), base);
    assert!(flags::parse(&cfg, "metal").unwrap().is_empty());
}

#[test]
fn malformed_per_source_flags_fail_closed() {
    for text in [
        "[build]\nkernel_nvcc_flags = 4",
        "[build.kernel_nvcc_flags]\nprep = true",
        "[build.kernel_nvcc_flags]\nprep = [4]",
        "[build.kernel_nvcc_flags]\nprep = [\"\"]",
        "[build.kernel_nvcc_flags]\n\"../prep\" = []",
    ] {
        assert!(flags::parse(&config(text), "nvidia").is_err(), "{text}");
    }
}
