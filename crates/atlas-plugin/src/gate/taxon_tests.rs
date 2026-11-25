// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn repo_root() -> PathBuf {
    // crates/atlas-plugin -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-taxon-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Build a minimal fake tree so the unit tests do not depend on the real
/// kernel layout, which changes as models are added.
fn fixture(name: &str) -> PathBuf {
    let root = tmp(name);
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("common")).unwrap();
    std::fs::create_dir_all(hw.join("modelA/nvfp4")).unwrap();
    std::fs::create_dir_all(hw.join("modelB/nvfp4")).unwrap();
    std::fs::write(
        hw.join("HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("modelA/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("modelB/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("common/shared.cu"), "__global__ void s() {}\n").unwrap();
    std::fs::write(hw.join("common/other.cu"), "__global__ void o() {}\n").unwrap();
    std::fs::write(hw.join("common/helper.cuh"), "#define H 1\n").unwrap();
    std::fs::write(
        hw.join("modelA/nvfp4/shared.cu"),
        "__global__ void s2() {}\n",
    )
    .unwrap();
    root
}

// ---------------------------------------------------------------------------
// Agreement with the real tree
// ---------------------------------------------------------------------------

/// ★ The load-bearing invariant: every target the walk finds must resolve
/// sources. `sources()` returning `None` means the gate falls back to
/// "affected" — correct but expensive — and returning an EMPTY set would be a
/// fail-open, since every empty set hashes alike.
#[test]
fn every_real_target_resolves_a_nonempty_source_set() {
    let root = repo_root();
    let targets = walk(&root);
    assert!(
        targets.len() >= 4,
        "the walk found {} targets — the kernel tree has many more, so the \
         walk is broken rather than the tree being small",
        targets.len()
    );
    for t in &targets {
        let srcs = sources(&root, t)
            .unwrap_or_else(|| panic!("{t}: sources() returned None — vendor table stale?"));
        assert!(!srcs.is_empty(), "{t}: empty source set");
    }
}

/// Every hardware dir in the tree must have a known vendor. A new backend
/// lands as `None` and its targets go to the expensive path silently; this
/// test makes that visible at the moment it is added.
#[test]
fn every_hardware_vendor_is_known_to_the_source_extension_table() {
    let root = repo_root();
    for t in walk(&root) {
        let v = vendor(&root, &t.hardware)
            .unwrap_or_else(|| panic!("{}: HARDWARE.toml has no vendor", t.hardware));
        assert!(
            source_ext(&v).is_some(),
            "hardware {} declares vendor {v:?}, which source_ext() does not know — \
             teach it the extension or its targets can never be skipped",
            t.hardware
        );
    }
}

/// `common/` holds no `MODEL.toml`, so it must never appear as a model. If it
/// did, a shared-kernel edit would be scoped to a model that does not exist.
#[test]
fn common_is_never_a_model() {
    for t in walk(&repo_root()) {
        assert_ne!(t.model, "common", "common/ resolved as a model");
    }
}

// ---------------------------------------------------------------------------
// Shadowing
// ---------------------------------------------------------------------------

#[test]
fn a_model_file_shadows_the_common_file_with_the_same_stem() {
    let root = fixture("shadow");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "nvfp4".into(),
    };
    let srcs = sources(&root, &t).unwrap();
    let shared: Vec<_> = srcs
        .iter()
        .filter(|p| p.file_stem().unwrap() == "shared")
        .collect();
    assert_eq!(shared.len(), 1, "one file per stem after shadowing");
    assert!(
        shared[0].to_string_lossy().contains("modelA/nvfp4"),
        "the model copy must win: {shared:?}"
    );
    assert!(
        srcs.iter()
            .any(|p| p.to_string_lossy().contains("common/other.cu")),
        "unshadowed common files stay in the set"
    );
}

/// Headers are not sources — they enter the hash only by being included. If
/// `sources()` returned them, an unused header would invalidate a target that
/// never sees it.
#[test]
fn headers_are_not_in_the_source_set() {
    let root = fixture("headers");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelB".into(),
        quant: "nvfp4".into(),
    };
    assert!(
        !sources(&root, &t)
            .unwrap()
            .iter()
            .any(|p| p.extension().unwrap() == "cuh"),
        ".cuh must not be a source"
    );
}

/// ★ Fail-closed: no resolvable sources is `None`, never `Some(vec![])`.
#[test]
fn an_unresolvable_target_is_none_not_empty() {
    let root = fixture("empty");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "does-not-exist".into(),
    };
    std::fs::remove_dir_all(root.join("kernels/gb10/common")).unwrap();
    assert!(
        sources(&root, &t).is_none(),
        "an empty resolution must be None — every empty set hashes alike"
    );
}

#[test]
fn an_unknown_vendor_resolves_to_none() {
    let root = fixture("vendor");
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"quantum-abacus\"\n",
    )
    .unwrap();
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "nvfp4".into(),
    };
    assert!(sources(&root, &t).is_none());
}

// ---------------------------------------------------------------------------
// Path -> node
// ---------------------------------------------------------------------------

#[test]
fn paths_map_to_the_right_nodes() {
    assert_eq!(hardware_of("kernels/gb10/common/x.cu"), Some("gb10"));
    assert_eq!(hardware_of("kernels/strix/qwen/nvfp4/x.cu"), Some("strix"));
    assert_eq!(hardware_of("crates/atlas-plugin/src/lib.rs"), None);

    assert_eq!(
        model_of("kernels/gb10/qwen3.6-27b/nvfp4/x.cu"),
        Some(("gb10", "qwen3.6-27b"))
    );
    assert_eq!(
        model_of("kernels/gb10/common/x.cu"),
        None,
        "a shared kernel belongs to no single model"
    );
    assert_eq!(
        model_of("kernels/gb10/HARDWARE.toml"),
        None,
        "a hardware-level file is not under a model"
    );
    assert_eq!(
        model_of("kernels/gb10/qwen3.6-27b/MODEL.toml"),
        Some(("gb10", "qwen3.6-27b"))
    );
}

/// A directory whose name merely starts with `kernels` is not the kernel tree.
#[test]
fn lookalike_paths_are_not_kernel_paths() {
    assert_eq!(hardware_of("kernels-old/gb10/common/x.cu"), None);
    assert_eq!(hardware_of("docs/kernels/gb10/x.cu"), None);
}

// ---------------------------------------------------------------------------
// Affected sets
// ---------------------------------------------------------------------------

#[test]
fn a_common_change_affects_every_target_on_that_hardware() {
    let root = fixture("affected-common");
    let hit = affected(&root, &["kernels/gb10/common/shared.cu".to_string()]);
    assert_eq!(hit.len(), 2, "both models inherit common: {hit:?}");
}

#[test]
fn a_model_change_affects_only_that_model() {
    let root = fixture("affected-model");
    let hit = affected(&root, &["kernels/gb10/modelA/nvfp4/shared.cu".to_string()]);
    assert_eq!(hit.len(), 1);
    assert_eq!(hit.iter().next().unwrap().model, "modelA");
}

#[test]
fn a_non_kernel_change_affects_no_target_here() {
    let root = fixture("affected-host");
    assert!(
        affected(
            &root,
            &["crates/atlas-plugin/src/gate/check.rs".to_string()]
        )
        .is_empty(),
        "host code is handled by the path boundary, not the taxonomy"
    );
}

// ---------------------------------------------------------------------------
// Span checks
// ---------------------------------------------------------------------------

#[test]
fn spans_are_reported_per_node_level() {
    let changed = vec![
        "kernels/gb10/qwen3.6-27b/nvfp4/a.cu".to_string(),
        "kernels/strix/qwen3.6-27b/nvfp4/a.cu".to_string(),
    ];
    assert_eq!(
        hardware_span(&changed).len(),
        2,
        "the AMD-port exemption case"
    );
    assert_eq!(
        model_span(&changed).len(),
        2,
        "same model name under two hardwares is two nodes"
    );

    let one_hw = vec![
        "kernels/gb10/modelA/nvfp4/a.cu".to_string(),
        "kernels/gb10/modelB/nvfp4/a.cu".to_string(),
    ];
    assert_eq!(hardware_span(&one_hw).len(), 1);
    assert_eq!(
        model_span(&one_hw).len(),
        2,
        "two models under one hardware — the case a split check can honestly fail"
    );
}

// ---------------------------------------------------------------------------
// Layered targets: `[model] kernel_source` and `[build] base_quant`
// ---------------------------------------------------------------------------

/// `redirect` compiles `source`'s nvfp4 kernels plus its own nvfp4 overlay;
/// `bare` redirects to `source` with no quant dir of its own; `source`'s `exl3`
/// composes on its `nvfp4` sibling.
fn layered_fixture(name: &str) -> PathBuf {
    let root = fixture(name);
    let hw = root.join("kernels/gb10");
    for d in ["source/nvfp4", "source/exl3", "redirect/nvfp4", "bare"] {
        std::fs::create_dir_all(hw.join(d)).unwrap();
    }
    std::fs::write(hw.join("source/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(
        hw.join("redirect/MODEL.toml"),
        "[model]\nkernel_source = \"source\"\n",
    )
    .unwrap();
    std::fs::write(
        hw.join("bare/MODEL.toml"),
        "[model]\nkernel_source = \"source\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("source/nvfp4/KERNEL.toml"), "[build]\n").unwrap();
    std::fs::write(hw.join("source/nvfp4/gemm.cu"), "__global__ void g() {}\n").unwrap();
    std::fs::write(hw.join("source/nvfp4/attn.cu"), "__global__ void a() {}\n").unwrap();
    std::fs::write(
        hw.join("source/exl3/KERNEL.toml"),
        "[build]\nbase_quant = \"nvfp4\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("source/exl3/exl3.cu"), "__global__ void e() {}\n").unwrap();
    std::fs::write(hw.join("redirect/nvfp4/KERNEL.toml"), "[build]\n").unwrap();
    std::fs::write(
        hw.join("redirect/nvfp4/attn.cu"),
        "__global__ void a2() {}\n",
    )
    .unwrap();
    root
}

fn target(model: &str, quant: &str) -> Target {
    Target {
        hardware: "gb10".into(),
        model: model.into(),
        quant: quant.into(),
    }
}

fn names(paths: &[PathBuf], root: &Path) -> Vec<String> {
    let base = root.join("kernels/gb10");
    let mut out: Vec<String> = paths
        .iter()
        .map(|p| {
            p.strip_prefix(&base)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    out.sort();
    out
}

/// A `kernel_source` target compiles the source's quant dir, then its own
/// quant dir over it by stem, as `resolve_targets` layers them.
#[test]
fn a_redirected_target_resolves_the_source_kernels_under_its_own_overlay() {
    let root = layered_fixture("redirect");
    assert_eq!(
        names(
            &sources(&root, &target("redirect", "nvfp4")).unwrap(),
            &root
        ),
        [
            "common/other.cu",
            "common/shared.cu",
            "redirect/nvfp4/attn.cu",
            "source/nvfp4/gemm.cu"
        ]
    );
}

/// `base_quant` composes the sibling quant dir under the quant's own.
#[test]
fn a_base_quant_target_resolves_its_sibling_under_its_own_kernels() {
    let root = layered_fixture("base");
    assert_eq!(
        names(&sources(&root, &target("source", "exl3")).unwrap(), &root),
        [
            "common/other.cu",
            "common/shared.cu",
            "source/exl3/exl3.cu",
            "source/nvfp4/attn.cu",
            "source/nvfp4/gemm.cu"
        ]
    );
}

/// The configs are the target's own MODEL.toml and every layer's KERNEL.toml.
#[test]
fn a_redirected_target_hashes_its_own_model_toml_and_every_layers_kernel_toml() {
    let root = layered_fixture("configs");
    assert_eq!(
        names(&configs(&root, &target("redirect", "nvfp4")), &root),
        [
            "HARDWARE.toml",
            "redirect/MODEL.toml",
            "redirect/nvfp4/KERNEL.toml",
            "source/nvfp4/KERNEL.toml"
        ]
    );
}

/// A redirected target's quants are the source's, so one with no quant dir of
/// its own is still a target, as the build resolves it.
#[test]
fn a_redirected_target_takes_its_quants_from_the_source() {
    let root = layered_fixture("walk");
    let all: Vec<String> = walk(&root).iter().map(|t| t.to_string()).collect();
    for key in [
        "gb10/bare/exl3",
        "gb10/bare/nvfp4",
        "gb10/redirect/exl3",
        "gb10/redirect/nvfp4",
    ] {
        assert!(all.contains(&key.to_string()), "{key} missing from {all:?}");
    }
}

/// A change to the source's kernels affects every target compiling them.
#[test]
fn a_source_change_affects_the_targets_that_redirect_to_it() {
    let root = layered_fixture("affected");
    let hit = affected(&root, &["kernels/gb10/source/nvfp4/gemm.cu".to_string()]);
    for model in ["source", "redirect", "bare"] {
        assert!(
            hit.contains(&target(model, "nvfp4")),
            "{model} must be affected: {hit:?}"
        );
    }
    assert!(!hit.contains(&target("modelA", "nvfp4")));
}

/// An unreadable MODEL.toml fails closed: the target resolves as affected.
#[test]
fn a_malformed_model_toml_resolves_to_none() {
    let root = layered_fixture("malformed");
    std::fs::write(root.join("kernels/gb10/redirect/MODEL.toml"), "[model\n").unwrap();
    assert!(sources(&root, &target("redirect", "nvfp4")).is_none());
}
