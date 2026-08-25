// SPDX-License-Identifier: AGPL-3.0-only

//! The load-bearing precondition behind `coverage::GOVERNANCE_LEDGER`.
//!
//! Split from `coverage_map_tests.rs` for the 500-LoC cap.
//!
//! Excluding `crates/atlas-governance` from every gate is safe for exactly one
//! reason: nothing that runs a model links it. That is a claim about the
//! DEPENDENCY GRAPH, not about the files, and it is the kind of claim that
//! silently stops being true — someone adds a ledger write to the server, and
//! an exclusion written today starts exempting code that now runs in the
//! serving path.
//!
//! So it is re-derived here from the manifests and the sources on every
//! `cargo test`, rather than trusted. If it ever fails, the exclusion must go,
//! not the test.

use super::coverage::REQUIRED;

/// Crates that can execute during a benchmark: the server, the model, the
/// shared core. If `atlas-governance` becomes reachable from any of them, the
/// exclusion is a lie.
const INFERENCE_CRATES: &[&str] = &["spark-server", "spark-model", "atlas-core"];

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate")
        .to_path_buf()
}

/// No inference crate may declare `atlas-governance` as a dependency.
#[test]
fn governance_is_not_a_dependency_of_any_inference_crate() {
    let root = repo_root();
    for krate in INFERENCE_CRATES {
        let manifest = root.join("crates").join(krate).join("Cargo.toml");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("{} is unreadable: {e}", manifest.display()));
        assert!(
            !text.contains("atlas-governance"),
            "{krate} now depends on atlas-governance — coverage.rs's GOVERNANCE_LEDGER \
             exclusion is no longer sound and must be removed"
        );
    }
}

/// Belt and braces: no inference source may name the crate either, which
/// catches a path dependency added under an alias.
#[test]
fn no_inference_source_references_the_governance_crate() {
    let root = repo_root();
    for krate in INFERENCE_CRATES {
        let src = root.join("crates").join(krate).join("src");
        let mut offenders: Vec<String> = Vec::new();
        visit(&src, &mut |path, body| {
            if body.contains("atlas_governance") {
                offenders.push(path.display().to_string());
            }
        });
        assert!(
            offenders.is_empty(),
            "{krate} references atlas_governance in {offenders:?} — GOVERNANCE_LEDGER \
             must be removed from coverage.rs"
        );
    }
}

/// The exclusion has to actually be installed on every required gate, or the
/// migration grant's reasoning does not hold for the gates that lack it.
#[test]
fn every_required_gate_excludes_the_governance_ledger() {
    for gate in REQUIRED.iter() {
        assert!(
            gate.excludes
                .iter()
                .any(|ex| ex.prefix == "crates/atlas-governance"),
            "{} does not exclude crates/atlas-governance",
            gate.id
        );
    }
}

/// Walk `dir` recursively, handing every `.rs` file's path and body to `f`.
fn visit(dir: &std::path::Path, f: &mut impl FnMut(&std::path::Path, &str)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit(&path, f);
        } else if path.extension().is_some_and(|e| e == "rs")
            && let Ok(body) = std::fs::read_to_string(&path)
        {
            f(&path, &body);
        }
    }
}
