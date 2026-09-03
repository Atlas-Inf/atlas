// SPDX-License-Identifier: AGPL-3.0-only

//! Content-pinned amnesty for the Avarok → Atlas-Inf migration.
//!
//! # The grant
//!
//! Moving the project from `Avarok-Cybersecurity/atlas` to `Atlas-Inf/atlas`
//! rewrote the org name wherever it appeared — doc comments, plugin metadata
//! strings, recipe test fixtures, one comment line in a `.cu` — and removed
//! LatticeDB, a dependency only the host-side governance ledger ever linked.
//! None of it is executable inference change. `PERF_PATHS` contains the string
//! `crates`, so all of it invalidated every gate anyway.
//!
//! That alone would cost ten fresh GPU legs. It landed together with a second
//! problem that makes the bill unpayable: the migration did not carry over the
//! feature branches the records were measured on, so **every** record's commit
//! is now absent from the remote and `git diff` cannot even be asked the
//! question (see `check.rs::invalidating_paths`). The gate reported `NONE` for
//! all ten benchmarks, which is not dischargeable by re-measuring — a fresh
//! record would be orphaned by the next branch deletion in exactly the same
//! way.
//!
//! The grant below covers only the final reviewed blobs of the paths that
//! migration touched. It is the same mechanism PR #701 used and the 2026-08-16
//! governance bootstrap before it: a table anyone can read, a pin no later
//! edit can inherit, and a test that demands removal after all ten records
//! have been re-earned.
//!
//! # What is NOT in here
//!
//! `crates/atlas-governance` is not amnestied. Its LatticeDB removal is a real
//! code deletion, not a rename, and pinning it as though it were inert would
//! be a lie in the table. It is excluded in `coverage.rs` instead, on the
//! standing `GATE_MACHINERY` rationale — nothing outside `atlas-plugin`'s own
//! gate bookkeeping and two governance CLI bins links it, and no benchmark
//! driver or serving path references it at all.
//!
//! # Why content-pinned rather than waived
//!
//! Each entry pins the exact blob OID this PR lands. `git rev-parse <head>:<path>`
//! names the CONTENT of the file at the commit being checked,
//! so the grant covers precisely the reviewed bytes: the moment anyone edits
//! either file again, the OID changes and invalidation applies exactly as
//! before. There is no time window and no path-level waiver to inherit —
//! a second edit to the taxonomy pays full price, as it should.
//!
//! # Fail-closed
//!
//! Every uncertainty is "not excused": a path not in the table, git missing
//! or failing, an unknown commit, the path absent at `head`, output that is
//! not a 40-hex OID, an OID that is not the pinned one. A false "not excused"
//! costs a re-run; a false "excused" would be a shipped regression behind a
//! green gate — the same asymmetry the rest of the gate is built on.
//!
//! # Residual risk, stated plainly
//!
//! This file cannot itself be in `BOUNDARY_FILES` without circularity: its
//! own landing would then invalidate everything it exists to protect. It is
//! covered only by `GATE_MACHINERY`'s cargo-test rationale, like the rest of
//! the gate bookkeeping. Compensations: the table's exact contents are pinned
//! by `the_table_is_exactly_the_migration_grant` (entry count, paths, OID
//! format), every application is logged loudly by `check.rs`, CODEOWNERS
//! review covers the gate directory, and the gate already executes
//! PR-checkout code — so this adds no new attack class, only a reviewed
//! exception to one rule.
//!
//! This grant is wider than #701's three files, and that is the honest cost of
//! a repo-wide rename: 38 migration paths, each pinned to one blob, every one
//! of which a reviewer can diff against `e0a634261b` and see is a string or a
//! comment — plus this PR's own two boundary-file edits, the same self-pin
//! #701 needed. The pin is what bounds it: the table cannot cover a single
//! byte anyone edits after this lands.
//!
//! # Removal condition
//!
//! EMPTY THE TABLE once every required gate has a record newer than
//! `AMNESTY_EPOCH`. At that point the grant protects nothing because every
//! record postdates the grant day and was earned against the amnestied
//! content. `amnesty_expires_once_every_gate_has_a_fresh_record` fails with
//! instructions when that day arrives, so the table cannot quietly outlive
//! its purpose.

use std::path::Path;

/// One excused path: the file, the exact blob its grant covers, and why.
#[derive(Debug, Clone, Copy)]
pub struct AmnestyEntry {
    pub path: &'static str,
    /// The 40-hex blob OID of the file AS THIS PR LANDS IT — computed with
    /// `git rev-parse <head>:<path>` (equivalently `git hash-object <path>`)
    /// once the content is final, in the pin phase. Until pinned it holds
    /// `"PENDING"`, which matches no blob and keeps the grant inert.
    pub head_blob_oid: &'static str,
    pub grant: &'static str,
}

/// End of the migration grant day: 2026-08-26T00:00:00Z. A record counts as
/// fresh only when it postdates the whole grant day.
///
/// Moved forward from the PR #701 value (2026-08-22T00:00:00Z) because the
/// records that retired *that* grant are the same ones this grant exists to
/// rescue: they were measured on branches the migration dropped. Leaving the
/// epoch behind would let `amnesty_expires_once_every_gate_has_a_fresh_record`
/// read those pre-migration records as proof this grant is spent.
pub const AMNESTY_EPOCH: u64 = 1_787_702_400;

/// The Avarok → Atlas-Inf migration grant.
///
/// Every entry is one path the migration touched inside `PERF_PATHS`, pinned
/// to the blob this PR lands. Diff any of them against `e0a634261b` — the
/// newest commit every committed record was measured at — and the change is a
/// rename in a comment, a URL, a container image name in a test fixture, or
/// the LatticeDB pruning that removed a host-only governance dependency.
///
/// `crates/atlas-governance` is deliberately absent; see the module docs.
///
/// The last two entries are this PR's own `BOUNDARY_FILES` edits. A boundary
/// file invalidates every gate by construction and no exclusion may exempt one
/// — that is the lock this list exists to be — so a change to the gate cannot
/// land without amnestying itself. PR #701 hit the identical wall and pinned
/// the same two files; see ADR-0015. There is no circularity: the table lives
/// in `amnesty.rs`, which is not a boundary file, so pinning `check.rs` and
/// `coverage.rs` does not perturb the blobs being pinned.
pub const ONE_TIME_AMNESTY: [AmnestyEntry; 40] = [
    AmnestyEntry {
        path: "crates/atlas-plugin/src/gate/check.rs",
        head_blob_oid: "3059ad9a0c64ba585e2b80e7a9bed6c40c782bc0",
        grant: "this PR's own boundary-file edit: commit_is_present, so a failed diff names the right fault",
    },
    AmnestyEntry {
        path: "crates/atlas-plugin/src/gate/coverage.rs",
        head_blob_oid: "6c5dc69bac336a7ee75dc0999126e14f1961198d",
        grant: "this PR's own boundary-file edit: the GOVERNANCE_LEDGER exclusion",
    },
    AmnestyEntry {
        path: "Cargo.lock",
        head_blob_oid: "b24675fe545e36daf569658fa3c7d73723a08df4",
        grant: "the LatticeDB removal pruned a host-only governance dependency; no inference crate ever linked it",
    },
    AmnestyEntry {
        path: "Cargo.toml",
        head_blob_oid: "17171ba4d0b4c9791d4ecffff02eedd1bee80e2e",
        grant: "the LatticeDB removal pruned a host-only governance dependency; no inference crate ever linked it",
    },
    AmnestyEntry {
        path: "crates/atlas-plugin/src/metadata.rs",
        head_blob_oid: "6ed89f88b1ac91ba0a22d2024aa80c90bdf11244",
        grant: "plugin metadata strings - author, URLs, contact - read by no model code",
    },
    AmnestyEntry {
        path: "crates/spark-model/src/layers/dflash_head/from_weights.rs",
        head_blob_oid: "185ca2b17c7ed538ef3f13668361ae9adac1dfef",
        grant: "Avarok to Atlas rename in comments and user-facing strings; no executable change",
    },
    AmnestyEntry {
        path: "crates/spark-model/src/layers/moe/mod.rs",
        head_blob_oid: "ccc89eac025e755262008dc89394ce199fc81702",
        grant: "Avarok to Atlas rename in comments and user-facing strings; no executable change",
    },
    AmnestyEntry {
        path: "crates/spark-model/src/layers/ops/moe_grouped_a.rs",
        head_blob_oid: "4f728ee45ac9a5f6284e885d005665d253600404",
        grant: "Avarok to Atlas rename in comments and user-facing strings; no executable change",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/metrics.rs",
        head_blob_oid: "a0769ca9ac91cc141ba06086ae9ed2dfb1753095",
        grant: "Avarok to Atlas rename in comments and user-facing strings; no executable change",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/recipe/fetch.rs",
        head_blob_oid: "9d103899ad629a4a95ee2387a8346c12b52ce28b",
        grant: "Avarok to Atlas rename in comments and user-facing strings; no executable change",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/recipe/fetch_tests.rs",
        head_blob_oid: "0b2d3b00ce0a2f1d5d6a5ece2f842bafb9c9b583",
        grant: "test-side rebrand string, absent from release builds",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/recipe/mod.rs",
        head_blob_oid: "bafb0d93669bd106d7068cb56c9dbc3ec749aa00",
        grant: "Avarok to Atlas rename in comments and user-facing strings; no executable change",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/recipe/yaml_tests.rs",
        head_blob_oid: "a127c218768b617f4a98b1317f976c807902df01",
        grant: "test-side rebrand string, absent from release builds",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/tui/data/catalogue_tests.rs",
        head_blob_oid: "6ad36c98ae1cdc459763a76ea608c2bde2c798c4",
        grant: "test-side rebrand string, absent from release builds",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/tui/render/render_tests.rs",
        head_blob_oid: "5a6ff415d00e8aeced1980c112a068dbe081a15f",
        grant: "test-side rebrand string, absent from release builds",
    },
    AmnestyEntry {
        path: "crates/spark-server/src/tui/report.rs",
        head_blob_oid: "2862ab0ec4ed82e407f8b05a3800130d855c7f2a",
        grant: "Avarok to Atlas rename in comments and user-facing strings; no executable change",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/deepseek-v4/deepseek-v4-flash-nvfp4-ep2.yaml",
        head_blob_oid: "93da466d0408f0e2e93e7395f25945ea28b51bce",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/gemma4/gemma-4-26b-a4b-nvfp4.yaml",
        head_blob_oid: "9d2ed7eaa11e3897332373e1759670684ae2c913",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/gemma4/gemma-4-31b-nvfp4.yaml",
        head_blob_oid: "a2dd054079a2fe18ba103dbada653ad885aebbef",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/minimax-m2.7/minimax-m2.7-nvfp4-ep2.yaml",
        head_blob_oid: "ec3f47a3fbacf4d8bcd0dd333692a3f57762c28a",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/mistral-small-4/mistral-small-4-119b-nvfp4.yaml",
        head_blob_oid: "9d0d779383fd7607e35df9664a9719b915afe014",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/nemotron-3-nano/nemotron-3-nano-30b-a3b-nvfp4.yaml",
        head_blob_oid: "236e71e8875397994c00295f358717b9676f7fc4",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/nemotron-3-super/nemotron-3-super-120b-a12b-nvfp4.yaml",
        head_blob_oid: "295530239cbd389a6ac100f083dfe403d9fd6398",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3-coder-next/qwen3-coder-next-fp8.yaml",
        head_blob_oid: "a2f72cf6d64f433ef9574e46fc833cfe45c6c962",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3-next/qwen3-next-80b-a3b-nvfp4.yaml",
        head_blob_oid: "cfef5f8fc4fff3e2e1ac54b4e7f684475141cf27",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3-vl/qwen3-vl-30b-a3b-nvfp4.yaml",
        head_blob_oid: "0bfb4feb23d5e7c4ea3754adb713ad9d37d6436b",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.5/qwen3.5-0.8b-bf16-atlas.yaml",
        head_blob_oid: "9e6ceb5e96b115936b2f16327c7a2f63dd898758",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.5/qwen3.5-122b-a10b-nvfp4-ep2.yaml",
        head_blob_oid: "78d3f996a10b40baae4671fe05bf6f13f62dd8db",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.5/qwen3.5-122b-a10b-nvfp4-single.yaml",
        head_blob_oid: "a31ae287c821c5946908720d48b7178931b5fb98",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.5/qwen3.5-27b-dense-nvfp4.yaml",
        head_blob_oid: "a92c45faf94703280574df79e9dbbc125e6bb963",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.5/qwen3.5-35b-a3b-nvfp4.yaml",
        head_blob_oid: "315c6f8ece74201d3ac00bd0f3bed55205b82e06",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-27b-fp8-mtp.yaml",
        head_blob_oid: "97f811ac89dc4a400bfd915f7251c65f76212580",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-27b-fp8.yaml",
        head_blob_oid: "207dc1be09a73d0c50116c81ed6257da911b3af8",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-27b-nvfp4-prefill-record.yaml",
        head_blob_oid: "bfd52947f245d971bb22c94be716b31b04b455bd",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-27b-nvfp4.yaml",
        head_blob_oid: "c842e8d9a26dfe3c45b251480be092079b2bb90b",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-bf16head.yaml",
        head_blob_oid: "4291442fd790729255fcac0bb921f0ba37446c34",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml",
        head_blob_oid: "55ba5b855579b4f230701ecb67e742ae8d2f9557",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head.yaml",
        head_blob_oid: "9862947f14ca4a7c5ea23dff8862ff47f2e32533",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "crates/spark-server/tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-nvfp4.yaml",
        head_blob_oid: "5b809147c3a6261d74d5b83f97c764a90a268bb1",
        grant: "recipe test fixture: container image name and maintainer strings, never read at serve time",
    },
    AmnestyEntry {
        path: "kernels/gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu",
        head_blob_oid: "9301c974a8b7ef12b0b1c7a6453b4133a6e1876c",
        grant: "a comment line in device source; the compiled kernel is byte-identical",
    },
];

/// Whether the one-time grant excuses `path` at `head`.
pub fn excused(root: &Path, head: &str, path: &str) -> bool {
    excused_by(root, head, path, &ONE_TIME_AMNESTY)
}

/// [`excused`] against an explicit table, so tests can pin real OIDs.
///
/// True iff `path` is in `table` AND the blob at `<head>:<path>` is exactly
/// the pinned 40-hex OID. Anything else — including any git failure — is
/// `false`.
pub fn excused_by(root: &Path, head: &str, path: &str, table: &[AmnestyEntry]) -> bool {
    let Some(entry) = table.iter().find(|e| e.path == path) else {
        return false;
    };
    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", &format!("{head}:{path}")])
        .stdin(std::process::Stdio::null())
        .output()
    else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let oid = String::from_utf8_lossy(&out.stdout).trim().to_string();
    oid.len() == 40 && oid.chars().all(|c| c.is_ascii_hexdigit()) && oid == entry.head_blob_oid
}
