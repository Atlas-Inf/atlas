// SPDX-License-Identifier: AGPL-3.0-only

// Re-emit atlas-rdma's `cfg(atlas_rdma_verbs)` (published through its `links`
// metadata) for the RDMA pair all-reduce; rustc cfgs never cross crates.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(atlas_rdma_verbs)");
    if std::env::var("DEP_ATLAS_RDMA_SHIM_HAS_VERBS").is_ok() {
        println!("cargo:rustc-cfg=atlas_rdma_verbs");
    }
}
