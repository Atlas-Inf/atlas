// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness-first paged Hy4 MLA/IndexPool. One stream per request cache;
//! rows are enqueued sequentially so circular SWA writes cannot evict keys
//! still needed by an earlier prefill row.

mod cache;
mod forward;
mod index;
mod kernels;
mod loader;
pub(crate) mod memory;
#[cfg(test)]
mod tests;

pub use cache::AttentionCache;
pub use loader::Hy4Attention;
