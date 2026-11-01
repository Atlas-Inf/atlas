// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_WARM_CTX` on a real device (ignored): the warm-turn oracle
//! of `warm_ctx_tests` on the CUDA backend, and a store round trip at the
//! GLM-5.3-Flash + DFlash2 shape (5 capture layers x 4096 x BF16 rows,
//! 2048-row windows). Copies only, no kernels:
//!
//!   cargo test --release -p spark-model --lib warm_ctx_gpu -- --ignored --nocapture
//!
//! The end-to-end check against a served model (a warm turn's first drafter
//! window vs a cold prefill of the same prompt) is
//! `scripts/dflash_warm_ctx_check.py`.

use anyhow::{Context, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

use super::WarmCtx;
use super::warm_ctx_store::{StoreRefs, WarmCtxStore, store_key};
use super::warm_ctx_tests::{Turn, W, prefill, store, tokens, window};

fn cuda() -> Result<AtlasCudaBackend> {
    AtlasCudaBackend::new(0, &[]).context("CUDA backend")
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_warm_turn_window_matches_its_cold_prefill_on_the_device() -> Result<()> {
    let gpu = cuda()?;
    for (first, second) in [(40, 44), (40, 41), (70, 75), (90, 120), (9, 13)] {
        let toks = tokens(second, 2);
        let store = store(&gpu, 4);
        let cold = window(
            &gpu,
            &mut prefill(&gpu, &store, WarmCtx::Exact, &toks, Turn::cold(second)),
        );
        let turn1 = Turn::cold(first);
        prefill(&gpu, &store, WarmCtx::Restore, &toks, turn1);
        let turn2 = Turn {
            skip: turn1.cut(),
            ..Turn::cold(second)
        };
        let warm = window(
            &gpu,
            &mut prefill(&gpu, &store, WarmCtx::Restore, &toks, turn2),
        );
        anyhow::ensure!(warm == cold, "{first}->{second}: {warm:?} != {cold:?}");
        anyhow::ensure!(warm.len() == W.min(second));
        store.free(&gpu);
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_store_round_trips_glm_windows_on_the_device() -> Result<()> {
    const ROW: usize = 5 * 4096 * 2;
    const WINDOW: usize = 2048;
    let gpu = cuda()?;
    let gpu: &dyn GpuBackend = &gpu;
    let stream = gpu.default_stream();
    let rows: Vec<u8> = (0..WINDOW * ROW).map(|i| (i * 131 % 251) as u8).collect();
    let acc = gpu.alloc(WINDOW * ROW)?;
    let out = gpu.alloc(WINDOW * ROW)?;
    gpu.copy_h2d(&rows, acc)?;
    let store = WarmCtxStore::reserve(gpu, 2, WINDOW, ROW)?;
    // A tail checkpoint 24 rows under a 45,000-token prompt end: the window
    // [42952, 45000) holds real rows [42952, 44976).
    let (ws, cut) = (42_952usize, 44_976usize);
    let key = store_key(0, &vec![7u32; cut]);
    let src = acc.offset(0);
    let saved = store.save(gpu, key, StoreRefs::default(), src, ws, cut, ROW, stream)?;
    anyhow::ensure!(saved.is_some());
    // The next turn (45,600 tokens) wants [43552, 44976).
    let want = 43_552usize;
    let lo = store
        .restore(gpu, key, cut, want, ROW, out, stream)?
        .map(|r| r.0);
    anyhow::ensure!(lo == Some(want), "{lo:?}");
    gpu.synchronize(stream)?;
    let mut got = vec![0u8; (cut - want) * ROW];
    gpu.copy_d2h(out, &mut got)?;
    let from = (want - ws) * ROW;
    anyhow::ensure!(got == rows[from..from + got.len()], "restored rows differ");
    anyhow::ensure!(
        store
            .restore(gpu, key, cut + 16, want, ROW, out, stream)?
            .is_none()
    );
    store.free(gpu);
    gpu.free(acc)?;
    gpu.free(out)?;
    Ok(())
}
