// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): a verify's refused rows, taken back, leave the
//! window ring as it was before them once the layer runs again.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::state -- --ignored --nocapture --test-threads=1

use anyhow::Result;
use spark_runtime::kernel_args::KernelLaunch;

use super::*;
use crate::layers::deepseek_v41::ops::Kernels;
use crate::layers::deepseek_v41::test_support::{
    Rng, activations, backend, download, le_bytes, upload,
};

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn refused_rows_keys_are_back_in_the_ring_when_the_layer_next_runs() -> Result<()> {
    let gpu = backend()?;
    let (stream, k) = (gpu.default_stream(), Kernels::load(gpu)?);
    let mut rng = Rng(0x051a_7e55);
    // A batch across the ring's wrap, and one that is all refused but its
    // first row.
    for (position, rows, accepted) in [(3 * WINDOW - 2, 6usize, 2usize), (40, KEPT_ROWS, 1)] {
        let ring = activations(&mut rng, WINDOW * HEAD_DIM);
        let batch = activations(&mut rng, rows * HEAD_DIM);
        let mut state = SeqState::alloc(gpu, 0, 16)?;
        gpu.copy_h2d(&le_bytes(&ring, u16::to_le_bytes), state.window)?;
        // The batch's own keys lie behind the unrolled window.
        let run = gpu.alloc((WINDOW + rows) * HEAD_DIM * 2)?;
        let own = upload(gpu, &le_bytes(&batch, u16::to_le_bytes))?;
        gpu.copy_d2d(own, run.offset(WINDOW * HEAD_DIM * 2), rows * HEAD_DIM * 2)?;
        KernelLaunch::new(gpu, k.window_unroll)
            .grid([(WINDOW * HEAD_DIM / 8).div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.window)
            .arg_ptr(run)
            .arg_ptr(state.kept_window)
            .arg_u32(position as u32)
            .arg_u32(rows as u32)
            .arg_u32(rows as u32)
            .launch(stream)?;
        state.kept = Some((position, rows));
        state.roll_back(gpu, 0, accepted, rows, stream)?;
        assert_eq!(state.taken_back, Some((position, accepted, rows)));
        state.restore_window(gpu, k.window_restore, stream)?;
        // Once: a layer that runs again has nothing more to give back.
        assert_eq!(state.taken_back, None);
        state.restore_window(gpu, k.window_restore, stream)?;
        gpu.synchronize(stream)?;
        let mut want = ring.clone();
        for row in 0..accepted {
            let slot = (position + row) % WINDOW;
            want[slot * HEAD_DIM..][..HEAD_DIM]
                .copy_from_slice(&batch[row * HEAD_DIM..][..HEAD_DIM]);
        }
        assert_eq!(download(gpu, state.window, WINDOW * HEAD_DIM)?, want);
        for p in [run, own] {
            gpu.free(p)?;
        }
        state.free(gpu)?;
    }
    Ok(())
}
