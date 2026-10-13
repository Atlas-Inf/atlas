// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE aux blob: what a snapshot reads back is what a restore writes.

use super::{parse_aux, write_aux_head};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

/// A carry and history through the snapshot's readback (`aux_d2h`, as
/// `snapshot_aux_into` does) and the restore's parse and upload: the history
/// and every carry byte come back, on any blob, pinned stage or not.
#[test]
fn a_restored_carry_is_the_saved_one_byte_for_byte() {
    let gpu = MockGpuBackend::new();
    // [(k - 1) * dilation, hc_mult * hidden] FP32 at a toy width; the
    // 9-step x 4-stream shape of qwen4_exp, hidden 32.
    let conv_bytes = 9 * 4 * 32 * 4;
    let carry: Vec<u8> = (0..conv_bytes).map(|i| (i * 7 % 251) as u8).collect();
    let live = gpu.alloc(conv_bytes).unwrap();
    gpu.copy_h2d(&carry, live).unwrap();
    let history: Vec<u32> = (0..6).map(|i| 151_000 + i * 17).collect();

    let mut blob = vec![0xAA; 3]; // a reused buffer: the head clears it
    write_aux_head(&mut blob, &history);
    let off = blob.len();
    blob.resize(off + conv_bytes, 0);
    crate::layers::aux_d2h::copy(&gpu, live, &mut blob[off..], 0).unwrap();
    assert_eq!(blob.len(), 4 + history.len() * 4 + conv_bytes);

    let (got_history, got_conv) = parse_aux(&blob, conv_bytes).unwrap();
    assert_eq!(got_history, history);
    let restored = gpu.alloc(conv_bytes).unwrap();
    gpu.copy_h2d_async(got_conv, restored, 0).unwrap();
    let mut back = vec![0u8; conv_bytes];
    gpu.copy_d2h(restored, &mut back).unwrap();
    assert_eq!(back, carry);

    // A blob for another carry size, or cut short, is refused.
    assert!(parse_aux(&blob, conv_bytes - 4).is_err());
    assert!(parse_aux(&blob[..blob.len() - 1], conv_bytes).is_err());
    assert!(parse_aux(&blob[..3], conv_bytes).is_err());
}
