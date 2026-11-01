// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::cell::Cell;

#[test]
fn real_projection_oracle_repoisons_and_checks_every_byte() {
    for fault in [0, 1, 2, 3] {
        let gpu = MockGpuBackend::new();
        let output = gpu.alloc(16).unwrap();
        let calls = Cell::new(0);
        let result = verify_output(
            &gpu,
            0,
            output,
            16,
            false,
            false,
            KernelHandle(1),
            KernelHandle(2),
            |k| {
                calls.set(calls.get() + 1);
                let mut before = [0; 16];
                gpu.copy_d2h(output, &mut before)?;
                assert_eq!(before, [0xff; 16], "fresh NaN poison before each launch");
                if !(fault == 1 && k.0 == 2) {
                    let mut value = [0x3fu8; 16];
                    if fault == 2 && k.0 == 2 {
                        value[15] = 0x40;
                    }
                    if fault == 3 && k.0 == 1 {
                        value.fill(0xff);
                    }
                    gpu.copy_h2d(&value, output)?;
                }
                Ok(())
            },
        );
        assert_eq!(result.is_ok(), fault == 0);
        assert_eq!(calls.get(), if fault == 3 { 1 } else { 2 });
    }
}

#[test]
fn oracle_refuses_capture_and_overlap_before_writes_or_callbacks() {
    for (capture, overlap) in [(true, false), (false, true)] {
        let gpu = MockGpuBackend::new();
        let output = gpu.alloc(16).unwrap();
        gpu.copy_h2d(&[7; 16], output).unwrap();
        let calls = Cell::new(0);
        assert!(
            verify_output(
                &gpu,
                0,
                output,
                16,
                capture,
                overlap,
                KernelHandle(1),
                KernelHandle(2),
                |_| {
                    calls.set(calls.get() + 1);
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(calls.get(), 0);
        let mut actual = [0; 16];
        gpu.copy_d2h(output, &mut actual).unwrap();
        assert_eq!(actual, [7; 16]);
    }
}

#[test]
fn oracle_does_not_launch_candidate_after_reference_error() {
    let gpu = MockGpuBackend::new();
    let output = gpu.alloc(16).unwrap();
    let calls = Cell::new(0);
    assert!(
        verify_output(
            &gpu,
            0,
            output,
            16,
            false,
            false,
            KernelHandle(1),
            KernelHandle(2),
            |_| {
                calls.set(calls.get() + 1);
                anyhow::bail!("reference failure")
            }
        )
        .is_err()
    );
    assert_eq!(calls.get(), 1);
}
