// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layer::ssm_batch::{SsmBatchView, SsmPoolView};
use spark_runtime::gpu::mock::MockGpuBackend;

fn layer(rows: usize, h_bytes: usize, conv_bytes: usize) -> SsmBatchLayer {
    let h = [DevicePtr(0x10000000)];
    let conv = [DevicePtr(0x20000000)];
    let pool = SsmPoolView::new(&h, &conv, h_bytes, h_bytes, conv_bytes, 4).unwrap();
    SsmBatchView::new(pool, DevicePtr(0x1000), &[3, 0, 2, 1][..rows])
        .unwrap()
        .layer(0)
        .unwrap()
}
fn buffer(address: u64, bytes: usize) -> KdaBuffer {
    KdaBuffer {
        ptr: DevicePtr(address),
        bytes,
    }
}
fn conv() -> KdaIndexedConv {
    KdaIndexedConv {
        input: buffer(0x30000000, 4 * CHANNELS * 2),
        weight: buffer(0x31000000, CHANNELS * 4 * 2),
        bias: None,
        output: buffer(0x32000000, 4 * CHANNELS * 2),
        input_stride: CHANNELS,
        output_stride: CHANNELS,
    }
}
fn recurrent() -> KdaIndexedRecurrent {
    KdaIndexedRecurrent {
        qkv: conv().output,
        gate: buffer(0x33000000, 4 * PLANE * 2),
        beta: buffer(0x34000000, 4 * 32 * 2),
        a_log: buffer(0x35000000, 32 * 4),
        dt_bias: buffer(0x36000000, PLANE * 4),
        output: buffer(0x37000000, 4 * PLANE * 2),
        lower_bound: -5.0,
    }
}

#[test]
fn typed_arguments_preserve_exact_cuda_abi_and_float_element_strides() {
    let s = KdaIndexedShape::select(4, 32, 128, 4).unwrap();
    let l = layer(4, PLANE * 128 * 4, CHANNELS * 4 * 4);
    assert_eq!(
        conv_args(s, l, conv()).unwrap(),
        [
            Arg::Ptr(l.conv_base()),
            Arg::Ptr(conv().input.ptr),
            Arg::Ptr(conv().weight.ptr),
            Arg::Ptr(DevicePtr::NULL),
            Arg::Ptr(conv().output.ptr),
            Arg::U32(12288),
            Arg::U32(4),
            Arg::U32(4),
            Arg::U32(12288),
            Arg::U32(12288),
            Arg::Ptr(l.slots()),
            Arg::U32(4),
            Arg::U64(49152)
        ]
    );
    let r = recurrent();
    assert_eq!(
        recurrent_args(s, l, r).unwrap(),
        [
            Arg::Ptr(r.qkv.ptr),
            Arg::Ptr(r.gate.ptr),
            Arg::Ptr(r.beta.ptr),
            Arg::Ptr(r.a_log.ptr),
            Arg::Ptr(r.dt_bias.ptr),
            Arg::Ptr(l.h_base()),
            Arg::Ptr(r.output.ptr),
            Arg::U32(4),
            Arg::U32(32),
            Arg::U32(128),
            Arg::F32(-5.0),
            Arg::Ptr(l.slots()),
            Arg::U32(4),
            Arg::U64(524288)
        ]
    );
    let mut c = conv();
    c.bias = Some(buffer(0x38000000, CHANNELS * 4));
    assert_eq!(
        conv_args(s, l, c).unwrap()[3],
        Arg::Ptr(DevicePtr(0x38000000))
    );
    let gpu = MockGpuBackend::new();
    kda_conv_indexed(&gpu, KernelHandle(1), s, l, c, 7).unwrap();
    kda_recurrent_indexed(&gpu, KernelHandle(2), s, l, r, 7).unwrap();
    assert_eq!(gpu.launch_count(), 2);
}

#[test]
fn malformed_present_metadata_and_every_required_span_fail_before_launch() {
    let s = KdaIndexedShape::select(4, 32, 128, 4).unwrap();
    let l = layer(4, PLANE * 128 * 4, CHANNELS * 4 * 4);
    let gpu = MockGpuBackend::new();
    assert!(kda_conv_indexed(&gpu, KernelHandle(0), s, l, conv(), 0).is_err());
    assert!(kda_recurrent_indexed(&gpu, KernelHandle(0), s, l, recurrent(), 0).is_err());
    let short = layer(4, 16, 16);
    assert!(conv_args(s, short, conv()).is_err());
    assert!(recurrent_args(s, short, recurrent()).is_err());
    assert!(conv_args(s, layer(3, PLANE * 128 * 4, CHANNELS * 4 * 4), conv()).is_err());
    for index in 0..4 {
        for null in [false, true] {
            let mut c = conv();
            c.bias = Some(buffer(0x38000000, CHANNELS * 4));
            let span = match index {
                0 => &mut c.input,
                1 => &mut c.weight,
                2 => c.bias.as_mut().unwrap(),
                _ => &mut c.output,
            };
            if null {
                span.ptr = DevicePtr::NULL;
            } else {
                span.bytes -= 1;
            }
            assert!(kda_conv_indexed(&gpu, KernelHandle(1), s, l, c, 0).is_err());
        }
    }
    for index in 0..6 {
        for null in [false, true] {
            let mut r = recurrent();
            let span = match index {
                0 => &mut r.qkv,
                1 => &mut r.gate,
                2 => &mut r.beta,
                3 => &mut r.a_log,
                4 => &mut r.dt_bias,
                _ => &mut r.output,
            };
            if null {
                span.ptr = DevicePtr::NULL;
            } else {
                span.bytes -= 1;
            }
            assert!(kda_recurrent_indexed(&gpu, KernelHandle(1), s, l, r, 0).is_err());
        }
    }
    for stride in [0, CHANNELS - 1, usize::MAX] {
        let mut c = conv();
        c.input_stride = stride;
        assert!(conv_args(s, l, c).is_err());
        c = conv();
        c.output_stride = stride;
        assert!(conv_args(s, l, c).is_err());
    }
    for lower_bound in [f32::NAN, f32::INFINITY, 0.0, 1.0] {
        assert!(
            recurrent_args(
                s,
                l,
                KdaIndexedRecurrent {
                    lower_bound,
                    ..recurrent()
                }
            )
            .is_err()
        );
    }
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn both_core_operations_are_validated_before_any_state_update() {
    let s = KdaIndexedShape::select(4, 32, 128, 4).unwrap();
    let l = layer(4, PLANE * 128 * 4, CHANNELS * 4 * 4);
    assert!(
        validate_kda_indexed_pair(s, l, KernelHandle(1), KernelHandle(2), conv(), recurrent())
            .is_ok()
    );
    for (conv_kernel, recurrent_kernel) in [(0, 2), (1, 0)] {
        assert!(
            validate_kda_indexed_pair(
                s,
                l,
                KernelHandle(conv_kernel),
                KernelHandle(recurrent_kernel),
                conv(),
                recurrent()
            )
            .is_err()
        );
    }
    let mut bad = recurrent();
    bad.output.bytes -= 1;
    assert!(
        validate_kda_indexed_pair(s, l, KernelHandle(1), KernelHandle(2), conv(), bad).is_err()
    );
    let mut unrelated = recurrent();
    unrelated.qkv.ptr = DevicePtr(0x39000000);
    assert!(
        validate_kda_indexed_pair(s, l, KernelHandle(1), KernelHandle(2), conv(), unrelated)
            .is_err()
    );
    let mut padded = conv();
    padded.output_stride += 16;
    padded.output.bytes += 4 * 16 * 2;
    assert!(conv_args(s, l, padded).is_ok());
    assert!(
        validate_kda_indexed_pair(s, l, KernelHandle(1), KernelHandle(2), padded, recurrent())
            .is_err()
    );
}

#[test]
fn exact_geometry_selects_only_independent_supported_widths() {
    for rows in 0..=9 {
        assert_eq!(
            KdaIndexedShape::select(rows, 32, 128, 4).is_some(),
            (2..=8).contains(&rows)
        );
    }
    for (heads, dim, width) in [
        (0, 128, 4),
        (64, 128, 4),
        (32, 64, 4),
        (32, 0, 4),
        (32, 128, 3),
        (usize::MAX, usize::MAX, usize::MAX),
    ] {
        assert!(KdaIndexedShape::select(4, heads, dim, width).is_none());
    }
    for rows in 2..=4 {
        let shape = KdaIndexedShape::select(rows, 32, 128, 4).unwrap();
        assert_eq!(shape.grid(false), [48, rows as u32, 1]);
        assert_eq!(shape.grid(true), [32, rows as u32, 1]);
        for recurrent in [false, true] {
            let grid = shape.grid(recurrent);
            let block = shape.block(recurrent);
            assert!(shape.validate_launch(grid, block, 0, recurrent).is_ok());
            for axis in 0..3 {
                let mut bad = grid;
                bad[axis] += 1;
                assert!(shape.validate_launch(bad, block, 0, recurrent).is_err());
                let mut bad = block;
                bad[axis] += 1;
                assert!(shape.validate_launch(grid, bad, 0, recurrent).is_err());
            }
            assert!(shape.validate_launch(grid, block, 4, recurrent).is_err());
        }
    }
}

#[test]
fn spans_reject_null_alignment_capacity_and_address_overflow() {
    let good = KdaBuffer {
        ptr: DevicePtr(0x1000),
        bytes: 128,
    };
    assert!(good.validate(128, 4).is_ok());
    assert!(good.validate(129, 4).is_err());
    for bad in [
        KdaBuffer {
            ptr: DevicePtr::NULL,
            ..good
        },
        KdaBuffer {
            ptr: DevicePtr(0x1001),
            ..good
        },
        KdaBuffer {
            ptr: DevicePtr(u64::MAX - 3),
            ..good
        },
        KdaBuffer {
            bytes: usize::MAX,
            ..good
        },
    ] {
        assert!(bad.validate(4, 4).is_err());
    }
    assert_eq!(row_bytes(4, 12288, 12288).unwrap(), 4 * 12288 * 2);
    assert_eq!(row_bytes(4, 12304, 12288).unwrap(), (3 * 12304 + 12288) * 2);
    assert!(row_bytes(4, 12287, 12288).is_err());
    assert!(row_bytes(4, usize::MAX, 12288).is_err());
}

#[test]
fn wrappers_submit_exact_rows_without_padding_or_hidden_allocations() {
    let gpu = MockGpuBackend::new();
    for rows in 2..=4 {
        let s = KdaIndexedShape::select(rows, 32, 128, 4).unwrap();
        let l = layer(rows, PLANE * 128 * 4, CHANNELS * 4 * 4);
        validate_kda_indexed_pair(s, l, KernelHandle(1), KernelHandle(2), conv(), recurrent())
            .unwrap();
        kda_conv_indexed(&gpu, KernelHandle(1), s, l, conv(), 0).unwrap();
        kda_recurrent_indexed(&gpu, KernelHandle(2), s, l, recurrent(), 0).unwrap();
    }
    let launches = gpu.launches_snapshot();
    for (pair, rows) in launches.chunks_exact(2).zip(2..=4) {
        assert_eq!((pair[0].grid, pair[0].block), ([48, rows, 1], [256, 1, 1]));
        assert_eq!((pair[1].grid, pair[1].block), ([32, rows, 1], [128, 1, 1]));
    }
    assert_eq!(gpu.launch_count(), 6);
    assert_eq!(gpu.alloc_count(), 0);
}
