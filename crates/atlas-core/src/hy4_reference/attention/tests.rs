// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::hy4_reference::{linear, rms};

fn rope(theta: f64) -> Hy4Rope {
    Hy4Rope {
        rope_type: "default".into(),
        rope_theta: theta,
        mrope_interleaved: true,
        mrope_section: [12, 10, 10],
    }
}
#[test]
fn window_is_causal_inclusive_and_exact_at_512_boundary() {
    assert_eq!(window_indices(0, 512).unwrap(), [0]);
    assert_eq!(
        window_indices(511, 512).unwrap(),
        (0..512).collect::<Vec<_>>()
    );
    assert_eq!(
        window_indices(512, 512).unwrap(),
        (1..513).collect::<Vec<_>>()
    );
    assert!(window_indices(0, 0).is_err());
}
#[test]
fn sink_changes_denominator_only_including_empty_and_extreme_cases() {
    assert_eq!(sink_softmax(&[], 1000.).unwrap(), Vec::<f32>::new());
    assert_eq!(sink_softmax(&[0., 0.], 0.).unwrap(), [1. / 3., 1. / 3.]);
    assert_eq!(sink_softmax(&[1000.], 1000.).unwrap(), [0.5]);
    assert_eq!(sink_softmax(&[-1000.], 1000.).unwrap(), [0.]);
    assert!(sink_softmax(&[f32::NAN], 0.).is_err());
    assert!(sink_softmax(&[], f32::INFINITY).is_err());
}
#[test]
fn sparse_mask_padding_future_window_and_sink_are_composed_before_gate() {
    let q = [1.];
    let k = [0., 0., 0., 0.];
    let v = [2., 4., 8., 16.];
    let a = attend(&q, &k, &v, 1, &[-1, 0, 1, 2, 3], 2, Some(2), 0., 1.).unwrap();
    assert_eq!(a, [4.]);
    assert_eq!(gate(&a, &[0.]).unwrap(), [2.]);
    let projected = linear(&gate(&a, &[0.]).unwrap(), &[2., -1.], 2).unwrap();
    let norm = rms(&projected, &[1., 2.], 1e-5).unwrap();
    assert!(norm[0] > 0. && norm[1] < 0.);
    assert_eq!(
        attend(&q, &[], &[], 1, &[-1], 0, None, 0., 1.).unwrap(),
        [0.]
    );
    assert_eq!(gate(&[], &[]).unwrap(), Vec::<f32>::new());
    assert!(attend(&q, &k, &v, 1, &[4], 2, None, 0., 1.).is_err());
    assert!(attend(&q, &k, &v, 1, &[0, 0], 2, None, 0., 1.).is_err());
    assert!(attend(&q, &k, &v, 1, &[-2], 2, None, 0., 1.).is_err());
    assert!(attend(&[], &[], &[], 1, &[], 0, None, 0., 1.).is_err());
    assert!(gate(&a, &[]).is_err());
}
#[test]
fn equal_axis_mrope_equals_text_and_indexer_has_split_half_storage() {
    let x: Vec<_> = (0..128).map(|v| bf16(v as f32 / 128.)).collect();
    let main = rotate(&x, [7; 3], &rope(1e7), false).unwrap();
    let index = rotate(&x, [7; 3], &rope(1e7), true).unwrap();
    assert_eq!(&main[..64], &x[..64]);
    for i in 0..32 {
        assert_eq!(main[64 + 2 * i], index[64 + i]);
        assert_eq!(main[64 + 2 * i + 1], index[96 + i]);
    }
    assert_ne!(main, rotate(&x, [7; 3], &rope(1e4), false).unwrap());
    let mut equal = rope(1e7);
    equal.mrope_interleaved = false;
    assert_eq!(main, rotate(&x, [7; 3], &equal, false).unwrap());
    assert_ne!(main, rotate(&x, [7, 8, 9], &rope(1e7), false).unwrap());
    assert!(rotate(&x, [-1; 3], &equal, false).is_err());
    equal.mrope_section = [0; 3];
    assert!(rotate(&x, [0; 3], &equal, false).is_err());
}
#[test]
fn tp2_partitions_both_attention_variants_without_deepseek_tp_rejection() {
    for global in [32, 64] {
        assert_eq!(
            HeadPlan::new(global, 0, 1).unwrap(),
            HeadPlan {
                start: 0,
                local: global
            }
        );
        let a = HeadPlan::new(global, 0, 2).unwrap();
        let b = HeadPlan::new(global, 1, 2).unwrap();
        assert_eq!(
            (a.start, a.local, b.start, b.local),
            (0, global / 2, global / 2, global / 2)
        );
    }
    for (g, r, w) in [(0, 0, 1), (31, 0, 2), (32, 2, 2), (32, 0, 4)] {
        assert!(HeadPlan::new(g, r, w).is_err());
    }
}
