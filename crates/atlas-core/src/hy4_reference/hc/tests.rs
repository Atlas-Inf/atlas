// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn params() -> HcParams {
    HcParams {
        hidden: 2,
        mult: 4,
        norm_eps: 1e-5,
        gate_eps: 1e-6,
        magnitude: 2.0,
    }
}
#[test]
fn independent_pre_post_and_head_have_no_stream_mixing() {
    let p = params();
    let x = [1., 2., 3., 4., 5., 6., 7., 8.];
    let (y, g) = pre(&x, &[0.; 64], &[1., 1.], &[0.; 8], p).unwrap();
    assert_eq!(y, [8., 10.]);
    assert!(g.iter().all(|v| (*v - 1.000001).abs() < 1e-7));
    let out = post(&[2., -2.], &x, &g, p).unwrap();
    assert_eq!(out, [3., bf16(2. - 2. * g[0]), 5., 2., 7., 4., 9., 6.]);
    assert_eq!(head(&x, &[0.; 32], &[1.], &[0.; 4], p).unwrap(), y);
}
#[test]
fn projection_uses_flattened_rms_and_separate_pre_post_scales() {
    let p = params();
    let x = [1.; 8];
    let mut w = [0.; 64];
    w[0] = 2.;
    w[32] = -3.;
    let (y, g) = pre(&x, &w, &[2., 3.], &[0.; 8], p).unwrap();
    let r = (1.0 + p.norm_eps).sqrt().recip();
    assert_eq!(y[0], bf16(sigmoid(4. * r) + 1.5 + 4. * p.gate_eps));
    assert!((g[0] - (2. * sigmoid(-9. * r) + p.gate_eps)).abs() < 1e-7);
    assert!((g[1] - 1.000001).abs() < 1e-7);
}
#[test]
fn epsilon_is_added_after_post_magnitude_and_highway_rounds_to_bf16() {
    let p = HcParams {
        gate_eps: 0.125,
        magnitude: 3.,
        ..params()
    };
    let (_, g) = pre(&[0.; 8], &[0.; 64], &[0.; 2], &[0.; 8], p).unwrap();
    assert_eq!(g, [1.625; 4]);
    let out = post(&[0.1, 0.2], &[0.1; 8], &g, p).unwrap();
    assert_eq!(out[0], bf16(bf16(0.1) + 1.625 * bf16(0.1)));
}
#[test]
fn invalid_empty_and_nonfinite_contracts_are_errors() {
    let p = params();
    assert!(pre(&[], &[], &[], &[], p).is_err());
    assert!(head(&[0.; 8], &[0.; 31], &[1.], &[0.; 4], p).is_err());
    assert!(post(&[], &[0.; 8], &[1.; 4], p).is_err());
    assert!(pre(&[f32::NAN; 8], &[0.; 64], &[1.; 2], &[0.; 8], p).is_err());
    for bad in [
        HcParams { mult: 3, ..p },
        HcParams { norm_eps: 0., ..p },
        HcParams { hidden: 0, ..p },
        HcParams {
            magnitude: f32::INFINITY,
            ..p
        },
    ] {
        assert!(bad.validate().is_err());
    }
}
