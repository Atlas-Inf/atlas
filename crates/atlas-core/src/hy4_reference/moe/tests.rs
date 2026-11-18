// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
fn routing() -> Routing {
    Routing {
        top_k: 2,
        renormalize: true,
        scaling: 3.998,
    }
}

#[test]
fn correction_bias_selects_but_does_not_weight() {
    let p = Routing {
        top_k: 1,
        ..routing()
    };
    let r = route(&[4., -4.], &[0., 10.], p).unwrap();
    assert_eq!(r, [(1, 3.998)]);
    let r = route(&[0., 0., 0.], &[0.; 3], routing()).unwrap();
    assert_eq!(r, [(0, 1.999), (1, 1.999)]);
    let r = route(
        &[0., 0.],
        &[0.; 2],
        Routing {
            renormalize: false,
            ..routing()
        },
    )
    .unwrap();
    assert_eq!(r, [(0, 1.999), (1, 1.999)]);
}
#[test]
fn extremes_empty_malformed_and_nan_fail_closed() {
    assert_eq!(
        route(&[-1000.; 2], &[0.; 2], routing()).unwrap(),
        [(0, 0.), (1, 0.)]
    );
    assert_eq!(
        route(&[1000.; 2], &[0.; 2], routing()).unwrap(),
        [(0, 1.999), (1, 1.999)]
    );
    for (l, b) in [
        (&[][..], &[][..]),
        (&[0.][..], &[0., 0.][..]),
        (&[f32::NAN, 0.][..], &[0., 0.][..]),
    ] {
        assert!(route(l, b, routing()).is_err());
    }
    assert!(
        route(
            &[0.; 2],
            &[0.; 2],
            Routing {
                top_k: 0,
                ..routing()
            }
        )
        .is_err()
    );
    assert!(swiglu(1., 2., Some(0.)).is_err());
    assert!(linear(&[], &[], 1).is_err());
    assert!(rms(&[1.], &[], 1e-5).is_err());
}
#[test]
fn clamp_is_on_raw_gate_upper_bound_only_and_not_on_dense_or_shared() {
    assert_eq!(
        swiglu(20., 20., Some(10.)).unwrap(),
        bf16(10. * sigmoid(10.) * 10.)
    );
    assert_eq!(
        swiglu(-20., 20., Some(10.)).unwrap(),
        bf16(-20. * sigmoid(-20.) * 10.)
    );
    assert_eq!(
        swiglu(20., -20., Some(10.)).unwrap(),
        bf16(10. * sigmoid(10.) * -10.)
    );
    assert_eq!(
        swiglu(20., 20., None).unwrap(),
        bf16(20. * sigmoid(20.) * 20.)
    );
}
#[test]
fn latent_projection_router_input_and_ep_shared_add_once() {
    let model = LatentMoe {
        router: &[1., 0., 0., 1.],
        bias: &[0., 0.],
        latent_down: &[1., 0.],
        latent_up: &[1., 2.],
        latent: 1,
        experts: vec![
            Mlp {
                gate: &[1.],
                up: &[1.],
                down: &[1.],
                intermediate: 1,
            },
            Mlp {
                gate: &[2.],
                up: &[1.],
                down: &[1.],
                intermediate: 1,
            },
        ],
        shared: Mlp {
            gate: &[1., 0., 0., 1.],
            up: &[1., 0., 0., 1.],
            down: &[1., 0., 0., 1.],
            intermediate: 2,
        },
        routing: Routing {
            scaling: 1.,
            ..routing()
        },
        limit: 10.,
        norm: &[1., 1.],
        eps: 1e-5,
    };
    let x = [1., 2.];
    let full = model.routed(&x, EpPlan::new(0, 1, 1, 2).unwrap()).unwrap();
    let a = model.routed(&x, EpPlan::new(0, 2, 2, 2).unwrap()).unwrap();
    let b = model.routed(&x, EpPlan::new(1, 2, 2, 2).unwrap()).unwrap();
    let sum: Vec<_> = a.iter().zip(&b).map(|(a, b)| bf16(a + b)).collect();
    for (a, b) in full.iter().zip(&sum) {
        assert!((a - b).abs() < 0.02);
    }
    let expected_route = route(&[1., 2.], model.bias, model.routing).unwrap();
    let y = bf16(
        expected_route[0].1 * bf16(2. * sigmoid(2.)) + expected_route[1].1 * bf16(sigmoid(1.)),
    );
    assert_eq!(full, [y, bf16(2. * y)]);
    let output = model.finish(&x, &sum).unwrap();
    let shared = model.shared.forward(&x, None).unwrap();
    let expected: Vec<_> = sum.iter().zip(shared).map(|(a, b)| bf16(a + b)).collect();
    assert_eq!(output, rms(&expected, &[1., 1.], 1e-5).unwrap());
    assert!(model.finish(&x, &[]).is_err());
}
