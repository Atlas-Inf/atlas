// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    hc::{self, HcParams},
    moe::{LatentMoe, Mlp, Routing},
    topology::EpPlan,
};
use serde_json::Value;

fn vec(v: &Value, key: &str) -> Vec<f32> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= tolerance, "index {i}: {a} vs {b}");
    }
}
#[test]
fn independent_hc_matches_cpu_torch_equation_snapshot() {
    let fixture: Value = serde_json::from_str(include_str!("torch_fixture.json")).unwrap();
    let h = &fixture["hc"];
    let p = HcParams {
        hidden: 4,
        mult: 4,
        norm_eps: 1e-5,
        gate_eps: 1e-6,
        magnitude: 2.,
    };
    let (y, post) = hc::pre(
        &vec(h, "x"),
        &vec(h, "fn"),
        &vec(h, "scale"),
        &vec(h, "base"),
        p,
    )
    .unwrap();
    close(&y, &vec(h, "collapsed"), 0.);
    close(&post, &vec(h, "post_gate"), 2e-6);
    close(
        &hc::post(&vec(h, "block"), &vec(h, "x"), &post, p).unwrap(),
        &vec(h, "scattered"),
        0.,
    );
    close(
        &hc::head(
            &vec(h, "x"),
            &vec(h, "head_fn"),
            &vec(h, "head_scale"),
            &vec(h, "head_base"),
            p,
        )
        .unwrap(),
        &vec(h, "head"),
        0.,
    );
}
#[test]
fn latent_moe_matches_cpu_torch_equation_snapshot() {
    let fixture: Value = serde_json::from_str(include_str!("torch_fixture.json")).unwrap();
    let m = &fixture["moe"];
    let router = vec(m, "router");
    let bias = vec(m, "bias");
    let down = vec(m, "down");
    let up = vec(m, "up");
    let eg = vec(m, "expert_gate");
    let eu = vec(m, "expert_up");
    let ed = vec(m, "expert_down");
    let sg = vec(m, "shared_gate");
    let su = vec(m, "shared_up");
    let sd = vec(m, "shared_down");
    let norm = vec(m, "norm");
    let x = vec(m, "x");
    let model = LatentMoe {
        router: &router,
        bias: &bias,
        latent_down: &down,
        latent_up: &up,
        latent: 2,
        experts: (0..3)
            .map(|i| Mlp {
                gate: &eg[i * 6..i * 6 + 6],
                up: &eu[i * 6..i * 6 + 6],
                down: &ed[i * 6..i * 6 + 6],
                intermediate: 3,
            })
            .collect(),
        shared: Mlp {
            gate: &sg,
            up: &su,
            down: &sd,
            intermediate: 3,
        },
        routing: Routing {
            top_k: 2,
            renormalize: true,
            scaling: 3.998,
        },
        limit: 0.1,
        norm: &norm,
        eps: 1e-5,
    };
    let routed = model.routed(&x, EpPlan::new(0, 1, 1, 3).unwrap()).unwrap();
    close(&routed, &vec(m, "routed"), 0.);
    close(
        &model.shared.forward(&x, None).unwrap(),
        &vec(m, "shared"),
        0.,
    );
    close(&model.finish(&x, &routed).unwrap(), &vec(m, "output"), 0.);
}
