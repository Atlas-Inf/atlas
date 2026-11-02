// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn valid() -> Params {
    let p = |i: u64| (i + 1) * 0x1000_0000;
    Params::new(
        p(0),
        p(1),
        p(2),
        p(3),
        p(4),
        p(5),
        p(6),
        p(7),
        ScratchLayout::new(4096).unwrap().bytes as usize,
        4096,
        0,
    )
    .unwrap()
}

#[test]
fn params_layout_matches_native_header() {
    assert_eq!(std::mem::size_of::<Params>(), 144);
    assert_eq!(std::mem::align_of::<Params>(), 8);
    assert_eq!(std::mem::offset_of!(Params, layout), 64);
    assert_eq!(std::mem::offset_of!(Params, tokens), 128);
    assert_eq!(std::mem::offset_of!(Params, stream), 136);
}

#[test]
fn insufficient_workspace_and_overlapping_inputs_are_rejected() {
    let p = valid();
    let build = |q, work, bytes| {
        Params::new(
            q, p.gate, p.beta, p.a_log, p.dt_bias, p.state, p.output, work, bytes, 4096, 0,
        )
    };
    assert_eq!(
        build(p.qkv, p.workspace, p.layout.bytes as usize - 1).unwrap_err(),
        Error::Invalid("native KDA scratch capacity exceeded")
    );
    assert_eq!(
        build(p.gate, p.workspace, p.layout.bytes as usize).unwrap_err(),
        Error::Invalid("native KDA operands overlap")
    );
    assert!(build(u64::MAX - 15, p.workspace, p.layout.bytes as usize).is_err());
    assert!(build(0, p.workspace, p.layout.bytes as usize).is_err());
    assert!(build(p.qkv, p.workspace + 16, p.layout.bytes as usize).is_err());
}

#[test]
fn canonical_state_has_one_in_place_owner() {
    let p = valid();
    assert_ne!(p.state, p.workspace);
    assert_ne!(p.state, p.output);
    assert_eq!(p.tokens, 4096);
    assert_eq!(p.reserved, 0);
}

#[cfg(not(all(feature = "cuda", atlas_glm_kda)))]
#[test]
fn uncompiled_operator_returns_unavailable_without_touching_fake_pointers() {
    assert!(!available());
    assert_eq!(preload(), Err(Error::Unavailable));
    // SAFETY: the uncompiled path never reads the numerical device addresses.
    assert_eq!(unsafe { launch(&valid()) }, Err(Error::Unavailable));
}
