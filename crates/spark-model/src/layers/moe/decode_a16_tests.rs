// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_GLM_FP4L_DECODE_A16`: the lane a routed FFN takes, the scopes that
//! decide it, and the `_a16` twins the target must ship.
use super::*;

#[test]
fn only_decode_rows_under_the_switch_leave_a4() {
    for stream in [false, true] {
        assert_eq!(lane(false, true, stream), Activations::A4);
        assert_eq!(lane(true, false, stream), Activations::A4);
        assert_eq!(lane(false, false, stream), Activations::A4);
    }
    // The stream twins when they take the batch, else the grouped W4A16 path.
    assert_eq!(lane(true, true, true), Activations::A16Stream);
    assert_eq!(lane(true, true, false), Activations::A16Grouped);
}

#[test]
fn scopes_nest_and_restore_what_they_found() {
    let rows = || DECODE_ROWS.with(|c| c.get());
    assert!(!rows());
    {
        let _outer = DecodeRows::enter();
        assert!(rows());
        {
            let _inner = DecodeRows::enter();
            assert!(rows());
        }
        // The inner scope restores the outer one's, not the default.
        assert!(rows());
    }
    assert!(!rows());
}

#[test]
fn a_mixed_batch_takes_its_prompt_rows_once() {
    assert_eq!(PromptRows::take(), None);
    {
        let _prompt = PromptRows::enter(12);
        // The routed FFN that splits the batch takes it; its halves see none.
        assert_eq!(PromptRows::take(), Some(12));
        assert_eq!(PromptRows::take(), None);
    }
    {
        let _outer = PromptRows::enter(3);
        {
            let _inner = PromptRows::enter(5);
        }
        assert_eq!(PromptRows::take(), Some(3));
    }
    assert_eq!(PromptRows::take(), None);
}

/// Every `_a16` twin `DecodeM16` looks up is declared in the GLM target.
#[test]
fn the_a16_twins_are_in_the_glm_target() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/gb10/glm-5.3-flash/nvfp4/glm_moe_decode_stream.cuh"),
    )
    .unwrap();
    for family in ["m16s", "m32s", "m64s"] {
        for kernel in ["a16_gate_up_silu_k128w_l2pf_p", "a16_k128w_l2pf_p"] {
            let name = format!("glm_moe_decode_{family}_{kernel}(");
            assert!(src.contains(&name), "{name}");
        }
    }
}
