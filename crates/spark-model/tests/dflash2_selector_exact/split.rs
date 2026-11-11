// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The rank split's two launches pick the fused kernel's tokens and
/// confidences, bit for bit, at the split's cut (the largest 16-row boundary
/// at or under half the vocabulary), at an uneven cut and with the lists in
/// either order.
#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn split_selector_matches_the_fused_kernel() -> Result<()> {
    let (gpu, stream) = setup()?;
    let (prod, conf) = kernels(&gpu)?;
    let (topk, walk, walk_conf) = split_kernels(&gpu, false)?;
    for case in CASES {
        let inp = inputs(&gpu, case)?;
        let tokens = fused(&gpu, stream, prod, &inp, GAMMA)?;
        let twin = fused(&gpu, stream, conf, &inp, 2 * GAMMA)?;
        for (cut, order) in [
            (VOCAB / 2 / 16 * 16, [0, 1]),
            (VOCAB / 2 / 16 * 16, [1, 0]),
            (1000, [0, 1]),
        ] {
            let got = split(&gpu, stream, topk, walk, &inp, cut, order, GAMMA)?;
            assert_eq!(
                got, tokens,
                "{} cut {cut} {order:?}: split picks differ",
                case.name
            );
            let got = split(&gpu, stream, topk, walk_conf, &inp, cut, order, 2 * GAMMA)?;
            assert_eq!(
                got, twin,
                "{} cut {cut} {order:?}: split confidences differ",
                case.name
            );
        }
        let (_, walk4, walk4_conf) = split_kernels(&gpu, true)?;
        let got = split4(&gpu, stream, topk, walk4, &inp, GAMMA)?;
        assert_eq!(
            got, tokens,
            "{}: the chain's four lists' picks differ",
            case.name
        );
        let got = split4(&gpu, stream, topk, walk4_conf, &inp, 2 * GAMMA)?;
        assert_eq!(
            got, twin,
            "{}: the chain's four lists' confidences differ",
            case.name
        );
        println!(
            "SPLIT {} matches the fused kernel (two and four lists)",
            case.name
        );
        for p in [
            inp.d_logits,
            inp.d_proj,
            inp.d_pred,
            inp.d_succ,
            inp.d_anchor,
            inp.d_ban,
        ] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
