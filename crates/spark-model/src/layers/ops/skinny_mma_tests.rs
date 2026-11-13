// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn names_match_the_kernels() {
    assert_eq!(kernel_name(Family::W4a16, 1), "skinny_w4a16_t1");
    assert_eq!(kernel_name(Family::Qg, 2), "skinny_qg_t2");
    assert_eq!(kernel_name(Family::W8a16, 2), "skinny_w8a16_t2");
}

#[test]
fn served_shapes_follow_k_and_the_row_cap() {
    let w4 = |m, k| serves(Family::W4a16, m, k);
    assert!(w4(1, 5120) && w4(16, 17408) && w4(9, 6144));
    assert!(w4(17, 5120) && w4(MAX_ROWS, 5120));
    assert!(!w4(0, 5120) && !w4(MAX_ROWS + 1, 5120));
    assert!(!w4(1, 5120 + 128) && !w4(4, 0));
    assert!(serves(Family::W8a16, 1, 5120 + 128) && !serves(Family::W8a16, 1, 5120 + 16));
}

#[test]
fn one_tile_to_eight_rows_two_past() {
    let k = SkinnyKernels {
        handles: [[KernelHandle(1), KernelHandle(2)]; 3],
        wide: [[KernelHandle(0); 3]; 3],
        o2: [KernelHandle(0); 3],
    };
    for m in 1..=8 {
        assert_eq!(k.kernel(Family::Qg, m).0, 1, "m={m}");
    }
    for m in 9..=16 {
        assert_eq!(k.kernel(Family::W8a16, m).0, 2, "m={m}");
    }
}

#[test]
fn a_wide_call_takes_the_narrowest_wide_kernel_holding_it() {
    let k = SkinnyKernels {
        handles: [[KernelHandle(1), KernelHandle(2)]; 3],
        wide: [[KernelHandle(4), KernelHandle(8), KernelHandle(16)]; 3],
        o2: [KernelHandle(0); 3],
    };
    assert!(k.wide_kernel(Family::W4a16, 16).is_none());
    assert_eq!(k.wide_kernel(Family::W4a16, 17).map(|h| h.0), Some(4));
    assert_eq!(k.wide_kernel(Family::Qg, 32).map(|h| h.0), Some(4));
    assert_eq!(k.wide_kernel(Family::Qg, 33).map(|h| h.0), Some(8));
    assert_eq!(k.wide_kernel(Family::W8a16, 72).map(|h| h.0), Some(16));
    let plan = |k: &SkinnyKernels, f, m| -> Vec<(u64, u32, u32)> {
        k.launches(f, m)
            .into_iter()
            .map(|(h, a, b, _)| (h.0, a, b))
            .collect()
    };
    assert_eq!(plan(&k, Family::W8a16, 72), vec![(16, 0, 72)]);
    let narrow = SkinnyKernels {
        wide: [[KernelHandle(0); 3]; 3],
        ..k
    };
    assert_eq!(
        plan(&narrow, Family::W4a16, 20),
        vec![(2, 0, 16), (1, 16, 4)]
    );
    let o2 = SkinnyKernels {
        o2: [KernelHandle(2), KernelHandle(2), KernelHandle(0)],
        ..k
    };
    assert_eq!(o2.launches(Family::W4a16, 64)[0].3, 2, "x8 holds 64 rows");
    assert_eq!(
        o2.launches(Family::W4a16, 65)[0].0.0,
        16,
        "past 64 rows w16"
    );
    assert_eq!(o2.launches(Family::W8a16, 64)[0].0.0, 8, "no FP8 x8");
    assert_eq!(o2.launches(Family::Qg, 16)[0].0.0, 2, "16 rows stay narrow");
}

#[test]
fn wide_calls_split_into_sixteen_row_launches() {
    assert_eq!(row_groups(1).collect::<Vec<_>>(), vec![(0, 1)]);
    assert_eq!(row_groups(16).collect::<Vec<_>>(), vec![(0, 16)]);
    assert_eq!(
        row_groups(41).collect::<Vec<_>>(),
        vec![(0, 16), (16, 16), (32, 9)]
    );
    assert_eq!(row_groups(MAX_ROWS).count(), 8);
}

#[test]
fn only_listed_kernels_are_taken() {
    let s = Skinny {
        kernels: SkinnyKernels {
            handles: [[KernelHandle(1), KernelHandle(2)]; 3],
            wide: [[KernelHandle(0); 3]; 3],
            o2: [KernelHandle(0); 3],
        },
        replaced: [vec![10, 11], vec![20], vec![30]],
    };
    assert!(s.takes(Family::W4a16, KernelHandle(11), 1, 5120));
    assert!(
        !s.takes(Family::W4a16, KernelHandle(20), 1, 5120),
        "a Q+gate handle"
    );
    assert!(
        !s.takes(Family::W4a16, KernelHandle(99), 1, 5120),
        "e.g. the FP32 logits GEMV"
    );
    assert!(s.takes(Family::W8a16, KernelHandle(30), 17, 5120));
    assert!(!s.takes(Family::W8a16, KernelHandle(30), MAX_ROWS + 1, 5120));
    assert!(!s.takes(Family::W8a16, KernelHandle(0), 1, 5120));
}

#[test]
fn lever_reads_exactly_one() {
    assert!(requested_from(Some("1")));
    assert!(!requested_from(None) && !requested_from(Some("0")) && !requested_from(Some("true")));
}
