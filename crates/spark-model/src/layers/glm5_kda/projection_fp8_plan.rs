// SPDX-License-Identifier: AGPL-3.0-only
//! CPU-only admission and scratch proof for the qualified KDA Lt FP8 route.

/// Widest qualified projection (the arena bound is checked by `plan`).
const MAX_M: u32 = u16::MAX as u32;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Span {
    pub ptr: u64,
    pub bytes: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    pub weight: u64,
    pub activation: u64,
}
pub(super) fn selected(
    enabled: bool,
    decode: bool,
    capture: bool,
    transposed: bool,
    m: u32,
    n: u32,
    k: u32,
) -> bool {
    enabled
        && !decode
        && !capture
        && transposed
        && (2048..=MAX_M).contains(&m)
        && n == 4096
        && k == 4096
}
pub(super) fn plan(
    scratch: Span,
    arena_rows: usize,
    m: u32,
    input: u64,
    output: u64,
    weight: u64,
    scales: u64,
) -> Result<Plan, &'static str> {
    if !(2048..=MAX_M).contains(&m) || m as usize > arena_rows {
        return Err("KDA Lt row capacity");
    }
    let weight_bytes = 4096usize * 4096;
    let activation_bytes = m as usize * 4096;
    let need = weight_bytes
        .checked_add(activation_bytes)
        .ok_or("KDA Lt scratch overflow")?;
    if scratch.bytes < need {
        return Err("KDA Lt scratch capacity");
    }
    let end = |span: Span| -> Result<u64, &'static str> {
        if span.ptr == 0 || !span.ptr.is_multiple_of(16) || span.bytes == 0 {
            return Err("KDA Lt pointer alignment");
        }
        span.ptr
            .checked_add(span.bytes as u64)
            .ok_or("KDA Lt pointer overflow")
    };
    end(scratch)?;
    let spans = [
        Span {
            ptr: scratch.ptr,
            bytes: need,
        },
        Span {
            ptr: input,
            bytes: activation_bytes * 2,
        },
        Span {
            ptr: output,
            bytes: activation_bytes * 2,
        },
        Span {
            ptr: weight,
            bytes: weight_bytes / 2,
        },
        Span {
            ptr: scales,
            bytes: weight_bytes / 16,
        },
    ];
    for (i, &span) in spans.iter().enumerate() {
        let e = end(span)?;
        for &prior in &spans[..i] {
            if span.ptr < end(prior)? && prior.ptr < e {
                return Err("KDA Lt live span alias");
            }
        }
    }
    Ok(Plan {
        weight: scratch.ptr,
        activation: scratch.ptr + weight_bytes as u64,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    const BASE: u64 = 0x1000_0000;
    fn fixture(m: u32, capacity: usize) -> Result<Plan, &'static str> {
        plan(
            Span {
                ptr: BASE,
                bytes: capacity,
            },
            4100,
            m,
            0x2000_0000,
            0x3000_0000,
            0x4000_0000,
            0x5000_0000,
        )
    }
    #[test]
    fn kda_lt_fp8_selection_preserves_decode_and_unqualified_rows() {
        for m in [1, 2, 3, 4, 5, 8, 128, 2047, MAX_M + 1] {
            assert!(!selected(true, false, false, true, m, 4096, 4096));
        }
        // A full chunk, and one carrying the widest riding verify rows.
        for m in [2048, 4096, 4100, MAX_M] {
            assert!(selected(true, false, false, true, m, 4096, 4096));
        }
        for (enabled, decode, capture, twin) in [
            (false, false, false, true),
            (true, true, false, true),
            (true, false, true, true),
            (true, false, false, false),
        ] {
            assert!(!selected(enabled, decode, capture, twin, 2048, 4096, 4096));
        }
        assert!(!selected(true, false, false, true, 2048, 2048, 4096));
    }
    #[test]
    fn kda_lt_fp8_plan_exact_capacity_and_tail() {
        for m in [2048, 4096, 4100] {
            let bytes = 4096 * 4096 + m as usize * 4096;
            let p = fixture(m, bytes).unwrap();
            assert_eq!(p.weight, BASE);
            assert_eq!(p.activation, BASE + 4096 * 4096);
            assert!(fixture(m, bytes - 1).is_err());
        }
    }
    #[test]
    fn kda_lt_fp8_rejects_owner_overflow_and_aliases() {
        let scratch = Span {
            ptr: BASE,
            bytes: 64 << 20,
        };
        assert!(
            plan(
                scratch,
                2047,
                2048,
                0x2000_0000,
                0x3000_0000,
                0x4000_0000,
                0x5000_0000
            )
            .is_err()
        );
        for slot in 0..4 {
            let mut p = [0x2000_0000, 0x3000_0000, 0x4000_0000, 0x5000_0000];
            p[slot] = BASE + 16;
            assert!(plan(scratch, 4100, 2048, p[0], p[1], p[2], p[3]).is_err());
        }
        assert!(
            plan(
                Span {
                    ptr: u64::MAX - 15,
                    bytes: 64 << 20
                },
                4100,
                2048,
                0x2000_0000,
                0x3000_0000,
                0x4000_0000,
                0x5000_0000
            )
            .is_err()
        );
        // Wider than the fixture's 4100-row arena.
        assert!(fixture(4101, 64 << 20).is_err());
    }
}
