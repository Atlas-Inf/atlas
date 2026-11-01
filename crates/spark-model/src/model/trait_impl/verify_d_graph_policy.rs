// SPDX-License-Identifier: AGPL-3.0-only
//! Default-off admission for the repaired single-draft distributed graph.

#[allow(clippy::too_many_arguments)]
pub(super) fn mtp1_graph_allowed(
    requested: bool,
    model_type: &str,
    tp: usize,
    ep: usize,
    active_capacity: u32,
    rows: usize,
    repair: bool,
    k2_diagnostic: bool,
) -> bool {
    requested
        && model_type == "glm5_next"
        && tp == 2
        && ep == 2
        && active_capacity == 1
        && rows == 2
        && repair
        && !k2_diagnostic
}

#[cfg(test)]
mod tests {
    use super::mtp1_graph_allowed as allowed;

    #[test]
    fn exact_repaired_c1_k2_is_opt_in() {
        assert!(allowed(true, "glm5_next", 2, 2, 1, 2, true, false));
        assert!(!allowed(false, "glm5_next", 2, 2, 1, 2, true, false));
    }

    #[test]
    fn unrelated_widths_topologies_and_diagnostic_stay_eager() {
        for rows in [0, 1, 3, 4, 5, 8, 32] {
            assert!(!allowed(true, "glm5_next", 2, 2, 1, rows, true, false));
        }
        for model in ["qwen3", "deepseek_v4", ""] {
            assert!(!allowed(true, model, 2, 2, 1, 2, true, false));
        }
        for (tp, ep, cap) in [(1, 2, 1), (2, 1, 1), (4, 2, 1), (2, 2, 2)] {
            assert!(!allowed(true, "glm5_next", tp, ep, cap, 2, true, false));
        }
        assert!(!allowed(true, "glm5_next", 2, 2, 1, 2, false, false));
        assert!(!allowed(true, "glm5_next", 2, 2, 1, 2, true, true));
    }
}
