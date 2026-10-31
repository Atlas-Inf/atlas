// SPDX-License-Identifier: AGPL-3.0-only

//! Adapted from Jason McCartney / Enntity's SparkGLM union grammar work.
//! Source commits and integration notes: `docs/porting/tool-json-union.json`.
//!
//! The branch decision of a tool-call-or-JSON union
//! (`GrammarSpec::ToolCallOrJson`) at its first output token.
//!
//! Masking the union token by token lets the model's runner-up decide: when
//! it means to answer with a fenced block (```` ```json ````, its top token,
//! masked), the tool opener can outrank `{` and every answer becomes a tool
//! call (live 2026-10-17: the TC-65 probe called a tool again 5/5). So the
//! branch is chosen by mass: the tool branch when the model puts more
//! probability on tool-branch tokens than on everything else (an answer in
//! any shape), else the answer; that position's mask is then the chosen
//! branch's alone and the model picks its best token within it.

/// Whether the tool branch outweighs the rest. `union` is the union's mask,
/// `answer` the answer branch's first-token mask (bit set = allowed); tool
/// tokens are those the union allows beyond the answer.
pub(super) fn prefers_tool(logits: &[f32], union: &[i32], answer: &[i32]) -> bool {
    let is_tool = |t: usize| {
        let (w, b) = (t / 32, t % 32);
        (union[w] & !answer[w]) >> b & 1 == 1
    };
    let max = logits
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return false;
    }
    let (mut tool, mut rest) = (0f64, 0f64);
    for (t, &l) in logits.iter().enumerate().take(union.len() * 32) {
        if l.is_finite() {
            let p = f64::from(l - max).exp();
            if is_tool(t) { tool += p } else { rest += p }
        }
    }
    tool > rest
}

/// Restrict `mask` to the chosen branch.
pub(super) fn restrict(mask: &mut [i32], answer: &[i32], tool: bool) {
    for (m, &a) in mask.iter_mut().zip(answer) {
        *m &= if tool { !a } else { a };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOL: usize = 40;
    const BRACE: usize = 7;
    const FENCE: usize = 3;

    fn masks() -> ([i32; 2], [i32; 2]) {
        let mut answer = [0i32; 2];
        answer[0] |= 1 << BRACE;
        let mut union = answer;
        union[TOOL / 32] |= 1 << (TOOL % 32);
        (union, answer)
    }

    fn logits(pairs: &[(usize, f32)]) -> Vec<f32> {
        let mut l = vec![-5.0f32; 64];
        for &(t, v) in pairs {
            l[t] = v;
        }
        l
    }

    /// The fence the union cannot emit still counts as answering.
    #[test]
    fn a_masked_fence_votes_for_the_answer_over_a_runner_up_tool_opener() {
        let (union, answer) = masks();
        let l = logits(&[(FENCE, 10.0), (TOOL, 4.0), (BRACE, 3.0)]);
        assert!(!prefers_tool(&l, &union, &answer));
        let mut mask = union;
        restrict(&mut mask, &answer, false);
        assert_eq!(mask, answer);
    }

    #[test]
    fn a_model_that_means_to_call_gets_the_tool_branch() {
        let (union, answer) = masks();
        let l = logits(&[(FENCE, 1.0), (TOOL, 9.0), (BRACE, 2.0)]);
        assert!(prefers_tool(&l, &union, &answer));
        let mut mask = union;
        restrict(&mut mask, &answer, true);
        assert_eq!(mask, [0, 1 << (TOOL % 32)]);
    }

    #[test]
    fn masked_out_logits_are_ignored() {
        let (union, answer) = masks();
        let mut l = logits(&[(TOOL, 9.0)]);
        l.iter_mut().for_each(|x| *x = f32::NEG_INFINITY);
        assert!(!prefers_tool(&l, &union, &answer), "nothing finite: answer");
        l[TOOL] = 1.0;
        assert!(prefers_tool(&l, &union, &answer));
    }
}
