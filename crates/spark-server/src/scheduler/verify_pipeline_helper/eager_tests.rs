// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::cancel_test_model::TestModel;
use crate::scheduler::{sched_ctx::SchedCtx, test_support::test_seq};

#[test]
fn eager_fp32_rows_apply_the_serial_pipeline_without_committing_shadow_history() {
    let model = TestModel {
        tokens: vec![],
        host_logits: true,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    let (mut a, _rx) = test_seq(vec![], 50, None, 20);
    a.finished = false;
    a.logit_bias = vec![(102, 6.0)];
    assert_eq!(
        verify_pick_eager_fp32(&model, &[101, 101], &mut a, &ctx).unwrap(),
        [102, 102]
    );
    assert!(a.output_tokens.is_empty());
    assert_eq!(a.remaining, 50);
    assert!(
        verify_pick_eager_fp32(&model, &[], &mut a, &ctx)
            .unwrap()
            .is_empty()
    );
    a.temperature = 0.7;
    assert!(verify_pick_eager_fp32(&model, &[101], &mut a, &ctx).is_err());
    assert!(a.output_tokens.is_empty());
}
