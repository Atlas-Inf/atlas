// SPDX-License-Identifier: AGPL-3.0-only

//! Shared scalar/verify copy-error policy; the existing immunity proof is unchanged.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::scheduler) enum CopyFailurePolicy {
    LegacyFallback,
    Propagate,
}

impl CopyFailurePolicy {
    pub(in crate::scheduler) fn immune(
        self,
        token: u32,
        history: &[u32],
        probe: impl FnOnce() -> anyhow::Result<bool>,
    ) -> anyhow::Result<bool> {
        let mut failure = None;
        let immune =
            crate::scheduler::fast_greedy::argmax_immune(token, history, || match probe() {
                Ok(positive) => positive,
                Err(error) => {
                    if self == Self::Propagate {
                        failure = Some(error);
                    }
                    false
                }
            });
        match failure {
            Some(error) => Err(error),
            None => Ok(immune),
        }
    }
}
