// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl Hy4Model {
    pub(super) fn unavailable<T>() -> Result<T> {
        anyhow::bail!(
            "Hy4 supports eager accepted text tokens only; no speculation, graph capture, DFly or vision"
        )
    }
}
