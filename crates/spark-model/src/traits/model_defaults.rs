// SPDX-License-Identifier: AGPL-3.0-only

//! Out-of-line bodies for non-trivial [`Model`] default methods, keeping the
//! trait definition in `model.rs` to declarations and one-line defaults.

use anyhow::Result;

use super::Model;

/// Default [`Model::ep_broadcast_disable_mtp_for_seq`].
pub(super) fn ep_broadcast_disable_mtp_for_seq<M: Model + ?Sized>(
    model: &M,
    seq_id: u32,
    disabled: bool,
) -> Result<()> {
    if disabled {
        model.ep_broadcast_cmd_for_seq(seq_id, 0xFFFF_FFF6)?;
        model.ep_broadcast_cmd(1)?;
    }
    Ok(())
}
