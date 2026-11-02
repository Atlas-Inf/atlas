// SPDX-License-Identifier: AGPL-3.0-only

//! Out-of-line bodies for non-trivial [`Model`] default methods, keeping the
//! trait definition in `model.rs` to declarations and one-line defaults.

use anyhow::Result;

use super::Model;
use crate::traits::SequenceState;

/// Head → worker: plant a prefix checkpoint in the addressed sequence's
/// prefill (`ATLAS_GLM_PC_INFLIGHT`). Followed by one word, the position.
pub const EP_CMD_PC_PLANT: u32 = 0xFFFF_FFED;

/// Default [`Model::pc_plant`]: send the worker the same request, then record
/// it. The prefill decides at each chunk, on both ranks alike, whether it is
/// still ahead (`pc_inflight`).
pub(super) fn pc_plant<M: Model + ?Sized>(
    model: &M,
    seq: &mut SequenceState,
    at: usize,
) -> Result<()> {
    // Convert before the command word: a failure between the two words would
    // leave the worker reading the next command as the position.
    let word = u32::try_from(at)?;
    model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, EP_CMD_PC_PLANT)?;
    model.ep_broadcast_cmd(word)?;
    seq.pc_plant_at = Some(at);
    Ok(())
}

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
