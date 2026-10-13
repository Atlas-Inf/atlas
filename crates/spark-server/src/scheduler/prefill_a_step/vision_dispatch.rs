// SPDX-License-Identifier: AGPL-3.0-only

//! Chunk-0 vision co-dispatch: the local slice base plus the worker's MTP
//! fence and vision-slice descriptor.

use super::*;

/// Point this request's chunk-0 vision splice at its slice, then send the EP
/// worker this request's MTP fence and the same slice descriptor.
pub(super) fn dispatch_chunk0_vision(
    model: &dyn Model,
    seq: &SequenceState,
    vision_slice: Option<VisionSlice>,
    image_pixels: &[spark_model::VisionItem],
    req_disable_mtp: bool,
) -> Result<()> {
    // Co-dispatch: point this request's chunk-0 splice/MRoPE at its slice of
    // the shared packed buf_out before the worker receives the same slice
    // descriptor. Single-chunk-fit is guaranteed upstream for this path.
    let vision_enabled = vision_slice.is_some() || !image_pixels.is_empty();
    if let Some(s) = vision_slice {
        model.set_vision_slice_base(
            s.patch_row_offset,
            s.grid_index_offset,
            s.num_images,
            s.patch_row_count,
        );
    } else {
        model.set_vision_slice_base(0, 0, 0, 0);
    }
    model.ep_broadcast_disable_mtp_for_seq(seq.slot_idx as u32, req_disable_mtp)?;
    let (vision_row_base, vision_grid_base, vision_owned_images, vision_slice_rows) = vision_slice
        .map_or((0, 0, 0, 0), |s| {
            (
                s.patch_row_offset,
                s.grid_index_offset,
                s.num_images,
                s.patch_row_count,
            )
        });
    model.ep_broadcast_vision_state_for_seq(
        seq.slot_idx as u32,
        vision_enabled,
        vision_row_base,
        vision_grid_base,
        vision_owned_images,
        vision_slice_rows,
    )?;
    Ok(())
}
