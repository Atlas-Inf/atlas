// SPDX-License-Identifier: AGPL-3.0-only

//! The unit of vision input handed to the model.

use std::sync::Arc;

/// One vision item: a still image, or a video, ready for the encoder.
///
/// `groups` holds the TEMPORAL GROUPS. A still image has exactly one; a video
/// has one group per `temporal_patch_size` frame window. Qwen pads a final
/// partial window by repeating its last frame, while GLM drops that window.
/// Each group is a full `grid_h x grid_w` patch plane built from
/// `temporal_patch_size` consecutive frames. Every group is shaped
/// identically — which is why the ViT consumes them on the same path and
/// needs no notion of time.
///
/// Grouping lives in the TYPE rather than in a parallel `groups_per_item`
/// vector carried alongside the pixels. Two vectors that must agree is
/// precisely the failure this feature is exposed to: the pad run, the encoder
/// rows and the MRoPE position stream all have to describe the same item, and
/// a desync between them is silent — fluent output, wrong answer.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionItem {
    /// The in-flight-rows permit this item holds (vision_admit).
    pub held: crate::vision_budget::HeldRows,
    /// Per temporal group: `[grid_h * grid_w, C * temporal_patch_size * patch^2]`.
    /// Clones share this immutable pixel storage so serial `n > 1` choices do
    /// not duplicate the image while each choice is in flight.
    pub groups: Arc<Vec<Vec<f32>>>,
    /// Pre-merge patch grid, identical across this item's groups.
    pub grid_h: usize,
    pub grid_w: usize,
    /// Content identity for the prefix cache: SHA-256 (truncated to 128 bits)
    /// of the preprocessed pixels and the grid. Two items with equal hashes
    /// produce the same encoder rows, so their KV may be shared; the prompt's
    /// placeholder tokens alone cannot tell two images apart (issue #58).
    pub content_hash: u128,
}

impl VisionItem {
    /// A still image: one temporal group.
    /// Ride an in-flight-rows permit (vision_admit; Enntity/atlas b060f987).
    pub fn holding(
        mut self,
        permit: std::sync::Arc<crate::vision_budget::VisionRowPermit>,
    ) -> Self {
        self.held = crate::vision_budget::HeldRows(Some(permit));
        self
    }

    pub fn image(pixels: Vec<f32>, grid_h: usize, grid_w: usize) -> Self {
        Self::video(vec![pixels], grid_h, grid_w)
    }

    /// A video (or any item): one entry per temporal group.
    pub fn video(groups: Vec<Vec<f32>>, grid_h: usize, grid_w: usize) -> Self {
        let content_hash = content_hash(&groups, grid_h, grid_w);
        Self {
            groups: Arc::new(groups),
            grid_h,
            grid_w,
            content_hash,
        }
    }

    /// The item, holding `permit` until its last copy drops.
    /// Several items of one request may share one permit.
    pub fn holding(
        mut self,
        permit: std::sync::Arc<crate::vision_budget::VisionRowPermit>,
    ) -> Self {
        self.held = crate::vision_budget::HeldRows(Some(permit));
        self
    }

    /// Temporal extent, in groups. 1 for a still.
    pub fn t_len(&self) -> usize {
        self.groups.len().max(1)
    }

    /// Merged tokens this item occupies in the prompt — the length of its pad
    /// run, and the number of embedding rows the encoder will return for it.
    /// The two are the same number by construction, which is the invariant
    /// the whole splice depends on.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        let sms = spatial_merge_size.max(1);
        self.t_len() * (self.grid_h / sms) * (self.grid_w / sms)
    }
}

/// SHA-256 over the grid and every group's pixels, truncated to 128 bits.
fn content_hash(groups: &[Vec<f32>], grid_h: usize, grid_w: usize) -> u128 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for dim in [groups.len(), grid_h, grid_w] {
        h.update((dim as u64).to_le_bytes());
    }
    for g in groups {
        h.update((g.len() as u64).to_le_bytes());
        for chunk in g.chunks(16 * 1024) {
            let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
            h.update(&bytes);
        }
    }
    let digest = h.finalize();
    u128::from_le_bytes(digest[..16].try_into().expect("SHA-256 is 32 bytes"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::VisionItem;

    #[test]
    fn content_hash_follows_pixels_and_grid() {
        let a = VisionItem::image(vec![0.25; 8], 2, 4);
        assert_eq!(
            a.content_hash,
            VisionItem::image(vec![0.25; 8], 2, 4).content_hash
        );
        let mut px = vec![0.25; 8];
        px[7] = 0.5;
        assert_ne!(a.content_hash, VisionItem::image(px, 2, 4).content_hash);
        assert_ne!(
            a.content_hash,
            VisionItem::image(vec![0.25; 8], 4, 2).content_hash
        );
        let clip = VisionItem::video(vec![vec![0.25; 4], vec![0.25; 4]], 2, 4);
        assert_ne!(
            a.content_hash, clip.content_hash,
            "grouping is part of the identity"
        );
    }

    #[test]
    fn clones_share_pixel_storage() {
        let item = VisionItem::image(vec![0.25; 8], 2, 4);
        let clone = item.clone();
        assert!(Arc::ptr_eq(&item.groups, &clone.groups));
    }
}
