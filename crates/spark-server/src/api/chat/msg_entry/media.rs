// SPDX-License-Identifier: AGPL-3.0-only

//! Message-entry media collection helpers (extracted from `msg_entry.rs`).

use super::*;

/// Append every media part on `m` to `media` **in content order**, growing
/// `image_pad_counts` in lockstep (each pad count is filled in later by the
/// preprocessor). Shared by the tool-message branch and the normal branch so
/// media rides every role uniformly — including tool results, the motivating
/// case for issue #165.
#[allow(clippy::result_large_err)]
pub(super) fn collect_message_media(
    m: &Message,
    media: &mut Vec<MediaInput>,
    image_pad_counts: &mut Vec<usize>,
    remote: &super::super::remote_image::RemoteImagePolicy,
) -> Result<(), Response> {
    // ORDER IS A CONTRACT, and the order is the CLIENT'S. The template emits
    // one marker per media item in the order they appear in the content
    // array, and `expand_vision_pads` walks pad tokens left to right
    // consuming one count each — so this one pass, `MsgEntry::media` and the
    // preprocessing loop below must all traverse `m.content` the same way.
    // Grouping by modality anywhere in that chain shows the model the items
    // in an order the caller never wrote, and does it silently: every count
    // still agrees with every other one.
    for part in &m.content {
        let (kind, data) = match part {
            ContentPart::Image(src) => (MediaKind::Image, &src.data),
            ContentPart::Video(src) => (MediaKind::Video, &src.data),
            ContentPart::Text(_) => continue,
        };
        media.push(MediaInput {
            kind,
            uri: resolve_media_uri(kind, data, remote)?,
        });
        image_pad_counts.push(0);
    }
    Ok(())
}

/// Resolve one media source to an encoder-input string, applying the
/// operator's remote-fetch policy. Shared by both modalities because the
/// policy and its failure modes are identical — only the noun in the message
/// differs.
#[allow(clippy::result_large_err)]
pub(super) fn resolve_media_uri(
    kind: MediaKind,
    data: &ImageData,
    remote: &super::super::remote_image::RemoteImagePolicy,
) -> Result<String, Response> {
    let noun = match kind {
        MediaKind::Image => "image",
        MediaKind::Video => "video",
    };
    match data {
        ImageData::Base64(s) => Ok(s.clone()),
        // Neither the encoder nor the video decoder fetches remote URLs
        // itself. Either the operator has granted this server the
        // capability, in which case the URL is resolved to a data: URI here,
        // or it has not, in which case this is a 400 naming the flag. What
        // must not happen is feeding the URL string onward: it would reach
        // the base64 decoder and fail with a confusing "base64 decode
        // failed" (PCND: fail fast, with the real reason).
        ImageData::Url(url) => {
            let shown: String = url.chars().take(120).collect();
            if !remote.enabled {
                return Err(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "{noun} URLs are not fetched by this server (got '{shown}'); \
                         send the {noun} as a base64 data: URI, or start the server \
                         with --vision-allow-remote-images to enable fetching"
                    ),
                ));
            }
            let remote_kind = match kind {
                MediaKind::Image => super::super::remote_image::RemoteMediaKind::Image,
                MediaKind::Video => super::super::remote_image::RemoteMediaKind::Video,
            };
            match super::super::remote_image::fetch_media_as_data_uri(url, remote, remote_kind) {
                Ok(data_uri) => Ok(data_uri),
                Err(why) => {
                    // The reason is surfaced rather than flattened to "could
                    // not fetch": the operator turned this on deliberately,
                    // and "resolves to a private address" and "exceeds the
                    // size cap" call for different fixes.
                    tracing::warn!("remote {noun} fetch refused: {shown}: {why}");
                    Err(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("could not fetch {noun} URL '{shown}': {why}"),
                    ))
                }
            }
        }
    }
}
