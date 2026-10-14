// SPDX-License-Identifier: AGPL-3.0-only
//
// `MsgEntry` + the pre-loop builder that turns the inbound
// `ChatCompletionRequest.messages` into the local representation
// used by every downstream phase (json_messages, loop detector,
// task pin, observation mask, …).
//
// Lifted out of `chat::chat_completions_inner` (wave 4g) so the
// orchestrator stays under the 500-LoC cap.

use axum::http::StatusCode;
use axum::response::Response;

use atlas_core::config::VisionConfig;

use crate::ir::{ContentPart, ImageData, MediaKind, Message, Role};

use super::super::compact::openai_error_response;
use crate::tokenizer::Glm5VisionPlaceholder;

mod glm5_vision;
#[cfg(test)]
use glm5_vision::checked_glm5_video_timestamps;

/// What the request path needs to decode a video: the operator's subprocess
/// policy and the sampling rate. Bundled so the signature does not grow two
/// more positional parameters that are always passed together.
pub(crate) struct VideoDecode<'a> {
    pub(crate) ffmpeg: &'a spark_model::video_decode_ffmpeg::FfmpegPolicy,
    pub(crate) fps: f32,
}

/// Per-message data: role, content text, optional structured
/// `tool_calls`, and image-part count for the Jinja vision-marker
/// expansion. `pub(super)` so `chat::chat_completions_inner` and
/// the other `chat/*` sub-files can read every field.
pub(crate) struct MsgEntry {
    pub(super) role: String,
    pub(super) content: String,
    /// Structured tool_calls for the Jinja template (arguments
    /// pre-parsed to dicts).
    pub(super) tool_calls: Option<Vec<serde_json::Value>>,
    /// Links a tool-result message to the assistant call it answers.
    pub(super) tool_call_id: Option<String>,
    /// This message's media parts, tagged and in the order the client
    /// sent them. When non-empty the json_messages builder emits a
    /// structured content array so the Jinja template can render one
    /// `<|vision_start|><|image_pad|><|vision_end|>` (or `<|video_pad|>`)
    /// marker per entry, in this order.
    ///
    /// A `(image_count, video_count)` pair lived here before and could not
    /// express order, so the markers came out grouped by modality — see
    /// [`crate::ir::Message::media_kinds`].
    pub(super) media: Vec<MediaKind>,
    /// Historical reasoning trace from a prior assistant turn (the
    /// `<think>...</think>` body). Forwarded from `IncomingMessage`
    /// and passed to the Jinja template so the template can
    /// rehydrate the historical `<think>` block. Empty/None ⇒ no
    /// `<think>` block emitted for this message — prevents the
    /// empty-`<think></think>` poisoning pattern that triggers
    /// premature `<|im_end|>` (vLLM/SGLang #131, MLC commit d75d64e).
    pub(super) reasoning_content: Option<String>,
}

/// Outputs of [`build_msg_entries`]. Bundled as a struct because
/// the caller threads each field through five later phases.
pub(super) struct BuildOut {
    pub(super) messages: Vec<MsgEntry>,
    pub(super) cwd_hint: Option<String>,
    pub(super) image_pixels: Vec<spark_model::VisionItem>,
    pub(super) image_pad_counts: Vec<usize>,
    /// GLM-5-specific placeholder geometry. Empty for Qwen/text families;
    /// the template renderer uses this to expand GLM's image/video marker
    /// triples after Jinja rendering.
    pub(super) vision_placeholders: Vec<Glm5VisionPlaceholder>,
}

/// One media item on its way to the vision path: what it is, and the
/// encoder-input string it resolved to (always a data: URI or raw base64 by
/// the time it is in here — remote URLs are fetched or refused at
/// collection time).
struct MediaInput {
    kind: MediaKind,
    uri: String,
}

#[allow(clippy::result_large_err)]
pub(super) fn build_msg_entries(
    admit: super::vision_admit::AdmitCtx<'_>,
    vision_config: Option<&VisionConfig>,
    vision_max_pixels: Option<usize>,
    remote_images: &super::remote_image::RemoteImagePolicy,
    video: &VideoDecode<'_>,
    input: &[Message],
    tools_active: bool,
    levers: &super::levers::ChatLevers,
    preserve_developer_role: bool,
) -> Result<BuildOut, Response> {
    let mut messages: Vec<MsgEntry> = Vec::with_capacity(input.len());
    let mut media: Vec<MediaInput> = Vec::new();
    let mut image_pad_counts: Vec<usize> = Vec::new();
    let mut vision_placeholders: Vec<Glm5VisionPlaceholder> = Vec::new();
    let mut consecutive_tool_errors: u32 = 0;
    // BW1 bash-wandering watchdog: tally tool-call productivity across the
    // conversation so a steering nudge can fire if the agent explores/runs
    // many commands without ever writing the deliverable (gap #9).
    let mut total_tool_calls: usize = 0;
    let mut productive_tool_calls: usize = 0;
    // P1-6 (2026-07-09): (index into `messages`, pre-hint original
    // text) for every tool-result entry, for duplicate-error masking
    // after the loop. Comparison must see the ORIGINAL text — the
    // injected hints vary with the escalation counter and would break
    // exact-match grouping.
    let mut tool_result_originals: Vec<(usize, String)> = Vec::new();
    // Atlas's prompt aids below (tool-result markers and hints, the cwd hint,
    // the content-free system drop); none for a verbatim-prompt model.
    let aids = !levers.verbatim_prompt;

    // F6 (2026-05-26): `last_query_index` was previously used to gate
    // an empty `<think>\n\n</think>\n\n` injection for historical
    // assistant turns. The Jinja template already does this gating
    // itself (via its own `ns.last_query_index` computation) and the
    // injection here was the source of empty-think poisoning. Removed.
    for m in input.iter() {
        let mut text = m.text();
        // F6: a failed tool result (Anthropic `is_error`, carried as
        // `Message::tool_error`) gets an explicit ASCII marker — chat-tuned
        // models have no structural error concept and otherwise hallucinate
        // success over error text. Rendered here (not in the adapters) so
        // every surface gets identical prompt bytes, and applied before the
        // error-hint scan below so hints see the final text.
        if m.tool_error && aids {
            text = format!("[tool error]\n{text}");
        }

        // Preserve structured tool_calls for the Jinja template.
        // Always extract from assistant messages — past turns may
        // carry tool_calls that the template MUST render even when
        // the current request didn't pass `tools`. `tc.arguments` is
        // already structured JSON in the IR (parsed at the adapter
        // boundary), so we forward it directly.
        let tool_calls_json = if m.role == Role::Assistant && !m.tool_calls.is_empty() {
            let parsed: Vec<serde_json::Value> = m
                .tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    })
                })
                .collect();
            Some(parsed)
        } else {
            None
        };

        // BW1: tally tool-call productivity (write/edit/build-run vs explore).
        if m.role == Role::Assistant && !m.tool_calls.is_empty() {
            for tc in &m.tool_calls {
                total_tool_calls += 1;
                if crate::hint_injector::tool_call_is_productive(&tc.name, &tc.arguments) {
                    productive_tool_calls += 1;
                }
            }
        }

        // Tool-response messages: pass raw content; Jinja template
        // handles `<tool_response>` wrapping and consecutive
        // grouping.
        if tools_active && m.role == Role::Tool {
            let mut text = text;
            // P1-6 (2026-07-09): record the pre-hint original at the
            // index this entry is about to occupy.
            tool_result_originals.push((messages.len(), text.clone()));
            if aids && crate::hint_injector::looks_like_error(&text) {
                consecutive_tool_errors += 1;
                crate::hint_injector::inject_hints(&mut text, consecutive_tool_errors);
            } else {
                consecutive_tool_errors = 0;
            }
            messages.push(MsgEntry {
                role: "tool".into(),
                content: text,
                tool_calls: None,
                tool_call_id: m.tool_call_id.clone(),
                media: m.media_kinds(),
                reasoning_content: None,
            });
            media::collect_message_media(m, &mut media, &mut image_pad_counts, remote_images)?;
            continue;
        }

        // Wave 3 (2026-05-26): `ATLAS_STRIP_REASONING_HISTORY=1` drops
        // historical reasoning_content entirely. Matches MLC commit
        // d75d64e (Apr 2026) `strip_reasoning_in_history` for qwen3,
        // whose PR description matches Atlas's Wave-1 failure mode
        // verbatim: echoing prior `<think>` traces makes the next turn
        // emit `<|im_end|>` prematurely AND seeds loop-attractor drift
        // on prior-failed-attempt token patterns (the `lean://` loop
        // observed in the Wave-1 opencode probe).
        let strip_reasoning = std::env::var("ATLAS_STRIP_REASONING_HISTORY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        // OpenAI's `developer` role is the successor of `system` (o-series
        // clients send it). Normalize at build time so every downstream
        // system-message scan (cwd hint, CWD injection, vacuous-strip) and
        // the template see one canonical role — previously the mapping
        // happened only at JSON render time, so `developer` messages
        // bypassed those scans.
        let role = match &m.role {
            Role::Other(r) if r == "developer" && !preserve_developer_role => "system".to_string(),
            r => r.as_wire().to_string(),
        };
        messages.push(MsgEntry {
            role,
            content: text,
            tool_calls: tool_calls_json,
            tool_call_id: m.tool_call_id.clone(),
            media: m.media_kinds(),
            // F1: forward reasoning_content for assistant messages only.
            // Wave 3: when strip_reasoning=true, drop it for ALL turns,
            // forcing the template back to the pre-F1 "clean content
            // only" rendering shape — but without re-introducing the
            // empty-`<think>\n\n</think>\n\n` poisoning, because F6's
            // template change skips the wrapper when reasoning_content
            // is empty.
            reasoning_content: if m.role == Role::Assistant && !strip_reasoning {
                m.reasoning
                    .as_ref()
                    .map(|r| r.text.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            },
        });
        media::collect_message_media(m, &mut media, &mut image_pad_counts, remote_images)?;
    }

    // P1-6 (2026-07-09): duplicate-error observation masking
    // (arXiv:2508.21433 pattern). Feeding an identical error text back
    // verbatim N times reinforces the failing-call attractor (45k
    // collapse: 6x "BadResource: FileSystem.readFile (/home/nologik)").
    // Mask the OLDER occurrences of a repeated error-shaped tool
    // result, keeping only the NEWEST verbatim (hints attach to the
    // newest, which is preserved untouched). Kill-switch
    // ATLAS_NO_ERROR_DEDUP=1 restores verbatim history. MUST run
    // before the vacuous-system removal below — the recorded indices
    // refer to the un-shifted `messages` vec.
    if tools_active && aids && !tail::error_dedup_disabled() {
        for (idx, replacement) in tail::duplicate_error_masks(&tool_result_originals) {
            messages[idx].content = replacement;
        }
    }

    // Extract working directory from the system message if present.
    let cwd_hint: Option<String> = messages.iter().find(|m| m.role == "system").and_then(|m| {
        for line in m.content.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("working directory")
                || lower.contains("working_directory")
                || lower.contains("cwd:"))
                && let Some(pos) = line.find(':')
            {
                let path = line[pos + 1..]
                    .trim()
                    .trim_matches(|c| c == '`' || c == '"' || c == '\'');
                if !path.is_empty() {
                    return Some(path.to_string());
                }
            }
        }
        None
    });

    // Inject CWD hint into the system message (NOT tool definitions —
    // those go to the Jinja template).
    if tools_active
        && aids
        && !levers.disable_cwd_hint_injection
        && let Some(ref cwd) = cwd_hint
    {
        let hints = format!("\n<environment>\nworking_directory: {cwd}\n</environment>");
        if let Some(first) = messages.first_mut()
            && first.role == "system"
        {
            first.content.push_str(&hints);
        }
    }

    // Neutralize a content-free leading system message. Clients (notably
    // Open WebUI's empty RAG/context template) inject a system message
    // carrying NO instruction — e.g. `"User Context:\n\n"` (trims to the
    // bare label `User Context:`). Models react to a content-free system
    // directive by producing terse / prematurely-terminated output
    // (isolated 2026-05-17: removing it 3x'd generation length on the
    // 3D-chess prompt). We can't fix the client, so Atlas adapts: treat
    // such a message as absent so a degenerate client prompt can't poison
    // generation. Conservative — only an empty body or a single short
    // bare `Label:` line qualifies; any substantive prompt is untouched.
    if aids
        && messages
            .first()
            .is_some_and(|m| m.role == "system" && tail::is_vacuous_system_content(&m.content))
    {
        let removed = messages.remove(0);
        tracing::info!(
            dropped = %removed.content.trim(),
            "Dropped content-free client system message (would bias the model toward terse output)"
        );
    }

    // Preprocess the media. One shared fail-fast point: if media was
    // supplied but the model has no vision encoder, reject the request
    // (issue #165) instead of silently dropping the user's input with a
    // 200 — the old text-only behavior lost images without any signal.
    //
    // ONE loop over the collected sequence, dispatching per item, so
    // `image_pixels[i]`, `image_pad_counts[i]` and the i-th rendered marker
    // are the same item by construction. Preprocessing images and videos in
    // separate passes would re-introduce the grouping this fix removed.
    let mut image_pixels: Vec<spark_model::VisionItem> = Vec::new();
    let has_video = media.iter().any(|m| m.kind == MediaKind::Video);
    let mut admission = super::vision_admit::Admission::begin(admit, media.len(), has_video)?;
    if !media.is_empty() {
        let Some(vcfg) = vision_config else {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "this model does not accept image or video input (no vision config)".to_string(),
            ));
        };
        let reserve_decode = media.iter().any(|m| m.kind == MediaKind::Video)
            && !vcfg.is_glm5_next
            && video.ffmpeg.max_total_decoded_pixels.is_some();
        for (idx, input) in media.iter().enumerate() {
            let (item, video_timestamps) = match input.kind {
                MediaKind::Image => {
                    let item = if let Some(omni) = &vcfg.omni {
                        spark_model::radio_preprocess::preprocess_image_omni(
                            &input.uri,
                            vcfg,
                            &omni.norm_mean,
                            &omni.norm_std,
                            vcfg.patch_size,
                            omni.downsample_ratio,
                            omni.max_num_patches,
                            omni.min_num_patches,
                        )
                        .map(|(pixels, gh, gw)| spark_model::VisionItem::image(pixels, gh, gw))
                    } else {
                        spark_model::vision_preprocess::preprocess_image_with_max_pixels(
                            &input.uri,
                            vcfg,
                            vision_max_pixels,
                        )
                        .map(|(pixels, gh, gw)| spark_model::VisionItem::image(pixels, gh, gw))
                    };
                    match item {
                        Ok(item) => (item, Vec::new()),
                        Err(e) => {
                            return Err(openai_error_response(
                                StatusCode::BAD_REQUEST,
                                format!("Image decode error: {e}"),
                            ));
                        }
                    }
                }
                MediaKind::Video => {
                    if reserve_decode {
                        let ceiling =
                            media::qwen_video_rows_ceiling(vcfg, vision_max_pixels, video);
                        admission.reserve_video(ceiling)?;
                    }
                    match spark_model::vision_preprocess::decode_gated(|| {
                        spark_model::video_preprocess::preprocess_video(
                            &input.uri,
                            vcfg,
                            vision_max_pixels,
                            video.fps,
                            video.ffmpeg,
                        )
                    }) {
                        Ok(v) => {
                            let timestamps = if vcfg.is_qwen3vl_video {
                                v.timestamps_hf.clone()
                            } else {
                                v.timestamps.clone()
                            };
                            (
<<<<<<< HEAD
                                spark_model::VisionItem {
                                    held: spark_model::vision_budget::HeldRows::default(),
                                    groups: v.groups,
                                    grid_h: v.grid_h,
                                    grid_w: v.grid_w,
                                },
=======
                                spark_model::VisionItem::video(v.groups, v.grid_h, v.grid_w),
>>>>>>> ab671c1e3 (prefix cache: content-keyed vision prompts (issue #58))
                                timestamps,
                            )
                        }
                        Err(e) => {
                            return Err(openai_error_response(
                                StatusCode::BAD_REQUEST,
                                format!("Video decode error: {e:#}"),
                            ));
                        }
                    }
                }
            };
            image_pad_counts[idx] = item.pad_count(vcfg.spatial_merge_size);
            admission.note(image_pad_counts[idx])?;
            if reserve_decode {
                admission.reconcile_video(image_pad_counts[idx])?;
            }
            if vcfg.is_glm5_next || vcfg.is_qwen3vl_video {
                vision_placeholders.push(glm5_vision::placeholder(
                    input.kind,
                    &item,
                    image_pad_counts[idx],
                    video_timestamps,
                    vcfg.is_qwen3vl_video,
                )?);
            }
            if input.kind == MediaKind::Video {
                // Logged at the media index, not a video ordinal: the index
                // is what lines the clip up with its pad run and its
                // encoder rows.
                tracing::info!(
                    "Video (media item {}): {} temporal groups, {}x{} patches, {} vision tokens",
                    idx,
                    item.t_len(),
                    item.grid_h,
                    item.grid_w,
                    image_pad_counts[idx],
                );
            }
            image_pixels.push(item);
        }
    }
    let admitted = admission.admit()?;
    for item in image_pixels.iter_mut() {
        *item = admitted.attach(std::mem::replace(
            item,
            spark_model::VisionItem::image(Vec::new(), 0, 0),
        ));
    }

    // BW1 bash-wandering watchdog: if the agent has run many tool calls with
    // no productive file output, append a steering nudge to the most recent
    // tool response (what the model reads just before its next action). Gated
    // by ATLAS_BASH_WANDER_WATCHDOG (PCND, default-off).
    if tools_active
        && aids
        && let Some(hint) = crate::hint_injector::bash_wander_hint(
            total_tool_calls,
            productive_tool_calls,
            levers.bash_wander,
        )
        && let Some(last_tool) = messages.iter_mut().rev().find(|e| e.role == "tool")
    {
        last_tool.content.push_str(&hint);
    }

    Ok(BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
        vision_placeholders,
    })
}

#[path = "msg_entry_tests.rs"]
mod msg_entry_tests;

// A sibling file rather than a module inside `msg_entry_tests.rs`: these cases
// carried that file past the 500-LoC cap.
#[path = "msg_entry/media.rs"]
mod media;
#[cfg(test)]
#[path = "msg_entry_media_order_tests.rs"]
mod msg_entry_media_order_tests;

#[path = "msg_entry/tail.rs"]
mod tail;
