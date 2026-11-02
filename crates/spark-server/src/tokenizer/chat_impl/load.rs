// SPDX-License-Identifier: AGPL-3.0-only
//! `ChatTokenizer::from_model_dir`/`from_model_dir_with` — tokenizer + template load (extracted; identical content).

use super::*;

impl ChatTokenizer {
    pub fn from_model_dir(
        model_dir: &Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        // MODEL.toml `[behavior].jinja_template`; empty = derive from
        // `model_type`.
        template_override: &str,
        repo_root: Option<&Path>,
        disable_template_overrides: bool,
        // MODEL.toml `[behavior].verbatim_prompt`: `tojson` spaced like
        // Python's `json.dumps`, as transformers renders it (otherwise the
        // process-wide `ATLAS_USE_HF_REF_JSON_DUMPS` choice), and an
        // effort-silent render keeps the template's own effort default.
        verbatim_prompt: bool,
    ) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // Template-source priority.
        //
        // Conceptually the default is now MODEL-FIRST: render off the
        // model's OWN `chat_template.jinja` / `tokenizer_config.json`.
        // Atlas's cross-cutting behaviors (autoclose-think,
        // think-control, F76 arg-parse) are applied in Rust
        // message-preprocessing (see `preprocess_for_render`), so a model
        // no longer needs a bespoke `jinja-templates/{model_type}.jinja`
        // override that is otherwise a byte-copy of its own template.
        // This is what makes `holo3_1_moe.jinja` REDUNDANT: Holo renders
        // correctly off its own template + Rust behaviors. (The override
        // file itself is still present for now only because
        // `tokenizer/tests.rs::render_holo_template_*` reads it directly;
        // it goes away together with those tests.)
        //
        // A `jinja-templates/{model_type}.jinja` override is OPT-IN by
        // FILE PRESENCE: dropping the file in is the explicit signal that
        // this model genuinely needs a template fix the Rust preprocessing
        // can't express (MiniMax's `_args.items()`, Gemma-4's
        // `strip_thinking`, etc.). We deliberately do NOT prefer the
        // model's own template when such a file exists — that would
        // silently undo those fixes. Instead, the operator opts OUT of all
        // overrides with `--disable-template-overrides`, which forces
        // every model onto its own template (relying purely on the Rust
        // behaviors).
        //
        // Priority (high → low):
        //   1. jinja-templates/<template> override, where <template> is
        //      MODEL.toml [behavior].jinja_template when set and
        //      {model_type}.jinja otherwise
        //      (opt-in: file present AND overrides not disabled)
        //   2. tokenizer_config.json / chat_template.jinja (the MODEL's own)
        //   3. Default ChatML fallback
        let override_tmpl = if disable_template_overrides {
            None
        } else {
            crate::tokenizer::jinja_helpers::load_override_template(
                model_type,
                template_override,
                repo_root,
            )
        };
        let chat_template = if let Some(override_tmpl) = override_tmpl {
            override_tmpl
        } else if let Some(config_tmpl) =
            crate::tokenizer::jinja_helpers::load_config_template(model_dir)?
        {
            config_tmpl
        } else {
            tracing::warn!("No chat template found — using default ChatML");
            crate::tokenizer::jinja_helpers::default_chatml_template(supports_thinking)
        };

        let build_env = |t: &str| {
            if verbatim_prompt {
                crate::tokenizer::jinja_helpers::build_jinja_env_with(
                    t,
                    crate::tokenizer::jinja_helpers::ToolJsonStyle::HfSpaced,
                )
            } else {
                crate::tokenizer::jinja_helpers::build_jinja_env(t)
            }
        };
        let jinja_env = build_env(&chat_template)?;

        // Load OpenAI-variant template if it exists (jinja-templates/openai/{model_type}.jinja).
        // This variant gates historical <think> wrappers on enable_thinking, preventing
        // spontaneous thinking during tool-use when thinking is disabled.
        //
        // It lives under `jinja-templates/` too, so `--disable-template-overrides`
        // skips it: the flag promises the MODEL's own template, and the chat
        // completions path renders through this variant whenever it exists. Before
        // this, Qwen3.6-35B kept the qwen3_5_moe variant under the flag — and that
        // variant ignores `preserve_thinking`, dropping prior-turn reasoning.
        let openai_jinja_env = if disable_template_overrides {
            None
        } else {
            crate::tokenizer::jinja_helpers::load_openai_template(model_type, repo_root).and_then(
                |tmpl| {
                    tracing::info!("Loaded OpenAI-variant Jinja template for {model_type}");
                    build_env(&tmpl).ok()
                },
            )
        };
        let chat_encoding = if model_type == "deepseek_v4" {
            tracing::info!("Using checkpoint-native DeepSeek-V4 message encoding");
            ChatEncoding::DeepseekV4
        } else {
            ChatEncoding::Jinja
        };

        // nemotron_h_omni: `<image>`(18) placeholders expand to
        // `<img>`(19) + N x id-18 + `</img>`(20) per the checkpoint's
        // processing_nemotron_h_omni.py. Resolve the three ids once.
        let omni_image_tokens = if model_type == "nemotron_h_omni" {
            let e = |t: &str| -> Option<u32> {
                tokenizer.encode(t, false).ok().and_then(|enc| {
                    let ids = enc.get_ids();
                    (ids.len() == 1).then(|| ids[0])
                })
            };
            match (e("<image>"), e("<img>"), e("</img>")) {
                (Some(img), Some(start), Some(end)) => Some((img, start, end)),
                _ => anyhow::bail!(
                    "nemotron_h_omni tokenizer is missing <image>/<img>/</img> special tokens"
                ),
            }
        } else {
            None
        };

        tracing::info!("Loaded tokenizer from {}", tokenizer_path.display());
        Ok(Self {
            tokenizer,
            large_encode_gate: crate::tokenizer::large_encode::configured(
                std::env::var("ATLAS_LARGE_ENCODE_CONCURRENCY")
                    .ok()
                    .as_deref(),
            )
            .map_err(anyhow::Error::msg)?,
            ids_only: crate::tokenizer::ids_only::configured(
                std::env::var("ATLAS_TOKENIZER_IDS_ONLY").ok().as_deref(),
            )
            .map_err(anyhow::Error::msg)?,
            eos_token_id,
            supports_thinking,
            glm_tool_markup_as_text: matches!(model_type, "glm5_next" | "qwen4_exp"),
            tool_call_ids_in_template: model_type == "glm5_next"
                || chat_encoding == ChatEncoding::DeepseekV4,
            chat_encoding,
            omni_image_tokens,
            chat_template,
            jinja_env,
            openai_jinja_env,
            verbatim_prompt,
        })
    }
}
