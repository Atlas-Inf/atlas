// SPDX-License-Identifier: AGPL-3.0-only

//! transformers' `{% generation %}` / `{% endgeneration %}` block marks the
//! assistant span for `return_assistant_tokens_mask` and renders its body
//! unchanged. minijinja has no such statement, so it becomes an always-true
//! `if` block with the same whitespace-control dashes (Thinking Machines
//! Inkling 2's template wraps each rendered assistant turn in one).

pub(super) fn neutralize_generation_tags(template: &str) -> String {
    let mut t = template.to_string();
    for (open, close) in [("{%", "%}"), ("{%-", "%}"), ("{%", "-%}"), ("{%-", "-%}")] {
        t = t.replace(
            &format!("{open} generation {close}"),
            &format!("{open} if true {close}"),
        );
        t = t.replace(
            &format!("{open} endgeneration {close}"),
            &format!("{open} endif {close}"),
        );
    }
    t
}

#[cfg(test)]
mod tests {
    use super::neutralize_generation_tags;

    fn render(t: &str) -> String {
        let mut env = minijinja::Environment::new();
        env.add_template("t", t).unwrap();
        env.get_template("t")
            .unwrap()
            .render(minijinja::context! { x => "hi" })
            .unwrap()
    }

    #[test]
    fn generation_blocks_render_their_body_with_the_same_whitespace_control() {
        let src = "a {%- generation -%} {{ x }} {%- endgeneration -%} b|{% generation %} {{ x }} {% endgeneration %}|";
        let t = neutralize_generation_tags(src);
        assert!(!t.contains("generation"));
        // as Jinja2 renders it: dashes trim around the block, plain tags keep the spaces
        assert_eq!(render(&t), "ahib| hi |");
    }

    #[test]
    fn templates_without_the_tag_are_unchanged() {
        let src = "{% for m in messages %}{{ m.content }}{% endfor %}";
        assert_eq!(neutralize_generation_tags(src), src);
    }
}
