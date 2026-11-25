// SPDX-License-Identifier: AGPL-3.0-only

//! `tojson` with Python keyword arguments, for templates that pass them.
//!
//! transformers renders chat templates with its own `tojson(x, ensure_ascii=False,
//! indent=None, separators=None, sort_keys=False)`, which is `json.dumps` with
//! those arguments. minijinja's builtin takes none of them, and Atlas keeps the
//! builtin for a bare `| tojson` (see `build_jinja_env`, ST-995). So only calls
//! that pass `sort_keys` or `separators` (Inkling 2's tool block, its tool-call
//! names and argument values) are routed to [`tojson_py`], which serializes as
//! `json.dumps` does: the given separators (default `", "` / `": "`), keys in
//! insertion order or sorted, non-ASCII kept unless `ensure_ascii`, and no HTML
//! escaping.

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Error, ErrorKind};

const KWARG_NAMES: [&str; 2] = ["sort_keys", "separators"];

/// Rename `tojson(...)` calls whose arguments name `sort_keys` or `separators`
/// to `tojson_py(...)`; every other `tojson` call is left as written.
pub(super) fn route_kwarg_calls(template: &str) -> String {
    const CALL: &str = "tojson(";
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(at) = rest.find(CALL) {
        let args_start = at + CALL.len();
        let args = &rest[args_start..args_start + args_len(&rest[args_start..])];
        out.push_str(&rest[..at]);
        out.push_str(if KWARG_NAMES.iter().any(|k| args.contains(k)) {
            "tojson_py("
        } else {
            CALL
        });
        rest = &rest[args_start..];
    }
    out.push_str(rest);
    out
}

/// Bytes up to the `)` closing a call whose arguments start `s`, skipping
/// nested parentheses and quoted strings (all of `s` if unclosed).
fn args_len(s: &str) -> usize {
    let (mut depth, mut quote, mut escaped) = (0usize, None::<char>, false);
    for (i, c) in s.char_indices() {
        match quote {
            Some(_) if escaped => escaped = false,
            Some(_) if c == '\\' => escaped = true,
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None => match c {
                '\'' | '"' => quote = Some(c),
                '(' => depth += 1,
                ')' if depth == 0 => return i,
                ')' => depth -= 1,
                _ => {}
            },
        }
    }
    s.len()
}

/// `json.dumps(value, ensure_ascii=…, separators=…, sort_keys=…)`.
pub(super) fn tojson_py(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let separators = kwargs.get::<Option<Vec<String>>>("separators")?;
    if kwargs
        .get::<Option<Value>>("indent")?
        .is_some_and(|v| !v.is_none())
    {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson: indent is not supported",
        ));
    }
    kwargs.assert_all_used()?;
    let (item, key) = match separators.as_deref() {
        None => (", ".to_string(), ": ".to_string()),
        Some([item, key]) => (item.clone(), key.clone()),
        Some(_) => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "tojson: separators must be (item, key)",
            ));
        }
    };
    let mut out = String::new();
    write_json(&value, sort_keys, (&item, &key), &mut out)?;
    if ensure_ascii {
        out = escape_non_ascii(&out);
    }
    Ok(Value::from_safe_string(out))
}

fn write_json(v: &Value, sort: bool, seps: (&str, &str), out: &mut String) -> Result<(), Error> {
    match v.kind() {
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if sort {
                keys.sort_by_key(|k| k.to_string());
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(seps.0);
                }
                out.push_str(&scalar(&Value::from(k.to_string()))?);
                out.push_str(seps.1);
                write_json(&v.get_item(k)?, sort, seps, out)?;
            }
            out.push('}');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, item) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(seps.0);
                }
                write_json(&item, sort, seps, out)?;
            }
            out.push(']');
        }
        _ => out.push_str(&scalar(v)?),
    }
    Ok(())
}

fn scalar(v: &Value) -> Result<String, Error> {
    serde_json::to_string(v).map_err(|e| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("tojson serialization failed: {e}"),
        )
    })
}

/// `\uXXXX` (surrogate pairs past the BMP) for every non-ASCII character.
fn escape_non_ascii(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{u:04x}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(src: &str, ctx: Value) -> String {
        let src = route_kwarg_calls(src);
        let mut env = minijinja::Environment::new();
        env.add_filter("tojson_py", tojson_py);
        env.add_template("t", &src).unwrap();
        env.get_template("t").unwrap().render(ctx).unwrap()
    }

    #[test]
    fn only_kwarg_calls_are_routed() {
        let t = r#"{{ a | tojson }}{{ b | tojson(sort_keys=true, separators=(",", ":")) }}{{ c | tojson(ensure_ascii=False) }}"#;
        assert_eq!(
            route_kwarg_calls(t),
            r#"{{ a | tojson }}{{ b | tojson_py(sort_keys=true, separators=(",", ":")) }}{{ c | tojson(ensure_ascii=False) }}"#
        );
    }

    #[test]
    fn matches_python_json_dumps() {
        let ctx = minijinja::context! {
            tool => serde_json::json!({"name": "f", "parameters": {"type": "object", "b": [1, 2.5], "a": "<é>"}}),
        };
        // json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert_eq!(
            render(
                r#"{{ tool | tojson(sort_keys=true, separators=(",", ":")) }}"#,
                ctx.clone()
            ),
            r#"{"name":"f","parameters":{"a":"<é>","b":[1,2.5],"type":"object"}}"#
        );
        // json.dumps(tool["parameters"], separators=(",", ":")): insertion order
        assert_eq!(
            render(
                r#"{{ tool.parameters | tojson(separators=(",", ":")) }}"#,
                ctx.clone()
            ),
            r#"{"type":"object","b":[1,2.5],"a":"<é>"}"#
        );
        // json.dumps(..., ensure_ascii=True, separators=(",", ":"))
        assert_eq!(
            render(
                r#"{{ tool.parameters.a | tojson(separators=(",", ":"), ensure_ascii=true) }}"#,
                ctx
            ),
            "\"<\\u00e9>\""
        );
    }
}
