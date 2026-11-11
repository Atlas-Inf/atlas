// SPDX-License-Identifier: AGPL-3.0-only
//! Recognized argument fields, scoped to one operation. Opaque document fields
//! are data, never recursively interpreted as destinations or commands.
use super::{BlockReason, Policy, scan};
use serde_json::Value;
use url::Url;

fn destination(key: &str) -> bool {
    matches!(
        key,
        "url" | "uri" | "endpoint" | "host" | "hostname" | "server" | "target" | "destination"
    )
}
fn command(key: &str) -> bool {
    matches!(
        key,
        "command" | "cmd" | "script" | "shell" | "exec" | "argv"
    )
}
fn payload(key: &str) -> bool {
    matches!(
        key,
        "body" | "data" | "json" | "payload" | "query" | "params" | "form" | "content"
    )
}
fn source(key: &str) -> bool {
    matches!(
        key,
        "file"
            | "files"
            | "file_path"
            | "filepath"
            | "source"
            | "source_path"
            | "local_path"
            | "input_file"
            | "upload_file"
    )
}
fn opaque(key: &str) -> bool {
    payload(key)
        || source(key)
        || matches!(
            key,
            "headers" | "text" | "code" | "note" | "notes" | "description" | "message" | "markdown"
        )
}
fn visit_strings(value: &Value, f: &mut impl FnMut(&str)) {
    match value {
        Value::String(s) => f(s),
        Value::Array(a) => a.iter().for_each(|v| visit_strings(v, f)),
        _ => {}
    }
}
fn sensitive_text(s: &str) -> bool {
    let decoded = scan::decode_percent(s);
    let lower = decoded.to_ascii_lowercase();
    scan::sensitive_source(&lower) || scan::private_key(&lower) || scan::token_literal(&lower)
}
fn sensitive_payload(value: &Value) -> bool {
    match value {
        Value::Object(o) => o.iter().any(|(key, v)| {
            (matches!(
                key.to_ascii_lowercase().as_str(),
                "api_key" | "access_token" | "secret_key" | "private_key" | "password"
            ) && !v.is_null())
                || sensitive_payload(v)
        }),
        Value::Array(a) => a.iter().any(sensitive_payload),
        Value::String(s) => sensitive_text(s),
        _ => false,
    }
}
fn sensitive_headers(value: &Value) -> bool {
    let Value::Object(headers) = value else {
        return sensitive_payload(value);
    };
    headers.iter().any(|(name, v)| {
        if scan::auth_header(name) {
            // Preserve normal API authentication, but never treat a private-key
            // block as an authentication-token exemption.
            let mut private_key = false;
            visit_strings(v, &mut |s| {
                private_key |= scan::private_key(&s.to_ascii_lowercase())
            });
            private_key
        } else {
            sensitive_payload(v)
        }
    })
}
fn quoted_arguments(value: &Value) -> String {
    let mut parts = Vec::new();
    visit_strings(value, &mut |s| {
        parts.push(format!("'{}'", s.replace('\'', "'\\''")))
    });
    parts.join(" ")
}

// Only known command-object members are executable; other object data remains opaque.
fn command_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(_) => quoted_arguments(value),
        Value::Object(o) => {
            let mut parts = Vec::new();
            for (key, v) in o {
                let key = key.to_ascii_lowercase();
                if command(&key) || matches!(key.as_str(), "program" | "executable") {
                    parts.push(command_text(v));
                }
            }
            for (key, v) in o {
                if key.eq_ignore_ascii_case("args") || key.eq_ignore_ascii_case("arguments") {
                    parts.push(quoted_arguments(v));
                }
            }
            parts.join(" ")
        }
        _ => String::new(),
    }
}
fn visit_destinations(value: &Value, f: &mut impl FnMut(&str)) {
    match value {
        Value::String(s) => f(s),
        Value::Array(a) => a.iter().for_each(|v| visit_destinations(v, f)),
        Value::Object(o) => {
            for (key, v) in o {
                if destination(&key.to_ascii_lowercase()) || key.eq_ignore_ascii_case("href") {
                    visit_destinations(v, f);
                }
            }
        }
        _ => {}
    }
}

impl Policy {
    pub(super) fn inspect_operation(&self, value: &Value) -> Result<(), BlockReason> {
        match value {
            Value::Array(a) => {
                for operation in a {
                    self.inspect_operation(operation)?;
                }
            }
            Value::Object(object) => {
                let mut untrusted = false;
                for (key, value) in object {
                    let key = key.to_ascii_lowercase();
                    if command(&key) {
                        let mut text = command_text(value);
                        if key != "argv" {
                            for (k, v) in object {
                                if k.eq_ignore_ascii_case("args")
                                    || k.eq_ignore_ascii_case("arguments")
                                {
                                    text.push(' ');
                                    text.push_str(&quoted_arguments(v));
                                }
                            }
                        }
                        self.inspect_command(&text)?;
                    } else if destination(&key) {
                        let mut outcome = Ok(());
                        visit_destinations(value, &mut |text| {
                            if outcome.is_err() {
                                return;
                            }
                            let decoded = scan::decode_percent(text);
                            let text = decoded.trim().trim_start_matches("//");
                            let url = if text.contains("://") {
                                Url::parse(text)
                            } else {
                                Url::parse(&format!("http://{text}"))
                            };
                            if let Ok(url) = url {
                                match self.inspect_destination(&url) {
                                    Ok(found) => untrusted |= found,
                                    Err(reason) => outcome = Err(reason),
                                }
                            }
                        });
                        outcome?;
                    }
                }
                if untrusted
                    && object.iter().any(|(key, v)| {
                        let key = key.to_ascii_lowercase();
                        if key == "headers" {
                            sensitive_headers(v)
                        } else {
                            (payload(&key) || source(&key)) && sensitive_payload(v)
                        }
                    })
                {
                    return Err(BlockReason::CredentialTransfer);
                }
                for (key, child) in object {
                    let key = key.to_ascii_lowercase();
                    if !opaque(&key) && !command(&key) && !destination(&key) {
                        self.inspect_operation(child)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
