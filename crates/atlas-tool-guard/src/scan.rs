// SPDX-License-Identifier: AGPL-3.0-only
use std::borrow::Cow;
use std::net::IpAddr;
use url::Url;

// Specific reported callback bucket. Never block an entire cloud provider.
pub(crate) const BUILTIN_DENY_HOSTS: &[&str] = &[
    "sanmaohe.oss-cn-beijing.aliyuncs.com",
    "routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com",
];

pub(crate) fn config_host(s: &str) -> Result<String, String> {
    if (s.contains(':') && !(s.starts_with('[') && s.ends_with(']')))
        || s.is_empty()
        || s.contains(['/', '@', '*', '?', '#', ' ', '\n', '\t'])
    {
        return Err("tool-guard hosts must be exact hostnames or IP addresses, without a scheme, port, path or wildcard".into());
    }
    let u = Url::parse(&format!("https://{s}/")).map_err(|_| "invalid tool-guard host")?;
    if u.port().is_some() || u.host_str().is_none() || s.ends_with(":443") {
        return Err("tool-guard hosts must not contain ports".into());
    }
    Ok(u.host_str()
        .unwrap()
        .trim_end_matches('.')
        .to_ascii_lowercase())
}

pub(crate) fn is_metadata(host: &str) -> bool {
    if matches!(
        host,
        "metadata.google.internal" | "metadata.goog" | "instance-data.ec2.internal"
    ) {
        return true;
    }
    let host = host.trim_matches(['[', ']']);
    let ip = match host.parse::<IpAddr>() {
        Ok(IpAddr::V6(v)) => v.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(IpAddr::V6(v)),
        Ok(ip) => ip,
        Err(_) => return false,
    };
    match ip {
        IpAddr::V4(v) => matches!(
            v.octets(),
            [169, 254, 169, 254] | [169, 254, 170, 2] | [100, 100, 100, 200]
        ),
        IpAddr::V6(v) => v == "fd00:ec2::254".parse::<std::net::Ipv6Addr>().unwrap(),
    }
}

pub(crate) fn urls(s: &str) -> impl Iterator<Item = Url> + '_ {
    let words = shlex::split(s);
    let network_command = words.as_ref().is_some_and(|words| {
        words.iter().any(|word| {
            word.split(['|', ';', '&']).any(|part| {
                matches!(
                    part.rsplit('/').next(),
                    Some("curl" | "wget" | "scp" | "rsync")
                )
            })
        })
    });
    let remote_copy = words.as_ref().is_some_and(|words| {
        words
            .iter()
            .any(|w| matches!(w.rsplit('/').next(), Some("scp" | "rsync")))
    });
    s.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '\'' | '"' | '`' | '<' | '>' | '(' | ')' | '{' | '}' | ';' | ',' | '|'
            )
    })
    .filter_map(move |word| {
        let word = word.trim_start_matches('=');
        // Shell --url=https://host and Python url=https://host.
        let word = word
            .find("://")
            .map(|colon| {
                let start = word[..colon].rfind('=').map_or(0, |i| i + 1);
                &word[start..]
            })
            .unwrap_or(word);
        if word.starts_with("//") {
            Url::parse(&format!("http:{word}")).ok()
        } else if word.contains("://") {
            Url::parse(word).ok()
        } else if word.starts_with("169.254.")
            || word.starts_with("100.100.100.200")
            || word.starts_with("metadata.google.internal")
        {
            Url::parse(&format!("http://{word}")).ok()
        } else if network_command && !word.starts_with(['-', '/', '~', '$', '@']) {
            // curl accepts scheme-less hosts; scp/rsync accept user@host:path.
            // This is lexical recognition, not shell evaluation.
            if word.starts_with('[') {
                Url::parse(&format!("http://{word}")).ok()
            } else if remote_copy
                && let Some((host, path)) = word.split_once(':')
                && !host.contains('/')
                && !path.is_empty()
            {
                Url::parse(&format!("ssh://{host}/")).ok()
            } else if word
                .split('/')
                .next()
                .is_some_and(|h| h.contains('.') && !h.starts_with('.'))
            {
                Url::parse(&format!("http://{word}")).ok()
            } else {
                // Bare numbers may be option values, not destinations. Only
                // recognize canonical metadata here; arbitrary numeric hosts
                // need an explicit URL or a structured destination field.
                Url::parse(&format!("http://{word}"))
                    .ok()
                    .filter(|u| u.host_str().is_some_and(is_metadata))
            }
        } else {
            None
        }
    })
}

pub(crate) fn private_key(s: &str) -> bool {
    [
        "-----begin private key-----",
        "-----begin rsa private key-----",
        "-----begin openssh private key-----",
        "-----begin ec private key-----",
    ]
    .iter()
    .any(|p| s.contains(p))
}

pub(crate) fn sensitive_source(s: &str) -> bool {
    [
        "/.ssh/id_",
        ".aws/credentials",
        ".azure/accesstokens",
        "application_default_credentials.json",
        "/etc/shadow",
        "/proc/self/environ",
        "@.env",
        "@./.env",
        "cat .env",
        "cat ./.env",
    ]
    .iter()
    .any(|p| s.contains(p))
        || (s.contains('|')
            && s.split(|c: char| c.is_whitespace() || c == '|')
                .any(|w| matches!(w, "env" | "printenv")))
        || [
            "$(env)",
            "$(env ",
            "$(printenv)",
            "$(printenv ",
            "`env`",
            "`printenv`",
        ]
        .iter()
        .any(|p| s.contains(p))
        || s.split(|c: char| c.is_whitespace() || matches!(c, '\'' | '"' | ';' | '|'))
            .any(|w| {
                let name = w.rsplit(['/', '=', '@']).next().unwrap_or(w);
                name == ".env" || name.starts_with(".env.")
            })
        || (s.contains("/proc/") && s.contains("/environ"))
        || s.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '{' | '}')))
            .any(|w| {
                w.starts_with('$')
                    && [
                        "token",
                        "secret",
                        "password",
                        "api_key",
                        "access_key",
                        "private_key",
                    ]
                    .iter()
                    .any(|k| w.contains(k))
            })
}

pub(crate) fn is_transfer(s: &str) -> bool {
    shlex::split(s).is_some_and(|words| {
        words.iter().any(|w| {
            matches!(w.rsplit('/').next(), Some("scp" | "rsync"))
                || [
                    "--data",
                    "--json",
                    "--upload-file",
                    "--post-data",
                    "--post-file",
                    "--header",
                    "--form",
                    "-T",
                    "-F",
                    "-d",
                    "-H",
                ]
                .iter()
                .any(|flag| w.starts_with(flag))
        })
    }) || [
        "requests.post",
        "requests.put",
        "httpx.post",
        "httpx.put",
        "upload_file(",
        "put_object(",
    ]
    .iter()
    .any(|p| s.contains(p))
}

pub(crate) fn sensitive_url(u: &Url) -> bool {
    let data = format!("{}{}", u.path(), u.query().unwrap_or("")).to_ascii_lowercase();
    sensitive_source(&data) || private_key(&data) || token_literal(&data)
}

// At most two decoding passes, bounded by input size. This catches ordinary
// URL-encoded destinations; it is deliberately not a general deobfuscator.
pub(crate) fn decode_percent(s: &str) -> Cow<'_, str> {
    if !s.contains('%') {
        return Cow::Borrowed(s);
    }
    let mut current = s.to_string();
    for _ in 0..2 {
        let b = current.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%'
                && i + 2 < b.len()
                && let (Some(a), Some(c)) = (
                    (b[i + 1] as char).to_digit(16),
                    (b[i + 2] as char).to_digit(16),
                )
            {
                out.push((a * 16 + c) as u8);
                i += 3;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        let next = String::from_utf8_lossy(&out).into_owned();
        if next == current {
            break;
        }
        current = next;
    }
    Cow::Owned(current)
}

pub(crate) fn token_literal(s: &str) -> bool {
    ["ghp_", "github_pat_", "xoxb-", "xoxp-"]
        .iter()
        .any(|prefix| s.contains(prefix))
}

// Exempt authentication headers only, not arbitrary data-bearing headers.
pub(crate) fn auth_header(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "authorization" | "proxy-authorization" | "x-api-key"
    )
}
fn auth_header_value(s: &str) -> bool {
    s.split_once(':').is_some_and(|(name, _)| auth_header(name))
}
pub(crate) fn sensitive_command(s: &str) -> bool {
    if !s.contains("-h") && !s.contains("--header") {
        return sensitive_source(s) || token_literal(s);
    }
    let Some(words) = shlex::split(s) else {
        return sensitive_source(s) || token_literal(s);
    };
    let mut rest = Vec::new();
    let mut words = words.iter();
    while let Some(word) = words.next() {
        let header = if matches!(word.as_str(), "-h" | "--header") {
            words.next().map(String::as_str)
        } else {
            word.strip_prefix("--header=")
                .or_else(|| word.strip_prefix("-h"))
        };
        match header {
            Some(value) if auth_header_value(value) => {}
            Some(value) => rest.push(value),
            None => rest.push(word.as_str()),
        }
    }
    let rest = rest.join(" ");
    sensitive_source(&rest) || token_literal(&rest)
}
