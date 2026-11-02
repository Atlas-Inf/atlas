<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Tool-call guard

Atlas blocks suspicious data-exfiltration tool calls by default. The same CPU
policy runs after tool parsing for Qwen and GLM, before a call is released to
OpenAI Chat Completions, Responses, or Anthropic clients. It needs no additional
model, GPU pass, network service, or changes to sampling/speculative decoding.

## Default checks

- Literal cloud metadata destinations, including IPv4-mapped IPv6 and numeric
  IPv4 URL forms normalized by the URL parser.
- The exact callback hosts listed in `crates/atlas-tool-guard/src/scan.rs`.
  These are specific reported indicators, not a cloud-provider or country ban.
- Recognizable credential-file/environment transfers in upload commands, private
  key material and structured credential payloads sent to a URL, and recognizable
  credentials in URL paths/queries. The standard
  Authorization, Proxy-Authorization and X-API-Key headers are allowed.
- Invalid JSON arguments, incomplete recognized streamed calls, or calls exceeding
  the configured byte limit.

Rules examine tool arguments only. Ordinary assistant text and reasoning are not
scanned. A native tool call recovered from reasoning by an existing parser still
passes through the guard. Mere prose describing a command remains prose.

The policy parses JSON escapes and up to two layers of percent encoding. It does
not execute or interpret shell, Python, JavaScript, base64 payloads, or arbitrary
programs. A URL hidden in dynamic code or split across separate calls can evade
these heuristics. An unlisted destination is not proof of safety. Network/DNS
access, redirects, rendering, and code executed later remain the responsibility
of the tool runner and application sandbox. Legacy `/v1/completions` returns text
and is outside this tool-call guard.

## Configuration

These are server startup flags, not request parameters:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tool-guard on\|off` | `on` | Enable or explicitly disable filtering |
| `--tool-guard-allow-host HOST` | none | Exact exception for callback and credential-transfer rules |
| `--tool-guard-deny-host HOST` | none | Additional exact blocked destination |
| `--tool-guard-max-bytes BYTES` | `1048576` | Per-call arguments and aggregate pending-call limit |

Host flags may repeat or contain comma-separated hosts. Wildcards, schemes,
ports and paths are rejected at startup. Host exceptions do not authorize
subdomains. Explicit denials and metadata checks take precedence over exceptions.
A deliberate metadata-access workflow requires disabling the guard; use a
separate, restricted endpoint for that workflow. The byte limit must be between
1 and 67108864. Invalid configuration fails before model loading.

Example: allow an organization's upload endpoint and deny another exact host:

```sh
spark serve /models/checkpoint \
  --tool-guard-allow-host uploads.example.org \
  --tool-guard-deny-host blocked.example.org
```

## Streaming and refusals

Each tool call's header and arguments are withheld until the complete call passes.
The limit bounds pending argument bytes, not total process memory; parsing and
serialization retain a small number of bounded copies. Argument fragments are
coalesced so buffering does not allocate an object per token. Regular text still streams normally. Buffering delays tool-argument
visibility until the closing tool marker; it adds no model inference.

On rejection, Atlas discards the pending calls, cancels further generation, and
returns a fixed refusal with `finish_reason: "content_filter"` on the OpenAI
Chat Completions surface and `guard_stop: "tool_call_guard"`. Anthropic returns
text with `stop_reason: "refusal"`. Responses currently renders the fixed notice
as `output_text` in a completed response; it does not expose the guard marker or
a structured refusal part. In every case the rejected call is withheld. The server logs a stable reason code,
without the rejected arguments. Explicit raw debug dumps can still contain
sensitive data; keep them private.

Earlier calls already released in the same streamed turn cannot be recalled.
Blocking responses reject the choice's whole set of calls when any one fails.
Clients must never execute incomplete arguments, a refusal, or code extracted
from ordinary text without their own authorization boundary.

## Evidence and limits

Pure policy tests and real Qwen/GLM parser tests cover fragmented arguments,
end-of-stream handling, malformed/oversized calls, allowed calls, configuration,
and refusal framing. The standalone `atlas-tool-guard` crate can be tested on a
CPU with `cargo test -p atlas-tool-guard`. Model throughput claims require separate
matched measurements; absence of an extra GPU pass is not a zero-overhead claim.

Reported callback indicators: [Cohere's investigation](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model)
and its linked [Crow report](https://github.com/nibor1896/Crow/issues/288).
The cause of the reported model behavior is undetermined; this feature makes no
claim about intent or model provenance.
