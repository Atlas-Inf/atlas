// SPDX-License-Identifier: AGPL-3.0-only
//! False-positive scan for the tool-call guard: extracts complete
//! `<tool_call>` argument JSON from receipts/logs and reports how many the
//! default-on policy would refuse. Usage: fp_scan <file>...
use atlas_tool_guard::Policy;

fn main() {
    let policy = Policy::new(true, &[], &[], atlas_tool_guard::DEFAULT_MAX_BYTES).unwrap();
    let (mut total, mut blocked) = (0usize, 0usize);
    let mut examples = Vec::new();
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        // BFCL/tool probes emit either JSON lines with "arguments" fields or
        // literal <tool_call>{json}</tool_call> runs.
        for (i, line) in text.lines().enumerate() {
            let mut rest = line;
            while let Some(start) = rest.find("\"arguments\"") {
                rest = &rest[start + 11..];
                let Some(js) = rest.find('"').map(|q| &rest[q + 1..]) else {
                    break;
                };
                let Some(end) = js.find('"') else { break };
                let args = js[..end].replace("\\\"", "\"").replace("\\n", "\n");
                rest = &rest[end..];
                total += 1;
                if let Err(r) = policy.inspect(&args) {
                    blocked += 1;
                    if examples.len() < 10 {
                        examples.push(format!(
                            "{path}:{i} {} {:?}",
                            r.code(),
                            &args[..args.len().min(160)]
                        ));
                    }
                }
            }
            let mut rest2 = line;
            while let Some(s) = rest2.find("<tool_call>") {
                rest2 = &rest2[s + 11..];
                let end = rest2.find("</tool_call>").unwrap_or(rest2.len());
                let body = &rest2[..end];
                if let Some(j) = body.find('{') {
                    let args = body[j..].trim_end_matches('}').to_string() + "}";
                    total += 1;
                    if let Err(r) = policy.inspect(&args) {
                        blocked += 1;
                        if examples.len() < 10 {
                            examples.push(format!(
                                "{path}:{i} {} {:?}",
                                r.code(),
                                &args[..args.len().min(160)]
                            ));
                        }
                    }
                }
                rest2 = &rest2[end..];
            }
        }
    }
    println!("fp-scan: {total} complete calls, {blocked} would be refused");
    for e in examples {
        println!("  {e}");
    }
}
