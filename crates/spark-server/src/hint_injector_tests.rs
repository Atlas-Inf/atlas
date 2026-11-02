// SPDX-License-Identifier: AGPL-3.0-only

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_eisdir_detection() {
        let mut text = "Error: EISDIR: illegal operation on a directory, read".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("file_path is a directory"));
    }

    #[test]
    fn test_eisdir_escalation() {
        let mut text = "Error: EISDIR: illegal operation on a directory, read".to_string();
        inject_hints(&mut text, 3);
        assert!(text.contains("<CRITICAL>"));
        assert!(text.contains("STOP using the Write tool"));
    }

    #[test]
    fn test_enoent_detection() {
        let mut text = "Error: ENOENT: no such file or directory".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("mkdir -p"));
    }

    #[test]
    fn test_generic_fallback() {
        let mut text = "Error: something went wrong".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("Bash as a fallback"));
    }

    #[test]
    fn test_no_hint_on_success() {
        let mut text = "File written successfully".to_string();
        let original = text.clone();
        inject_hints(&mut text, 0);
        assert_eq!(text, original);
    }

    #[test]
    fn test_specific_wins_over_generic() {
        let mut text = "Error: EISDIR: illegal operation on a directory".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("file_path is a directory"));
        assert!(!text.contains("Tool call failed"));
    }

    #[test]
    fn test_read_error() {
        let mut text =
            "Error: ENOENT: no such file or directory, open './test/foo.txt'".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("Verify the path"));
    }

    #[test]
    fn test_task_delegation() {
        let mut text = "Error: Unknown agent type:  is not a valid agent type".to_string();
        inject_hints(&mut text, 2);
        assert!(text.contains("STOP using the Task tool"));
    }

    #[test]
    fn test_edit_mismatch() {
        let mut text = "Error: old_string not found in file".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("Read the file first"));
    }

    #[test]
    fn bw1_classify_productive_vs_explore() {
        use serde_json::json;
        // write/edit tools → productive
        assert!(tool_call_is_productive(
            "write",
            &json!({"filePath":"src/main.rs","content":"fn main(){}"})
        ));
        assert!(tool_call_is_productive("Edit", &json!({})));
        // bash that writes/builds/runs → productive
        assert!(tool_call_is_productive(
            "bash",
            &json!({"command":"cargo run --release"})
        ));
        assert!(tool_call_is_productive(
            "bash",
            &json!({"command":"cat > src/main.rs << 'EOF'"})
        ));
        // bash exploration → NOT productive
        assert!(!tool_call_is_productive(
            "bash",
            &json!({"command":"ls -la /tmp"})
        ));
        assert!(!tool_call_is_productive(
            "bash",
            &json!({"command":"cat Cargo.toml"})
        ));
        // read/glob → NOT productive
        assert!(!tool_call_is_productive("read", &json!({"filePath":"x"})));
        assert!(!tool_call_is_productive("glob", &json!({"pattern":"**/*"})));
    }

    #[test]
    fn bw1_hint_threshold_and_escalation() {
        // Below threshold or any productive call → no hint.
        assert!(bash_wander_hint_inner(4, 0).is_none());
        assert!(bash_wander_hint_inner(20, 1).is_none());
        // At/over threshold with zero productive → standard nudge.
        let h = bash_wander_hint_inner(5, 0).expect("should fire");
        assert!(h.contains("PROGRESS WATCHDOG"));
        assert!(h.contains("write"));
        // High count → critical escalation.
        let c = bash_wander_hint_inner(10, 0).expect("should fire");
        assert!(c.contains("CRITICAL"));
    }

    mod repeat_notice {
        use super::super::super::*;
        use crate::ir::message::ToolCall;
        use crate::ir::{ContentPart, Message, Role};

        fn msg(role: Role, t: &str) -> Message {
            Message {
                role,
                content: vec![ContentPart::Text(t.into())],
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
                reasoning: None,
                tool_error: false,
            }
        }

        fn call(id: &str, cmd: &str) -> Message {
            let mut m = msg(Role::Assistant, "");
            m.tool_calls = vec![ToolCall {
                id: id.into(),
                name: "bash".into(),
                arguments: serde_json::json!({ "command": cmd }),
            }];
            m
        }

        fn round(conv: &mut Vec<Message>, id: &str, cmd: &str, out: &str) {
            conv.push(call(id, cmd));
            conv.push(msg(Role::Tool, out));
        }

        #[test]
        fn counts_identical_trailing_rounds_ignoring_call_ids() {
            let mut conv = vec![msg(Role::User, "find the MTP docs")];
            round(&mut conv, "a", "ls docs", "REPRODUCE.md");
            for id in ["b", "c", "d"] {
                round(&mut conv, id, "grep -rn MTP docs/REPRODUCE.md", "===");
            }
            assert_eq!(identical_tool_round_run(&conv), 3);
        }

        #[test]
        fn a_changed_result_or_call_breaks_the_run() {
            let mut conv = vec![msg(Role::User, "build it")];
            round(&mut conv, "a", "cargo check", "error[E0425]");
            round(&mut conv, "b", "cargo check", "warning: unused");
            assert_eq!(
                identical_tool_round_run(&conv),
                1,
                "progressing results are not a loop"
            );
            round(&mut conv, "c", "cargo test", "warning: unused");
            assert_eq!(
                identical_tool_round_run(&conv),
                1,
                "a different call starts a new run"
            );
        }

        #[test]
        fn zero_unless_the_conversation_ends_on_a_tool_result() {
            let mut conv = vec![msg(Role::User, "x")];
            round(&mut conv, "a", "ls", "a.txt");
            round(&mut conv, "b", "ls", "a.txt");
            conv.push(msg(Role::User, "thanks"));
            assert_eq!(identical_tool_round_run(&conv), 0);
        }

        #[test]
        fn repeated_errors_are_left_to_the_error_hints() {
            let mut conv = vec![msg(Role::User, "x")];
            round(&mut conv, "a", "cat nope", "Error: ENOENT: no such file");
            round(&mut conv, "b", "cat nope", "Error: ENOENT: no such file");
            assert_eq!(identical_tool_round_run(&conv), 0);
        }

        #[test]
        fn notice_names_the_run_and_keeps_tools_open() {
            let n = repeat_notice(3);
            assert!(n.contains("3 times"));
            assert!(n.contains("different"));
            assert!(n.starts_with("\n\n"));
        }
    }
}
