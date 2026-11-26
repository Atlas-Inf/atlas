# Ling-3.2-air on Spark: speculation, the think-phase serial rule, and the production knob

## Summary

Production serving of Ling-3.2-air (bailing_moe_v3_vl + NEXTN single-draft MTP) at real
context showed two symptoms after PR #129/#130/#134: every real-context request logged
"MTP drafter prefill failed (continuing without): ling MLA kv-only: 64 rows at 128
exceed the scratch cap", and long-context requests sat at 10-15 tok/s instead of the
~30+ the MTP arm delivers.

Root causes, in order of impact:

1. **The WARN text is a pre-#134 binary, not live code.** The message format
   `{rows} rows at {seq_len_start} exceed the scratch cap` was removed in PR #134
   (`e7966b073`): main (`205f0486c`) emits `{rows} rows exceed the scratch cap {cap}`
   and the position-constrained bound is gone (only the chunk width occupies
   scratch now). A binary built from `205f0486c` cannot emit the "at 128" text
   (`git grep "rows at {"` on main: only a doc comment matches). A serve that logs
   it is running a stale `target/release/spark` — the deploy that shows this
   symptom needs a rebuild from main, nothing else.
2. **The 10-15 tok/s at long context is the think-phase serial rule, by design.**
   Speculation NEVER runs while a sequence is inside its think block unless
   `ATLAS_MTP_SPEC_THINK=1` / `ATLAS_DFLASH_SPEC_THINK=1` (or the model's verify
   is bit-exact: `spec_think_for`, qwen4_exp / qwen3_5 levers). Ling's batch-2
   K=2 verify is deterministic per step (PR #134 gates) but is NOT bit-exact
   against the M=1 serial forward — the 2026-08-16 bisect behind
   `spec_dispatch_eligible` measured this flip breaking agentic trajectories
   deterministically (10/10 without spec-in-think, failures with), so the default
   guards it. A request whose entire generation stays inside think (an
   analysis-heavy doc summary that exhausts `max_tokens` reasoning) then runs
   plain serial decode for every token: `serial=1.00 mtp=0.00 p1=0.000`, ~11-15
   tok/s at 1.5-2.9k context. The drafter's prompt-context KV is still built
   correctly (coverage lines: 2857-token prompt -> 2856 drafter rows, no
   warnings) — those tokens are simply never consumed while inside think.

## The knob, measured (f341 / GB10, exact production flags:
## `--max-prefill-tokens 32768 --max-batch-size 1 --ssm-cache-slots 2
## --speculative --num-drafts 1`, bf16 lm head)

| request | default (serial in think) | `ATLAS_MTP_SPEC_THINK=1` |
|---|---|---|
| 1571-tok doc, 250-tok answer | 11.4 tok/s, p1=0.600, serial=0.96 | **30.5 tok/s, p1=0.887**, serial=0.11 |
| 2857-tok doc, 250-tok answer | 10.8 tok/s, p1=0.000, serial=1.00 | **30.7 tok/s, p1=0.901**, serial=0.00 |
| 1571-tok doc, 700-tok answer | 18.3 tok/s, p1=0.000 (all-think) | **27.9 tok/s, p1=0.660**, serial=0.00 |
| 40-tok prompt, 700-tok story  | ~23 tok/s (exits think early)   | healthy, p1=0.61-0.63 |

Read: on document workloads (long reasoning inside think, short answers) the knob
is worth ~2.7x delivered tok/s and the drafter acceptance at 2.5k+ context is high
(p1 0.89-0.90, tok_step 1.89-1.90). Merge-gate criterion "p1 >= 0.78 at ~2.5k
context, zero drafter warnings" passes on both prod shapes with the knob.

## Risk and recommendation

`ATLAS_MTP_SPEC_THINK=1` on Ling re-enables speculation during thinking WITHOUT
bit-exact verify: the measured failure mode is a T=0 token flip on a low-margin
reasoning token, which can derail a long agentic reasoning chain. The K=2 verify
is step-deterministic (same process, same arm -> same tokens; PR #134 battery),
so the risk is specifically "slightly different logits than the serial path would
have produced", not run-to-run nondeterminism. Recommended posture: enable the
knob for interactive/summarization workloads after an A/B on the agentic task
suite; keep it off (the default) wherever trajectory exactness against the serial
path matters. Both postures need the production serve rebuilt from main first —
with a pre-#134 binary the kv-only refusal silently kills the drafter context on
any prompt past ~190 tokens regardless of the knob.
