# Gemini context — Atlas Strix Qwen3.8 port

Read `AGENTS.md` first — it is the full contributor guide and applies to you.

## The MLPerf ST leg IS the Atlas bfcl-subset ST leg (62/10/10)

The MLPerf edge-agentic ST accuracy leg is DEFINED as the BFCL golden draw:
non_live 62 / live 10 / hallucination 10, subset_floor 25, n=995, seed 42,
temp 0, NO param overrides. Atlas's own runner IS that leg — running

    spark benchmark run bfcl-subset --url http://127.0.0.1:8081 --model <checkpoint>

with no overrides IS running the MLPerf ST leg. There is no separate MLPerf
ST gate, and the external harness (`inference-endpoint`, ~/endpoints) is an
alternative runner for the same leg, not a different gate. Do NOT confuse the
mix with 12/23/46 (the `results_st996_*` local experiment draw, n=1004 — not
MLPerf). Strix Qwen3.8-unsloth baseline on this leg: overall 84.22 /
normalized 83.68 (run-1788366618469517340.json, 2026-09-02, AzeezStrix,
preservation recipe — see kernels/strix-hip/qwen3.8-27b/BENCH.toml).
