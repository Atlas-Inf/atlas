# DFlash2 run scripts (AzeezStrix, gfx1151)

Every number in `docs/porting/QWEN38_STRIX_PORT.md` §2026-09-15 came from
these scripts' outdirs under `~/dp4a-ab/out/` on strix.

- `dflash_ab.sh` — sustained A/B driver: one serve arm per invocation
  (serial / MTP K4 / DFlash γ / Option-B / GEMV toggles), 3×N MinHeap +
  prose + JSON rows, parity + accept capture. Writes
  `dflash-<tag>-<TS>/` with `results.jsonl`, `fingerprint-<arm>.txt`,
  `serve-<arm>.log`, `<arm>-accept.txt`, per-request texts, `SUMMARY.md`.
- `dflash_overnight.sh` — the overnight chain: ST-995 (bfcl-subset
  golden draw) then ST-996 (bfcl_v4 12/23/46) legs. Writes
  `dflash-overnight-<TS>/` with `chain.log`, `<leg>-fingerprint.txt`,
  `<leg>-serve.log`, `<leg>-bench.log`, `<leg>-accept.txt`.
- `dflash_agentic_only.sh` — standalone MLPerf agentic-coding 2.5h leg
  (inference-endpoint harness, long-context serve, optional prefix
  caching + pre-probe). Writes `dflash-agentic-<TS>/` with
  `agentic-fingerprint.txt`, `agentic-serve.log`, `agentic-harness.log`.
- `dflash_handover.sh` — watcher that detects the chain's LEG2 banner,
  stops the chain/serve, records the ST-995 result, launches the
  agentic-only leg. Log: `~/dp4a-ab/out/dflash-handover.log`.
- `dflash_conc.sh` — DFlash concurrency probe (parallel requests vs
  serial arm). Writes `dflash-conc-<TS>/`.
