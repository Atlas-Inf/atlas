# Release notes & shipped-image record

One file per shipped image so `:latest`'s provenance is answerable at a glance.
When `/atlas-release publish` promotes a verified image (see
[`.claude/skills/atlas-release`](../../.claude/skills/atlas-release/SKILL.md)),
record it here as `docs/releases/<git-sha>.md` with:

- the **git SHA** the image was built from (also stamped into the image as
  `org.opencontainers.image.revision` — `docker inspect` it),
- the moving tags it received (`:latest` / `:dev` / `:nightly` / `:<semver>`),
- the **serve-matrix verdict** (`tests/gate_results.py` PASS + the results table),
- notable engine changes since the previous shipped SHA.

This closes the "is `:latest` the merged code?" gap: the answer is the newest file
here, cross-checked against the image's revision label.

## Verifying on a single GB10

The serve-matrix leg runs on any single-node GB10 with the checkpoints cached.
`scripts/release/verify-gb10-local.sh <sha7>` runs the whole pipeline below —
worktree, build, matrix, coherence, gate, save — including the rootless/sudo
handling in the notes; the per-stage commands are:

```bash
python3 tests/run_all_models.py --roster tests/rosters/gb10-cached.json
python3 scripts/test_coherence.py
python3 tests/gate_results.py
```

`tests/rosters/gb10-cached.json` is the scoped roster that passed on reiner
(jobs 513/521): five models, one round each — Flash-Next NVFP4 (MTP), 27B
DFlash2, Lightning-30B, 35B-A3B NVFP4, 35B-A3B FP8. `kv_dtype: "bf16"` is not
optional on the NVFP4 entries — QSA refuses an nvfp4 KV cache ("QSA selection
requires a plain BF16 KV cache"). The Lightning entry carries
`"env": {"ATLAS_DFLASH_OPTION_B": "1"}` (roster `env` field): the Lightning
DSpark product path's `validate()` refuses to boot without it.

Notes from the reiner runs, in the order they bite:

- **Rootless Docker.** `run_all_models.py` issues `sudo docker`, and
  `--network host` containers join rootlesskit's network namespace, not the
  host's. Two consequences: a pass-through `sudo` shim must sit on PATH (a
  `bin/sudo` that just `exec "$@"`), and clients that must reach the
  `0.0.0.0`-bound serve have to run inside the namespace:
  `nsenter -U --preserve-credentials -n -t "$(pgrep -u "$(id -u)" -x dockerd)" <cmd>`.
- **`test_coherence.py` needs a reasoning-off serve** (`--disable-thinking`):
  its Edge Cases section reads `message.content` after `max_tokens` 10–20,
  and a thinking serve spends that budget inside `<think>` — the client sees
  empty content and reports a false FAIL.
- **The tok/s bar is liveness-only until baselines are blessed.** Seed them
  once with `gate_results.py --update-baselines`, then commit
  `tests/baselines/` — uncommitted baselines mean every run measures only
  that it ran, not that it's fast enough.
