# ADR-0017: A gate record is only as durable as the commit it names

**Status:** Accepted
**Date:** 2026-08-25
**Extends:** ADR-0013 (coverage by content, never by ancestry)

## Context

ADR-0013 removed the ancestry precondition from `record_covers` because Atlas
squash-merges, and a squash gives the landed commit a new sha with no parent
link back to the branch the record was measured on. `git diff A B` compares
trees and needs no history relationship, so the diff alone was kept.

The diff needs one thing the ADR did not state: **both commits must be objects
this clone actually holds.** Ancestry was dropped; existence was assumed. That
assumption held for as long as the branches stayed on the remote.

The move from `Avarok-Cybersecurity/atlas` to `Atlas-Inf/atlas` did not carry
the feature branches over. `Atlas-Inf/atlas` has ten branches: `main`, two port
branches, six dependabot branches, and one bot branch. Every branch a gate
record was ever measured on is gone.

The effect on `main` was total and silent:

```
gate check for 20f1c3a322
  NONE  agentic-webserver — latest record is for e0a634261b — git cannot diff
        that commit against this one; is it in this clone? (the gate job needs
        `fetch-depth: 0`)
  … the same for all ten required gates
10 bench(es) still need a passing gate record
```

Of 502 committed records, **11** name a commit reachable from anything the
remote still has, and those 11 cover only five of the ten required gates. Five
gates — `concurrency-sweep`, `decode-floor`, `ssm-state-poisoning-gate`,
`video-fidelity`, `vision-fidelity` — have zero. No re-run fixes this, and
`fetch-depth: 0` was already set: the objects are not on the remote to fetch.

Three consequences, in order:

1. `PR benchmark gate` is the only red job on `main` and on every PR.
2. `dev-release.yml` chains off CI's verdict and skips whenever it is red. It
   has skipped on all eight runs since the migration.
3. So the Releases page is empty. All nine release targets — including
   `linux-x86_64-*` and `windows-x86_64-*` — build green on every run and are
   discarded when the one-day artifact retention expires.

A GPU perf gate whose records were orphaned by a repo move had, transitively,
stopped anyone from downloading a binary.

## Decision

**Three changes, each addressing a different one of the three faults.**

### 1. Say which operand is missing

`invalidating_paths` returned `None` for any failed diff, and the caller
rendered every `None` as "is it in this clone? (the gate job needs
`fetch-depth: 0`)". A shallow clone and a deleted branch are indistinguishable
at that level and have opposite remedies. Pointing at a setting that is already
correct cost the entire investigation.

`check::commit_is_present` now splits the two, and the deleted-branch arm says
so and says that no fetch depth can recover it.

### 2. Restore the commits the committed records name

The records are evidence, and re-keying them to a commit that happens to exist
would be falsifying it. The commits are restored instead, under
`refs/tags/gate-record/<sha>` — a namespace that states why the object is
retained, is fetched by `actions/checkout` alongside `refs/heads/*`, and cannot
be confused with a live branch. `e0a634261b` carries 51 commits and 420 objects
not already in `main`.

**A record's commit must stay reachable for as long as the record is
committed.** Deleting a branch after merge is fine; deleting it while a record
names it silently retires that record.

### 3. Amnesty what the migration actually changed

With the commits reachable the gate gives a real verdict, and it is a red one:
42 `PERF_PATHS` files differ between `e0a634261b` and `main`. Every one is
inert:

| what changed | files | why it cannot move a number |
| --- | --- | --- |
| `Avarok` → `Atlas` in doc comments | 6 | comments; one is a `.cu` comment line |
| plugin metadata strings | 1 | author, URLs, contact |
| recipe test fixtures | 27 | container image name, maintainer |
| `Cargo.toml` / `Cargo.lock` | 2 | LatticeDB pruning |
| `crates/atlas-governance` | 4 | host-only ledger — see below |
| **total** | **42** | |

The first four rows go into `ONE_TIME_AMNESTY` — Tom Turney's mechanism from
PR #701, reused exactly as its docs anticipate. Each of the 38 entries pins the
blob OID this PR lands, so the grant covers the reviewed bytes and nothing
else: edit any of those files again and it invalidates at full price.

Two more entries pin this change's own edits to `check.rs` and `coverage.rs`.
Both are in `BOUNDARY_FILES`, which invalidates every gate by construction and
which no exclusion may exempt — that is the lock the list exists to be. So a
change to the gate cannot land without amnestying itself. #701 hit the same
wall and pinned the same two files. There is no circularity: the table lives in
`amnesty.rs`, which is not a boundary file, so pinning `check.rs` and
`coverage.rs` does not perturb the blobs being pinned.

`crates/atlas-governance` is deliberately **not** amnestied. Its change is a
real code deletion, and pinning it as inert would be a lie in a table whose
whole value is that a reviewer can trust it. It gets a standing exclusion,
`GOVERNANCE_LEDGER`, on the same rationale `GATE_MACHINERY` already carries:
the ledger never runs a model. Its entire reverse-dependency set is
`atlas-plugin`, and within that crate the only references are `src/gate/**`
(already excluded) and two `ledger_*` CLI bins. `spark-server`, `spark-model`
and `atlas-core` do not name it.

That is a dependency-graph claim, so `governance_exclusion_tests` re-derives it
from the manifests and sources on every `cargo test`. If someone links the
ledger into the server, the exclusion fails a test rather than quietly
exempting code that now runs there.

## Consequences

`main` goes green, so `dev-release.yml` publishes `bNNNN` prereleases again and
the Linux and Windows executables become downloadable for the first time since
the migration.

The grant is 38 paths where #701's was three. That is the honest price of a
repo-wide rename, and the pin is what bounds it. `AMNESTY_EPOCH` moves to
2026-08-26T00:00:00Z: the records that retired the #701 grant are the same ones
this grant exists to rescue, so leaving the epoch at 2026-08-22 would let
`amnesty_expires_once_every_gate_has_a_fresh_record` read pre-migration records
as proof this grant is already spent.

**This does not lower the bar for a real regression.** Every path outside the
pinned 38 invalidates exactly as before, and the 38 stop being excused the
moment anyone edits them.

### What this does not fix

A record still dies if its commit becomes unreachable, and the tag namespace is
a convention that a future migration can forget just as this one did. The
durable fix is for a record to carry the content it was measured against —
per-path blob OIDs, or the tree hash — so coverage needs no commit object at
all. Records already committed have no such field, so it cannot resolve this
incident; it is the right shape for the next one.
