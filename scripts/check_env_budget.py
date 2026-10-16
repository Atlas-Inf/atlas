#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Ratchet on the number of distinct `ATLAS_*` names the crates reference.

Every `"ATLAS_…"` string literal in `crates/**/*.rs` counts once, whichever
helper reads it (`env::var`, `opt_in`, `tunable`, `env_usize`, …), so routing
a new knob through a helper does not dodge the budget. The ceiling lives in
`scripts/atlas_env_budget.txt` and only ever moves down: CI fails when the
count rises above it, and asks for the file to be lowered when pruning brings
the count below it. See #117 (repo diet) and the CONTRIBUTING rule that a
neutral opt-in arm does not merge.

Stdlib only; the runner's system python3 suffices.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUDGET_FILE = ROOT / "scripts" / "atlas_env_budget.txt"
NAME = re.compile(r'"(ATLAS_[A-Z0-9_]+)"')


def names() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "crates").rglob("*.rs"):
        if "target" in path.parts:
            continue
        found.update(NAME.findall(path.read_text(encoding="utf-8", errors="replace")))
    return found


def main() -> int:
    budget = int(BUDGET_FILE.read_text().split()[0])
    count = len(names())
    if count > budget:
        print(
            f"FAIL: crates reference {count} distinct ATLAS_* names, over the budget of {budget}.\n"
            "A new env knob needs an old one removed, or a CLI flag / MODEL.toml field instead.\n"
            "An opt-in arm measured neutral does not merge (CONTRIBUTING); see #117."
        )
        return 1
    if count < budget:
        print(
            f"FAIL: {count} distinct ATLAS_* names, below the budget of {budget}. "
            f"Lower scripts/atlas_env_budget.txt to {count} so the ratchet keeps the win."
        )
        return 1
    print(f"ATLAS_* env budget: {count} / {budget}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
