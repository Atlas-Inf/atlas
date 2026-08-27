#!/usr/bin/env python3
"""Every kernel the qwen4_exp path asks for must exist in its target's PTX.

WHY. Atlas resolves kernels by two STRINGS — a module name and an entry-point
name — and the startup audit is fail-closed, so a typo in either is a server
that refuses to boot. That is the good case. The bad case is a name that
resolves in a DIFFERENT model's shadow: `hyper_connection` belongs to
DeepSeek-V4's Sinkhorn mHC as well as to this target's low-rank one, and the
same name over a different argument list is a segfault or, worse, plausible
numbers.

Neither failure is visible to `cargo check`: the names are string literals.
This walks the target's real `.cu` files (following the symlinks into
`qwen3.6-35b-a3b`, plus everything inherited from `common/`), collects the
`extern "C" __global__` entry points per module, and checks every
`kernel(...)` / `try_kernel(...)` call in the qwen4_exp Rust against them.

It does NOT compile anything, so it cannot catch a signature mismatch — only a
name that is not there. Run it before pushing; it takes milliseconds and it is
the one class of startup failure a laptop can rule out.

Exit 0 clean, 1 with unresolvable names, 2 if the target or its KERNEL.toml is
missing (which is itself a finding).
"""

from __future__ import annotations

import collections
import glob
import os
import re
import sys

TARGET = "kernels/gb10/qwen3.8-flash-next/nvfp4"
COMMON = "kernels/gb10/common"

# The Rust that resolves kernels for this model. Kept explicit rather than
# globbed over the crate: the point is to check THIS port's names, and a
# wildcard would quietly start covering (or stop covering) other families.
RUST_PATHS = [
    "crates/spark-model/src/layers/ple/*.rs",
    "crates/spark-model/src/layers/qsa*.rs",
    "crates/spark-model/src/layers/ops/qwen4exp*.rs",
    "crates/spark-model/src/layers/qwen3_ssm/kernel_select.rs",
]

# Resolved by name CONSTRUCTION rather than a literal (`format!("{base}_sigmoid")`),
# so the regex below cannot see them. Checked explicitly, because these are the
# three kernels whose absence would mean the GDN output gate silently uses SiLU
# on all 36 linear-attention layers.
CONSTRUCTED = {
    "norm": [
        "gated_rms_norm",
        "gated_rms_norm_sigmoid",
        "gated_rms_norm_f32_input",
        "gated_rms_norm_f32_input_sigmoid",
        "gated_rms_norm_prefill",
        "gated_rms_norm_prefill_sigmoid",
    ]
}


def module_map(target: str) -> dict[str, str]:
    """`[modules]` from KERNEL.toml: file stem -> module name."""
    path = os.path.join(target, "KERNEL.toml")
    if not os.path.exists(path):
        print(f"{path}: missing", file=sys.stderr)
        sys.exit(2)
    out, inside = {}, False
    for line in open(path):
        if line.strip() == "[modules]":
            inside = True
            continue
        if inside and line.startswith("["):
            break
        m = re.match(r'\s*([A-Za-z0-9_]+)\s*=\s*"([^"]+)"', line)
        if inside and m:
            out[m.group(1)] = m.group(2)
    return out


def entry_points(target: str) -> dict[str, set[str]]:
    """module -> {entry point names} for everything this target can resolve."""
    mods = module_map(target)
    found: dict[str, set[str]] = collections.defaultdict(set)
    # A shadow file shadows the common file of the same name, so the shadow is
    # walked first and common/ only contributes stems the shadow lacks.
    seen_stems: set[str] = set()
    for directory in (target, COMMON):
        for shown in sorted(glob.glob(os.path.join(directory, "*.cu"))):
            stem = os.path.basename(shown)[:-3]
            if directory == COMMON and stem in seen_stems:
                continue
            seen_stems.add(stem)
            module = mods.get(stem, stem)
            src = open(os.path.realpath(shown), encoding="utf-8").read()
            for m in re.finditer(
                r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(', src
            ):
                found[module].add(m.group(1))
    return found


def asks() -> list[tuple[str, str, str]]:
    """(module, entry, file) for every kernel lookup in the qwen4_exp path."""
    out = []
    for pattern in RUST_PATHS:
        for path in sorted(glob.glob(pattern)):
            src = open(path, encoding="utf-8").read()
            for m in re.finditer(r'kernel\(\s*"([\w.]+)"\s*,\s*"(\w+)"\s*\)', src):
                out.append((m.group(1), m.group(2), path))
            for m in re.finditer(
                r'try_kernel\(\s*\w+\s*,\s*"([\w.]+)"\s*,\s*"(\w+)"\s*\)', src
            ):
                out.append((m.group(1), m.group(2), path))
    return out


def main() -> int:
    if not os.path.isdir(TARGET):
        print(f"{TARGET}: missing", file=sys.stderr)
        return 2
    available = entry_points(TARGET)
    total = sum(len(v) for v in available.values())
    requested = sorted(set(asks()))
    bad = 0

    for module, name, path in requested:
        if name in available.get(module, set()):
            continue
        bad += 1
        elsewhere = sorted(m for m, v in available.items() if name in v)
        hint = (
            f"resolves in {elsewhere} instead — a module-name collision is the "
            "dangerous case, not the missing one"
            if elsewhere
            else "not in ANY module this target can see"
        )
        print(f"MISSING {module}::{name} ({os.path.basename(path)}): {hint}")

    for module, names in CONSTRUCTED.items():
        for name in names:
            if name not in available.get(module, set()):
                bad += 1
                print(f"MISSING {module}::{name} (built by name construction)")

    # `[expected_absent]` documents the probe-and-fall-back twins this target
    # deliberately does not ship, and is what takes the fail-closed startup
    # audit from 6 unresolved kernels to 0. A STALE entry — a name listed as
    # absent that is in fact present — is the mirror-image hazard: it tells the
    # audit to stop looking at something that changed.
    absent_stale = 0
    absent_total = 0
    toml_path = os.path.join(os.path.dirname(TARGET), "MODEL.toml")
    if os.path.exists(toml_path):
        toml = open(toml_path, encoding="utf-8").read()
        for module, body in re.findall(
            r"\[expected_absent\.([\w]+)\]\n(.*?)(?=\n\[|\Z)", toml, flags=re.S
        ):
            for name in re.findall(r"^([A-Za-z0-9_]+)\s*=", body, flags=re.M):
                absent_total += 1
                if name in available.get(module, set()):
                    absent_stale += 1
                    print(
                        f"STALE expected_absent.{module}::{name}: listed as absent "
                        "but this target ships it"
                    )
    bad += absent_stale

    checked = len(requested) + sum(len(v) for v in CONSTRUCTED.values())
    if bad:
        print(f"\nqwen4_exp kernel names: {bad} of {checked} unresolvable")
        return 1
    print(
        f"qwen4_exp kernel names: OK "
        f"({checked} checked against {total} entry points in "
        f"{len(available)} modules; {absent_total} expected_absent entries, "
        "none stale)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
