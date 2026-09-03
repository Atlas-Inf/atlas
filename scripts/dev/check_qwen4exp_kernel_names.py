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
# The qwen4_exp-specific code, PLUS the init paths of every layer a qwen4_exp
# model builds: 36 Qwen3SsmLayer, 12 Qwen3AttentionLayer, the MoE. Those inits
# are where the fail-closed startup audit's lookups happen, so a name that does
# not resolve there is a server that refuses to boot.
RUST_PATHS = [
    "crates/spark-model/src/layers/ple/*.rs",
    "crates/spark-model/src/layers/qsa*.rs",
    "crates/spark-model/src/layers/ops/qwen4exp*.rs",
    "crates/spark-model/src/layers/qwen3_ssm/init*.rs",
    "crates/spark-model/src/layers/qwen3_ssm/kernel_select.rs",
    "crates/spark-model/src/layers/qwen3_attention/init*.rs",
    "crates/spark-model/src/layers/moe/init*.rs",
    "crates/spark-model/src/layers/moe.rs",
    "crates/spark-model/src/layers.rs",
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
            found[module].update(declared_kernels(src))
            found[module].update(macro_kernels(src))
    return found


def _paged_concat_suffixes() -> set[str]:
    """Suffixes the `.cuh` templates append to `KERNEL_NAME`.

    `prefill_paged_compute.cuh` declares BOTH
    `extern "C" __global__ void KERNEL_NAME(...)` and
    `... PAGED_CONCAT(KERNEL_NAME, _64)(...)`, so one `#define` yields two
    entry points. Read the suffixes out of the templates rather than hardcoding
    `_64`, so a new variant does not silently become a false MISSING.
    """
    suffixes = {""}
    for path in glob.glob(os.path.join(COMMON, "*.cuh")) + glob.glob(
        os.path.join(TARGET, "*.cuh")
    ):
        try:
            text = open(os.path.realpath(path), encoding="utf-8").read()
        except OSError:
            continue
        for m in re.finditer(r"PAGED_CONCAT\(\s*KERNEL_NAME\s*,\s*(_\w+)\s*\)", text):
            suffixes.add(m.group(1))
    return suffixes


_SUFFIXES = None


def macro_kernels(src: str) -> set[str]:
    """Entry points generated through `#define KERNEL_NAME` + an included template.

    21 files in this tree name their kernel with a macro and `#include` a `.cuh`
    that declares it, so the name never appears in an `extern "C"` line and no
    scan of declarations alone can see it. Without this, every one of them reads
    as MISSING — 7 of them are on the qwen4_exp path.
    """
    global _SUFFIXES
    if _SUFFIXES is None:
        _SUFFIXES = _paged_concat_suffixes()
    names: set[str] = set()
    for m in re.finditer(r"#define\s+KERNEL_NAME\s+(\w+)", src):
        for suffix in _SUFFIXES:
            names.add(m.group(1) + suffix)
    return names


# Attributes and keywords that can sit between `extern "C" __global__` and the
# entry point's NAME. A regex demanding `void <name>(` contiguously misses every
# kernel that carries a launch bound or wraps the line, and this tree is full of
# both:
#
#     extern "C" __global__ void __launch_bounds__(128, 1)
#     gated_delta_rule_decode(
#
#     extern "C" __global__
#     __launch_bounds__(128, 3)
#     void w4a16_gemm_t_m128(
#
# Under-reporting the available set is the dangerous direction: it makes this
# script emit a false MISSING for a kernel that is right there, which is the
# failure mode that wastes an afternoon. So the prologue is SCANNED rather than
# matched: skip keywords, skip balanced attribute arguments, and the last
# identifier before the parameter list is the name.
_SKIP = {"void", "__global__", "static", "inline", "__inline__", "extern"}


def _object_macros(src: str) -> dict[str, str]:
    """`#define NAME other_identifier` in one file.

    A kernel may be declared through an ALIAS rather than its own name:

        #ifndef ATLAS_PREFILL_ENTRY
        #define ATLAS_PREFILL_ENTRY inferspark_prefill
        #endif
        extern "C" __global__ void ATLAS_PREFILL_ENTRY(

    so the scanner below has to resolve the alias or it reports the macro's
    name, which resolves against nothing and reads as a MISSING kernel.
    """
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^\s*#define\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*$", src, re.M)
    }


def declared_kernels(src: str) -> set[str]:
    """Every `extern "C" __global__` entry point name in one translation unit."""
    macros = _object_macros(src)

    def resolve(name: str) -> str:
        # Bounded, so a `#define A B` / `#define B A` pair cannot spin.
        for _ in range(4):
            nxt = macros.get(name)
            if nxt is None or nxt == name:
                break
            name = nxt
        return name

    names: set[str] = set()
    for m in re.finditer(r'extern\s+"C"\s+__global__', src):
        i = m.end()
        name = None
        # Bounded scan: a prologue longer than this is not a declaration.
        limit = min(len(src), i + 400)
        while i < limit:
            ch = src[i]
            if ch.isspace():
                i += 1
                continue
            if src.startswith("__launch_bounds__", i):
                i += len("__launch_bounds__")
                # Skip its balanced argument list.
                while i < limit and src[i].isspace():
                    i += 1
                if i < limit and src[i] == "(":
                    depth = 0
                    while i < limit:
                        if src[i] == "(":
                            depth += 1
                        elif src[i] == ")":
                            depth -= 1
                            if depth == 0:
                                i += 1
                                break
                        i += 1
                continue
            token = re.match(r"[A-Za-z_]\w*", src[i:])
            if token:
                word = token.group(0)
                i += len(word)
                if word not in _SKIP:
                    name = word
                continue
            if ch == "(":
                break
            # Anything else (a `*`, a stray token) means this is not a shape we
            # understand; stop rather than guess a name.
            break
        if name:
            names.add(resolve(name))
    return names


# `gpu.kernel(...)` is FAIL-CLOSED: absent means the server refuses to boot.
# `try_kernel(...)` returns a 0-handle the caller gates on, so an absence there
# is a documented fallback, not a defect — probe-and-fall-back twins live in
# other models' shadows on purpose. Conflating the two would make this script
# flag ~28 legitimate fallbacks.
_HARD = re.compile(r'(?<!try_)\bkernel\(\s*"([\w.]+)"\s*,\s*"(\w+)"\s*\)')
_SOFT = re.compile(r'try_kernel\(\s*[\w.]+\s*,\s*"([\w.]+)"\s*,\s*"(\w+)"\s*\)')


def asks() -> tuple[list[tuple[str, str, str]], int]:
    """(hard lookups, soft-lookup count) across the swept paths."""
    hard, soft = [], 0
    for pattern in RUST_PATHS:
        for path in sorted(glob.glob(pattern)):
            src = open(path, encoding="utf-8").read()
            for m in _HARD.finditer(src):
                hard.append((m.group(1), m.group(2), path))
            soft += len(_SOFT.findall(src))
    return hard, soft


def main() -> int:
    if not os.path.isdir(TARGET):
        print(f"{TARGET}: missing", file=sys.stderr)
        return 2
    available = entry_points(TARGET)
    total = sum(len(v) for v in available.values())
    hard_asks, soft_count = asks()
    requested = sorted(set(hard_asks))
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
        f"({checked} fail-closed lookups checked against {total} entry points "
        f"in {len(available)} modules; {soft_count} try_kernel fallbacks not "
        f"required; {absent_total} expected_absent entries, none stale)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
