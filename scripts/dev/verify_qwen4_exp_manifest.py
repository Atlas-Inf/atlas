#!/usr/bin/env python3
"""Diff Atlas's qwen4_exp weight manifest against a real checkpoint.

    python scripts/dev/verify_qwen4_exp_manifest.py <checkpoint_dir>

Works on a directory of safetensors (the tiny checkpoint from
make_tiny_qwen4_exp.py) or on one holding only `model.safetensors.index.json`
(a published checkpoint whose weights you have not downloaded -- the index
alone carries every tensor NAME, so name coverage is checkable for free;
shapes are only checked for shards actually present).

Only `model.visual.*` is excluded, by design: the vision tower is independent
of the language model and its dimensions live in a separate config block.

Quantization scale siblings ARE covered -- the manifest emits them from the
checkpoint's declared `weight_block_size` and `modules_to_not_convert`.
"""
import json, pathlib, struct, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def manifest_for(config_path, stacked_mtp=False):
    base = ["cargo", "run", "-q", "--release", "-p", "atlas-core",
            "--example", "qwen4exp_manifest"]
    tail = ["--", str(config_path)] + (["--stacked-mtp"] if stacked_mtp else [])
    for extra in ([], ["--no-default-features", "--features", "metal"]):
        done = subprocess.run(base + extra + tail,
                              cwd=ROOT, capture_output=True, text=True)
        if done.returncode == 0:
            return {t["name"]: tuple(t["shape"]) for t in json.loads(done.stdout)}
        err = done.stderr
    raise SystemExit(f"cargo run failed:\n{err}")


def safetensors_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    return head


def main():
    ckpt = pathlib.Path(sys.argv[1])
    # Expert layout is a per-release packaging choice, not an architectural one,
    # so try both rather than making the caller know which this checkpoint used.
    manifest = manifest_for(ckpt / "config.json")
    stacked = manifest_for(ckpt / "config.json", stacked_mtp=True)

    actual = {}
    index = ckpt / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"]) if index.exists() else set()
    for shard in sorted(ckpt.glob("*.safetensors")):
        for name, meta in safetensors_header(shard).items():
            actual[name] = tuple(meta["shape"])
    names |= set(actual)

    visual = {n for n in names if n.startswith("model.visual.")}
    core = names - visual
    if len(core - set(stacked)) < len(core - set(manifest)):
        manifest = stacked
        print("  (MTP experts are stored stacked in this release)")
    scales = {n for n in names if n.endswith(".weight_scale_inv")}

    missing = sorted(core - set(manifest))
    unexpected = sorted(set(manifest) - core)
    orphan = [s for s in scales if s[: -len("_scale_inv")] not in manifest]
    nonexpert = []
    mismatched = [(n, manifest[n], actual[n])
                  for n in sorted(set(actual) & set(manifest))
                  if actual[n] != manifest[n]]

    print(f"checkpoint      : {ckpt}")
    print(f"  names         : {len(names)} ({len(visual)} visual, {len(scales)} fp8 scales)")
    print(f"  core          : {len(core)}  (scales included)")
    print(f"manifest        : {len(manifest)}")
    print(f"shapes present  : {len(set(actual) & set(manifest))}")
    for label, rows in (("missing from manifest", missing),
                        ("unexpected in manifest", unexpected),
                        ("scales with no base weight", orphan)):
        print(f"  {label:30s}: {len(rows)}")
        for row in rows[:6]:
            print(f"      {row}")
    print(f"  {'shape mismatches':30s}: {len(mismatched)}")
    for n, want, got in mismatched[:6]:
        print(f"      {n}: manifest {list(want)} vs checkpoint {list(got)}")

    bad = len(missing) + len(unexpected) + len(orphan) + len(mismatched)
    print("\n" + ("MANIFEST MATCHES THE CHECKPOINT" if not bad else f"{bad} DISCREPANCIES"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
