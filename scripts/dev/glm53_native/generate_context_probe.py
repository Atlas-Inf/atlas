#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Reproduce the October 12 archive-retrieval request without contacting a server.

Example: python3 generate_context_probe.py --records 2000 > request.json
Record count is not token count; tokenize the resulting request with the model
and template used for the benchmark. Each request contains 20 buried keys.
"""

import argparse
import json
import random
import sys


WORDS = (
    "harbor orchard granite lantern copper meadow river canvas compass cedar "
    "silver garden timber vessel amber ridge valley horizon forest bridge island "
    "wind rain book house lane tree chair desk clock marble wagon blue green soft "
    "tall short hill stone metal glass"
).split()


def record_count(value):
    count = int(value)
    if count < 20:
        raise argparse.ArgumentTypeError("records must be at least 20")
    return count


def build_request(records, model):
    rng = random.Random(20261012)
    expected = {
        f"KEY_{i:02}": "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=10))
        for i in range(20)
    }
    lines = [
        "Archive record " + str(i) + ": " + " ".join(rng.choices(WORDS, k=32)) + "."
        for i in range(records)
    ]
    for j, (key, value) in enumerate(expected.items()):
        lines[int((j + 0.5) * records / 20)] += (
            " VERIFIED KEY " + key + ": " + value + "."
        )
    prompt = (
        "Read this archive and retrieve the 20 VERIFIED KEY records. "
        "The ordinary records are filler.\n"
        + "\n".join(lines)
        + "\nReturn only a JSON object mapping KEY_00 through KEY_19 "
        "to their exact recorded values."
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 90210,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=record_count, required=True)
    parser.add_argument("--model", default="glm-5.3-flash-local-port")
    args = parser.parse_args()
    json.dump(build_request(args.records, args.model), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
