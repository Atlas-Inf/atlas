#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Reconstruct the recorded synthetic content workloads without an inference server.

Only Python's standard library is required. The adjacent manifest records the
original tokenizer-selected source block counts and expected hashes. This tool
reproduces request bodies; it does not tokenize, benchmark, or contact a service.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random


KINDS = ("structured", "code", "prose")
SITES = ("Harbor", "Cedar", "Summit", "Orchard", "Delta", "Lakeside")
PRODUCTS = ("battery", "sensor", "pump", "valve", "relay", "filter")
NOTES = (
    "Inspection report is pending.",
    "Replacement batch passed inspection.",
    "Delivery date changed twice.",
    "Supplier disputes the failure count.",
    "Operators requested a follow-up visit.",
    "Test equipment was calibrated yesterday.",
)
INSTRUCTIONS = {
    "structured": (
        "Analyze the following JSON Lines records. Return only a JSON object with "
        "keys overview, ranked_findings, and actions. Include at least 30 distinct "
        "findings with record IDs, quantitative evidence, operational impact, and "
        "a detailed recommendation per finding. Compare sites and suppliers. Use "
        "the actual data, avoid duplicate boilerplate, and do not merely copy "
        "records. Make the report detailed enough to exceed 2500 words."
    ),
    "code": (
        "Review this generated Python module and write a substantial "
        "production-quality replacement module. Consolidate duplicated logic, "
        "preserve documented behavior, explicitly handle missing fields and "
        "invalid types, use typed dataclasses where useful, and include diverse "
        "unit tests. Output Python code only. Implement the replacement, "
        "including validation, aggregation, deduplication, sorting, chunking, "
        "and error reporting. Produce at least 200 lines of useful code with "
        "no placeholder bodies."
    ),
    "prose": (
        "Write a detailed analytical report from these operational field notes. "
        "Explain tensions between procurement, quality, finance, and maintenance; "
        "compare evidence across sites, discuss uncertainty, and propose a "
        "practical sequence of actions with owners. Write at least 2500 words "
        "of coherent prose in varied paragraphs, drawing specific examples "
        "from the notes. Avoid lists of repeated sentences, copying the notes "
        "verbatim, or generic filler."
    ),
}


def code_body(method: int, product: str, scale: int, limit: int) -> str:
    """Return one of the original six synthetic function templates."""
    bodies = (
        "    total = 0.0\n"
        "    for item in records:\n"
        '        amount = float(item.get("amount", 0))\n'
        '        discount = float(item.get("discount", 0))\n'
        "        total += amount * (1 - discount / 100)\n"
        f"    return round(total * {scale}, 2)",
        "    counts = {}\n"
        "    for item in records:\n"
        f'        key = item.get("{product}", "unknown")\n'
        '        counts[key] = counts.get(key, 0) + int(item.get("units", 1))\n'
        "    return {key: value for key, value in counts.items() "
        f"if value >= {scale}}}",
        '    ordered = sorted(records, key=lambda row: row.get("timestamp", ""))\n'
        "    selected = []\n"
        "    for row in ordered:\n"
        f'        if row.get("status") != "deleted" and len(selected) < {limit}:\n'
        "            selected.append(dict(row))\n"
        "    return selected",
        "    seen = set()\n"
        "    result = []\n"
        "    for row in records:\n"
        '        identity = row.get("id")\n'
        "        if identity not in seen:\n"
        "            seen.add(identity)\n"
        "            result.append(row)\n"
        f"    return result[-{limit}:]",
        "    errors = []\n"
        "    for index, row in enumerate(records):\n"
        '        if not isinstance(row.get("name"), str):\n'
        '            errors.append((index, "missing name"))\n'
        '        if row.get("quantity", 0) < 0:\n'
        '            errors.append((index, "negative quantity"))\n'
        "    return errors",
        "    batches = []\n"
        "    position = 0\n"
        "    while position < len(records):\n"
        f"        batches.append(records[position:position + {scale}])\n"
        f"        position += {scale}\n"
        "    return batches",
    )
    return bodies[method]


def prose_variants(
    site: str, product: str, units: int, failed: int, delay: int, cost: float
) -> tuple[str, ...]:
    """Keep text and ordering unchanged from the measured fixture generator."""
    return (
        f"At {site}, the team received {units} {product} components during the "
        f"latest delivery window. Inspectors rejected {failed} items, although "
        "the supplier argued that transport conditions had affected the "
        f"readings. The shipment arrived {delay} hours behind schedule. Staff "
        "agreed to preserve the test records before arranging another inspection.",
        f"The planning meeting at {site} exposed a disagreement about the "
        f"{product} order. Purchasing focused on the unit cost of ${cost:.2f}, "
        f"while maintenance pointed to {failed} failed inspections out of "
        f"{units} units. The operations lead asked both teams to explain how "
        f"a delay of {delay} hours would affect the next shift.",
        f"A technician at {site} described how the {product} line had changed "
        f"since the previous review. The new shipment contained {units} units, "
        f"but {failed} needed additional testing. Rather than extrapolate from "
        "a small sample, the team reserved a separate bench and documented the "
        "conditions under which each measurement was taken.",
        f"The report from {site} separated confirmed observations from "
        f"assumptions. The {product} delivery was {delay} hours late, and the "
        f"quoted price was ${cost:.2f} per unit. The cause of the {failed} "
        "rejected parts remained unresolved. Managers scheduled a supplier "
        "call and assigned an owner to collect photographs and instrument records.",
        f"By the end of the week, the {site} warehouse had counted {units} "
        f"{product} items. A supervisor noted that {failed} items had inconsistent "
        f"labels. Finance wanted to close the invoice at ${cost:.2f} per item, "
        "but the receiving team asked for a written exception before releasing "
        "payment. Nobody had yet quantified the cost of the "
        f"{delay}-hour interruption.",
        f"At the handover in {site}, staff debated whether to keep a larger "
        f"buffer of {product} stock. The latest order of {units} units had "
        f"arrived after a {delay}-hour delay. Of those units, {failed} were set "
        "aside for review. The team wanted a plan that balanced storage costs, "
        "traceability, and the risk of another interruption.",
    )


def make_request(kind: str, seed: int, source_blocks: int) -> dict:
    """Recreate a body using the recorded tokenizer-selected block count."""
    if kind not in KINDS or not 1 <= source_blocks <= 260:
        raise ValueError("Invalid workload kind or source block count")
    rng = random.Random(seed)
    nonce = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(24))
    blocks = []
    # Preserve draw order, including draws unused by a particular content type.
    for index in range(260):
        site = rng.choice(SITES)
        product = rng.choice(PRODUCTS)
        units = rng.randrange(40, 900)
        failed = rng.randrange(0, units // 8 + 1)
        delay = rng.randrange(0, 90)
        cost = round(rng.uniform(15, 300), 2)
        if kind == "structured":
            record = {
                "id": f"{nonce[:4]}-{index:04d}",
                "site": site,
                "product": product,
                "units": units,
                "failed": failed,
                "delay_hours": delay,
                "cost_usd": cost,
                "supplier": rng.choice(("Arden", "Morrow", "Kepler", "Tern")),
                "priority": rng.choice(("normal", "high", "urgent")),
                "note": rng.choice(NOTES),
            }
            blocks.append(json.dumps(record))
        elif kind == "code":
            scale = rng.randrange(2, 21)
            limit = rng.randrange(50, 501)
            body = code_body(index % 6, product, scale, limit)
            blocks.append(
                f"def process_{nonce[:4]}_{index}(records):\n"
                f'    """Process {site.lower()} {product} records '
                f'for export job {index}."""\n{body}\n'
            )
        else:
            variants = prose_variants(site, product, units, failed, delay, cost)
            blocks.append(variants[rng.randrange(len(variants))])
    text = (
        f"Case reference {nonce}.\n{INSTRUCTIONS[kind]}\n\nSOURCE MATERIAL:\n"
        + "\n\n".join(blocks[:source_blocks])
        + "\n\nEND SOURCE MATERIAL. Begin your response now."
    )
    return {
        "messages": [{"role": "user", "content": text}],
        "enable_thinking": False,
        "temperature": 0,
        "repetition_penalty": 1.0,
        "min_tokens": 1024,
        "max_tokens": 1024,
        "seed": seed,
    }


def body_sha256(body: dict) -> str:
    """Hash canonical UTF-8 JSON: sorted keys, compact separators, ASCII escapes."""
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSONL file; refuses overwrite"
    )
    args = parser.parse_args()
    manifest_path = Path(__file__).with_name("workload-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in manifest["workloads"]:
        body = make_request(item["kind"], item["seed"], item["source_blocks"])
        if body_sha256(body) != item["body_sha256"]:
            parser.error(f"Request body hash mismatch: {item['id']}")
        rows.append({**item, "body": body})
    try:
        with args.output.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
    except OSError as error:
        parser.error(str(error))
    print(f"Wrote {len(rows)} verified request bodies to {args.output}")


if __name__ == "__main__":
    main()
