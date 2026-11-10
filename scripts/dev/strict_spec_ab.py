#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""A/B harness for strict structured-output speculation (`ATLAS_STRICT_SPEC`).

Drives one live OpenAI-compatible server through a fixed set of
`response_format` workloads and writes every reply to a JSON record. Run it
once per arm (lever off, lever on: same binary, same flags, fresh boot each),
then `compare` the two records: wall tok/s per case, JSON/schema validity, and
whether the temperature-0 replies of the two arms are identical.

  strict_spec_ab.py run --url http://127.0.0.1:8081 --arm off --out off.json
  strict_spec_ab.py run --url http://127.0.0.1:8081 --arm on  --out on.json
  strict_spec_ab.py compare off.json on.json

Rates are wall rates over whole requests (prefill included, prompts are
short); the server's own `Done:` lines carry the decode-only rate and the
`serial=`/`mtp=` split, which is what proves the lever engaged. The record
holds no URL or host name.

A third arm, `serial`, is the lever on with every structured request on a
verify of its own (`ATLAS_GRAMMAR_SERIAL_VERIFY` set); `compare` takes any two
records, baseline first.

Exit status: `run` 0 when every request returned; `compare` 0 when the second
record has no invalid structured reply.
"""
import argparse
import concurrent.futures
import datetime
import json
import statistics
import sys
import time
import urllib.request

ITEM = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "category": {"type": "string", "enum": ["tool", "toy", "food", "book"]},
        "price_cents": {"type": "integer"},
        "in_stock": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["name", "category", "price_cents", "in_stock", "description"],
}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"items": {"type": "array", "items": ITEM}},
    "required": ["items"],
}
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "catalogue", "strict": True, "schema": SCHEMA},
}
JSON_OBJECT = {"type": "json_object"}

THEMES = ["a hardware shop", "a seaside kiosk", "a mountain hut", "a night market",
          "a school fair", "a ferry terminal", "a museum shop", "a farm stand"]


def prompt(i, structured):
    theme = THEMES[i % len(THEMES)]
    if structured:
        return (f"Write a catalogue of exactly 12 products sold at {theme}. For each give a "
                "name, a category (tool, toy, food or book), a price in cents, whether it is "
                "in stock, and a two-sentence description. Reply with the JSON object only.")
    return f"Write a detailed 400-word description of a day at {theme}."


# name -> (response_format, structured prompt, extra request fields, concurrency levels)
CASES = {
    "schema": (JSON_SCHEMA, True, {"temperature": 0, "reasoning_effort": "none"}, None),
    "object": (JSON_OBJECT, True, {"temperature": 0, "reasoning_effort": "none"}, [1]),
    "plain": (None, False, {"temperature": 0, "reasoning_effort": "none"}, [1]),
    "sampled": (JSON_SCHEMA, True,
                {"temperature": 0.8, "seed": 1234, "reasoning_effort": "none"}, [1, 4]),
    "think": (JSON_SCHEMA, True, {"temperature": 0, "reasoning_effort": "low"}, [1]),
    "logprobs": (JSON_SCHEMA, True, {"temperature": 0, "reasoning_effort": "none",
                                     "logprobs": True, "top_logprobs": 3}, [1]),
}
# One structured request beside three plain ones: the plain ones must not be
# held serial by the strict one, and the strict one must stay valid.
MIXED = "mixed"


def body(model, case, i, max_tokens):
    rfmt, structured, extra, _ = CASES[case]
    b = {"model": model, "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": prompt(i, structured)}], **extra}
    if rfmt is not None:
        b["response_format"] = rfmt
    return b


def post(url, b):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            reply = json.load(r)
    except Exception as e:  # noqa: BLE001 - recorded, the run goes on
        return {"error": f"{type(e).__name__}: {e}", "wall_s": time.time() - t0}
    choice = reply["choices"][0]
    return {
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (reply.get("usage") or {}).get("completion_tokens", 0),
        "wall_s": time.time() - t0,
    }


def check_schema(obj, schema, path="$"):
    """Minimal validator: types, required, enum, no extra keys, array items."""
    t = schema.get("type")
    if t == "integer":
        if not isinstance(obj, int) or isinstance(obj, bool):
            return [f"{path}: expected integer, got {obj!r}"]
    elif t == "boolean":
        if not isinstance(obj, bool):
            return [f"{path}: expected boolean, got {obj!r}"]
    elif t in ("object", "string", "array"):
        want = {"object": dict, "string": str, "array": list}[t]
        if not isinstance(obj, want):
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    errs = []
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object":
        props = schema.get("properties", {})
        errs += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in obj]
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected {k}" for k in obj if k not in props]
        for k, v in obj.items():
            if k in props:
                errs += check_schema(v, props[k], f"{path}.{k}")
    if t == "array":
        for n, v in enumerate(obj):
            errs += check_schema(v, schema.get("items", {}), f"{path}[{n}]")
    return errs


def invalid(case, reply):
    """Why a reply is not a valid structured answer, or None."""
    if "error" in reply:
        return reply["error"]
    rfmt = CASES[case][0]
    if rfmt is None:
        return None
    try:
        obj = json.loads(reply["content"])
    except ValueError as e:
        return f"not JSON ({reply.get('finish_reason')}): {e}"
    if rfmt is JSON_OBJECT:
        return None if isinstance(obj, dict) else "json_object reply is not an object"
    errs = check_schema(obj, SCHEMA)
    return "; ".join(errs[:3]) if errs else None


def batch(url, model, jobs, max_tokens):
    """Fire `jobs` ((case, prompt index) pairs) together; replies plus the batch wall."""
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
        futs = [pool.submit(post, url, body(model, c, i, max_tokens)) for c, i in jobs]
        replies = [f.result() for f in futs]
    wall = time.time() - t0
    for (c, i), r in zip(jobs, replies):
        r.update(case=c, prompt=i, invalid=invalid(c, r))
    tokens = sum(r.get("completion_tokens", 0) for r in replies)
    return {"wall_s": wall, "tokens": tokens, "tok_s": tokens / wall if wall else 0.0,
            "replies": replies}


def run(args):
    with urllib.request.urlopen(args.url + "/v1/models", timeout=30) as r:
        model = args.model or json.load(r)["data"][0]["id"]
    levels = [int(c) for c in args.concurrency.split(",")]
    plan = []
    for case, (_, _, _, own) in CASES.items():
        for c in own or levels:
            plan.append((f"{case}@c{c}", [(case, i) for i in range(c)]))
    plan.append((f"{MIXED}@c4", [("schema", 0), ("plain", 1), ("plain", 2), ("plain", 3)]))
    # One discarded request so the first measured one is not a cold start.
    post(args.url, body(model, "plain", 7, 32))
    rows = {}
    for key, jobs in plan:
        rows[key] = [batch(args.url, model, jobs, args.max_tokens) for _ in range(args.reps)]
        rates = [b["tok_s"] for b in rows[key]]
        bad = sum(1 for b in rows[key] for r in b["replies"] if r["invalid"])
        print(f"{key:14s} tok/s {statistics.median(rates):7.1f} "
              f"({min(rates):.1f}-{max(rates):.1f})  invalid {bad}", flush=True)
    record = {"arm": args.arm, "model": model, "commit": args.commit, "note": args.note,
              "date": datetime.date.today().isoformat(), "reps": args.reps,
              "max_tokens": args.max_tokens, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    errors = sum(1 for bs in rows.values() for b in bs for r in b["replies"] if "error" in r)
    print(f"wrote {args.out}; {errors} request error(s)")
    return 1 if errors else 0


def compare(args):
    with open(args.off) as f:
        off = json.load(f)
    with open(args.on) as f:
        on = json.load(f)
    # Columns carry the records' own arm names; two records of one arm (the
    # noise floor) read `off` and `off#2`.
    a_name, b_name = off["arm"], on["arm"]
    if a_name == b_name:
        b_name += "#2"
    for name, rec in ((a_name, off), (b_name, on)):
        print(f"- `{name}`: commit {rec['commit']}, model {rec['model']}, {rec['date']}, "
              f"reps {rec['reps']}, max_tokens {rec['max_tokens']}; {rec['note']}")
    print(f"\n| case | {a_name} tok/s (min-max) | {b_name} tok/s (min-max) | "
          f"{b_name}/{a_name} | invalid {a_name} | invalid {b_name} | identical |"
          "\n|---|---|---|---|---|---|---|")
    on_invalid, details = 0, []
    for key, off_bs in off["rows"].items():
        on_bs = on["rows"].get(key)
        if on_bs is None:
            continue

        def rate(bs):
            rs = [b["tok_s"] for b in bs]
            return statistics.median(rs), f"{statistics.median(rs):.1f} ({min(rs):.1f}-{max(rs):.1f})"

        def replies(bs):
            return [r for b in bs for r in b["replies"]]

        (m_off, s_off), (m_on, s_on) = rate(off_bs), rate(on_bs)
        r_off, r_on = replies(off_bs), replies(on_bs)
        bad_off = [r for r in r_off if r["invalid"]]
        bad_on = [r for r in r_on if r["invalid"]]
        on_invalid += len(bad_on)
        details += [f"- {key} ({arm}): {r['invalid']}"
                    for arm, bad in ((a_name, bad_off), (b_name, bad_on)) for r in bad[:2]]
        # Sampled replies are not expected to match across arms.
        if key.startswith("sampled"):
            same = "n/a"
        else:
            pairs = list(zip(r_off, r_on))
            eq = sum(1 for a, b in pairs if a.get("content") == b.get("content"))
            same = f"{eq}/{len(pairs)}"
            for a, b in pairs:
                x, y = a.get("content", ""), b.get("content", "")
                if x != y:
                    at = next((n for n, (p, q) in enumerate(zip(x, y)) if p != q),
                              min(len(x), len(y)))
                    details.append(f"- {key} prompt {a['prompt']}: first difference at char "
                                   f"{at}: {a_name} {x[at:at + 40]!r} / "
                                   f"{b_name} {y[at:at + 40]!r}")
                    break
        ratio = f"{m_on / m_off:.2f}" if m_off else "n/a"
        print(f"| {key} | {s_off} | {s_on} | {ratio} | {len(bad_off)}/{len(r_off)} | "
              f"{len(bad_on)}/{len(r_on)} | {same} |")
    if details:
        print("\n" + "\n".join(details))
    print(f"\n{'FAIL' if on_invalid else 'PASS'}: {on_invalid} invalid structured "
          f"reply(ies) in `{b_name}`")
    return 1 if on_invalid else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8081")
    r.add_argument("--arm", required=True, choices=["off", "serial", "on"],
                   help="off: lever unset; serial: lever on, one verify per structured "
                        "request; on: lever on, batched verify")
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--concurrency", default="1,2,4,8",
                   help="levels for the json_schema case; keep within --max-num-seqs")
    r.add_argument("--max-tokens", type=int, default=800)
    r.add_argument("--commit", default="unknown", help="build commit, recorded")
    r.add_argument("--note", default="", help="serve flags and env, recorded (no host names)")
    c = sub.add_parser("compare")
    c.add_argument("off")
    c.add_argument("on")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
