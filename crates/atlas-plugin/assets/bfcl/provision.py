# SPDX-License-Identifier: AGPL-3.0-only
"""Materialize the BFCL v4 single-turn table as JSONL, once, into ~/.atlas.

Ported from mlcommons/endpoints `inference_endpoint/dataset_manager/predefined/
bfcl_v4/__init__.py` (Apache-2.0, NVIDIA CORPORATION). The row shape here is
deliberately identical to that adapter's, because the recorded BFCL scores we
compare against were produced from it:

  * messages come from the nested `question` field, flattened, with a null
    `content` coerced to "" (valid per the OpenAI API, rejected by several
    local servers);
  * tools are converted by bfcl-eval's OWN `convert_to_tool` with the
    GORILLA_TO_OPENAPI mapping, so the schemas the model sees are the reference
    schemas rather than a re-implementation of them;
  * `tool_choice` is sent explicitly — some servers stall when tools are
    present but tool_choice is omitted.

Emits one JSON object per line and prints a summary to stdout for the caller.
The sampling draw is NOT done here: it is deterministic (`head(n)` per subset)
and lives on the Rust side so the pane can show the resulting n before the run.
"""
from __future__ import annotations

import argparse
import hashlib
import copy
import json
import pathlib
import sys

SINGLE_TURN_SUBSETS = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
]


def _tools_from_functions(functions):
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool

    return convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)


def _messages_from_question(question):
    return [
        {"role": msg["role"], "content": msg.get("content") or ""}
        for turn in question
        for msg in turn
    ]


def _data_dir() -> pathlib.Path:
    import bfcl_eval

    return pathlib.Path(bfcl_eval.__file__).parent / "data"


def _load_ground_truths(data_dir: pathlib.Path, subset: str) -> dict:
    path = data_dir / "possible_answer" / f"BFCL_v4_{subset}.json"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "id" in row:
            out[row["id"]] = row.get("ground_truth", [])
    return out


def _preprocessed_functions(sample: dict, subset: str) -> list:
    """The function doc BFCL hands to the MODEL. Not the one it hands the checker.

    `bfcl_eval` rewrites the doc per language before the model sees it
    (`add_language_specific_hint_to_function_doc`). For Java and JavaScript
    that rewrite sets EVERY parameter's type to `string` and says so in the
    description ("This is Java integer type parameter in string
    representation."), because those ground truths are source literals rather
    than JSON values. `ast_checker` enforces the same contract from the other
    side: for those two languages it rejects a non-string argument.

    Skipping the step does not make the prompt smaller, it makes it
    unwinnable -- we told the model `pageNo: integer`, it correctly emitted
    `3`, and the checker demanded `"3"`. On the pinned n=995 draw that alone
    held simple_java to 41.94% and simple_javascript to 25.81% against
    simple_python's 89.92%.

    ONLY the model-visible `tools` get this. `func_description` stays RAW,
    because the checker converts types itself through `JAVA_TYPE_CONVERSION`
    /`JS_TYPE_CONVERSION`, and neither table has a `string` key -- handing it
    a preprocessed doc raises KeyError inside `ast_checker`, which `score.py`
    catches as a zero. That failure is SILENT and scores the whole subset
    0.0, so the split is load-bearing, not stylistic.

    Applied to every subset because that is what BFCL does: on Python ones the
    rewrite only appends " Note that the provided function is in Python 3
    syntax." to each description.

    Delegates to `bfcl_eval` instead of restating the rule, so their
    preprocessing cannot silently desync from ours, and raises rather than
    falling back -- a benchmark that quietly provisions the wrong prompt
    reports a wrong number, which is worse than not running.
    """
    from bfcl_eval.utils import add_language_specific_hint_to_function_doc

    functions = sample.get("function", [])
    entry = {"id": sample.get("id", ""), "function": copy.deepcopy(functions)}
    add_language_specific_hint_to_function_doc([entry])
    processed = entry["function"]

    # Java/JS is the case that matters and the case that silently degrades, so
    # assert the postcondition rather than trusting the call went through.
    if subset.endswith(("java", "javascript")):
        for fn in processed:
            props = fn.get("parameters", {}).get("properties", {})
            bad = {k: v.get("type") for k, v in props.items() if v.get("type") != "string"}
            if bad:
                raise AssertionError(
                    f"{sample.get('id')}: language preprocessing left non-string "
                    f"parameter types {bad}; the model would be asked for JSON "
                    f"values the checker rejects"
                )
    return processed


def _rows_for_subset(data_dir: pathlib.Path, subset: str):
    path = data_dir / f"BFCL_v4_{subset}.json"
    if not path.exists():
        return []
    truths = _load_ground_truths(data_dir, subset)
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        question = sample.get("question", [])
        if not question:
            continue
        sample_id = sample.get("id", "")
        functions = sample.get("function", [])
        tool_functions = _preprocessed_functions(sample, subset)
        ground_truth = truths.get(sample_id, sample.get("ground_truth", []))
        rows.append(
            {
                "subset": subset,
                "sample_id": sample_id,
                "messages": _messages_from_question(question),
                "tools": _tools_from_functions(tool_functions),
                "tool_choice": "auto",
                "ground_truth": json.dumps(ground_truth) if ground_truth else "[]",
                "func_description": json.dumps(functions),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path of the dataset.jsonl to write")
    args = ap.parse_args()

    try:
        data_dir = _data_dir()
    except ImportError as e:
        print(f"bfcl-eval is not importable: {e}", file=sys.stderr)
        return 2
    if not data_dir.is_dir():
        print(f"bfcl-eval has no data directory at {data_dir}", file=sys.stderr)
        return 2

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    digest = hashlib.sha256()
    total = 0
    # Subsets are written in SINGLE_TURN_SUBSETS order and, within a subset, in
    # file order. The draw takes `head(n)` per subset, so this ordering IS the
    # sample selection — shuffling here would silently change the benchmark.
    with out_path.open("w") as f:
        for subset in SINGLE_TURN_SUBSETS:
            rows = _rows_for_subset(data_dir, subset)
            counts[subset] = len(rows)
            total += len(rows)
            for row in rows:
                line = json.dumps(row, sort_keys=True)
                f.write(line + "\n")
                digest.update(line.encode())

    if total == 0:
        print("no BFCL samples were found in the bfcl-eval data directory", file=sys.stderr)
        return 2

    print(json.dumps({"total": total, "sha256": digest.hexdigest(), "subsets": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
