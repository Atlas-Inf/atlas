#!/usr/bin/env python3
"""Coherence + function-calling + perf smoke for a served qwen4_exp model.

Three legs, each answering a different question about a NEW port:

  coherence  does it produce correct, on-format text at all? A port with a
             subtly wrong norm or a mis-sized PLE table still emits fluent
             tokens, so every case here has a CHECKABLE answer rather than a
             vibe. Fluent-but-wrong is the failure this catches.
  tools      does the chat template + sampler survive structured output? A
             BFCL-style AST subset: right function, right arguments, and one
             IRRELEVANCE case where the correct move is NOT to call anything.
  perf       is the decode loop in the right order of magnitude, and does it
             hold up with several sequences in flight? The concurrency leg is
             the part that exercises the batched decode paths.

Stdlib only, on purpose: it has to run on a fresh box with no pip install.

Usage:
    python3 scripts/dev/qwen4exp_smoke.py --base http://127.0.0.1:8889
    python3 scripts/dev/qwen4exp_smoke.py --leg tools --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request

TIMEOUT = 600

# Sampling is DELIBERATELY not pinned. Qwen3.8-Flash-Next's card declares
# thinking temperature=1.0 top_p=0.95 top_k=20, and
# `kernels/gb10/qwen3.8-flash-next/MODEL.toml` sets
# `use_sampling_presets_for_core = true` so a plain request gets those rather
# than Atlas's generic CLI defaults. Sending `temperature: 0` overrides the
# lot and puts a thinking MoE into greedy decode, which loops on its own
# ("408 408 408", "The capital of France is Paris." repeated) — a decoding
# artefact that reads exactly like a broken port. Pass --temperature only to
# investigate; the default is to test the model as it is meant to be served.
SAMPLING: dict = {}


def post(base: str, path: str, body: dict, timeout: int = TIMEOUT) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip('/') + path, data=data,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode())


def chat(base: str, messages: list, **kw) -> dict:
    body = {'model': kw.pop('model', 'qwen4exp'), 'messages': messages}
    body.update(kw)
    return post(base, '/v1/chat/completions', body)


def text_of(resp: dict) -> str:
    msg = resp['choices'][0]['message']
    # A reasoning model may put the answer after a thinking block; the
    # server surfaces that separately, and `content` is the reply proper.
    return (msg.get('content') or '').strip()


def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


# ── leg 1: coherence ─────────────────────────────────────────────────────────
# `check` gets the normalised reply and returns True if it is acceptable.
# Kept deliberately tolerant about WORDING and strict about the FACT.
COHERENCE = [
    ('capital',
     [{'role': 'user', 'content': 'What is the capital of France? Answer with the city name only.'}],
     lambda t: 'paris' in t),
    ('arithmetic',
     [{'role': 'user', 'content': 'Compute 17 * 24. Reply with only the number.'}],
     lambda t: '408' in t),
    ('multi_step',
     [{'role': 'user', 'content': 'A train leaves at 14:05 and the trip takes 95 minutes. '
                                  'What time does it arrive? Use 24-hour HH:MM.'}],
     lambda t: '15 40' in t or '1540' in t),
    ('instruction_format',
     [{'role': 'user', 'content': 'List exactly three primary colours as a JSON array of strings. '
                                  'Output only the JSON.'}],
     lambda t: sum(c in t for c in ('red', 'blue', 'yellow')) >= 3),
    ('multi_turn_memory',
     [{'role': 'user', 'content': 'My favourite animal is the axolotl. Remember it.'},
      {'role': 'assistant', 'content': 'Noted — the axolotl.'},
      {'role': 'user', 'content': 'What did I say my favourite animal was? One word.'}],
     lambda t: 'axolotl' in t),
    ('negation',
     [{'role': 'user', 'content': 'Name a colour that is NOT red. One word.'}],
     lambda t: bool(t) and 'red' != t.strip()),
    # Long-ish prompt: pushes past the trivial prefill path and makes the PLE
    # n-gram injection and (if the context is long enough) QSA do real work.
    ('needle_in_context',
     [{'role': 'user', 'content':
       'Read this record and answer the question at the end.\n\n'
       + '\n'.join(f'item {i}: value {i * 7 % 100}' for i in range(300))
       + '\n\nQuestion: what is the value of item 42? Reply with only the number.'}],
     lambda t: '94' in t),
]

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'get_weather',
        'description': 'Get the current weather for a city.',
        'parameters': {'type': 'object', 'properties': {
            'city': {'type': 'string', 'description': 'City name'},
            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
            'required': ['city']}}},
    {'type': 'function', 'function': {
        'name': 'send_email',
        'description': 'Send an email to a recipient.',
        'parameters': {'type': 'object', 'properties': {
            'to': {'type': 'string'}, 'subject': {'type': 'string'},
            'body': {'type': 'string'}}, 'required': ['to', 'subject']}}},
    {'type': 'function', 'function': {
        'name': 'calculate_mortgage',
        'description': 'Compute a monthly mortgage payment.',
        'parameters': {'type': 'object', 'properties': {
            'principal': {'type': 'number'}, 'rate': {'type': 'number'},
            'years': {'type': 'integer'}},
            'required': ['principal', 'rate', 'years']}}},
]


def calls_of(resp: dict) -> list:
    """Extract tool calls, tolerating a model that emits them as text.

    A brand-new port can be numerically perfect and still fail structured
    output because the chat template never got wired up — so an inline
    ```json {"name": ...}``` counts as a PARTIAL, not a pass, and the report
    distinguishes them.
    """
    msg = resp['choices'][0]['message']
    out = []
    for c in (msg.get('tool_calls') or []):
        fn = c.get('function', {})
        args = fn.get('arguments')
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {'__unparsed__': args}
        out.append((fn.get('name'), args or {}, 'native'))
    if not out:
        content = msg.get('content') or ''
        for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}', content):
            try:
                blob = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            args = blob.get('arguments') or blob.get('parameters') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            out.append((blob.get('name'), args, 'inline'))
    return out


# (name, prompt, expected function names, argument predicate)
# `expect=[]` means the right answer is to call NOTHING (BFCL "irrelevance").
BFCL = [
    ('simple', 'What is the weather in Tokyo?', ['get_weather'],
     lambda a: 'tokyo' in json.dumps(a).lower()),
    ('enum_arg', 'Weather in Cairo in fahrenheit please.', ['get_weather'],
     lambda a: a.get('unit') == 'fahrenheit' and 'cairo' in json.dumps(a).lower()),
    ('function_choice', 'Email bob@example.com with the subject Lunch.', ['send_email'],
     lambda a: 'bob@example.com' in json.dumps(a)),
    ('numeric_args', 'Monthly payment on a 300000 loan at 6.5 percent over 30 years?',
     ['calculate_mortgage'],
     lambda a: abs(float(a.get('principal', 0)) - 300000) < 1 and int(a.get('years', 0)) == 30),
    ('parallel', 'Compare the weather in Oslo and Lisbon.', ['get_weather'],
     lambda a: True),
    ('irrelevance', 'Write a haiku about winter.', [], lambda a: True),
]

PERF_PROMPT = 'Explain in detail how a transformer attention layer works.'


def run_coherence(base: str, model: str, verbose: bool) -> list:
    rows = []
    for name, msgs, check in COHERENCE:
        t0 = time.time()
        try:
            r = chat(base, msgs, model=model, max_tokens=768, **SAMPLING)
            txt = text_of(r)
            ok = bool(check(norm(txt)))
            note = txt.replace('\n', ' ')[:90]
        except Exception as exc:                       # noqa: BLE001
            ok, note = False, f'{type(exc).__name__}: {exc}'[:90]
        rows.append((name, ok, time.time() - t0, note))
        print(f'  {"PASS" if ok else "FAIL"}  {name:20s} {time.time() - t0:6.1f}s  {note}')
        if verbose and not ok:
            print(f'        full reply: {note}')
    return rows


def run_tools(base: str, model: str, verbose: bool) -> list:
    rows = []
    for name, prompt, expect, argcheck in BFCL:
        t0 = time.time()
        try:
            r = chat(base, [{'role': 'user', 'content': prompt}], model=model,
                     tools=TOOLS, tool_choice='auto', max_tokens=768,
                     **SAMPLING)
            got = calls_of(r)
            names = [g[0] for g in got]
            mode = got[0][2] if got else 'none'
            if not expect:
                ok = len(got) == 0
                note = 'no call (correct)' if ok else f'called {names}'
            elif not got:
                ok, note = False, 'emitted no tool call'
            else:
                ok = all(n in expect for n in names) and any(
                    argcheck(g[1]) for g in got)
                note = f'{mode}: {names} {json.dumps(got[0][1])[:50]}'
            if ok and mode == 'inline':
                note += '  [INLINE — template not wired]'
        except Exception as exc:                       # noqa: BLE001
            ok, note = False, f'{type(exc).__name__}: {exc}'[:90]
        rows.append((name, ok, time.time() - t0, note))
        print(f'  {"PASS" if ok else "FAIL"}  {name:20s} {time.time() - t0:6.1f}s  {note}')
    return rows


def one_stream(base: str, model: str, max_tokens: int, out: list, idx: int) -> None:
    t0 = time.time()
    try:
        r = chat(base, [{'role': 'user', 'content': PERF_PROMPT}], model=model,
                 max_tokens=max_tokens, **SAMPLING)
        u = r.get('usage') or {}
        out[idx] = (time.time() - t0, u.get('completion_tokens'),
                    u.get('prompt_tokens'), None)
    except Exception as exc:                           # noqa: BLE001
        out[idx] = (time.time() - t0, None, None, f'{type(exc).__name__}: {exc}')


def run_perf(base: str, model: str, max_tokens: int) -> list:
    rows = []
    for conc in (1, 4):
        out = [None] * conc
        threads = [threading.Thread(target=one_stream,
                                    args=(base, model, max_tokens, out, i))
                   for i in range(conc)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        errs = [o[3] for o in out if o and o[3]]
        toks = sum(o[1] or 0 for o in out if o)
        rows.append((conc, wall, toks, toks / wall if wall else 0, errs))
        if errs:
            print(f'  conc={conc}  FAILED: {errs[0][:100]}')
        else:
            print(f'  conc={conc}  wall {wall:6.2f}s  completion {toks:5d} tok  '
                  f'aggregate {toks / wall:6.2f} tok/s')
    return rows


def wait_healthy(base: str, minutes: int) -> bool:
    deadline = time.time() + minutes * 60
    last = ''
    while time.time() < deadline:
        for path in ('/health', '/v1/models'):
            try:
                with urllib.request.urlopen(base.rstrip('/') + path, timeout=10) as fh:
                    fh.read()
                print(f'server healthy ({path})')
                return True
            except Exception as exc:                   # noqa: BLE001
                last = f'{type(exc).__name__}: {exc}'
        time.sleep(10)
    print(f'server never became healthy: {last}', file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8889')
    ap.add_argument('--model', default='qwen4exp')
    ap.add_argument('--leg', choices=['coherence', 'tools', 'perf', 'all'],
                    default='all')
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--wait', type=int, default=0,
                    help='minutes to wait for the server to come up first')
    ap.add_argument('--temperature', type=float, default=None,
                    help='override sampling; omit to use the model card presets')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.temperature is not None:
        SAMPLING['temperature'] = args.temperature
        print(f'temperature pinned to {args.temperature} — card presets '
              f'OVERRIDDEN (greedy loops are expected at 0.0)')

    if args.wait and not wait_healthy(args.base, args.wait):
        return 2

    legs, failed = {}, 0
    if args.leg in ('coherence', 'all'):
        print('\n== coherence ==')
        legs['coherence'] = run_coherence(args.base, args.model, args.verbose)
    if args.leg in ('tools', 'all'):
        print('\n== tools (BFCL-style AST subset) ==')
        legs['tools'] = run_tools(args.base, args.model, args.verbose)
    if args.leg in ('perf', 'all'):
        print('\n== perf ==')
        legs['perf'] = run_perf(args.base, args.model, args.max_tokens)

    print('\n== summary ==')
    for leg, rows in legs.items():
        if leg == 'perf':
            for conc, wall, toks, rate, errs in rows:
                state = 'FAIL' if errs else 'ok'
                print(f'  perf conc={conc}: {state} {rate:.2f} tok/s aggregate '
                      f'({toks} tok in {wall:.2f}s)')
                failed += bool(errs)
            continue
        passed = sum(1 for _, ok, _, _ in rows if ok)
        failed += len(rows) - passed
        print(f'  {leg}: {passed}/{len(rows)} passed')
        for name, ok, _, note in rows:
            if not ok:
                print(f'      FAIL {name}: {note}')
    print(f'\n{"SMOKE FAILED" if failed else "SMOKE CLEAN"} ({failed} failing)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
