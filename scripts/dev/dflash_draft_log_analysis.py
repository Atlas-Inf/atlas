#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline analysis of logged DFlash verifies (Atlas GLM), for draft-policy ideas.

Input: a server log captured with
    ATLAS_DFLASH_FIXED_WIDTH=7 ATLAS_LIGHTNING_VERIFY_TOKEN_TRACE=1 ATLAS_DFLASH_CONF_LOG=1
(single stream; one line pair per verify, `Done:` closes each request). Canonical
verify makes the emitted text independent of the verify width, so a width-7 log is
the uncensored record: every draft is compared with the true continuation.

Reports
  1. per-position survival / conditional acceptance, and "late hits" (a draft that is
     right although an earlier one was wrong: shift/synonym errors);
  2. first-failure token classes and top (draft -> target) confusions;
  3. width policies replayed on the logged steps (replay approximation: each logged
     step is scored independently as if its width had been chosen at that step):
       fixed w, oracle w (= accepted, >=1), the production confidence rule
       (a port of dflash_conf_width.rs), and a cross-validated logistic rule on
       richer features (conf, depth, draft token class, previous-step outcome);
     each as tokens/step, rows/step and tok/s under step_ms = A + B*rows.
Usage: draft_log_analysis.py LOG [--tokenizer tokenizer.json] [--A 33.8 --B 6.3]
       draft_log_analysis.py --selftest
"""
import argparse, ast, json, math, re, sys
from collections import Counter, defaultdict

import numpy as np

TRACE = re.compile(r"LIGHTNING VERIFY TOKEN TRACE slot=(\d+) last=(\d+) drafts=(\[[^\]]*\]) raw=(\[[^\]]*\]) processed=(\[[^\]]*\]) accepted=(\d+)")
CONF = re.compile(r"DFLASH CONF drafted=(\d+) accepted=(\d+) conf=(\[[^\]]*\])")
DONE = re.compile(r"Done: (\d+) tokens")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MAXD = 7

# ---------------------------------------------------------------- parsing

def parse(path):
    reqs, cur, pending = [], [], None
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = TRACE.search(line)
        if m:
            if pending: cur.append(pending)
            pending = dict(last=int(m[2]), drafts=ast.literal_eval(m[3]),
                           verified=ast.literal_eval(m[5]), acc=int(m[6]), conf=None)
            continue
        m = CONF.search(line)
        if m and pending is not None and pending["conf"] is None:
            pending["conf"] = [float(x) for x in ast.literal_eval(m[3])]
            continue
        if DONE.search(line):
            if pending: cur.append(pending); pending = None
            if cur: reqs.append(cur)
            cur = []
    if pending: cur.append(pending)
    if cur: reqs.append(cur)
    return reqs

def annotate(reqs):
    """Rebuild each request's continuation and mark every draft right/wrong
    against it (not only the accepted prefix)."""
    steps = []
    for r, req in enumerate(reqs):
        seq, pos = [], []
        for s in req:
            pos.append(len(seq))
            seq += s["verified"][: s["acc"] + 1]
        for i, s in enumerate(req):
            p = pos[i]
            truth = seq[p: p + len(s["drafts"])]
            s["hit"] = [d == t for d, t in zip(s["drafts"], truth)]
            s["truth"] = truth
            s["req"], s["idx"] = r, i
            s["prev_acc"] = req[i - 1]["acc"] if i else -1
            s["prev_n"] = len(req[i - 1]["drafts"]) if i else 0
            s["copy"] = bool(s["conf"]) and s["conf"][0] >= 1.0
            steps.append(s)
    return steps

# ---------------------------------------------------------------- token classes

def load_classes(path):
    if not path: return None
    tj = json.load(open(path))
    vocab = tj["model"]["vocab"]
    added = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    cls = {}
    for tok, i in list(vocab.items()) + list(added.items()):
        if tok in added: c = "special"
        elif tok.startswith("Ċ") or "\n" in tok: c = "newline"
        elif tok.startswith("Ġ"):
            body = tok[1:]
            c = "word" if body[:1].isalnum() else "sp_punct"
        elif tok[:1].isalnum(): c = "cont"
        else: c = "punct"
        cls[i] = c
    return cls

# ---------------------------------------------------------------- policies

EDGES = [-1.5, -1.0, -0.7, -0.5, -0.35, -0.22, -0.12, -0.06, -0.03, -0.01]
PRIOR = [0.12, 0.16, 0.25, 0.35, 0.42, 0.48, 0.55, 0.62, 0.68, 0.76, 0.95, 0.90, 0.55]
COPY_BIN, COPY_REPLY_BIN, BINS = 11, 12, 13

def cbin(c):
    if c >= 2.0: return COPY_REPLY_BIN
    if c >= 1.0: return COPY_BIN
    return sum(c >= e for e in EDGES)

class ConfPolicy:
    """Port of dflash_conf_width.rs (Calibration + Policy::choose, one owner)."""
    def __init__(self, tau=0.3, probe=64, A=33.8, B=6.3):
        self.v = np.zeros((MAXD, BINS)); self.a = np.zeros((MAXD, BINS))
        self.tau, self.probe, self.A, self.B = tau, probe, A, B
        self.steps = 0
    def rate(self, d, b):
        pooled = (self.a[:, b].sum() + 8 * PRIOR[b]) / (self.v[:, b].sum() + 8)
        return (self.a[d, b] + 8 * pooled) / (self.v[d, b] + 8)
    def survival(self, conf):
        out, alive = [], 1.0
        for d, c in enumerate(conf[:MAXD]):
            alive *= self.rate(d, cbin(c)); out.append(alive)
        return out
    def choose(self, conf, mx=MAXD):
        self.steps += 1
        if self.probe and self.steps % self.probe == 0: return mx
        price = self.tau / self.B
        surv = self.survival(conf); exp, best, bv = 1.0, 1, -1e9
        for w in range(1, mx + 1):
            exp += surv[w - 1]
            v = exp - price * (self.A + self.B * (w + 1))
            if v > bv: best, bv = w, v
        return best
    def record(self, conf, drafted, acc):
        copy = conf[0] >= 1.0
        rng = [cbin(conf[0])] if copy else list(range(COPY_BIN))
        self.v[:, rng] *= 0.995; self.a[:, rng] *= 0.995
        for d in range(min(drafted, acc + 1, MAXD)):
            b = cbin(conf[d]); self.v[d, b] += 1
            if d < acc: self.a[d, b] += 1

def features(s, d, cls):
    c = s["conf"][d] if s["conf"] else 0.0
    f = [1.0, c, min(c, 0) ** 2, d, float(s["prev_acc"] >= 0 and s["prev_acc"] == s["prev_n"]),
         max(s["prev_acc"], 0) / MAXD,
         float(s["conf"] is not None and d > 0 and s["conf"][d - 1] > -0.06)]
    if cls is not None:
        k = cls.get(s["drafts"][d], "other")
        f += [float(k == x) for x in ("word", "cont", "punct", "newline", "sp_punct")]
        if d > 0:
            f.append(float(k == "cont" and cls.get(s["drafts"][d - 1], "") in ("word", "cont")))
        else:
            f.append(float(k == "cont" and cls.get(s["last"], "") in ("word", "cont")))
    return f

def fit_logistic(X, y, l2=1e-2, iters=50):
    X, y = np.asarray(X), np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w

def conditional_rows(steps, cls):
    X, y = [], []
    for s in steps:
        for d in range(min(len(s["drafts"]), s["acc"] + 1, MAXD)):
            X.append(features(s, d, cls)); y.append(d < s["acc"])
    return X, y

def score(steps, widths, A, B):
    tok = sum(min(s["acc"], w) + 1 for s, w in zip(steps, widths))
    rows = sum(w + 1 for w in widths)
    ms = sum(A + B * (w + 1) for w in widths)
    n = len(steps)
    return tok / n, rows / n, 1000 * tok / ms

# ---------------------------------------------------------------- report

def report(steps, cls, A, B):
    full = [s for s in steps if len(s["drafts"]) >= MAXD and not s["copy"]]
    print(f"verifies {len(steps)} (full-width drafter verifies {len(full)}, copies {sum(s['copy'] for s in steps)})")
    if not full:
        print("no width-7 drafter verifies: run with ATLAS_DFLASH_FIXED_WIDTH=7"); return
    accs = np.array([s["acc"] for s in full])
    print(f"mean accepted at width 7: {accs.mean():.3f}  tokens/step if verified at 7: {accs.mean() + 1:.3f}")
    print("accepted-count histogram:", np.bincount(accs, minlength=MAXD + 1).tolist())
    surv = [(accs >= k).mean() for k in range(1, MAXD + 1)]
    cond = [((accs >= k).sum() / max((accs >= k - 1).sum(), 1)) for k in range(1, MAXD + 1)]
    print("survival      :", " ".join(f"{x:.3f}" for x in surv))
    print("conditional p_k:", " ".join(f"{x:.3f}" for x in cond))
    late = [np.mean([s["hit"][k] for s in full if s["acc"] < k and len(s["hit"]) > k]) for k in range(1, MAXD)]
    print("P(draft k right | an earlier draft wrong), k=2..7:", " ".join(f"{x:.3f}" for x in late))
    shift = np.mean([any(s["drafts"][k + 1] == s["truth"][k] for k in range(s["acc"], min(len(s["truth"]), MAXD - 1)))
                     for s in full if s["acc"] < MAXD - 1])
    print(f"first miss followed by a one-token shift match (draft k+1 == truth k): {shift:.3f}")

    if cls is not None:
        fc, ok = Counter(), Counter()
        for s in full:
            for d in range(min(s["acc"] + 1, MAXD)):
                k = cls.get(s["drafts"][d], "other")
                (ok if d < s["acc"] else fc)[k] += 1
        print("conditional acceptance by draft token class:")
        for k in sorted(set(fc) | set(ok), key=lambda k: -(fc[k] + ok[k])):
            print(f"   {k:9s} n={fc[k] + ok[k]:6d}  p={ok[k] / (fc[k] + ok[k]):.3f}")
    conf_pairs = Counter((s["drafts"][s["acc"]], s["truth"][s["acc"]]) for s in full
                         if s["acc"] < len(s["truth"]))
    print("top first-failure (draft, target) id pairs:", conf_pairs.most_common(12))

    print(f"\nwidth policies, replayed over the {len(full)} logged steps (step_ms = {A} + {B}*rows):")
    print("   policy              tok/step rows/step   tok/s")
    for w in range(1, MAXD + 1):
        t, r, v = score(full, [w] * len(full), A, B); print(f"   fixed {w}             {t:8.3f} {r:9.3f} {v:7.2f}")
    t, r, v = score(full, [max(s["acc"], 1) for s in full], A, B)
    print(f"   oracle (w=acc)      {t:8.3f} {r:9.3f} {v:7.2f}")
    if all(s["conf"] for s in full):
        pol = ConfPolicy(A=A, B=B); ws = []
        for s in full:
            w = pol.choose(s["conf"]); ws.append(w); pol.record(s["conf"], w, min(s["acc"], w))
        t, r, v = score(full, ws, A, B)
        print(f"   conf rule (prod)    {t:8.3f} {r:9.3f} {v:7.2f}")
        # cross-validated logistic rule, split by request parity
        res = []
        for fold in (0, 1):
            tr = [s for s in full if s["req"] % 2 != fold]; te = [s for s in full if s["req"] % 2 == fold]
            if not tr or not te: continue
            X, y = conditional_rows(tr, cls); wv = fit_logistic(X, y)
            price = 0.3 / B
            for s in te:
                alive, exp, best, bv = 1.0, 1.0, 1, -1e9
                for d in range(MAXD):
                    alive *= 1 / (1 + math.exp(-np.dot(wv, features(s, d, cls))))
                    exp += alive
                    val = exp - price * (A + B * (d + 2))
                    if val > bv: best, bv = d + 1, val
                res.append((s, best))
        if res:
            t, r, v = score([s for s, _ in res], [w for _, w in res], A, B)
            print(f"   logistic (2-fold)   {t:8.3f} {r:9.3f} {v:7.2f}")
        # discrimination of conf alone at each depth
        aucs = []
        for d in range(MAXD):
            pts = [(s["conf"][d], d < s["acc"]) for s in full if s["acc"] >= d]
            pos = [c for c, a in pts if a]; neg = [c for c, a in pts if not a]
            if pos and neg:
                allc = np.array(pos + neg); rk = allc.argsort().argsort() + 1
                aucs.append((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        print("AUC of selector conf for draft k accepted | k-1 accepted:", " ".join(f"{a:.3f}" for a in aucs))

# ---------------------------------------------------------------- self-test

def selftest():
    import random, tempfile
    rnd = random.Random(1); lines = []
    for r in range(6):
        for i in range(60):
            conf = [-abs(rnd.gauss(0.3, 0.4)) for _ in range(MAXD)]
            acc = 0
            while acc < MAXD and rnd.random() < math.exp(conf[acc]) * 0.95: acc += 1
            drafts = [rnd.randrange(1000) for _ in range(MAXD)]
            ver = drafts[:acc] + [rnd.randrange(1000, 2000)] + [rnd.randrange(1000) for _ in range(MAXD - acc)]
            lines.append(f"x LIGHTNING VERIFY TOKEN TRACE slot=0 last=5 drafts={drafts} raw={ver} processed={ver} accepted={acc}")
            lines.append(f"x DFLASH CONF drafted={MAXD} accepted={acc} conf=[{', '.join(f'{c:.3f}' for c in conf)}]")
        lines.append("x Done: 300 tokens (length) 40.0 tok/s, TTFT=1.0ms, p1=0.7")
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False); f.write("\n".join(lines)); f.close()
    report(annotate(parse(f.name)), None, 33.8, 6.3)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?"); ap.add_argument("--tokenizer")
    ap.add_argument("--A", type=float, default=33.8); ap.add_argument("--B", type=float, default=6.3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); sys.exit()
    report(annotate(parse(a.log)), load_classes(a.tokenizer), a.A, a.B)
