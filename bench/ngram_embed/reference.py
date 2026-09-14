"""Reference for n-gram scaled embeddings (LongCat / Qwen3.8-Flash-Next family).

PROVENANCE
----------
Written against two primary sources:

  * "Scaling Language Models with Scaled Embeddings", arXiv:2601.21204.
  * The MIT-licensed reference implementation published by Meituan at
    https://huggingface.co/meituan-longcat/LongCat-Flash-Lite
    (`modeling_longcat_ngram.py`, Copyright (c) 2025 Meituan). This file is an
    independent derivation of the mechanism that file describes; see LICENSE in
    this directory for the MIT notice that derivation carries.

No third-party dependency, deliberately: these fixtures must be regenerable from
a bare Python in CI, and every quantity here is either exact integer arithmetic
or a matmul over test-sized matrices. A numpy reference that cannot be re-run on
the machine reviewing it is not much of a reference.

THE MECHANISM
-------------
`K*(N-1)` lookup tables (K = emb_split_num, N = emb_neighbor_num). Table
`index = (i-2)*K + j`, for n-gram size `i` and split `j`, holds

    T(index) = ratio * vocab + 2*index + 1

rows at width `hidden // (K*(N-1))`. The consecutive ODD offsets give the K
tables of one n-gram size mutually near-coprime row counts, so a hash collision
in one split is independent of the others.

A row id is a polynomial rolling hash over TOKEN IDS ONLY -- never hidden state:

    id_t(i, j) = ( x_t + sum_{d=1..i-1} shift_d(x)_t * (V^d mod T) ) mod T

`shift_d` is a right shift by d that RESETS at document boundaries: positions
within d tokens of a segment start contribute 0 rather than reaching across an
EOS. Segments end AT an EOS (inclusive).

Because ids depend only on token ids they are deterministic, prefetchable and
speculative-decode-safe -- and a decode step needs only the last N-1 tokens of
history, which is what makes this cheap to serve.

Fusion, with every addend scaled by 1/(1 + K*(N-1)):

    out_t = ( word[x_t] + sum_index proj_index( table_index[ id_t(index) ] ) ) / 13
"""

from __future__ import annotations


class NgramDims:
    """The n-gram trio plus the dimensions derived from it."""

    def __init__(self, vocab_size, hidden_size, ratio, neighbor_num, split_num, eos_token_id):
        if neighbor_num < 2:
            raise ValueError("emb_neighbor_num must be >= 2 (it is the largest n-gram size)")
        if split_num < 1:
            raise ValueError("emb_split_num must be >= 1")
        if ratio < 1:
            raise ValueError("ngram_vocab_size_ratio must be >= 1")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.ratio = int(ratio)
        self.neighbor_num = int(neighbor_num)
        self.split_num = int(split_num)
        self.eos_token_id = int(eos_token_id)
        if self.hidden_size % self.num_tables:
            raise ValueError(
                f"hidden_size {self.hidden_size} must divide evenly by "
                f"{self.num_tables} n-gram tables"
            )

    @property
    def num_tables(self) -> int:
        return self.split_num * (self.neighbor_num - 1)

    @property
    def table_dim(self) -> int:
        return self.hidden_size // self.num_tables

    def table_rows(self, index: int) -> int:
        return self.ratio * self.vocab_size + 2 * index + 1

    def vocab_mods(self, ngram: int, split: int):
        """[V^1 mod T, ..., V^(ngram-1) mod T] for one table."""
        t = self.table_rows((ngram - 2) * self.split_num + split)
        mods, power = [], 1
        for _ in range(ngram - 1):
            power = (power * self.vocab_size) % t
            mods.append(power)
        return mods


def shift_right_ignore_eos(ctx, n, eos):
    """out[t] = ctx[t-n], except where [t-n, t] would cross a document boundary.

    Segments end AT an EOS token (inclusive). A segment of length <= n
    contributes nothing: no position in it sits far enough from the start to
    look back n tokens without leaving the document.
    """
    out = [0] * len(ctx)
    prev = 0
    for pos, tok in enumerate(ctx):
        if tok == eos:
            end = pos + 1
            if end - prev > n:
                out[prev + n : end] = ctx[prev : end - n]
            prev = end
    if len(ctx) - prev > n:
        out[prev + n :] = ctx[prev : len(ctx) - n]
    return out


def ngram_ids(dims, ctx):
    """Row ids for every table over `ctx`, as a list of `num_tables` lists."""
    # A shift by d is shared across every split of every n-gram size using it.
    shifts = {d: shift_right_ignore_eos(ctx, d, dims.eos_token_id) for d in range(1, dims.neighbor_num)}

    out = [None] * dims.num_tables
    for ngram in range(2, dims.neighbor_num + 1):
        for split in range(dims.split_num):
            index = (ngram - 2) * dims.split_num + split
            t = dims.table_rows(index)
            mods = dims.vocab_mods(ngram, split)
            ids = []
            for pos, x in enumerate(ctx):
                acc = x
                for d, m in enumerate(mods, start=1):
                    acc += shifts[d][pos] * m
                ids.append(acc % t)
            out[index] = ids
    return out


def max_accumulator(dims):
    """Worst-case value of the pre-modulo accumulator, across every table.

    This is the number that decides whether the hash is evaluable in 64-bit
    integers. Each addend is bounded by (V-1)*mod and there are at most N-1 of
    them, plus the token id itself. An implementation should REFUSE a config
    whose accumulator would wrap rather than discover it later as garbage
    logits -- Python's ints would not wrap here, but Rust's u64 would.
    """
    worst = 0
    for ngram in range(2, dims.neighbor_num + 1):
        for split in range(dims.split_num):
            mods = dims.vocab_mods(ngram, split)
            acc = (dims.vocab_size - 1) + sum((dims.vocab_size - 1) * m for m in mods)
            worst = max(worst, acc)
    return worst


def fuse(dims, ctx, seq_len, word, tables, projs):
    """Fused embedding for the LAST `seq_len` positions of `ctx`.

    word:   [vocab][hidden]                tables: num_tables x [T(index)][table_dim]
    projs:  num_tables x [hidden][table_dim]   (nn.Linear layout: out_features first)
    """
    ids = ngram_ids(dims, ctx)
    out = [list(word[t]) for t in ctx[-seq_len:]]
    for index in range(dims.num_tables):
        table, proj = tables[index], projs[index]
        for pos, row_id in enumerate(ids[index][-seq_len:]):
            row = table[row_id]
            dest = out[pos]
            for h in range(dims.hidden_size):
                pr = proj[h]
                dest[h] += sum(row[c] * pr[c] for c in range(dims.table_dim))
    scale = 1.0 / (1 + dims.num_tables)
    return [[v * scale for v in row] for row in out]
