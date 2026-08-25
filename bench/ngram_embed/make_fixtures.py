#!/usr/bin/env python3
"""Emit cross-language fixtures for the Rust n-gram implementation.

    python3 bench/ngram_embed/make_fixtures.py > bench/ngram_embed/fixtures.json

Two fixture families:

  `id_cases`    -- integer row ids at REAL LongCat-Flash-Lite dimensions plus
                   synthetic shapes, over token streams chosen to exercise the
                   document-boundary resets (leading EOS, back-to-back EOS,
                   trailing EOS, segments shorter than the shift distance).
  `fuse_cases`  -- the full fused embedding at toy dimensions with weights
                   carried in the fixture, so the Rust side validates the whole
                   gather/project/scale chain and not merely the hashing.

Weights are exact binary fractions (k/64), so they survive f32 and f64 alike and
a cross-language mismatch means a real disagreement rather than a rounding
artefact.
"""

import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from reference import NgramDims, fuse, max_accumulator, ngram_ids


def det_weights(rows, cols, seed):
    """Deterministic exact-binary-fraction weights in [-0.5, 0.5).

    A 64-bit LCG (Knuth's constants) purely so the fixture is reproducible; the
    VALUES are written into the fixture, so no other language has to reproduce
    this generator.
    """
    state = seed & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            row.append(((state >> 40) % 64 - 32) / 64.0)
        out.append(row)
    return out


def id_case(name, dims, tokens):
    return {
        "name": name,
        "vocab_size": dims.vocab_size,
        "hidden_size": dims.hidden_size,
        "ngram_vocab_size_ratio": dims.ratio,
        "emb_neighbor_num": dims.neighbor_num,
        "emb_split_num": dims.split_num,
        "eos_token_id": dims.eos_token_id,
        "tokens": tokens,
        "max_accumulator": max_accumulator(dims),
        "expected_ids": ngram_ids(dims, tokens),
    }


def main():
    cases = []

    # Real LongCat-Flash-Lite dimensions (config.json @ HF main, 2026-08-25):
    # vocab 131072, hidden 3072, ratio 78, N=4, K=4, eos 2.
    lite = NgramDims(131072, 3072, 78, 4, 4, 2)
    cases.append(id_case("lite_plain", lite, [11, 523, 9001, 44, 130000, 7, 88, 4]))
    # EOS mid-stream: every position after it must reset rather than reach back.
    cases.append(id_case("lite_mid_eos", lite, [11, 523, 2, 44, 130000, 7, 88, 4]))
    # Leading EOS, adjacent EOS pair, and a trailing EOS -- the three boundary
    # shapes where an off-by-one in the segment walk shows up.
    cases.append(id_case("lite_edge_eos", lite, [2, 5, 2, 2, 9, 13, 21, 2]))
    # A decode step: n-1 carried context tokens plus one new token, the
    # narrowest window the mechanism is defined on.
    cases.append(id_case("lite_decode_window", lite, [412, 98, 7, 1234]))
    # Every token an EOS: no segment is ever longer than the shift distance, so
    # every shifted contribution must vanish and ids collapse to x_t % T.
    cases.append(id_case("lite_all_eos", lite, [2, 2, 2, 2, 2]))

    # Synthetic shapes: a different (N, K) and a 2-gram-only degenerate case,
    # to keep the index math from being overfit to N=4, K=4.
    cases.append(id_case("synth_n3_k2", NgramDims(64, 24, 3, 3, 2, 2), [5, 9, 2, 31, 17, 0, 63, 8]))
    cases.append(id_case("synth_n2_k1", NgramDims(32, 8, 2, 2, 1, 2), [5, 9, 2, 31, 17, 0, 3, 8]))

    fuse_cases = []
    fd = NgramDims(32, 24, 2, 4, 4, 2)
    word = det_weights(fd.vocab_size, fd.hidden_size, 0x5EED)
    tables = [det_weights(fd.table_rows(i), fd.table_dim, 0xA11CE + i) for i in range(fd.num_tables)]
    projs = [det_weights(fd.hidden_size, fd.table_dim, 0xB0B + i) for i in range(fd.num_tables)]
    for name, ctx, seq_len in [
        ("prefill_full", [7, 19, 3, 28, 2, 11, 5, 30], 8),
        ("decode_one", [19, 3, 28, 11], 1),
        ("chunk_tail", [7, 19, 3, 28, 2, 11, 5, 30], 3),
    ]:
        fuse_cases.append({
            "name": name,
            "vocab_size": fd.vocab_size,
            "hidden_size": fd.hidden_size,
            "ngram_vocab_size_ratio": fd.ratio,
            "emb_neighbor_num": fd.neighbor_num,
            "emb_split_num": fd.split_num,
            "eos_token_id": fd.eos_token_id,
            "ctx": ctx,
            "seq_len": seq_len,
            "expected": fuse(fd, ctx, seq_len, word, tables, projs),
        })

    json.dump(
        {
            "_source": "bench/ngram_embed/make_fixtures.py -- regenerate, do not hand-edit",
            "id_cases": cases,
            "fuse_weights": {"word": word, "tables": tables, "projs": projs},
            "fuse_cases": fuse_cases,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
