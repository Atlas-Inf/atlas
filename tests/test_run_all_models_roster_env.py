#!/usr/bin/env python3
"""Self-test for the roster `env` field in tests/run_all_models.py.

Pure stdlib (unittest); no server, no GPU, no docker. Covers the roster
parse (with and without `env`), the `-e K=V` docker flags with shell
quoting, and the env-key validator that fails fast on a bad key.

    python3 -m unittest tests.test_run_all_models_roster_env   # from repo root
    python3 tests/test_run_all_models_roster_env.py            # direct
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_models as H  # noqa: E402


def write_roster(rounds):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False)
    json.dump(rounds, f)
    f.close()
    return f.name


class RosterEnvTests(unittest.TestCase):
    def test_roster_without_env_defaults_empty(self):
        path = write_roster([[{"host": "head", "label": "a", "model": "m/x"}]])
        try:
            spec = H.load_roster(path)[0][0][1]
        finally:
            os.unlink(path)
        self.assertEqual(spec.env, {})

    def test_roster_env_parsed_into_spec(self):
        path = write_roster([[{
            "host": "head",
            "label": "ltn",
            "model": "m/lightning",
            "env": {"ATLAS_DFLASH_OPTION_B": "1"},
        }]])
        try:
            spec = H.load_roster(path)[0][0][1]
        finally:
            os.unlink(path)
        self.assertEqual(spec.env, {"ATLAS_DFLASH_OPTION_B": "1"})

    def test_docker_flags_emit_quoted_env(self):
        spec = H.TestSpec(
            "x", "m/x",
            env={"ATLAS_DFLASH_OPTION_B": "1", "ATLAS_NOTE": "a b"},
        )
        flags = H.env_docker_flags(spec)
        # Sorted by key; values with spaces survive ssh + bash -lc as one
        # shlex-quoted token.
        self.assertIn("-e ATLAS_DFLASH_OPTION_B=1", flags)
        self.assertIn("-e 'ATLAS_NOTE=a b'", flags)

    def test_no_env_no_flags(self):
        self.assertEqual(H.env_docker_flags(H.TestSpec("x", "m/x")), "")

    def test_bad_env_key_rejected(self):
        for bad in ("lowercase", "9FOO", "FOO-BAR", "FOO BAR"):
            path = write_roster([[{
                "label": "x", "model": "m/x", "env": {bad: "1"},
            }]])
            try:
                with self.assertRaises(SystemExit):
                    H.load_roster(path)
            finally:
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
