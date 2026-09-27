#!/usr/bin/env python3
"""Self-test for tests/run_all_models.py's readiness markers.

Pure stdlib (unittest); no server, no GPU, no docker. Pins the ready-line
contract against `serve_router.rs::ready_line` — the server prints
`Server live and ready at {host}:{port} running {model}` post-bind, with
wildcard binds rendered as 127.0.0.1 and IPv6 literals bracketed.

    python3 -m unittest tests.test_run_all_models_ready_marker   # from repo root
    python3 tests/test_run_all_models_ready_marker.py            # direct
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_models as H  # noqa: E402


class ReadyMarkerTests(unittest.TestCase):
    def test_wildcard_bind_renders_as_loopback(self):
        # SERVE_BIND is 0.0.0.0; ready_line renders it as 127.0.0.1.
        self.assertEqual(
            H.ready_marker(8888),
            "Server live and ready at 127.0.0.1:8888 running ",
        )

    def test_explicit_ipv4_bind(self):
        self.assertEqual(
            H.ready_marker(9999, bind="10.0.0.7"),
            "Server live and ready at 10.0.0.7:9999 running ",
        )

    def test_ipv6_bind_is_bracketed(self):
        self.assertEqual(
            H.ready_marker(8888, bind="fd00::1"),
            "Server live and ready at [fd00::1]:8888 running ",
        )
        # The wildcard v6 bind also maps to 127.0.0.1, matching ready_line.
        self.assertEqual(
            H.ready_marker(8888, bind="::"),
            "Server live and ready at 127.0.0.1:8888 running ",
        )

    def test_legacy_listening_marker_accepted(self):
        log = "INFO Listening on 0.0.0.0:8888\n"
        state = H.log_ready_state(log, H.ready_marker(8888), H.legacy_ready_marker(8888))
        self.assertEqual(state, "ready")

    def test_current_ready_line_accepted(self):
        log = "INFO Server live and ready at 127.0.0.1:8888 running qwen3.8\n"
        state = H.log_ready_state(log, H.ready_marker(8888), H.legacy_ready_marker(8888))
        self.assertEqual(state, "ready")

    def test_wrong_bind_fails_fast(self):
        # Bound 127.0.0.1 *inside the container* — unreachable for this
        # harness; must not read as ready.
        log = "INFO Server live and ready at 127.0.0.1:9999 running qwen3.8\n"
        state = H.log_ready_state(log, H.ready_marker(8888), H.legacy_ready_marker(8888))
        self.assertEqual(state, "wrong_bind")

    def test_legacy_wrong_bind_fails_fast(self):
        log = "INFO Listening on 127.0.0.1:8888\n"
        state = H.log_ready_state(log, H.ready_marker(8888), H.legacy_ready_marker(8888))
        self.assertEqual(state, "wrong_bind")

    def test_modelless_line_is_not_ready(self):
        log = "INFO Server live at 127.0.0.1:8888 — no model loaded yet\n"
        state = H.log_ready_state(log, H.ready_marker(8888), H.legacy_ready_marker(8888))
        self.assertEqual(state, "waiting")

    def test_other_log_noise_waits(self):
        log = "INFO loading weights\nINFO Server live at [::]:8888 — no model loaded yet\n"
        state = H.log_ready_state(log, H.ready_marker(8888), H.legacy_ready_marker(8888))
        self.assertEqual(state, "waiting")


if __name__ == "__main__":
    unittest.main()
