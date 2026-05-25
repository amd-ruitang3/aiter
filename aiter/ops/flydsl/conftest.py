# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pytest configuration for FlyDSL ops tests.

Scope-limited to ``aiter/ops/flydsl/`` so the ``--run-slow`` opt-in and
the ``slow`` marker do not leak into the rest of the aiter test suite.

Adds:

* CLI flag ``--run-slow`` -- opt-in to long-running parametrized cases
  (the 407-shape ``Qwen3Next-trace-*`` performance sweep in
  ``test_flydsl_linear_attention_prefill.py::TestPerformance``).

* Marker ``@pytest.mark.slow`` -- registered locally so that emitting
  it from a parametrize block does not produce ``PytestUnknownMarkWarning``.

* Collection hook -- if ``--run-slow`` is not passed, every test
  collected under this conftest's directory that carries the ``slow``
  marker is skipped (deselected at runtime via dynamic skip, so that
  ``pytest --collect-only`` still shows the test exists).

To run the slow trace shapes (in addition to the default cases):

    pytest -sv --run-slow \
        aiter/ops/flydsl/test_flydsl_linear_attention_prefill.py::TestPerformance

To run only the slow shapes:

    pytest -sv --run-slow -k Qwen3Next-trace \
        aiter/ops/flydsl/test_flydsl_linear_attention_prefill.py::TestPerformance
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="opt in to long-running tests under aiter/ops/flydsl/ "
             "(e.g. the 407-shape Qwen3Next-trace prefill sweep). "
             "Slow tests are skipped by default.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: opt-in long-running tests (gate with --run-slow).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(
        reason="long-running test; pass --run-slow to enable"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
