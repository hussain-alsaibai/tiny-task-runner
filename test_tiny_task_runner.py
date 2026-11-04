"""Smoke tests for tiny-task-runner.

Added 2026-11-04. This repo had no test suite and no CI, which is how a
module that could not be imported shipped unnoticed.

The guard is deliberately small: a module that cannot be imported is a module
that cannot be used, and that is the cheapest defect to catch.
"""
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

MODULE = "tiny_task_runner"
ROOT = pathlib.Path(__file__).resolve().parent


def test_module_imports():
    """The module must import cleanly."""
    import importlib

    mod = importlib.import_module(MODULE)
    assert mod is not None


def test_has_public_api():
    import importlib

    mod = importlib.import_module(MODULE)
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert len(public) > 3, "suspiciously few public names: %s" % public


def test_a_version_is_declared_somewhere():
    """A consumer must be able to find a version.

    It may live on the module (__version__ or VERSION) or in the packaging
    metadata. What must not happen is no version anywhere.
    """
    import importlib

    mod = importlib.import_module(MODULE)
    if getattr(mod, "__version__", None) or getattr(mod, "VERSION", None):
        return
    pattern = re.compile(r"version\s*[=:]\s*[\"']?[0-9]")
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        path = ROOT / name
        if path.exists() and pattern.search(path.read_text(encoding="utf-8")):
            return
    pytest.fail("no version declared on the module or in the packaging metadata")
