"""
Refuse a test session that is not reading this checkout.

The two distributions live under ``packages/``, and a suite that imports
pyCAFE from anywhere else is quoting numbers about some other copy of the
code. When that other copy predates the split, every module that moved
raises ``ImportError`` at collection and the twenty tracebacks say nothing
about the cause, so the interpreter is checked once, before collection,
and the path it would have read is printed instead.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Import name -> the source root this repo would have it read.
SOURCES = {
    "pycafe": REPO / "packages" / "pycafe" / "src",
    "pycafe_vibro": REPO / "packages" / "pycafe-vibro" / "src",
}


def _package_dir(name):
    """
    Directory ``name`` would be imported from, or None if it is not importable.

    A package left half-installed can raise rather than return, and that is
    the same answer as far as this check is concerned: not importable.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def pytest_configure(config):
    for name, source in SOURCES.items():
        if not source.is_dir():
            # Only one of the two distributions is checked out here.
            continue

        found = _package_dir(name)
        expected = source / name

        if found == expected:
            continue

        where = f"reads it from\n    {found}" if found else "cannot import it"
        raise pytest.UsageError(
            f"\n{name} is not the one in this repository: this interpreter\n"
            f"    {sys.executable}\n"
            f"  {where}\n"
            f"  and the sources under test are\n    {expected}\n\n"
            "Install this checkout into the environment you are running:\n"
            "    pip install -e packages/pycafe -e packages/pycafe-vibro\n"
            "or run the suite with the interpreter that already has it."
        )
