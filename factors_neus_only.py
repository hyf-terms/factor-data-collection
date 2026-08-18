"""Legacy-compatible entry point for historical sparse-factor IC runs.

New production tests must use ``factors_neus_only2.py``.  This wrapper keeps
the former skip-invalid-date behavior so old results remain reproducible.
"""

from __future__ import annotations

import sys

import factors_neus_only2 as _implementation


# Preserve the old module's import surface, including helpers whose names begin
# with an underscore and are used by the existing test suite.
globals().update(
    {
        name: value
        for name, value in vars(_implementation).items()
        if not name.startswith("__")
    }
)

_strict_run_factor_test_pipeline = _implementation.run_factor_test_pipeline


def run_factor_test_pipeline(*args, **kwargs):
    """Run the historical policy unless the caller explicitly overrides it."""
    kwargs.setdefault("cross_section_policy", "skip")
    return _strict_run_factor_test_pipeline(*args, **kwargs)


def main() -> None:
    """Keep CLI compatibility while defaulting this legacy entry point to skip."""
    if "--cross-section-policy" not in sys.argv:
        sys.argv.extend(["--cross-section-policy", "skip"])
    _implementation.main()


if __name__ == "__main__":
    main()
