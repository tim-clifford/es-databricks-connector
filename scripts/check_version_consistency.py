#!/usr/bin/env python3
"""Gate: every place that names the connector's version agrees with pyproject.toml.

WHY THIS EXISTS
---------------
The version is written in several files that no test imports, so a stale one fails silently and in
the worst possible way: `integration_tests/config/test_config.yml` pins the wheel the integration
tier INSTALLS, so a stale pin there means the whole live tier validates the PREVIOUS release while
reporting success. That happened (it sat at 0.5.0 while the source was 0.6.0), which is what
prompted this script.

Reference 5 of the connector skill says a manual check that becomes fully mechanical should be
promoted to `scripts/` rather than left as a review step. This is that promotion: item 1 of the
doc-review checklist (version consistency), now enforced.

Checks, all against `pyproject.toml [project].version` as the single source of truth:
  1. `src/databricks_es_connector/__init__.py` `__version__`
  2. `HANDOFF.md`'s header version (it is deliberately pinned to a release)
  3. every `databricks_es_connector-<version>-py3-none-any.whl` reference in tracked text files

Exits non-zero and names every mismatch. Run from the repo root.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files whose wheel references must match. Globs, relative to the repo root. Deliberately does NOT
# include dist/ (built artifacts legitimately accumulate old versions).
WHEEL_REF_GLOBS = ("README.md", "RELEASING.md", "HANDOFF.md", "integration_tests/**/*.yml",
                   "integration_tests/**/*.md", "integration_tests/**/*.py", ".agents/**/*.md")

_WHEEL_RE = re.compile(r"databricks_es_connector-(\d+\.\d+\.\d+)-py3-none-any\.whl")


def expected_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        sys.exit("could not find [project].version in pyproject.toml")
    return m.group(1)


def check(want: str) -> list[str]:
    problems: list[str] = []

    # 1. __init__.py __version__
    init = (ROOT / "src/databricks_es_connector/__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M)
    if not m:
        problems.append("src/databricks_es_connector/__init__.py: no __version__ found")
    elif m.group(1) != want:
        problems.append(f"src/databricks_es_connector/__init__.py: __version__ = {m.group(1)!r}, "
                        f"expected {want!r}")

    # 2. HANDOFF.md header (pinned to a release on purpose)
    handoff = (ROOT / "HANDOFF.md").read_text().splitlines()[0]
    found = re.findall(r"(\d+\.\d+\.\d+)", handoff)
    if not found:
        problems.append(f"HANDOFF.md: header names no version ({handoff!r})")
    elif want not in found:
        problems.append(f"HANDOFF.md: header says {found}, expected {want!r} -- {handoff!r}")

    # 3. every wheel filename reference
    seen_any = False
    for pattern in WHEEL_REF_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                for ref in _WHEEL_RE.finditer(line):
                    seen_any = True
                    if ref.group(1) != want:
                        rel = path.relative_to(ROOT)
                        problems.append(f"{rel}:{i}: wheel pinned to {ref.group(1)}, "
                                        f"expected {want}")
    if not seen_any:
        # A silent zero-match would make this gate vacuous, which is the failure mode it exists to
        # prevent. Treat it as a problem so a moved/renamed file cannot disable the check quietly.
        problems.append("no wheel references found at all -- the globs no longer match anything, "
                        "so this check is not actually verifying anything")
    return problems


def main() -> int:
    want = expected_version()
    problems = check(want)
    print(f"expected version (pyproject.toml): {want}")
    if problems:
        print(f"\n{len(problems)} version inconsistency(ies):")
        for p in problems:
            print(f"  - {p}")
        print("\nFix each, or bump pyproject.toml if the version itself is what changed.")
        return 1
    print("OK: __init__, HANDOFF header, and every wheel reference agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
