#!/usr/bin/env python3
"""Verify requirements.txt is an exact match to the built wheel's resolved dependency closure.

Release gate (step 3 of the release checklist): the wheel declares only the abstract range
`elasticsearch>=8,<9` in pyproject.toml, so requirements.txt (the pinned closure used for security
scanning) can silently drift from what a fresh install of the wheel actually pulls in. This script
resolves the wheel's dependency closure in a throwaway venv and compares it, name+version exact,
against requirements.txt. Exits non-zero (and prints the diff) on any mismatch, so it can gate a
release in CI or a Makefile target.

What "match" means here: the set of {package==version} lines. We compare the resolved closure of
installing the built wheel against the pinned set in requirements.txt. Packages that requirements.txt
documents as runtime-only-excluded (pyspark/pandas/numpy/pyarrow, provided by the Databricks runtime)
are never in the wheel's closure and never in requirements.txt, so they don't appear on either side.

Usage:
    python scripts/check_requirements_match.py [--wheel dist/xxx.whl] [--requirements requirements.txt]

With no --wheel, uses the newest wheel in dist/. Requires network (pip resolves the closure).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import venv


def _newest_wheel(dist_dir: str) -> str:
    wheels = sorted(glob.glob(os.path.join(dist_dir, "*.whl")), key=os.path.getmtime)
    if not wheels:
        sys.exit(f"error: no wheel found in {dist_dir}/ (build one first: python -m build --wheel)")
    return wheels[-1]


def _parse_requirements(path: str) -> dict[str, str]:
    """Return {normalized_name: version} for exact `name==version` lines. Comments/blank ignored.

    Names are normalized per PEP 503 (lowercase, runs of -_. collapse to a single -) so
    `typing_extensions` and `typing-extensions` compare equal.
    """
    pins: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)
            if not m:
                # A non-pinned or complex line (marker, range). requirements.txt is a flat pinned
                # closure by design; surface anything that isn't so the author fixes the file.
                sys.exit(f"error: {path} line is not a simple name==version pin: {line!r}")
            pins[_normalize(m.group(1))] = m.group(2)
    return pins


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _resolve_closure(wheel: str) -> dict[str, str]:
    """Install the wheel into a throwaway venv and return its full {name: version} closure
    (excluding pip/setuptools/wheel bootstrap and the connector itself)."""
    with tempfile.TemporaryDirectory() as tmp:
        env_dir = os.path.join(tmp, "venv")
        venv.create(env_dir, with_pip=True)
        pip = os.path.join(env_dir, "bin", "pip")
        try:
            subprocess.run([pip, "install", "--quiet", "--disable-pip-version-check", wheel],
                           check=True)
        except subprocess.CalledProcessError:
            sys.exit("error: could not install the wheel to resolve its dependency closure "
                     "(network required: pip resolves transitive deps from the index).")
        out = subprocess.run([pip, "freeze", "--all"], check=True, capture_output=True, text=True)

    closure: dict[str, str] = {}
    bootstrap = {"pip", "setuptools", "wheel"}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or "==" not in line:
            continue
        name, _, version = line.partition("==")
        norm = _normalize(name)
        if norm in bootstrap or norm == "databricks-es-connector":
            continue
        closure[norm] = version
    return closure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel", help="path to the wheel (default: newest in dist/)")
    ap.add_argument("--requirements", default="requirements.txt")
    ap.add_argument("--dist-dir", default="dist")
    args = ap.parse_args()

    wheel = args.wheel or _newest_wheel(args.dist_dir)
    print(f"wheel:        {wheel}")
    print(f"requirements: {args.requirements}")

    declared = _parse_requirements(args.requirements)
    resolved = _resolve_closure(wheel)

    missing = {n: declared[n] for n in declared.keys() - resolved.keys()}          # in reqs, not installed
    extra = {n: resolved[n] for n in resolved.keys() - declared.keys()}            # installed, not in reqs
    mismatched = {n: (declared[n], resolved[n])
                  for n in declared.keys() & resolved.keys() if declared[n] != resolved[n]}

    if not (missing or extra or mismatched):
        print(f"\nOK: requirements.txt exactly matches the wheel's resolved closure "
              f"({len(resolved)} packages).")
        return 0

    print("\nMISMATCH: requirements.txt does not match the wheel's resolved closure.")
    if extra:
        print("\n  In the wheel's closure but NOT pinned in requirements.txt (add + re-annotate):")
        for n in sorted(extra):
            print(f"    + {n}=={extra[n]}")
    if missing:
        print("\n  Pinned in requirements.txt but NOT in the wheel's closure (remove, or drift):")
        for n in sorted(missing):
            print(f"    - {n}=={missing[n]}")
    if mismatched:
        print("\n  Version mismatch (requirements.txt vs resolved):")
        for n in sorted(mismatched):
            d, r = mismatched[n]
            print(f"    ~ {n}: requirements.txt {d} != resolved {r}")
    print("\nRegenerate requirements.txt (see the command in its header), re-annotate, and re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
