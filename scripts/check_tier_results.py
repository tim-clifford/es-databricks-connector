#!/usr/bin/env python3
"""Gate: the integration tier actually RAN every test it claims to cover.

WHY THIS EXISTS
---------------
`dbx_test` treats a `run_setup` failure as "this fixture ran zero tests", not as a failure:

    _execute_setup() -> False  ->  "Skipping tests due to setup failure"  ->  results = []

`get_results()` then derives `total = len(results) = 0`, so `failed` and `errors` are 0 too. Zero
failures reads as success everywhere downstream: the fixture is reported as one passing `all_tests`
entry, the run summary prints `Failed: 0`, and the console ends with `All tests passed`.

That is not hypothetical. `test_bulk_write_roundtrip.py` stopped running entirely when
`require_existing_index=True` landed in 0.6.0 (its setup wrote to an index it never created), and
SIX consecutive tier runs reported it green. The total stayed at 84 the whole time because the
fixture silently dropped 8 named tests in the same change that added a 7-test fixture, so two
unrelated deltas very nearly cancelled. A human reading `84/84` and `All tests passed` had no signal
at all, which is exactly the class of silent failure this library exists to eliminate in its own
behavior; the test tier deserves the same treatment.

A count that only looks stable because two errors cancel is what a gate catches and a reviewer does
not. Per the connector skill's rule that a mechanizable check belongs in `scripts/` rather than in a
checklist someone can skip, this enforces two invariants on the newest tier results:

  1. **No fixture reported zero tests.** A zero-test fixture is a skipped fixture, never a pass.
  2. **Every fixture's reported test count matches the number of `test_*` methods in its source.**
     This is the stronger check: it catches a fixture that ran only PART of its tests, and it
     catches the `all_tests` collapse (1 reported vs N in source) that hides a setup failure.

Also fails when it finds no results at all, or no fixtures inside them, so a moved directory or an
empty run cannot make the gate vacuous.

Usage:
    python scripts/check_tier_results.py                 # newest .dbx-test-results/*/results.json
    python scripts/check_tier_results.py --results PATH   # a specific results.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / ".dbx-test-results"
FIXTURE_DIR = ROOT / "integration_tests"


def newest_results() -> Path:
    """The results.json from the most recent tier run, by directory name (timestamped)."""
    candidates = sorted(RESULTS_DIR.glob("*/results.json"))
    if not candidates:
        sys.exit(f"no results found under {RESULTS_DIR.relative_to(ROOT)}/*/results.json -- "
                 "run the integration tier first (RELEASING.md step 2)")
    return candidates[-1]


def source_test_counts() -> dict[str, int]:
    """{fixture module stem: number of test_* methods declared in it}.

    Parsed with `ast` rather than imported: these are Databricks notebooks whose top level calls
    `dbutils`, so importing them off-cluster is impossible.
    """
    counts: dict[str, int] = {}
    for path in sorted(FIXTURE_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        n = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                n += sum(1 for item in node.body
                         if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and item.name.startswith("test_"))
        counts[path.stem] = n
    return counts


def reported_counts(results: dict) -> dict[str, list[str]]:
    """{fixture stem: [reported test names]} from a results.json."""
    by_fixture: dict[str, list[str]] = {}
    for test in results.get("tests", []):
        # `notebook` is a workspace path; the stem is the fixture module name.
        stem = re.sub(r"\.py$", "", (test.get("notebook") or "").rstrip("/").split("/")[-1])
        by_fixture.setdefault(stem, []).append(test.get("test_name") or "<unnamed>")
    return by_fixture


def check(results_path: Path) -> list[str]:
    problems: list[str] = []
    results = json.loads(results_path.read_text())
    reported = reported_counts(results)
    expected = source_test_counts()

    if not reported:
        return [f"{results_path.relative_to(ROOT)} contains no test entries at all, so this check "
                "would verify nothing"]
    if not expected:
        return [f"no test_*.py fixtures found under {FIXTURE_DIR.relative_to(ROOT)}, so this check "
                "would verify nothing"]

    for stem, names in sorted(reported.items()):
        got = len(names)
        # 1. zero tests, or the single opaque entry dbx_test emits when setup failed
        if got == 0:
            problems.append(f"{stem}: reported ZERO tests -- a skipped fixture is not a pass")
            continue
        if names == ["all_tests"]:
            want = expected.get(stem)
            detail = f" ({want} test_* methods in source)" if want else ""
            problems.append(
                f"{stem}: reported a single opaque 'all_tests' entry{detail}. Either run_setup "
                "failed (dbx_test reports that as zero tests, which reads as success) or the "
                "fixture does not end with dbutils.notebook.exit(json.dumps(run_notebook_tests()))")
            continue
        # 2. partial runs: fewer (or more) reported than the source declares
        if stem not in expected:
            problems.append(f"{stem}: reported {got} test(s) but there is no "
                            f"integration_tests/{stem}.py to compare against")
        elif got != expected[stem]:
            problems.append(f"{stem}: reported {got} test(s) but {expected[stem]} test_* method(s) "
                            f"are declared in integration_tests/{stem}.py")

    # A fixture that exists in source but is missing from the results entirely: it never ran, and
    # nothing in the summary would say so.
    for stem, want in sorted(expected.items()):
        if want and stem not in reported:
            problems.append(f"{stem}: {want} test(s) in source but the fixture is ABSENT from the "
                            "results -- it never ran")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=None,
                    help="path to a results.json (default: the newest tier run)")
    args = ap.parse_args()

    results_path = args.results or newest_results()
    problems = check(results_path)

    print(f"tier results: {results_path.relative_to(ROOT) if results_path.is_relative_to(ROOT) else results_path}")
    if problems:
        print(f"\n{len(problems)} problem(s) -- the tier did not run what it claims to cover:")
        for p in problems:
            print(f"  - {p}")
        print("\nA fixture whose setup fails reports 0 tests, 0 failures, which prints as a PASS. "
              "Fix the fixture and re-run the tier; do not release on these results.")
        return 1
    total = sum(len(v) for v in reported_counts(json.loads(results_path.read_text())).values())
    print(f"OK: every fixture ran, and each one's count matches its source ({total} tests).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
