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
checklist someone can skip, this enforces three invariants on the newest tier results:

  1. **No fixture reported zero tests.** A zero-test fixture is a skipped fixture, never a pass.
  2. **A fixture's reported test names EQUAL its source's `test_*` method names**, in both
     directions. Names, not counts: a count cannot tell "ran 12 of 13" from "ran 13, one of which
     was parametrized into two" (see below), and it silently accepts one test disappearing while
     another is added. Both directions, because either half alone is a false pass: a source method
     with no result never ran, and a RESULT with no source method means the results describe some
     other revision of this repo (a stale `results.json`, or a test renamed since), so nothing in
     them can be trusted about this checkout.
  3. **Only a `passed` result counts as coverage**, as an allow-list rather than a deny-list of the
     bad statuses. `dbx_test` reports `skipped`, `xpassed`, `failed` and `error` as ordinary entries
     alongside `passed`, so a `@pytest.mark.skip` would otherwise keep every name matching while the
     test verifies nothing. `xfailed` is allowed but warned: it ran, and it pins a known-broken path
     rather than a working one. An unrecognized status fails closed.

Names rather than counts, because `dbx_test` supports `@pytest.mark.parametrize` and expands ONE
source method into N reported results named `test_thing[1]`, `test_thing[2]`, ... A count check
therefore fails a perfectly healthy parametrized fixture, and it also cannot see one test vanishing
while another is added, nor say WHICH test is missing. Reported names are matched back to their
source method by stripping the `Class.` prefix and any `[...]` suffix, so a parametrized method is
satisfied by any of its expansions.

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


def _display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (a --results path may be outside the repo)."""
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def _is_discoverable_test(name: str) -> bool:
    """Mirror `dbx_test`'s own discovery rule for a method name.

    Its `_get_test_methods` does:

        if not name.startswith("test_") or name.startswith("test__"): continue

    so a `test__`-prefixed method is deliberately NOT a test (it is a helper that happens to sort
    near the tests). This gate must demand exactly the set the framework runs: requiring more means
    failing a release over a test that was never going to run.
    """
    return name.startswith("test_") and not name.startswith("test__")


def source_test_names() -> dict[str, set[str]]:
    """{fixture module stem: {names of test_* methods the framework would DISCOVER in it}}.

    Parsed with `ast` rather than imported: these are Databricks notebooks whose top level calls
    `dbutils`, so importing them off-cluster is impossible. That makes this an independent
    reimplementation of `dbx_test`'s discovery, so it has to match the framework's rules rather than
    guess at them, in both directions:

    - **Only TOP-LEVEL classes.** `dbx_test` discovers via `dir(self)` on the fixture instance, which
      sees the class and its bases but never an inner class. So a test method on a nested class is
      never run, and demanding it would fail a healthy release. A base class at module level IS
      reachable through inheritance, which is why every top-level class is scanned and not just the
      `NotebookTestFixture` subclass.
    - **`test__`-prefixed methods excluded**, per `_is_discoverable_test`.

    Names are collected into a SET, so an inherited method counts once no matter how many classes in
    the file expose it.
    """
    names: dict[str, set[str]] = {}
    for path in sorted(FIXTURE_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        found: set[str] = set()
        # tree.body, not ast.walk: walking would descend into nested classes the framework cannot see.
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                found.update(item.name for item in node.body
                             if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and _is_discoverable_test(item.name))
        names[path.stem] = found
    return names


# `dbx_test` reports a test as `Class.method`, and a parametrized method as `Class.method[param]`.
# Strip both decorations to recover the source method name.
_REPORTED_NAME_RE = re.compile(r"^(?:.*\.)?(?P<method>[A-Za-z_]\w*)(?:\[.*\])?$")

# `dbx_test` emits exactly six statuses (grep `status="` in its testing.py): passed, skipped,
# xfailed, xpassed, failed, error. Only `passed` is evidence that a test asserted something about
# working behavior, so coverage is an ALLOW-LIST rather than a deny-list of the bad ones. Written
# this way on purpose: a deny-list silently admits anything it forgot, which is how an earlier
# revision of this gate counted `failed` and `error` results as coverage and then printed
# "ran and passed" over them. An unrecognized status must fail closed, not pass by default.
#
#   skipped  the test did not run at all; a @pytest.mark.skip otherwise keeps every name matching
#            while the test verifies nothing
#   xfailed  ran, but pins a known-broken path rather than a working one -> allowed, warned
#   xpassed  an xfail-marked test unexpectedly PASSED, so the "known bug" premise is now wrong and
#            the marker is stale -> not coverage of the working behavior it claims
#   failed / error  the test ran and did NOT pass; a green tier summary cannot coexist with these,
#            so seeing one here means the summary and the per-test results disagree
_COVERING = {"passed"}
_ALLOWED_WITH_WARNING = {"xfailed"}


def _source_method(reported_name: str) -> str:
    """`TestX.test_a[2]` -> `test_a`. Returns the input unchanged if it does not parse."""
    m = _REPORTED_NAME_RE.match(reported_name.strip())
    return m.group("method") if m else reported_name


def reported_by_fixture(results: dict) -> dict[str, list[dict]]:
    """{fixture stem: [raw test entries]} from a results.json."""
    by_fixture: dict[str, list[dict]] = {}
    for test in results.get("tests", []):
        # `notebook` is a workspace path; the stem is the fixture module name.
        stem = re.sub(r"\.py$", "", (test.get("notebook") or "").rstrip("/").split("/")[-1])
        by_fixture.setdefault(stem, []).append(test)
    return by_fixture


def check(results_path: Path) -> tuple[list[str], list[str], int]:
    """Return (problems, warnings, tests_counted). Empty `problems` means the gate passes."""
    problems: list[str] = []
    warnings: list[str] = []
    results = json.loads(results_path.read_text())
    reported = reported_by_fixture(results)
    expected = source_test_names()
    shown = _display(results_path)

    if not reported:
        return ([f"{shown} contains no test entries at all, so this check would verify nothing"],
                warnings, 0)
    if not expected:
        return ([f"no test_*.py fixtures found under {FIXTURE_DIR.relative_to(ROOT)}, so this check "
                 "would verify nothing"], warnings, 0)

    counted = 0
    for stem, entries in sorted(reported.items()):
        # 1. zero tests, or the single opaque entry dbx_test emits when setup failed
        if not entries:
            problems.append(f"{stem}: reported ZERO tests -- a skipped fixture is not a pass")
            continue
        if [e.get("test_name") for e in entries] == ["all_tests"]:
            want = len(expected.get(stem, ()))
            detail = f" ({want} test_* methods in source)" if want else ""
            problems.append(
                f"{stem}: reported a single opaque 'all_tests' entry{detail}. Either run_setup "
                "failed (dbx_test reports that as zero tests, which reads as success) or the "
                "fixture does not end with dbutils.notebook.exit(json.dumps(run_notebook_tests()))")
            continue
        if stem not in expected:
            problems.append(f"{stem}: reported {len(entries)} test(s) but there is no "
                            f"integration_tests/{stem}.py to compare against")
            continue
        if not expected[stem]:
            # A fixture file with no discoverable test at all. Whatever the results say about it
            # cannot be checked against anything, so accepting it would be accepting an unverifiable
            # claim. Usually a fixture whose tests were renamed into `test__` helpers, or one that
            # only defines run_setup.
            problems.append(f"integration_tests/{stem}.py declares no discoverable test_* method, "
                            f"yet the results report {len(entries)} for it -- nothing there can be "
                            "verified; give it real tests or delete the file")
            continue

        # 2. the reported set and the source set must be EQUAL, in both directions, and every
        #    reported result must have actually asserted something.
        covered: set[str] = set()
        for entry in entries:
            name = entry.get("test_name") or ""
            status = (entry.get("status") or "").lower()
            method = _source_method(name)
            if status in _COVERING:
                covered.add(method)
                counted += 1
                continue
            if status in _ALLOWED_WITH_WARNING:
                # It ran, so the method is covered, but say so out loud: it pins a known-broken path
                # rather than a working one.
                warnings.append(f"{stem}.{name}: {status} (a known-broken path, not a working one)")
                covered.add(method)
                counted += 1
                continue
            # Anything else (skipped, xpassed, failed, error, or a status this script has never seen)
            # is NOT coverage. Left out of `covered` so it surfaces below as missing, naming the
            # status so the reason is unambiguous.
            problems.append(f"{stem}.{name}: reported as {status!r}, which is not evidence the test "
                            "asserted anything about working behavior")

        for missing in sorted(expected[stem] - covered):
            problems.append(f"{stem}: source declares {missing}() but no passing result reports it "
                            f"-- it did not run, was renamed without being wired up, or did not pass")
        # The other direction, which a "does the source set fit inside the reported set" check would
        # miss: a result naming a test the source does not declare. That means the results and the
        # checkout disagree, so NOTHING here can be trusted to describe this code -- typically a
        # stale results.json from before a rename or deletion, or a run against a different revision.
        # Left unchecked it also inflates the reported total, which is the number a human reads.
        for orphan in sorted(covered - expected[stem]):
            problems.append(f"{stem}: results report {orphan}() but the source declares no such test "
                            "-- these results do not match this checkout (stale run, or a test "
                            "renamed/deleted since); re-run the tier")

    # A fixture that exists in source but is missing from the results entirely: it never ran, and
    # nothing in the summary would say so.
    for stem, want in sorted(expected.items()):
        if want and stem not in reported:
            problems.append(f"{stem}: {len(want)} test(s) in source but the fixture is ABSENT from "
                            "the results -- it never ran")
    return problems, warnings, counted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=None,
                    help="path to a results.json (default: the newest tier run)")
    args = ap.parse_args()

    results_path = args.results or newest_results()
    problems, warnings, counted = check(results_path)

    print(f"tier results: {_display(results_path)}")
    for w in warnings:
        print(f"  ! {w}")
    if problems:
        print(f"\n{len(problems)} problem(s) -- the tier did not run what it claims to cover:")
        for p in problems:
            print(f"  - {p}")
        print("\nA fixture whose setup fails reports 0 tests, 0 failures, which prints as a PASS. "
              "Fix the fixture and re-run the tier; do not release on these results.")
        return 1
    print(f"OK: every test_* method in every fixture ran and passed ({counted} results).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
