"""Regression tests for `scripts/check_tier_results.py`, the tier-results release gate.

Why this file exists: that script is the gate protecting against a green test tier that did not
actually run its tests, and it had NO automated coverage. Four review rounds each found a real defect
in it, every one verified with a throwaway script that left nothing behind, so the next edit could
silently reintroduce any of them. A gate against silent failure that is itself unguarded is the same
mistake one level up.

Every test below pins a shape that was, at some point, wrong:

  - a fixture reporting a single opaque `all_tests` entry (the original live bug: `run_setup` raised,
    `dbx_test` reported zero tests, and zero tests means zero failures, which reads as success)
  - `failed` / `error` / `xpassed` counted as coverage (a deny-list that forgot them)
  - a partial `@pytest.mark.parametrize` run passing because the case suffix was stripped before
    comparison, so 3-of-4 cases looked identical to 4-of-4
  - the name comparison running in one direction only, so a result naming a test the source does not
    declare (a stale results.json) passed
  - the gate demanding `test__`-prefixed helpers and nested-class methods that `dbx_test` never
    discovers, which would fail a HEALTHY release

The gate is loaded by path (`scripts/` is not a package) and pointed at a temporary fixture directory,
so these tests never depend on the real `integration_tests/` tree or on a live tier run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "_check_tier_results", ROOT / "scripts" / "check_tier_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_check_tier_results"] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load_gate()


@pytest.fixture
def tier(tmp_path, monkeypatch):
    """A harness that writes a fixture source + a results.json and runs the gate over them.

    Returns `run(source, reported)`, where `reported` is a list of (test_name, status) tuples for
    one fixture named `test_thing`, and the result is the gate's `(problems, warnings, counted)`.
    """
    fixtures = tmp_path / "integration_tests"
    fixtures.mkdir()
    monkeypatch.setattr(gate, "FIXTURE_DIR", fixtures)

    def run(source: str, reported, stem: str = "test_thing"):
        (fixtures / f"{stem}.py").write_text("import pytest\n" + source)
        payload = {"summary": {}, "tests": [
            {"notebook": f"/Workspace/Users/x/{s}", "test_name": n, "status": st}
            for s, n, st in ((stem, n, st) for n, st in reported)]}
        results = tmp_path / "results.json"
        results.write_text(json.dumps(payload))
        return gate.check(results)

    return run


PLAIN = """
class TestThing:
    def test_a(self): pass
    def test_b(self): pass
"""


# --- the shape that started it all ---------------------------------------------------------------

def test_all_tests_collapse_is_rejected(tier):
    """`dbx_test` reports a setup failure as one opaque `all_tests` entry. It is never a pass."""
    problems, _, _ = tier(PLAIN, [("all_tests", "passed")])
    assert problems, "a single opaque all_tests entry must fail the gate"
    assert "all_tests" in problems[0]


def test_complete_run_passes(tier):
    problems, warnings, counted = tier(
        PLAIN, [("TestThing.test_a", "passed"), ("TestThing.test_b", "passed")])
    assert problems == []
    assert warnings == []
    assert counted == 2


def test_partial_run_names_the_missing_test(tier):
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", "passed")])
    assert len(problems) == 1
    assert "test_b" in problems[0]


def test_fixture_absent_from_results_is_rejected(tier):
    problems, _, _ = tier(PLAIN, [], stem="test_thing")
    assert problems, "a fixture with no results at all never ran"


# --- statuses: coverage is an allow-list, so anything unknown fails closed ------------------------

@pytest.mark.parametrize("status", ["failed", "error", "xpassed", "skipped", "brand_new_status"])
def test_non_passing_status_is_not_coverage(tier, status):
    """Only `passed` proves a test asserted something. A deny-list once let these through."""
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", status), ("TestThing.test_b", "passed")])
    assert problems, f"status {status!r} must not count as coverage"
    assert any(status in p for p in problems)


def test_xfailed_counts_but_warns(tier):
    """It ran, so the method is covered, but it pins a known-broken path: say so."""
    problems, warnings, _ = tier(
        PLAIN, [("TestThing.test_a", "xfailed"), ("TestThing.test_b", "passed")])
    assert problems == []
    assert len(warnings) == 1 and "xfailed" in warnings[0]


def test_status_is_case_insensitive(tier):
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", "PASSED"), ("TestThing.test_b", "passed")])
    assert problems == []


# --- both directions ------------------------------------------------------------------------------

def test_orphan_result_is_rejected(tier):
    """A result naming a test the source does not declare means the results are stale."""
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", "passed"), ("TestThing.test_b", "passed"),
                                  ("TestThing.test_ghost", "passed")])
    assert problems, "a result with no matching source test must fail"
    assert "test_ghost" in problems[0]


def test_renamed_test_reports_both_halves(tier):
    """The missing source test AND the unexpected result, so the cause is unambiguous."""
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", "passed"),
                                  ("TestThing.test_b_renamed", "passed")])
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "test_b" in joined and "test_b_renamed" in joined


def test_duplicate_names_cannot_pad_a_fixture(tier):
    """Reporting one test twice must not stand in for the test that never ran."""
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", "passed"), ("TestThing.test_a", "passed")])
    assert problems and "test_b" in problems[0]


# --- parametrize: expanded per case, so a partial run is visible ----------------------------------

PARAM = """
class TestThing:
    @pytest.mark.parametrize("x", [1, 2, 3])
    def test_p(self, x): pass
"""


def test_all_parametrized_cases_present_passes(tier):
    problems, _, counted = tier(
        PARAM, [(f"TestThing.test_p[{i}]", "passed") for i in (1, 2, 3)])
    assert problems == []
    assert counted == 3


def test_partial_parametrized_run_is_rejected(tier):
    """3-of-4 once de-parametrized to the same single name as 4-of-4, hiding the gap."""
    problems, _, _ = tier(PARAM, [(f"TestThing.test_p[{i}]", "passed") for i in (1, 2)])
    assert problems, "a parametrized method missing a case must fail"
    assert "test_p[3]" in problems[0]


def test_parametrize_ids_are_honored(tier):
    src = """
class TestThing:
    @pytest.mark.parametrize("x", [1, 2], ids=["lo", "hi"])
    def test_p(self, x): pass
"""
    problems, _, _ = tier(src, [("TestThing.test_p[lo]", "passed"),
                                ("TestThing.test_p[hi]", "passed")])
    assert problems == []


def test_parametrize_multi_arg_case_ids(tier):
    """`dbx_test` joins a tuple's values with '-'."""
    src = """
class TestThing:
    @pytest.mark.parametrize("x,y", [(1, 2), (3, 4)])
    def test_p(self, x, y): pass
"""
    problems, _, _ = tier(src, [("TestThing.test_p[1-2]", "passed"),
                                ("TestThing.test_p[3-4]", "passed")])
    assert problems == []


def test_non_literal_parametrize_falls_back_to_bare_name(tier):
    """The case count is not knowable statically, so require only the method: never fail a healthy run."""
    src = """
PARAMS = [1, 2, 3]

class TestThing:
    @pytest.mark.parametrize("x", PARAMS)
    def test_p(self, x): pass
"""
    problems, _, _ = tier(src, [("TestThing.test_p[1]", "passed")])
    assert problems == []


# --- discovery parity: demand exactly what dbx_test would run ------------------------------------

def test_dunder_prefixed_method_is_not_required(tier):
    """`dbx_test` skips `test__*` names, so demanding one would fail a healthy release."""
    src = """
class TestThing:
    def test_a(self): pass
    def test__helper(self): pass
"""
    problems, _, _ = tier(src, [("TestThing.test_a", "passed")])
    assert problems == []


def test_nested_class_method_is_not_required(tier):
    """`dir(self)` never sees an inner class, so its methods are never discovered."""
    src = """
class TestThing:
    def test_a(self): pass
    class Inner:
        def test_deep(self): pass
"""
    problems, _, _ = tier(src, [("TestThing.test_a", "passed")])
    assert problems == []


def test_module_level_base_class_tests_are_required(tier):
    """A base class at module level IS reachable through inheritance, so its tests must run."""
    src = """
class _Base:
    def test_inherited(self): pass

class TestThing(_Base):
    def test_own(self): pass
"""
    problems, _, _ = tier(src, [("TestThing.test_own", "passed")])
    assert problems and "test_inherited" in problems[0]


def test_async_test_is_required(tier):
    src = """
class TestThing:
    async def test_async(self): pass
"""
    problems, _, _ = tier(src, [])
    assert problems, "an async test method still has to run"


# --- cannot verify nothing -----------------------------------------------------------------------

def test_fixture_with_no_discoverable_tests_is_rejected(tier):
    """Nothing to compare against means accepting an unverifiable claim."""
    src = """
class TestThing:
    def helper(self): pass
"""
    problems, _, _ = tier(src, [("TestThing.test_whatever", "passed")])
    assert problems and "no discoverable test" in problems[0]


def test_results_with_no_entries_at_all_is_rejected(tier, tmp_path):
    results = tmp_path / "empty.json"
    results.write_text(json.dumps({"summary": {"total": 91}, "tests": []}))
    (tmp_path / "integration_tests" / "test_thing.py").write_text(PLAIN)
    problems, _, _ = gate.check(results)
    assert problems and "no test entries" in problems[0]


def test_unknown_fixture_in_results_is_rejected(tier):
    """A result naming a fixture with no source file cannot be checked."""
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", "passed"), ("TestThing.test_b", "passed")])
    assert problems == []
    # now the same results, but attributed to a fixture that does not exist in source
    problems, _, _ = tier(PLAIN, [("TestThing.test_a", "passed")], stem="test_ghost_fixture")
    assert problems, "results for a fixture with no source file must fail"


# --- name parsing --------------------------------------------------------------------------------

@pytest.mark.parametrize("reported,expected", [
    ("TestX.test_a", "test_a"),
    ("test_a", "test_a"),
    ("TestX.test_a[1]", "test_a"),
    ("Mod.TestX.test_a[2]", "test_a"),
    ("TestX.test_a[x.y]", "test_a"),
    ("TestX.test_a[a]b]", "test_a"),
])
def test_source_method_strips_class_and_case(reported, expected):
    assert gate._source_method(reported) == expected


@pytest.mark.parametrize("reported,expected", [
    ("TestX.test_a", "test_a"),
    ("TestX.test_a[2]", "test_a[2]"),
    ("test_a[2]", "test_a[2]"),
    ("Mod.TestX.test_a[lo]", "test_a[lo]"),
])
def test_strip_class_keeps_the_case_suffix(reported, expected):
    """The case suffix has to survive, or a partial parametrized run becomes invisible again."""
    assert gate._strip_class(reported) == expected


@pytest.mark.parametrize("name,discoverable", [
    ("test_a", True),
    # `test_` is a legal (if useless) test name and the framework DOES discover it: its rule is
    # startswith("test_") and not startswith("test__"), which `test_` satisfies. Verified against
    # the real framework, so the gate must agree rather than being tidier than it.
    ("test_", True),
    ("test__helper", False),
    ("helper", False),
    ("_test_a", False),
])
def test_is_discoverable_test_mirrors_the_framework(name, discoverable):
    assert gate._is_discoverable_test(name) is discoverable
