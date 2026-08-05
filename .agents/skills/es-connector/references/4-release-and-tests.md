# Reference 4: Testing and releasing

The authoritative release procedure is `RELEASING.md` in the repo root, follow it, don't duplicate
it. This file is the orientation around it: the two test tiers, how to run each, and the FEVM
specifics that are easy to forget. The deterministic gates live in `scripts/`; this skill points at
them, it never re-implements them.

## Two test tiers

**`tests/`, pure-Python, fast, no infra (`pytest`).** The inner loop. Covers `coerce_value`,
`read_coerce` (including the round-trip oracle), config validation, `classify_bulk_result` / result
merging, the streaming glue, and the PURE helpers in `spark_prep.py` (`_type_has_timestamp`,
`_type_is_arrow_hostile`, `_hostile_columns_from_describe`, etc.). The transform core is at 100%
line coverage. This is where refactor safety mostly lives: a behavior-changing edit to the transform
layer fails an oracle test here.

```bash
python -m pytest -q                    # ~228 tests, sub-second
python -m pytest -q --cov=databricks_es_connector --cov-report=term-missing   # coverage
```

**`integration_tests/`, live Spark + ES on FEVM serverless (`dbx_test`).** The release gate. Covers
what `tests/` structurally cannot: `sanitize_for_arrow` (VARIANT/INTERVAL + the `df.schema`-throws-
on-Connect constraint), `normalize_timestamps_for_utc` (real `unix_millis`), the real `bulk_write`
`mapInPandas` round-trip, `read_index` sliced/​paged reads, deletes, streaming, and the ES-behavior
fixtures (timezone under a non-UTC session, dynamic-mapping coercion). The Spark-side function bodies
in `spark_prep.py` are ONLY provable here.

## Running the integration tier on FEVM (the muscle memory)

Profile `fe-vm-tim-clifford-classic-dsl-lite`; catalog `tim_clifford_classic_dsl_lite_catalog`;
wheel Volume `/Volumes/tim_clifford_classic_dsl_lite_catalog/es_poc/artifacts/`; secret scope
`es_poc` (keys `hosts`, `username`, `password`). ES creds come from the scope, never plaintext;
`verify_certs=False` is sandbox-only.

```bash
# 1. Build + upload the wheel FIRST (the fixtures %pip-install it by path from the Volume, so a
#    stale wheel there is what gets tested). Verify local md5 == uploaded md5. (RELEASING.md step 1.)
python -m build --wheel && md5 dist/databricks_es_connector-*.whl
VOL=/Volumes/tim_clifford_classic_dsl_lite_catalog/es_poc/artifacts
databricks fs cp dist/databricks_es_connector-*.whl "dbfs:$VOL/" --overwrite \
  --profile fe-vm-tim-clifford-classic-dsl-lite

# 2. Sync fixtures to the workspace, then run. Re-importing a dir can leave stale .py-suffixed
#    notebooks alongside the suffixless ones -> DELETE the stale copies so each fixture runs once.
databricks workspace import-dir integration_tests \
  /Workspace/Users/<you>/es_connector_integration --overwrite \
  --profile fe-vm-tim-clifford-classic-dsl-lite

dbx_test run \
  --tests-dir /Workspace/Users/<you>/es_connector_integration \
  --workspace-tests --profile fe-vm-tim-clifford-classic-dsl-lite \
  --config integration_tests/config/test_config.yml
```

`test_config.yml` pins the wheel path, it must name the version you're releasing;
`scripts/check_version_consistency.py` enforces that. Any downstream consumer that installs the wheel
is a useful second live check on a release candidate, but it is not part of this repo's gate: the
integration tier above is what has to be green.

## The release gates (`scripts/`, all hard gates in RELEASING.md step 3)

- **`scripts/check_requirements_match.py`**, resolves the built wheel's dependency closure in a
  throwaway venv and compares it name+version exact against `requirements.txt` (which pyproject's
  abstract `elasticsearch>=8,<9` range lets drift). Exit 1 on any missing/extra/version mismatch.
  Run it with the SAME Python minor version used to regenerate `requirements.txt` (env markers can
  otherwise cause a spurious mismatch).
- **`scripts/check_readme_sync.py`**, asserts every shipped module, integration fixture, and release
  script is referenced by name in the README that should list it (whole-tree invariant; catches the
  drift a diff-scoped review can't). Exit 1 with the gaps. Matching is word-boundary aware
  (`transform.py` is NOT satisfied by `read_transform.py`).
- **`scripts/check_tier_results.py`**, asserts the tier RAN what it claims to cover: every fixture
  appears in the newest `results.json` and its reported test count equals the number of `test_*`
  methods in its source. `Failed: 0` cannot establish this on its own, because `dbx_test` turns a
  `run_setup` exception into "zero tests ran" and derives the failure count from the tests that ran,
  so a skipped fixture reports as a pass. `test_bulk_write_roundtrip` was silently not running for
  six tier runs while the summary printed `84/84` and `All tests passed`; the total looked stable only
  because the 8 tests it dropped were nearly offset by a new 7-test fixture. Exit 1 on a zero-test
  fixture, an opaque `all_tests` entry, a partial run, or a fixture missing entirely.
- **`scripts/check_version_consistency.py`**, asserts `pyproject.toml`'s version matches
  `__init__.__version__`, `HANDOFF.md`'s header, and every wheel-filename reference in the docs, this
  skill, and `integration_tests/config/test_config.yml`. That last file is the reason it exists: it
  pins the wheel the live integration tier INSTALLS, so a stale pin silently validates the previous
  release while reporting success. Also fails if it matches ZERO wheel references, so a moved file
  cannot make the gate vacuous.

```bash
python scripts/check_tier_results.py          # exit 0 = the tier ran every test it covers
python scripts/check_requirements_match.py    # exit 0 = match
python scripts/check_readme_sync.py           # exit 0 = docs enumerate the code
python scripts/check_version_consistency.py   # exit 0 = every version/wheel reference agrees
```

## Provenance discipline (a trap I have hit)

The wheel that runs must be the wheel you built. Always confirm the chain: local build md5 ==
uploaded-Volume md5 == the version the fixtures/demos install. A stale Volume wheel once made a probe
show "correct" behavior for reverted code. Verify against live state; never assume what you wrote is
what's running. Distinguish proven-live (ran it, saw the result) from tested (a check passed) from
designed (wrote it, didn't run it).
