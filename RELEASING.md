# Releasing

The checklist for cutting a tagged release. Every release MUST pass these four steps in order.
Step 3 is enforced by a script (`scripts/check_requirements_match.py`, exits non-zero on drift);
the others are manual but must be done and their evidence recorded in the release notes.

Do not tag or publish without explicit sign-off. Never push to `main` to cut a release; tag from a
reviewed, merged commit.

## Prerequisites

- A clean checkout of the commit you intend to tag (usually the merge commit on `main`).
- The version in `pyproject.toml` `[project].version` already bumped to the target (e.g. `0.4.2`).
- A Databricks profile with access to the FEVM workspace (default
  `fe-vm-tim-clifford-classic-dsl-lite`) and the `es_poc` secret scope, for the integration tier.
- `gh` authenticated against the repo.

## The four steps

### 1. Run the integration tests (live Spark + ES on FEVM)

The pure-Python unit tests are the fast inner loop; the release gate is the **integration tier**,
which exercises the Spark-side paths (`sanitize_for_arrow`, the `unix_millis` timestamp
normalization) and the real `bulk_write` / `read_index` round-trip against live Elasticsearch.

```bash
# fast local gate first (must be green)
python -m pytest -q                                  # expect: all passed

# then the live integration tier on serverless.
# Upload the fixtures to a workspace path (once, or after any fixture change):
databricks workspace import-dir integration_tests \
  /Workspace/Users/<you>/es_connector_integration --overwrite --profile <profile>
# (delete any stale .py-suffixed notebooks left by a re-import so each fixture runs once)

dbx_test run \
  --tests-dir /Workspace/Users/<you>/es_connector_integration \
  --workspace-tests --profile <profile> \
  --config integration_tests/config/test_config.yml
```

Record the result line (e.g. `Total: 85, Passed: 85, Failed: 0`) in the release notes. Any failure
blocks the release.

> The integration `test_config.yml` pins the wheel path on the FEVM Volume. It must point at the
> version you are releasing. Build + upload (step 2) BEFORE running the integration tier so the tests
> run against the release wheel, not a stale one. Confirm the uploaded wheel's md5 matches your local
> build (see step 2) so you know exactly what ran.

### 2. Rebuild the wheel (and regenerate requirements.txt)

```bash
rm -f dist/databricks_es_connector-*.whl
python -m build --wheel                              # writes dist/databricks_es_connector-<version>-py3-none-any.whl
md5 dist/databricks_es_connector-*.whl               # note the md5; this is the release artifact's identity
```

Regenerate the pinned dependency closure (the build does NOT do this; the wheel declares only
`elasticsearch>=8,<9`, so `requirements.txt` drifts otherwise):

```bash
python -m venv /tmp/req && /tmp/req/bin/pip install "elasticsearch>=8,<9"
/tmp/req/bin/pip freeze > requirements.txt           # then re-annotate + restore the header comment
```

> **Use one Python minor version for both the regeneration above and the step-3 check**, ideally the
> one the Databricks runtime ships (`requirements.txt`'s header records which version resolved it).
> Transitive deps can carry environment markers (`; python_version < "3.11"`), so resolving on a
> different interpreter than the checker runs on can produce a spurious mismatch.

Upload the wheel to the FEVM Volume so step 1's integration run uses it, and verify the uploaded
copy is byte-identical to your local build:

```bash
VOL=/Volumes/<catalog>/<schema>/<volume>
databricks fs cp dist/databricks_es_connector-*.whl "dbfs:$VOL/" --overwrite --profile <profile>
databricks fs cp "dbfs:$VOL/databricks_es_connector-<version>-py3-none-any.whl" /tmp/uploaded.whl \
  --overwrite --profile <profile>
md5 /tmp/uploaded.whl                                # MUST equal the local md5 above
```

### 3. Run the release gates (HARD GATES)

Two scripts, both must exit `0`. Run them with the **same Python minor version** used to regenerate
`requirements.txt` in step 2 (see the caveat there).

**3a. `requirements.txt` exactly matches the wheel's closure:**

```bash
python scripts/check_requirements_match.py           # uses the newest wheel in dist/
# or explicitly:
python scripts/check_requirements_match.py --wheel dist/databricks_es_connector-<version>-py3-none-any.whl
```

Exits `0` and prints `OK: requirements.txt exactly matches ...` when the pinned closure matches the
wheel's resolved dependency tree, name+version exact. Exits `1` and prints the diff (missing / extra
/ version-mismatch) otherwise. On a non-zero exit, regenerate `requirements.txt` (step 2),
re-annotate, and re-run until green.

**3b. the docs enumerate the code** (every module / fixture / release script is documented):

```bash
python scripts/check_readme_sync.py
```

Exits `0` when every shipped module, integration fixture, and release script is referenced by name
in the README(s) that should list it; exits `1` with the gaps otherwise. This catches the drift a
diff-scoped code review structurally can't (a file added without updating the README's file list).
On a non-zero exit, add the missing file(s) to the named doc(s) with a one-line description and
re-run.

**A non-zero exit from either gate blocks the release.**

### 4. Tag and attach the wheel to the release

```bash
git tag v<version> <reviewed-commit-sha>
git push origin v<version>
gh release create v<version> \
  dist/databricks_es_connector-<version>-py3-none-any.whl \
  --title "v<version>" --notes "<summary + the step-1 integration result line + the step-2 md5>"
```

Confirm the wheel is attached:

```bash
gh release view v<version> --json assets --jq '.assets[].name'
```

## Definition of done for a release

- [ ] `pytest` green, integration tier green (result line recorded)
- [ ] wheel rebuilt from the tagged commit; local md5 == uploaded md5 (recorded)
- [ ] `scripts/check_requirements_match.py` exits 0
- [ ] `scripts/check_readme_sync.py` exits 0
- [ ] tag pushed from the reviewed commit; wheel attached to the GitHub release (verified)
