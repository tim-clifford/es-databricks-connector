# Integration tests (live Spark + Elasticsearch)

A second test tier that runs on **real Databricks serverless compute** via
[`dbx_test`](https://github.com/jsparhamii/dbx_test), covering what the pure-Python `tests/` suite
structurally cannot.

## Why this exists

`tests/` is fast, infra-free, and covers the pure-Python layer (`coerce_value`,
`classify_bulk_result`, `_merge_partition_results`, `EsConfig`) with hand-built inputs and a
monkeypatched ES client. Two things can't be tested there and only manifest on a live serverless
session:

- **`sanitize_for_arrow`** — the Spark-side VARIANT/INTERVAL serialization, including the fact that
  `df.schema` *throws* on a VARIANT column under Spark Connect (the reason the connector uses
  `DESCRIBE`). `tests/test_spark_prep.py` explicitly defers this to "a live cluster".
- **The `bulk_write` `mapInPandas` path** — real Arrow conversion of every dtype, partition
  fan-out, and the result schema (`total_input` / `error_samples`) surviving the Spark→driver trip.

These fixtures also serve as **live regression coverage for the 0.3.1 fidelity fixes** (non-string
map keys, float32 widening, pre-epoch timestamp floor).

This tier is **not** part of the fast local gate — it needs a workspace, the connector wheel on a
Volume, and (for the round-trip) a reachable ES + the `es_poc` secret scope. Keep running `pytest`
for the fast inner loop; run this before a release or on connector PRs.

## Fixtures

- **`test_sanitize_for_arrow.py`** — Spark only, no ES. Asserts VARIANT→JSON-string,
  scalar INTERVAL→Spark string form, plain columns untouched, idempotency, and the
  `df.schema`-throws-on-Connect constraint.
- **`test_bulk_write_roundtrip.py`** — live ES. Writes a wide, edge-case DataFrame through the
  connector, reads it back, and asserts the documented transforms plus the 0.3.1 result contract.
  Uses a throwaway index (`connector-integration-roundtrip`), recreated and dropped per run.

## Running

Prerequisites in the target workspace: the connector wheel on the Volume referenced in
`config/test_config.yml`, and (for the round-trip) the `es_poc` secret scope + a network path to ES.

```bash
# from the connector repo root
pip install -e ".[dev]"
pip install git+https://github.com/jsparhamii/dbx_test.git

# upload the fixtures to a workspace path, then:
dbx_test run \
  --tests-dir /Workspace/Users/<you>/es_connector_integration \
  --profile <profile> \
  --config integration_tests/config/test_config.yml
```

`dbx_test` installs the framework + the connector wheel into an inline serverless environment,
runs each fixture as a notebook, and reports console + JUnit + JSON results (JUnit is CI-friendly).

## Notes on the dbx_test dependency

- Install from `main` (`git+https://github.com/jsparhamii/dbx_test.git`).
- Each fixture ends with `run_notebook_tests(<TheFixtureClass>)`, passing the class **explicitly**.
  The no-argument form auto-discovers via a frame walk that doesn't reach the notebook's globals
  through the wrapper, so it silently finds zero fixtures and reports a hollow pass. Always name the
  class.

