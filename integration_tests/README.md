# Integration tests (live Spark + Elasticsearch)

A second test tier that runs on **real Databricks serverless compute** via
[`dbx_test`](https://github.com/jsparhamii/dbx_test), covering what the pure-Python `tests/` suite
structurally cannot.

## Why this exists

`tests/` is fast, infra-free, and covers the pure-Python layer (`coerce_value`,
`classify_bulk_result`, `_merge_partition_results`, `EsConfig`, and the `make_foreach_batch`
streaming glue) with hand-built inputs and a stubbed ES client. Two things can't be tested there and
only manifest on a live serverless session:

- **`sanitize_for_arrow`**: the Spark-side VARIANT/INTERVAL serialization, including the fact that
  `df.schema` *throws* on a VARIANT column under Spark Connect (the reason the connector uses
  `DESCRIBE`). `tests/test_spark_prep.py` explicitly defers this to "a live cluster".
- **The `bulk_write` `mapInPandas` path**: real Arrow conversion of every dtype, partition
  fan-out, and the result schema (`total_input` / `error_samples`) surviving the Spark→driver trip.

These fixtures also serve as **live regression coverage for the 0.3.1 fidelity fixes** (non-string
map keys, float32 widening, pre-epoch timestamp floor).

This tier is **not** part of the fast local gate: it needs a workspace, the connector wheel on a
Volume, and (for the round-trip) a reachable ES + the `es_poc` secret scope. Keep running `pytest`
for the fast inner loop; run this before a release or on connector PRs.

## Fixtures

Each fixture owns one concern:

- **`test_datatype_coverage.py`**: live ES. The exhaustive datatype-fidelity test: one wide row
  covering **every** Spark type + edge cases (byte/short/int/long, float32 widening, double,
  decimal incl. an 18-sig-fig value that proves the documented precision loss, date/timestamp incl.
  pre-epoch floor and `timestamp_ntz` UTC interpretation, binary, unicode string, bool,
  struct/nested-struct/partial-null-struct, map incl. non-string keys/empty-map/null-value,
  array/empty-array/array-of-struct/null-element, non-finite floats, VARIANT at every depth,
  INTERVAL), plus an all-NULL row proving the null contract (field present as JSON null, not
  absent), plus a reverse check that the connector adds no unexpected fields. Asserts each value
  against the README "Datatype coverage" contract. Throwaway index, dropped per run.
- **`test_bulk_write_roundtrip.py`**: live ES. Owns the **write-result contract**: `bulk_write`
  returns `written`/`deleted`/`errors`/`total_input`/`error_samples` correctly, including a
  deliberately-rejected doc (so the error path + `error_samples` are exercised, not just the clean
  path), idempotent re-write via a deterministic `_id`, and a duplicate-id-within-one-input case
  proving the collapse (counts succeed, ES doc count is lower). Does not re-assert per-type transforms.
- **`test_deletes_roundtrip.py`**: live ES. Owns the **delete-propagation contract** end-to-end:
  with `has_deletes=True`, flagged rows delete-by-`_id` while unflagged rows index; `deleted` is
  counted exactly; the delete-flag column is never indexed; and a delete of an absent `_id` is a
  **404 no-op** (neither a delete nor an error): the `classify_bulk_result` suppression rule that
  unit tests cover in isolation, proven live. Includes an idempotent re-delete.
- **`test_streaming_sink.py`**: live ES + a throwaway Delta table + a UC-Volume checkpoint. Owns
  the **streaming sink**: `make_foreach_batch` as a real `foreachBatch` on `readStream` +
  `trigger(availableNow=True)`. Proves one doc per source row and restart idempotency (re-running
  the same checkpoint after appends writes only the new rows; an empty re-run is a clean no-op),
  measured by the ES doc count. The only place `stream.py` runs against genuine Structured Streaming
  rather than a stub. (`on_batch`'s contract is unit-tested; it isn't re-checked here because
  serverless `foreachBatch` runs server-side and can't feed a driver-local capture.)
- **`test_sanitize_for_arrow.py`**: Spark only, no ES. The Spark-side transform in isolation:
  VARIANT→JSON-string at any depth, scalar INTERVAL→Spark string form, plain columns untouched,
  idempotency, Arrow-collectability, and the `df.schema`-throws-on-Connect constraint.

## Running

Prerequisites in the target workspace: the connector wheel on the Volume referenced in
`config/test_config.yml`, and (for the round-trip / deletes / streaming fixtures) the `es_poc`
secret scope + a network path to ES. The streaming fixture additionally needs a UC catalog/schema
to hold a throwaway Delta table and a UC Volume for its checkpoint (dbfs:/tmp checkpoints fail with
`INSUFFICIENT_PERMISSIONS` on serverless); it uses the catalog/schema/Volume named at the top of
`test_streaming_sink.py`, adjust those constants for a different workspace. Every fixture creates
and drops its own throwaway index/table/checkpoint.

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

- Installed from `main` (`git+https://github.com/jsparhamii/dbx_test.git`). Each fixture ends with
  the documented no-arg `run_notebook_tests()`, which discovers the fixture class in the notebook's
  scope.

