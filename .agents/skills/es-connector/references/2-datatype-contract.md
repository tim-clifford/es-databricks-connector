# Reference 2: The datatype contract (the five-places rule)

When you add support for a new Spark datatype, or change how an existing one is written/read, the
change is only complete when **all of these move together**. Skipping one is how fidelity silently
breaks. This checklist is the connector's most important review gate.

## The places, in dependency order

1. **`src/databricks_es_connector/spark_serialize.py` -> `build_ndjson`**
   The write transform: it builds the whole `_bulk` action line in Spark with `to_json` (Catalyst,
   JVM). Add or adjust a branch here (the recursive `_rewrite_date_ntz` / `_null_nonfinite` walks over
   struct/array/map, or the header/source expressions). This is column-level Catalyst, NOT per-row
   Python: you work with pyspark `Column` expressions and `df.schema` types, and `to_json` does the
   actual serialization. If a type needs Arrow-hostile handling or timestamp normalization first, that
   is the sixth place (`spark_prep.py`) below.

2. **`src/databricks_es_connector/read_transform.py` -> `read_coerce`**
   The EXACT inverse for the declared type token (`"timestamp"`, `"decimal(10,2)"`, `"struct<...>"`,
   etc.). If the write side is genuinely one-way (e.g. variant->string), the inverse is a documented
   passthrough, say so in a comment and in the README.

3. **`tests/test_read_transform.py`** (the read half) **AND an `integration_tests/` fixture** (the
   write<->read whole). `build_ndjson` only runs in Spark, so the round-trip oracle lives in the
   integration tier now, not offline. Unit-test `read_coerce` against the stored form directly
   (`read_coerce(<epoch-millis / base64 / float / JSON string>, "type") == x`, or `== <documented
   delta>` if one-way) for a fast red-before-green on the read inverse; add positive AND edge cases
   (null, empty container, nesting, the boundary that makes it one-way).

4. **An `integration_tests/` fixture** (live Spark + ES) -- the write<->read oracle
   Prove it end-to-end, because `tests/` can't run `to_json`, Spark's Arrow conversion, or ES itself.
   Usually `test_datatype_coverage.py` (the wide one-row-per-type matrix) and/or
   `test_read_roundtrip.py`. For timestamps, also `test_timezone_utc.py` under a non-UTC session.

5. **README "Datatype coverage" + "Read fidelity" tables** (and `.agents/skills/es-connector/references/1-fidelity-model.md`)
   The documented contract customers read. Update the write table, the read-inverse table, and the
   one-way-delta list if the delta set changed. Keep ref 1 in this skill in sync.

## The sixth place: Arrow-hostile or timestamp types

`src/databricks_es_connector/spark_prep.py` is involved when the type:
- **can't cross Arrow** (VARIANT, INTERVAL), it must be serialized to a string in `sanitize_for_arrow`
  *before* `mapInPandas`. Update `_type_is_arrow_hostile` / the serialization branch, and note that
  detection is TYPE-position aware (a field *named* `interval` must not trip it, see the regex and
  its unit tests in `tests/test_spark_prep.py`).
- **is a `TimestampType`**, `normalize_timestamps_for_utc` converts it to epoch-millis via
  `unix_millis` at every nesting depth (`_type_has_timestamp` / `_rewrite_timestamps` /
  `_epoch_struct_type`). Only `TimestampType`, never NTZ or Date.

The pure planning helpers here (`_type_has_timestamp`, `_epoch_struct_type`,
`_hostile_columns_from_describe`, `_type_is_arrow_hostile`, `_is_scalar_interval_type`) are unit-
tested in `tests/test_spark_prep.py`; the Spark-executing function bodies are covered only by the
integration tier.

## Worked example: adding `timestamp_ntz` read support (what actually happened in 0.4.1)

The write side already produced epoch-millis for NTZ, but `read_coerce` had no `timestamp_ntz`
branch, so it fell through to the unknown-token passthrough and returned the raw epoch-millis **int**
instead of a datetime. The complete fix touched:
1. `read_transform.py`: added `_epoch_millis_to_naive_datetime` + a `"timestamp_ntz"` branch
   returning a NAIVE datetime (the wall-clock Spark expects), symmetric with the write reading it as
   UTC to pick the epoch.
2. `tests/test_read_transform.py`: `test_timestamp_ntz_roundtrip_is_naive` (asserts value AND
   `tzinfo is None` AND `isinstance datetime`, i.e. not the raw int), proven red before the fix.
3. `integration_tests/test_read_roundtrip.py`: an `s_ts_ntz` column read back naive against live ES;
   `test_datatype_coverage.py` / `test_timezone_utc.py` assert NTZ is unaffected by the session zone.
4. README: `timestamp_ntz` rows added to both the write and read-inverse tables.
(the write side needed no change here, it was already correct, which is itself the tell: always
check whether the gap is on the write side (`build_ndjson`), the read side (`read_coerce`), or both.)

## Review checklist for a PR that touches transforms

- [ ] `build_ndjson` (`to_json`) and `read_coerce` are still exact inverses (or the delta is newly documented).
- [ ] A `tests/test_read_transform.py` case covers the read inverse against the stored form, shown to FAIL without the change.
- [ ] An integration fixture covers the write<->read round-trip live (and under a non-UTC session if it's a timestamp).
- [ ] README write + read tables and ref 1 updated; one-way-delta list correct.
- [ ] If Arrow-hostile/timestamp: `spark_prep.py` handles it, with `tests/test_spark_prep.py` cases.
- [ ] `scripts/check_readme_sync.py` passes (new fixture/module documented).
- [ ] `scripts/check_version_consistency.py` passes (only relevant if the change ships with a
      version bump, but it is cheap and catches a stale wheel pin in the integration config).
