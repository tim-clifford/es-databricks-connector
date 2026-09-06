# Reference 1: The fidelity model

The connector's core promise: a value written from Spark to Elasticsearch and read back with the
same declared schema is **unchanged**, except for two documented one-way deltas. This file is the
authoritative per-type map. It mirrors the README "Datatype coverage" (write) and "Read fidelity"
(read) tables; if you change a transform, update those tables and this file together.

The write serializer is `spark_serialize.build_ndjson`, which builds the whole `_bulk` action line
in Spark with `to_json` (there is one write path; the old per-row Python `coerce_value` path was
removed in 0.9.0). The read inverse is `read_transform.read_coerce`.

## Per-type: write transform, stored form, read inverse

| Spark type | `build_ndjson` / `to_json` writes (ES `_source`) | `read_coerce` reads back | Exact? |
|---|---|---|---|
| `string`, `boolean` | unchanged | unchanged | yes |
| `byte`/`short`/`int`/`long` | one JSON number (width not preserved in ES) | `int(value)` to the declared width | yes (value; width is the declared type's) |
| `double` | unchanged; non-finite (`inf`/`-inf`/`NaN`) -> JSON `null` | `float(value)` | yes, except non-finite -> null (one-way) |
| `float` (32-bit) | its **short decimal repr** (shortest string round-tripping to the same float32, e.g. `0.1`) | `float(value)` | yes (reads back into a `FLOAT` unchanged) |
| `decimal(p,s)` | **full precision** JSON number | `Decimal(str(value))` | integer-valued: exact; fractional: **one-way past ~15-17 sig figs** (lost on the read float-parse) |
| `date` | epoch-millis (midnight UTC) | `date` (UTC date component) | yes |
| `timestamp` | epoch-millis of the true UTC instant (via `unix_millis` in Spark) | aware UTC `datetime` | yes to the ms; **one-way: sub-ms floored** |
| `timestamp_ntz` | epoch-millis of the wall-clock read as UTC | **naive** `datetime` (zone dropped) | yes to the ms |
| `binary` | **base64 string** | `base64.b64decode` -> `bytes` | yes |
| `struct` / `map` | nested object (recursed); non-string map keys stringified by `to_json` to their own form (a temporal key -> ISO string, unlike a temporal VALUE -> epoch-millis) | recurse per field/value type; keys stay strings | yes (keys stay strings, one-way) |
| `array` | array (recursed) | list (recursed); a bare ES scalar is wrapped to `[x]` | yes |
| `null` (any type) | JSON `null` (field kept) | `None` | yes |
| `variant` | **JSON string** (serialized in `sanitize_for_arrow`) | the JSON string (caller re-parses with `parse_json`) | one-way: caller must re-parse |
| `interval` | **string** (Spark's string form) | the string | one-way |

## The two documented one-way deltas (the ONLY acceptable losses)

1. **Decimal fractional precision** beyond double's ~15-17 significant figures. `to_json` writes the
   decimal at full precision, so an integer-valued decimal round-trips exactly; a value with a
   fractional part past ~15-17 sig figs loses its low digits when the stored JSON number is parsed to
   a float on read. Mitigation the README documents: `CAST(col AS STRING)` in Spark before writing,
   declare `StringType` on read, exact.
2. **Sub-millisecond timestamp** precision. `unix_millis` floors to the millisecond (ES `date` is
   ms-resolution by default). Mitigation: map as `date_nanos` and send nanos yourself.

**`float` (32-bit) is NOT a delta anymore** (it was in 0.8.x and earlier, when the per-row path stored
the exact widened double). `to_json` renders a `FLOAT` as its short decimal repr, which reads back
into a `FLOAT` unchanged. Dropping float32 took the delta count from three to two; that was a
deliberate 0.9.0 contract change (the single-path consolidation), not a slip.

If a change would introduce a THIRD one-way delta, that is a contract change: it must be added to the
README tables, this file, and called out explicitly to the user, not slipped in.

## Non-finite floats and nulls (write side, `spark_serialize.build_ndjson`)

- `to_json` renders a non-finite float as the quoted string `"NaN"`/`"Infinity"`/`"-Infinity"`, which
  ES's strict parser rejects, so `build_ndjson` replaces `inf`/`-inf`/`NaN` with JSON `null` at ANY
  nesting depth (recursive walk over struct/array/map values). This is **not counted**:
  `coerced_nonfinite` in the result is always 0. Detect a non-finite in Spark before the write if you
  need to know one became null (e.g. an upstream divide-by-zero).
- A null or non-finite `id_field` value -> a null action line -> `make_ndjson_partition_writer`
  RAISES, failing the write unconditionally rather than shipping `"_id": null` (which ES could
  auto-assign, duplicating the row on replay).
- A `NaN` used as a semantic signal is lost (becomes null). Documented; flag it if a user relies on it.

## The `_source`-vs-indexed distinction (why ES coercion does NOT break the round-trip)

This is the key insight that resolves most "did the connector lose my data?" questions:

- **`read_index` reads from `_source`**, which Elasticsearch stores **verbatim** as the JSON that was
  indexed (`read.py` uses `h.get("_source", {})`).
- ES dynamic-mapping **coercion** (a float `1.5` into a field first mapped as `long` indexes as `1`)
  and keyword **`ignore_above`** truncation affect only the **indexed / queryable** value, never
  `_source`.
- Therefore the connector round-trip is **faithful** even under dynamic mapping. The surprise only
  appears when someone **queries ES directly** (aggregation, sort, `term`) and reads the indexed
  value. See [3-es-gotchas.md](3-es-gotchas.md). Proven live in
  `integration_tests/test_dynamic_mapping_coercion.py`.

## How to verify a fidelity claim (do this, don't reason in your head)

1. **Offline read inverse first**: add a case to `tests/test_read_transform.py` feeding `read_coerce`
   the stored form (epoch-millis, base64, a float, ...) and asserting the Python value. Cheap
   red-before-green for the read half.
2. **Then the live write->read round-trip** (the whole oracle now, since `to_json` only runs in
   Spark): an `integration_tests/` fixture on FEVM (`test_datatype_coverage.py`,
   `test_read_roundtrip.py`), ideally under a NON-UTC session (that's what surfaced the timezone bug
   a UTC-only test hid). This is where write<->read fidelity is proven end to end.
3. For an ES-behavior claim, consult Elastic's docs or probe a real index. Never assert ES behavior
   from memory; the timezone corruption hid behind a plausible test for multiple sessions.
