# Reference 1: The fidelity model

The connector's core promise: a value written from Spark to Elasticsearch and read back with the
same declared schema is **unchanged**, except for three documented one-way deltas. This file is the
authoritative per-type map. It mirrors the README "Datatype coverage" (write) and "Read fidelity"
(read) tables; if you change a transform, update those tables and this file together.

## Per-type: write transform, stored form, read inverse

| Spark type | `coerce_value` writes (ES `_source`) | `read_coerce` reads back | Exact? |
|---|---|---|---|
| `string`, `boolean` | unchanged | unchanged | yes |
| `byte`/`short`/`int`/`long` | one JSON number (width not preserved in ES) | `int(value)` to the declared width | yes (value; width is the declared type's) |
| `double` | unchanged; non-finite (`inf`/`-inf`/`NaN`) -> JSON `null` | `float(value)` | yes, except non-finite -> null (one-way) |
| `float` (32-bit) | its exact 32-bit value widened to double | `float(value)` | **one-way: float32 widening** (`0.1f` -> `0.10000000149011612`) |
| `decimal(p,s)` | **float** | `Decimal(str(value))` | **one-way past ~15-17 sig figs** |
| `date` | epoch-millis (midnight UTC) | `date` (UTC date component) | yes |
| `timestamp` | epoch-millis of the true UTC instant (via `unix_millis` in Spark) | aware UTC `datetime` | yes to the ms; **one-way: sub-ms floored** |
| `timestamp_ntz` | epoch-millis of the wall-clock read as UTC | **naive** `datetime` (zone dropped) | yes to the ms |
| `binary` | **base64 string** | `base64.b64decode` -> `bytes` | yes |
| `struct` / `map` | nested object (recursed); non-string map keys stringified | recurse per field/value type | yes (keys stay strings, one-way) |
| `array` | array (recursed) | list (recursed); a bare ES scalar is wrapped to `[x]` | yes |
| `null` (any type) | JSON `null` (field kept) | `None` | yes |
| `variant` | **JSON string** (serialized in `sanitize_for_arrow`) | the JSON string (caller re-parses with `parse_json`) | one-way: caller must re-parse |
| `interval` | **string** (Spark's string form) | the string | one-way |

## The three documented one-way deltas (the ONLY acceptable losses)

1. **Decimal precision** beyond double's ~15-17 significant figures. Mitigation the README documents:
   `CAST(col AS STRING)` in Spark before writing, declare `StringType` on read, exact.
2. **Sub-millisecond timestamp** precision. `unix_millis` / `coerce_value` floor to the millisecond
   (ES `date` is ms-resolution by default). Mitigation: map as `date_nanos` and send nanos yourself.
3. **Float32 widening**. A Spark `FLOAT` holds the nearest float32; the connector stores that exact
   value widened to double (faithful to what Spark held, not to the source literal). Use `DOUBLE` to
   avoid it.

If a change would introduce a FOURTH one-way delta, that is a contract change: it must be added to
the README tables, this file, and called out explicitly to the user, not slipped in.

## Non-finite floats and nulls (write side, `transform.py`)

- `None`, pandas `NaN`/`NaT`/`pd.NA` -> JSON `null` (via `_is_null`, which also guards `_id`
  derivation: a `NaN` id must be rejected, not stringified to `"nan"` and collapsed).
- `inf` / `-inf` -> JSON `null` (ES's strict parser rejects the `Infinity` token; a raw inf would
  fail the whole doc and lose it to the error count).
- `NaN` used as a semantic signal is lost (becomes null). Documented; flag it if a user relies on it.

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

1. **Offline oracle first**: add a case to `tests/test_read_transform.py`:
   `_roundtrip(x, "type") == x` (or the documented delta). This is the cheapest red-before-green.
2. **Then live** if the type touches Spark-side prep (VARIANT/INTERVAL/timestamp): an
   `integration_tests/` fixture on FEVM, ideally run under a NON-UTC session (that's what surfaced
   the timezone bug that a UTC-only test hid).
3. For an ES-behavior claim, consult Elastic's docs or probe a real index. Never assert ES behavior
   from memory; the timezone corruption hid behind a plausible test for multiple sessions.
