# Reference 3: Elasticsearch behavior gotchas

These are the recurring customer/reviewer questions. All three are **standard Elasticsearch
behavior**, not connector bugs, but the connector's design (schema-agnostic writes, relies on ES
dynamic mapping, no explicit mapping created) makes users hit them, so the connector's docs must
explain them. Each is verified (live serverless probes and/or Elastic docs), not asserted from memory.

## 1. Timezone: `timestamp` epoch is session-independent

**The hazard:** Spark's Arrow export renders a `TimestampType` instant to a naive pandas Timestamp
using `spark.sql.session.timeZone`, i.e. the session-LOCAL wall-clock with the zone dropped. Reading
that naive value as UTC would RELABEL the wall-clock rather than convert it, shifting the stored epoch
by the session's offset (e.g. -5h under `America/New_York`).

**How the connector avoids it:** `normalize_timestamps_for_utc` (`spark_prep.py`) converts every
`TimestampType` (at any nesting depth) to its true epoch-millis via Spark `unix_millis` **before** the
Arrow export. `unix_millis` operates on the instant, so it is independent of the session zone. No
session mutation. Mirrors elasticsearch-hadoop's approach.

**Why a UTC-only test cannot prove this:** a symmetric round-trip under a single UTC session cancels
the error on both sides, so it only surfaces under a NON-UTC session. That is why
`integration_tests/test_timezone_utc.py` and `test_datatype_coverage.py` run under
`America/New_York`. When reasoning about timezones, run real code under a non-UTC session.

**ES side (verified):** ES stores `date` fields as UTC epoch-millis internally; a numeric epoch is
taken as the literal UTC instant with no per-index tz shift. Only date STRINGS without an offset get
an assumed-UTC parse; only query-time aggregations take an optional `time_zone`. So sending
epoch-millis (what the connector does) is unambiguous. `timestamp_ntz` and `date` are already
session-independent and are deliberately left untouched.

## 2. `term` on a dynamically-mapped string returns nothing (use `.keyword` or `match`)

A dynamically-mapped string field becomes `text` (analyzed: lowercased, tokenized) with a `.keyword`
sub-field (exact). A `term` query does NOT analyze its input, so `{"term": {"region": "us"}}` compares
your raw input against analyzed tokens and usually misses. Query the exact sub-field
(`{"term": {"region.keyword": "us"}}`) or use `{"match": {"region": "us"}}`.

- This is universal ES behavior. The connector's only role: it creates no explicit mapping, so it
  relies on ES dynamic mapping, which produces the `text`+`.keyword` shape, so connector users hit
  this by default.
- Caveat when advising: if the user pre-created an EXPLICIT mapping (`region` as `keyword`), then the
  bare field is correct and `.keyword` doesn't exist, the opposite advice. Tell them to check with
  `GET /<index>/_mapping/field/<field>`. Documented in the README Reading section.

## 3. Dynamic-mapping coercion / `ignore_above`: `_source` faithful, indexed value changed

If no mapping is pre-created, ES infers each field's type from the **first** document and keeps it.
A later doc whose value doesn't fit is coerced **for indexing** (a float `1.5` into a field first
seen as an integer indexes as `1`; a keyword string past `ignore_above`, default 256 chars, isn't
indexed). Verified against Elastic docs: **`_source` keeps the original value verbatim in every
case.**

Consequences:
- **Connector round-trip: faithful.** `read_index` reads `_source`, so it returns `1.5` and the full
  long string regardless of coercion. Not a connector bug.
- **Direct ES query: sees the coerced/truncated value.** A `sum` aggregation reads the coerced `long`
  (returns 20, not 20.7, for `10 + 10.7`); a `term` on `.keyword` can't find a string that exceeded
  `ignore_above`. This is the surprise, and it's ES behavior.
- **A `_source` round-trip therefore cannot verify a mapping.** This is the trap to watch for in
  review: a write-then-`read_index` comparison compares the data against a verbatim copy of itself, so
  it passes no matter how wrong the mapping is. Any check claiming to validate a mapping must read the
  INDEXED value (`"fields": [...]`, an aggregation, or `GET /_mapping`). Verified live: `amount`
  dynamically mapped `long`, write `100.75` -> `errors=0`, round-trip `100.75`, indexed `100`,
  `range > 100.5` finds nothing.
- **Advice:** for heterogeneous-type or long-string fields, pre-create an explicit mapping.

Proven live in `integration_tests/test_dynamic_mapping_coercion.py` (both halves, with the two
"surprise" assertions being genuinely red-able if ES ever stopped coercing/truncating). Documented in
the README "Dynamic-mapping gotcha" note. The demos repo carries a shared `es_verify/` helper that
performs the indexed-value comparison.

## General principle

For ANY Elasticsearch-behavior claim: verify against Elastic's current docs or probe a live index,
and distinguish what ES *indexes* from what it *stores in `_source`*, that single distinction
resolves most fidelity confusion. The connector reads `_source`; most "data changed" reports are
actually "the indexed value differs," which is an ES-mapping matter the customer controls.
