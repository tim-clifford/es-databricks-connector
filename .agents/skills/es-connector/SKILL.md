---
name: es-connector
description: >
  Maintain, review, refactor, and release the databricks-es-connector, a serverless-safe,
  bidirectional Spark<->Elasticsearch library (bulk_write / read_index). Use when changing any file
  under src/databricks_es_connector/, adding or altering a Spark datatype's write transform or read
  inverse, touching the write path (sanitize_for_arrow / timestamp normalization / coerce_value /
  bulk), the read path (read_index / read_coerce), debugging a data-fidelity or round-trip question,
  reasoning about Elasticsearch behavior (dynamic mapping, term/keyword, timezone, coercion), running
  the integration tier on FEVM serverless, reviewing repo documentation for drift (README, HANDOFF,
  RELEASING, the skill itself), or cutting a release. Also use when the user mentions the
  connector, bulk_write, read_index, EsConfig/EsWriteConfig/EsReadConfig, datatype fidelity, the
  round-trip, doc review, or es-databricks-connector.
---

# databricks-es-connector maintenance

This is a **stable maintenance-and-review** skill, not an authoring one. The connector is a small
(~1,300-line) library; the recurring work is *changing it safely and proving fidelity*, not writing
new code from a template. The single most important property to protect is **round-trip data
fidelity**: a value written from Spark to ES and read back must be unchanged, except for the deltas
the README explicitly documents as one-way.

## Architecture in one screen

**Public API** (`src/databricks_es_connector/__init__.py`): `bulk_write`, `read_index`,
`make_foreach_batch`, `reconcile_or_raise`, configs (`EsWriteConfig`, `EsReadConfig`, `EsConnection`,
and the `EsConfig` alias = `EsWriteConfig`), the pure transforms `coerce_value` / `read_coerce` /
`to_es_source` / `sanitize_for_arrow`, and the exceptions `EsWriteError` / `AmbiguousDeleteFlag` /
`ReadSchemaMismatch`.

**Write path** (`bulk_write` in `bulk.py`) runs, in order:
1. `sanitize_for_arrow(df)` (`spark_prep.py`, Spark-side): serializes Arrow-hostile columns
   (VARIANT/INTERVAL, at any nesting depth) to strings, because they can't cross Arrow into
   `mapInPandas`. Reads the schema via `DESCRIBE` over a temp view (NOT `df.schema`, which throws on
   a VARIANT column under Spark Connect).
2. `normalize_timestamps_for_utc(df)` (`spark_prep.py`, Spark-side): converts every `TimestampType`
   (at any depth) to an epoch-millis long via `unix_millis`, so the stored epoch is the true UTC
   instant regardless of `spark.sql.session.timeZone`. Runs AFTER sanitize (sanitize strips the
   VARIANT columns that would break `df.schema`).
3. `_preflight(df, cfg)` (driver-side, once): validates `drop_fields` names against `df.columns` and
   checks the target index exists, so a misspelled name fails closed instead of silently pruning
   nothing or writing to an auto-created index. Runs after sanitize (`df.columns` is then safe) and
   before any row is written. Never inside the `mapInPandas` closure, which would put an HTTP
   round-trip on every executor.
4. `df.mapInPandas(writer, ...)` (per-partition, executor-side): each row dict goes through
   `coerce_value` (`transform.py`), the pure-Python value shaper, then `build_action` /
   `to_es_source` build the ES bulk action; `helpers.streaming_bulk` ships it, yielding one result
   per document so each outcome is classified individually (`classify_bulk_result`).
5. `_merge_partition_results(rows)` sums the per-partition counts and derives `unaccounted`.
   `raise_on_error=True` then applies `reconcile_or_raise`.

**Read path** (`read_index` in `read.py`): opens a Point-in-Time, fans out
`spark.range(num_slices).mapInPandas(...)` (pass `num_slices=1` for a single unsliced reader on small
reads), and coerces each `_source` value back to the caller's **declared Spark type** via
`read_coerce` (`read_transform.py`).
The read schema is required, there is no mapping inference (several write transforms are one-way and
can't be inverted from the stored value alone).

**Serverless / Spark Connect constraints** (these shape the whole design, do not "simplify" them
away): no RDD APIs (`mapInPandas` only); `df.schema` / `df.dtypes` / `df.columns` all throw on a
VARIANT column (use `DESCRIBE`); executors build their own ES client from a frozen config (nothing
non-serializable crosses the wire).

## The fidelity contract (read this before touching any transform)

`coerce_value` (write) and `read_coerce` (read) MUST stay exact inverses, except for the documented
one-way deltas: **decimal** precision beyond ~15-17 sig figs, **sub-millisecond timestamp** floor,
**float32** widening. Everything else round-trips exactly. See
[references/1-fidelity-model.md](references/1-fidelity-model.md) for the full per-type table (stored
form, inverse, and which deltas are expected), and the crucial `_source`-vs-indexed distinction that
explains why ES dynamic-mapping coercion does NOT break the connector round-trip.

## The five-places rule (the highest-value invariant)

When you add or change how a Spark datatype is handled, **five things must move together** or fidelity
silently breaks. This is the connector's single most error-prone surface:

1. `transform.py::coerce_value`: the write transform (Spark value -> ES `_source`).
2. `read_transform.py::read_coerce`: the exact read inverse for the declared type.
3. `tests/test_read_transform.py`: the pure round-trip **oracle** (`read_coerce(coerce_value(x)) == x`).
4. An `integration_tests/` fixture: the same round-trip proven live against real Spark + ES.
5. The README "Datatype coverage" + "Read fidelity" tables: the documented contract.

`spark_prep.py` (VARIANT/INTERVAL, timestamp normalization) is a sixth place when the type is
Arrow-hostile or a timestamp. See [references/2-datatype-contract.md](references/2-datatype-contract.md)
for the exact checklist and worked examples.

## Elasticsearch gotchas that generate questions

Timezone handling, `term`/`.keyword` on dynamically-mapped strings, and dynamic-mapping coercion
(`_source` faithful vs. indexed value coerced, so a `_source` round-trip cannot verify a mapping).
These are the recurring customer/reviewer questions. See
[references/3-es-gotchas.md](references/3-es-gotchas.md).

## Testing and releasing

Two tiers: `tests/` (pure-Python, `pytest`, the fast gate, the transform core is 100% line-covered)
and `integration_tests/` (live Spark + ES on FEVM serverless via `dbx_test`, the release gate). The
Spark-side code in `spark_prep.py` can only be proven in the integration tier. Releases follow
`RELEASING.md` with two hard gate scripts (`scripts/check_requirements_match.py`,
`scripts/check_readme_sync.py`). See
[references/4-release-and-tests.md](references/4-release-and-tests.md) for how to run each and the
exact FEVM commands.

## Documentation review (pre-change and at review time)

The prose docs (README, `integration_tests/README.md`, `RELEASING.md`, `HANDOFF.md`, and THIS skill)
are not executable, so nothing fails when they drift; `scripts/check_readme_sync.py` catches only
missing file listings, not stale descriptions or wrong versions. Run the holistic doc-review
checklist in [references/5-doc-review.md](references/5-doc-review.md) as a **pre-change step** when
starting a session that will modify the repo, and as part of **code review** for any PR that changes
behavior, the public API, datatype handling, the release process, or the file set. The skill itself
is the doc most prone to silent drift (it restates the others), so it is explicitly in scope.

## Workflow (what to load, when)

| Task | Load |
|------|------|
| Add / change a datatype's handling | refs 1, 2 (and update all five/six places) |
| Debug a round-trip / fidelity question | ref 1 (then reproduce with a `tests/test_read_transform.py` oracle case) |
| Answer an ES-behavior question (timezone, term, coercion) | ref 3 |
| Touch the Spark-side write prep (`spark_prep.py`) | refs 1, 4 (only the integration tier can prove it) |
| Run the integration tier on FEVM / cut a release | ref 4 + `RELEASING.md` |
| Review a PR touching transforms | refs 1, 2 (verify the five-places rule held) |
| Start a work session / review any behavior or file-set change | ref 5 (holistic doc-review checklist) |

## Critical rules

- **Never let `coerce_value` and `read_coerce` drift apart.** A change to one needs the inverse in
  the other, or the round-trip oracle in `tests/test_read_transform.py` must be updated with a
  *deliberate, documented* reason (a new one-way delta added to the README). Red-before-green: a new
  fidelity test must fail without the change.
- **`df.schema` is forbidden on a possibly-VARIANT DataFrame** (Spark Connect throws). Use the
  `DESCRIBE`-over-temp-view path already in `spark_prep.py`.
- **Timestamp normalization runs after sanitize, never before**: sanitize removes the VARIANT
  columns that would make the `df.schema` walk in `normalize_timestamps_for_utc` throw.
- **Only `TimestampType` is normalized to epoch-millis**, never `TimestampNTZType` (zoneless,
  defined as UTC) or `DateType` (no time-of-day); `unix_millis` rejects NTZ anyway.
- **The pure-Python layers must stay importable without Spark**: pyspark is imported lazily inside
  functions in `spark_prep.py` / `read.py`. Keep it that way so `tests/` runs with no Spark.
- **Verify claims against live ES, not memory or intuition.** The timezone bug hid behind a
  plausible-looking test for multiple sessions; the fix was only found by running real code on
  serverless under a non-UTC session. For any ES-behavior claim, check Elastic's docs or probe a
  live index. Distinguish proven-live from tested-offline from designed.
- **Docs are part of the change.** Adding a module/fixture/script means updating the README(s);
  `scripts/check_readme_sync.py` gates it, but write the entry as you go.
- **Review the docs holistically before changing the repo and at review time.** Run
  [references/5-doc-review.md](references/5-doc-review.md): version consistency, public API, datatype
  tables, release process, HANDOFF status, and this skill vs. the code. Fix drift in the same PR that
  caused it, never in a deferred "docs catch-up" pass. This is the only backstop that keeps the prose
  docs and the skill itself honest, since they have no failing test.
