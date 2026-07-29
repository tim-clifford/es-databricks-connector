# databricks-es-connector

Serverless-safe bulk export from Databricks/Spark to Elasticsearch.

Schema-agnostic Python library: given a Spark DataFrame and an `EsConfig`, it writes
rows to an Elasticsearch index via the `elasticsearch-py` client, parallelized
across executors with `mapInPandas` (serverless-safe), with gzip request compression
and deterministic document IDs for idempotent upserts.

Built because the `elasticsearch-spark` connector cannot run on serverless compute
(no third-party Spark JARs), has no Spark 4 / DBR 17+ build, and offers no request
compression. The Python client has none of those limits.

> **Maturity: 0.3.0.** The mechanism is proven and every valid Spark datatype is exportable with no
> caller pre-processing; inserts, upserts, and deletes are supported. Production hardening items (TLS/CA trust,
> API-key auth worked example, FIPS/FedRAMP, error dead-lettering, index templates/ILM) are not yet
> built. See [HANDOFF.md](HANDOFF.md) for the known limitations and pre-production checklist.

## Why this shape

- **Serverless works.** Uses `mapInPandas`, not RDD APIs (`foreachPartition` is blocked
  on serverless). Throughput still scales across executors.
- **Egress-minimized.** `http_compress=True` gzips the bulk body (~5–10x on JSON); field
  pruning drops columns you don't need indexed. Both are levers for cross-cloud cost.
- **Idempotent.** Deterministic `_id` (from a configurable column) means checkpoint
  replays and backfills upsert instead of duplicating.
- **Schema-agnostic.** Knows nothing about any customer schema or data model — you supply
  the DataFrame and the target index.

## Usage (batch)

> **First time in a workspace?** Install the library and create the ES secret scope before
> running any of the snippets below — see [Deploying to a workspace](#deploying-to-a-workspace).
> The `EsConfig` examples assume `databricks_es_connector` is importable and secrets exist.

```python
from databricks_es_connector import EsConfig, bulk_write

cfg = EsConfig(
    hosts="https://es-host:9200",
    api_key=dbutils.secrets.get("es", "api_key"),
    index="my-index",
    id_field="doc_id",       # deterministic _id => idempotent
    http_compress=True,      # gzip the request body
    drop_fields=("raw_data", "unmapped"),  # prune before indexing
    verify_certs=True,
)
result = bulk_write(gold_df, cfg)   # -> {"written": N, "deleted": D, "errors": M, "total_input": T, "error_samples": [...]}
```

## Usage (streaming)

The library exposes `make_foreach_batch(cfg, on_batch=None)`, a `foreachBatch` function that
bulk-writes each micro-batch:

```python
from databricks_es_connector import EsConfig, make_foreach_batch

cfg = EsConfig(hosts=..., api_key=..., index=..., id_field="doc_id")
(spark.readStream.table("catalog.schema.source_table")
   .writeStream
   .foreachBatch(make_foreach_batch(cfg, on_batch=lambda bid, r: print(bid, r)))
   .option("checkpointLocation", "/Volumes/<cat>/<schema>/<vol>/es_ckpt")  # UC Volume
   .trigger(availableNow=True)   # only availableNow/Once on serverless
   .start())
```

Notes for serverless:

- **Only `availableNow`/`Once` triggers work** on serverless. Each run drains the source
  commits appended since the last run; the checkpoint tracks the offset (incremental tailing).
- **The checkpoint must live on a UC Volume** (`dbfs:/tmp` paths fail with
  `INSUFFICIENT_PERMISSIONS`). One checkpoint dir = one logical stream.
- **Run streaming as a Databricks job, not interactively.** Driving a `foreachBatch` stream
  cell-by-cell on serverless / Spark Connect is unreliable (`query.start()` can intermittently
  hang). Wrap it in a job that calls `spark.streams.awaitAnyTermination()`. See
  [HANDOFF.md](HANDOFF.md) for the details and the at-least-once / freshness caveats.

## Usage (deletes)

By default the connector only inserts and updates (replace-by-`_id`). To also propagate
**deletes**, set `has_deletes=True` and name a `delete_flag_column`: rows whose flag is truthy
are sent to ES as a delete-by-`_id`; all other rows index as usual.

```python
cfg = EsConfig(
    hosts=..., api_key=..., index="my-index",
    id_field="doc_id",              # required for deletes — you delete by _id
    has_deletes=True,
    delete_flag_column="_is_delete",  # truthy row => delete that _id from ES
)
result = bulk_write(prepared_df, cfg)   # -> {"written": N, "deleted": D, "errors": M, "total_input": T, "error_samples": [...]}
```

The connector is schema-agnostic about *how* you decide the latest state of a row: it deletes
exactly the rows you flag and indexes the rest. A common source is a Delta **Change Data Feed** —
you dedup to one record per id in Spark and set the flag from `_change_type == 'delete'`. That
ordering/dedup work stays in your pipeline, not in the connector, because it needs your business
sequencing column (e.g. `event_ts`) and the sort should run distributed in Spark rather than in
executor memory. A typical `foreachBatch` shape:

```python
from pyspark.sql import functions as F, Window

def upsert_latest(batch_df, batch_id):
    # keep the latest record per id: business time first, commit version as tie-breaker
    w = Window.partitionBy("id").orderBy(
        F.col("event_ts").desc_nulls_last(),
        F.col("_commit_version").desc(),
    )
    latest = (
        batch_df
        .withColumn("rn", F.row_number().over(w)).where("rn = 1").drop("rn")
        .withColumn("_is_delete", F.col("_change_type") == "delete")
    )
    make_foreach_batch(cfg, on_batch=on_batch)(latest, batch_id)  # cfg has has_deletes=True
```

**Ordering caveat:** dedup like the above is only authoritative *within* one micro-batch. Across
batches, Elasticsearch applies writes in arrival/commit order, so a late-arriving row whose
`event_ts` is older than a doc already in ES — but committed in a later batch — can overwrite the
newer doc. If your source can deliver out-of-order events across batches, make `event_ts` globally
authoritative with ES external versioning (`version` = `event_ts` epoch-millis,
`version_type=external_gte`) so ES itself rejects stale writes regardless of batch order.

## Configuration (`EsConfig`)

Every knob for both batch and streaming lives on the `EsConfig` dataclass
(`src/databricks_es_connector/config.py`). It is frozen and serializable so it can be shipped
to Spark executors; the Elasticsearch client is built from it *inside* each partition, never on
the driver.

**Connection**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `hosts` | `str` | — | **Yes** | e.g. `"https://host:9200"`. Empty string raises `ValueError`. |
| `api_key` | `str \| None` | `None` | one-of\* | Preferred auth. Base64 `id:key` or an encoded key. |
| `basic_auth` | `tuple \| None` | `None` | one-of\* | `("user", "pass")`. Sandbox/dev only; prefer `api_key` in prod. |
| `verify_certs` | `bool` | `True` | No | Set `False` for self-signed sandbox boxes. Library default is secure. |
| `ca_certs` | `str \| None` | `None` | No | Path to a CA bundle when pinning. Mutually exclusive with `verify_certs=False`. |

\* **Auth is required**: you must set exactly one of `api_key` or `basic_auth`, or the
constructor raises `ValueError`. Setting `ca_certs` together with `verify_certs=False` also
raises (pick one).

**Write behavior**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `index` | `str` | `""` | **Yes** | Target index. `bulk_write`/`build_action` raise if empty. |
| `id_field` | `str \| None` | `None` | No | Column used as the deterministic `_id` → idempotent upserts. If unset, ES assigns random IDs (replays duplicate). If set, the column must be non-null in every row. |
| `http_compress` | `bool` | `True` | No | gzip the bulk request body (the egress-cost lever). |
| `chunk_size` | `int` | `500` | No | Docs per bulk request. |
| `request_timeout` | `int` | `60` | No | Per-request timeout (seconds). |
| `max_retries` | `int` | `3` | No | Client-side retries per request. |
| `retry_on_timeout` | `bool` | `True` | No | Retry on timeout as well as connection errors. |

**Doc shaping**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `drop_fields` | `tuple` | `()` | No | Columns the client chooses to **skip** to shrink payload (egress opt-out), e.g. `("raw_data", "unmapped")`. |

**Deletes**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `has_deletes` | `bool` | `False` | No | `False` (default) = every row is an index/upsert (historical behavior, unchanged). `True` routes rows whose `delete_flag_column` is truthy to an ES delete-by-`_id`. |
| `delete_flag_column` | `str \| None` | `None` | when `has_deletes` | Boolean-ish column; a truthy value (`True`/`1`/`"true"`/…) deletes that `_id`. Null/absent = not a delete. The column is pruned from the `_source` of kept rows so it is never indexed. |

When `has_deletes=True` the constructor requires both `id_field` (you cannot delete without an
`_id`) and `delete_flag_column`, and raises `ValueError` otherwise. Setting `delete_flag_column`
while `has_deletes=False` also raises — a flag column that does nothing is a misconfiguration, not
a silent no-op.

Delete idempotency: deleting a doc that isn't in ES returns a 404, which the connector treats as
an expected no-op (not an error) — so checkpoint replays, filtered rows, and re-processed deletes
stay clean. This suppression is scoped strictly to `delete` + `404`; a 404 on an index, or any
other status on a delete, is still counted in `errors`.

See [Usage (deletes)](#usage-deletes) above for the recommended Change-Data-Feed pattern and the
cross-batch ordering caveat.

## The write result (`bulk_write` return value)

`bulk_write` returns a dict summarizing the write:

```python
{
  "written": 998,          # index/upsert ops that succeeded
  "deleted": 0,            # successful delete-by-id ops (non-zero only with has_deletes)
  "errors": 2,             # docs Elasticsearch rejected (exact count)
  "total_input": 1000,     # rows handed to the writer
  "error_samples": [       # bounded diagnostics for rejected docs (up to 20)
    {"_id": "abc", "op_type": "index", "status": 400,
     "reason": "failed to parse field [ts] of type [date]"},
  ],
}
```

- **Reconcile with `total_input`.** `written + deleted + errors` should equal `total_input` minus
  any delete-404 no-ops. If it is **less**, some rows were lost below the per-document level (for
  example a chunk-level serialization/transport error), which the per-doc `errors` count cannot
  see — so check this equality if exactness matters.
- **`error_samples` is a breadcrumb, not a dead-letter queue.** It retains up to the first 20
  failures (id, op, HTTP status, ES reason) so a failed write is diagnosable instead of an opaque
  count. The `errors` count is always exact; only the retained sample list is capped, so a batch
  that fails wholesale can't exhaust memory. If you need every failed row durably, capture them
  from your own pipeline — the connector does not persist them.

## Datatype coverage

Every valid Spark column can be exported with **no caller pre-processing** — hand `bulk_write` any
DataFrame. Values are transformed on the way to Elasticsearch as follows. These transforms are
**by design**: they are the deltas to expect between the Spark row and the ES `_source`, not bugs.

| Spark type | ES `_source` value | Example (input → stored) |
|---|---|---|
| `string`, `boolean` | unchanged | `"hi"` → `"hi"`; `true` → `true` |
| `byte`/`short`/`int`/`long` | unchanged (all integer widths become one JSON number; width not preserved) | `5` → `5` |
| `double` | unchanged; non-finite values (`Infinity`/`-Infinity`/`NaN`) become `null` (no JSON representation) | `1.5` → `1.5`; `Infinity` → `null` |
| `float` (32-bit) | its **exact 32-bit value widened to double** — the stored number shows the float32 rounding, not the literal you typed (see note) | `0.1` (float) → `0.10000000149011612` |
| `decimal(p,s)` | **float** (precision lost beyond ~15–17 sig figs) | `Decimal("1.50")` → `1.5` |
| `date` / `timestamp` | **epoch milliseconds** (integer), floored to the millisecond (sub-ms precision dropped) | `2021-01-01T00:00:00Z` → `1609459200000` |
| `binary` | **base64 string** | `b"\x01\x02"` → `"AQI="` |
| `struct` / `map` | nested object (recursed). Non-string `map` keys are rendered to strings (JSON keys must be strings) using the same transform as the value type | `{a: 1}` → `{"a": 1}`; `map<int,_>` `{1: "x"}` → `{"1": "x"}` |
| `array` | array (recursed) | `[1, 2]` → `[1, 2]` |
| `null` (any type) | present as JSON `null` (the field is kept, its value is `null`) | `None` → `null` |
| `variant` | **string containing serialized JSON** (see below) | `{"k": 1}` → `"{\"k\":1}"` |
| `interval` | **string** (see below) | `INTERVAL '1 02:03:04' DAY TO SECOND` → `"INTERVAL '1 02:03:04' DAY TO SECOND"` |

**Why `float` looks "changed":** a Spark `FLOAT` is 32-bit and can't represent `0.1` exactly — it
holds the nearest float32, whose true value is `0.10000000149011612`. The connector stores that
exact value (widened to a 64-bit double) rather than reformatting it back to `0.1`, so the stored
number is faithful to what Spark actually held, not to the source literal. Use `DOUBLE` if you need
`0.1` to store as `0.1`.

**Timestamp precision:** epoch-millis is floored to the containing millisecond, consistently for
pre- and post-epoch instants (matches Spark/Java `unix_millis`). Elasticsearch `date` is
millisecond-resolution by default; sub-millisecond (microsecond) precision from a Spark `timestamp`
is not preserved. Map the field as `date_nanos` and send nanos yourself if you need finer than ms.

### Arrow-hostile types: `variant` and `interval` (handled automatically)

`VARIANT` and `INTERVAL` have no Apache Arrow representation, so they cannot cross into
`mapInPandas` directly. `bulk_write` handles this for you — any column whose type **contains** one
of these at **any nesting depth** is serialized to a **string** before export, so no caller action
is required; hand `bulk_write` the raw DataFrame. The string form depends on the type:

- **`variant`** (at any depth, e.g. `variant`, `struct<...,v:variant>`, `array<struct<...,v:variant>>`)
  → a **JSON string**. Example: a `variant` holding `{"k": 1, "nested": [2, 3]}` is stored as the
  string `"{\"k\":1,\"nested\":[2,3]}"`.
- **`interval`** (day-time or year-month) → its **Spark string form**. Example:
  `INTERVAL '1 02:03:04' DAY TO SECOND` is stored as the string `"INTERVAL '1 02:03:04' DAY TO SECOND"`.

Round-trip consequence to expect: such a column lands in ES as a **string**, not a queryable nested
object — so map it as `keyword`/`text`, not `object`.

When adding fields, add a matching entry to the ES index mapping or ES will dynamic-map (and guess)
the type.

## Developing the library

Work on the library locally (on your machine, not Databricks) — set up an environment
and run the unit tests:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # editable install + test deps
pytest                        # pure-Python tests; no Spark or ES needed
```

To package a release, see [Building the wheel](#building-the-wheel) below.

## Distribution

Private Databricks Field Engineering artifact. Not published to PyPI. Distributed to
customers as a built `.whl` (upload to a Unity Catalog Volume, or via Delta Share),
installed in a notebook with `%pip install /Volumes/.../databricks_es_connector-*.whl`.

### Dependencies

The wheel's only direct runtime dependency is `elasticsearch>=8,<9` (declared in
`pyproject.toml`). [`requirements.txt`](requirements.txt) lists the full resolved
dependency tree with pinned versions — direct and transitive — for security scanning
(SCA / vulnerability review). Spark/pandas are provided by the Databricks runtime and
are not bundled.

### Building the wheel

The build packages only `src/databricks_es_connector/` (see
`[tool.hatch.build.targets.wheel]` in `pyproject.toml`); `tests/` is not shipped.

```bash
python -m venv .venv && source .venv/bin/activate   # if not already in one
pip install build                                   # PEP 517 build frontend
python -m build --wheel                             # writes dist/databricks_es_connector-<version>-py3-none-any.whl
```

The version comes from `[project].version` in `pyproject.toml`; bump it there before
building a new release. The wheel lands in `dist/` (git-ignored).

**When cutting a release, regenerate [`requirements.txt`](requirements.txt).** The build does
*not* read that file — the wheel declares only the abstract range `elasticsearch>=8,<9`, so a
fresh install resolves transitive versions at install time and drifts from the pinned snapshot.
`requirements.txt` is a point-in-time closure for security scanning; keep it matched to the
release by regenerating it (see the command in its header) whenever dependencies or the pin
change.

Verify the wheel contains only the library (no tests leakage):

```bash
python -c "import zipfile, glob; print('\n'.join(zipfile.ZipFile(sorted(glob.glob('dist/*.whl'))[-1]).namelist()))"
```

### Deploying to a workspace

Upload the wheel to a Unity Catalog Volume (persists across compute; works on serverless):

```bash
databricks fs cp dist/databricks_es_connector-*.whl \
  dbfs:/Volumes/<catalog>/<schema>/<volume>/ --profile <profile>
```

Then install it in a notebook and restart Python:

```python
%pip install /Volumes/<catalog>/<schema>/<volume>/databricks_es_connector-<version>-py3-none-any.whl
%restart_python
```

For customer hand-off, share the same `.whl` via a UC Volume in their workspace or
through Delta Share — no code changes, only the `EsConfig` endpoint/auth differ.

## Repo layout

```
src/databricks_es_connector/   # the library (this is what ships in the .whl)
  config.py                    #   EsConfig: connection + write behavior (see Configuration above)
  transform.py                 #   pure-Python row shaping (timestamps, numpy/arrays, decimal, binary, pruning)
  bulk.py                      #   executor-side mapInPandas bulk write (batch entry point)
  stream.py                    #   foreachBatch helper for Structured Streaming
  spark_prep.py                #   sanitize_for_arrow: Arrow-hostile types (VARIANT/INTERVAL) -> JSON string
tests/                         # unit tests for the pure-Python layer (no Spark/ES needed)
integration_tests/             # live-Spark/ES tests run on Databricks serverless via dbx_test
  test_datatype_coverage.py    #   every Spark datatype + edge cases, round-tripped through ES
  test_bulk_write_roundtrip.py #   the bulk_write result contract (counts, total_input, error_samples)
  test_sanitize_for_arrow.py   #   the Spark-side VARIANT/INTERVAL serialization, no ES
  config/test_config.yml       #   dbx_test config (profile, wheel, serverless env)
```

Two test tiers: `tests/` is the fast, infra-free gate (pure-Python, run with `pytest`);
`integration_tests/` runs on a live serverless workspace + Elasticsearch to cover the Spark-side
code the unit tests can't reach — see [`integration_tests/README.md`](integration_tests/README.md).

The built wheel contains only `databricks_es_connector`; neither `tests/` nor `integration_tests/`
is part of the customer artifact.

## License & Attribution

**Copyright © Databricks, Inc.** — Developed and maintained by Databricks Forward Deployed Engineering. Available to support customers and the broader community in connecting Databricks to Elasticsearch. For production support and customization, contact your Databricks account team.

---

**Built with 💜 by Databricks Forward Deployed Engineering**
