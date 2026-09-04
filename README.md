# databricks-es-connector

Serverless-safe, bi-directional transfer between Databricks/Spark and Elasticsearch.

Python library. **Write:** given a Spark DataFrame and an `EsWriteConfig`, it writes rows to an
Elasticsearch index via the `elasticsearch-py` client, parallelized across executors with
`mapInPandas` (serverless-safe), with gzip request compression and deterministic document IDs for
idempotent upserts. The writer is schema-agnostic: hand it any DataFrame, no pre-processing.
**Read:** given an `EsReadConfig` and a *declared* Spark schema (the reader does not infer one),
`read_index` pulls an index back into a DataFrame, distributed across executors via a sliced
Point-in-Time scroll.

Built because the `elasticsearch-spark` connector cannot run on serverless compute
(no third-party Spark JARs), has no Spark 4 / DBR 17+ build, and offers no request
compression. The Python client has none of those limits.

> **Maturity.** The mechanism is proven for both directions: every valid Spark datatype is
> exportable with no caller pre-processing (inserts, upserts, deletes), and an index can be read back
> into a DataFrame against a declared schema with write→read fidelity (see
> [Read fidelity](#read-fidelity-write--read)). Production hardening items (TLS/CA trust, API-key auth
> worked example, FIPS/FedRAMP, error dead-lettering, index templates/ILM) are not yet built. See
> [HANDOFF.md](HANDOFF.md) for the known limitations and pre-production checklist.

## Contents

- [Why this shape](#why-this-shape)
- [Writing to Elasticsearch](#writing-to-elasticsearch)
  - [Batch (`bulk_write`)](#batch-bulk_write)
  - [Streaming (`make_foreach_batch`)](#streaming-make_foreach_batch)
  - [Deletes](#deletes)
  - [The write result (`bulk_write` return value)](#the-write-result-bulk_write-return-value)
- [Reading from Elasticsearch](#reading-from-elasticsearch)
  - [`read_index`](#read_index)
  - [Read fidelity (write → read)](#read-fidelity-write--read)
- [Configuration](#configuration)
  - [Connection (shared by both configs)](#connection-shared-by-both-configs)
  - [Write behavior (`EsWriteConfig`)](#write-behavior-eswriteconfig)
  - [Read behavior (`EsReadConfig`)](#read-behavior-esreadconfig)
- [Datatype coverage (write transforms + read inverse)](#datatype-coverage-write-transforms--read-inverse)
  - [Arrow-hostile types: `variant` and `interval`](#arrow-hostile-types-variant-and-interval-handled-automatically)
- [Developing the library](#developing-the-library)
- [Distribution](#distribution)
  - [Dependencies](#dependencies)
  - [Building the wheel](#building-the-wheel)
  - [Deploying to a workspace](#deploying-to-a-workspace)
- [Repo layout](#repo-layout)

## Why this shape

**Shared by both directions:**

- **Serverless works.** Both the write and the read path use `mapInPandas`, not RDD APIs
  (`foreachPartition` and custom `DataSource`/RDD readers are blocked on serverless). Work still
  fans out across executors: writes partition the DataFrame, reads fan out one task per index slice.
- **Egress-minimized.** `http_compress=True` gzips both directions (~5-10x on JSON): the request
  body on write, and the ES response on read. A lever for cross-cloud cost either way.

**Writing** (`bulk_write` / `make_foreach_batch`):

- **Idempotent.** Deterministic `_id` (from a configurable column) means checkpoint
  replays and backfills upsert instead of duplicating.
- **Field pruning.** `drop_fields` drops columns you don't need indexed before the write, a further
  write-side egress lever on top of compression.
- **Schema-agnostic on write.** The writer knows nothing about your schema or data model, you
  hand it any DataFrame and a target index, and every Spark datatype is exportable with no
  pre-processing (see [Datatype coverage](#datatype-coverage-write-transforms--read-inverse)).

**Reading** (`read_index`):

- **Distributed snapshot read.** One Elasticsearch Point-in-Time gives a consistent snapshot, and a
  sliced scroll fans out across executors (one task per shard/slice) so a full-index export scales.
- **Explicit schema required.** Unlike the writer, the reader must be handed a Spark schema: several
  stored values are ambiguous and can't be inverted from `_source` alone, so the caller declares the
  intended types. This is the deliberate trade-off for exact write→read fidelity; see
  [Read fidelity](#read-fidelity-write--read).

## Writing to Elasticsearch

Write a Spark DataFrame to an index with `bulk_write` (batch) or `make_foreach_batch` (streaming),
configured with an [`EsWriteConfig`](#write-behavior-eswriteconfig).

### Batch (`bulk_write`)

> **First time in a workspace?** Install the library and create the ES secret scope before
> running any of the snippets below, see [Deploying to a workspace](#deploying-to-a-workspace).
> The examples assume `databricks_es_connector` is importable and secrets exist.

```python
from databricks_es_connector import EsWriteConfig, bulk_write

cfg = EsWriteConfig(
    hosts="https://es-host:9200",
    api_key=dbutils.secrets.get("es", "api_key"),
    index="my-index",
    id_field="doc_id",       # deterministic _id => idempotent
    http_compress=True,      # gzip the request body
    drop_fields=("raw_data", "unmapped"),  # prune before indexing
    verify_certs=True,
)
result = bulk_write(gold_df, cfg)   # -> {"written": N, "errors": M, "unaccounted": U, ...}

# Or let the library enforce it: raises EsWriteError if any doc was rejected or any row lost.
result = bulk_write(gold_df, cfg, raise_on_error=True)
```

`bulk_write` returns the counts and leaves the verdict to you (`raise_on_error=False` by default);
see [The write result](#the-write-result-bulk_write-return-value) for what to check. The **streaming**
entry point defaults the other way and raises, because there a swallowed error silently advances the
checkpoint past the lost rows.

### Streaming (`make_foreach_batch`)

The library exposes `make_foreach_batch(cfg, on_batch=None, on_error="raise")`, a `foreachBatch`
function that bulk-writes each micro-batch:

```python
from databricks_es_connector import EsWriteConfig, make_foreach_batch

cfg = EsWriteConfig(hosts=..., api_key=..., index=..., id_field="doc_id")
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
- **`on_batch` result shape.** The dict passed to your `on_batch` callback always carries the same
  keys `bulk_write` returns (`written`/`deleted`/`errors`/`ignored`/`coerced_nonfinite`/
  `total_input`/`unaccounted`/`error_samples`). An **empty** micro-batch (no rows) skips the write
  and additionally sets `empty: True`; that flag is present only on skipped batches, so read it with
  `result.get("empty")`, not `result["empty"]`.

#### A failed micro-batch fails the stream (`on_error`, default `"raise"`)

Structured Streaming commits a micro-batch's checkpoint offset when `foreachBatch` **returns
normally**, with no visibility into what happened to the documents inside. So a helper that returned
after a failed write would advance the offset past the rejected rows and they would never be
retried: silent, permanent loss behind a green job.

`make_foreach_batch` therefore **raises `EsWriteError` by default**, failing the batch so Spark
retries it and the checkpoint holds. With `id_field` set, that retry is an idempotent upsert rather
than a duplicate.

```python
fb = make_foreach_batch(cfg)                      # raise, checkpoint holds
fb = make_foreach_batch(cfg, on_error="log")      # warn, checkpoint ADVANCES (rows not retried)
fb = make_foreach_batch(cfg, on_error="ignore")   # same, silently
```

A write counts as failed when `errors > 0` (ES rejected documents) or `unaccounted > 0` (rows lost
below the per-document level). Delete-404 no-ops (`ignored`) are not failures, so a CDF replay that
deletes already-absent docs does not trip it. `on_batch` runs *before* the policy, so a metrics or
dead-letter hook still sees the failing batch.

> **Triage note.** `ConnectionError`, `ConnectionTimeout` and `SerializationError` are not `ApiError`
> subclasses, so the connector's `raise_on_exception=False` does not suppress them: a network failure
> propagates, fails the Spark task, and is retried by Spark. Only per-document rejections are returned
> as counts, which is what the `on_error` policy acts on.

### Deletes

By default the connector only inserts and updates (replace-by-`_id`). To also propagate
**deletes**, set `has_deletes=True` and name a `delete_flag_column`: rows whose flag is truthy
are sent to ES as a delete-by-`_id`; all other rows index as usual.

```python
cfg = EsWriteConfig(
    hosts=..., api_key=..., index="my-index",
    id_field="doc_id",              # required for deletes: you delete by _id
    has_deletes=True,
    delete_flag_column="_is_delete",  # truthy row => delete that _id from ES
)
result = bulk_write(prepared_df, cfg)   # -> {"written": N, "deleted": D, "errors": M, "total_input": T, "error_samples": [...]}
```

The connector is schema-agnostic about *how* you decide the latest state of a row: it deletes
exactly the rows you flag and indexes the rest. A common source is a Delta **Change Data Feed**:
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
`event_ts` is older than a doc already in ES, but committed in a later batch, can overwrite the
newer doc. If your source can deliver out-of-order events across batches, make `event_ts` globally
authoritative with ES external versioning (`version` = `event_ts` epoch-millis,
`version_type=external_gte`) so ES itself rejects stale writes regardless of batch order.

### The write result (`bulk_write` return value)

`bulk_write` returns a dict summarizing the write:

```python
{
  "written": 998,          # index/upsert ops that succeeded
  "deleted": 0,            # successful delete-by-id ops (non-zero only with has_deletes)
  "errors": 2,             # docs Elasticsearch rejected (exact count)
  "ignored": 0,            # delete-404 no-ops (deleting an already-absent doc: expected)
  "coerced_nonfinite": 0,  # inf/-inf/NaN values that had to become JSON null to be sent
  "total_input": 1000,     # rows handed to the writer
  "unaccounted": 0,        # rows that produced NO per-document outcome (loss below that level)
  "overcounted": 0,        # more outcomes than input rows: impossible, so a bug in THIS library
  "error_samples": [       # bounded diagnostics for rejected docs (up to 20)
    {"_id": "abc", "op_type": "index", "status": 400,
     "reason": "failed to parse field [ts] of type [date]"},
  ],
}
```

- **`unaccounted` is the reconciliation check, pre-computed.** Every input row yields exactly one of
  `written`/`deleted`/`errors`/`ignored`, so any shortfall means rows vanished below the per-document
  level (e.g. a chunk-level serialization/transport error) where the `errors` count structurally
  cannot see them. `ignored` is part of the identity precisely so an expected delete-404 no-op does
  not masquerade as loss. `bulk_write(df, cfg, raise_on_error=True)` (or `reconcile_or_raise(result)`)
  turns both signals into an `EsWriteError`; the streaming path does this by default.
- **`overcounted` should always be 0.** It counts the reverse discrepancy, more per-document outcomes
  than input rows, which is impossible by construction and so indicates a counting bug in this
  library rather than anything wrong with your data. It is reported and logged but does **not** fail
  the write, because failing a healthy write over our defect would wedge a streaming pipeline in a
  retry loop on a batch that can never pass. It is a separate field rather than a negative
  `unaccounted` so that it can never net against real loss: the two are accumulated per partition, so
  one partition over-counting cannot cancel out another partition genuinely losing rows. If you ever
  see a non-zero value, please report the result dict.
- **`coerced_nonfinite` catches invisible nulls.** `inf`/`-inf`/`NaN` have no JSON representation
  (ES rejects the bare `Infinity`/`NaN` tokens), so they must become JSON null to be sent at all.
  That is the right behavior, but it means an upstream divide-by-zero lands in ES as a null with no
  error and no error sample. A non-zero count here says real numbers became nulls.
- **`error_samples` is a breadcrumb, not a dead-letter queue.** It retains up to the first 20
  failures (id, op, HTTP status, ES reason) so a failed write is diagnosable instead of an opaque
  count. The `errors` count is always exact; only the retained sample list is capped, so a batch
  that fails wholesale can't exhaust memory. If you need every failed row durably, capture them
  from your own pipeline, the connector does not persist them.
- **Duplicate `id_field` values collapse, and reconciliation won't flag it.** If `id_field` is
  set and two input rows share the same id, the deterministic `_id` makes the later row **upsert
  over** the earlier one, so ES ends up with fewer documents than rows you sent. Every op reports
  success (`written` counts each op, not each surviving doc), so `written + deleted + errors` still
  equals `total_input`: the reconciliation identity passes even though the ES doc count is lower.
  This is the same idempotency that makes replays safe, but if you expect a 1:1 row→document
  mapping, ensure `id_field` is unique across the input (e.g. dedup upstream) or leave it unset to
  let ES assign random ids.

## Reading from Elasticsearch

### `read_index`

Read an Elasticsearch index back into a Spark DataFrame. **You must declare the target Spark
schema**: the reader does not infer it (see [Read fidelity](#read-fidelity-write--read) for why).

```python
from databricks_es_connector import EsReadConfig, read_index
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType

schema = StructType([
    StructField("doc_id", StringType()),
    StructField("event_ts", TimestampType()),   # stored as epoch-millis, read back as timestamp
    StructField("n", LongType()),
])

cfg = EsReadConfig(
    hosts="https://es-host:9200",
    api_key=dbutils.secrets.get("es", "api_key"),
    index="my-index",
    id_field="doc_id",               # filled from the ES _id if absent from _source
    query={"term": {"region.keyword": "us"}},  # optional raw ES query DSL; omit for match_all (.keyword: see below)
    pit_keep_alive="5m",             # sliding window: covers the gap between PIT touches, not the whole job (see note)
)
df = read_index(spark, cfg, schema)  # distributed: one Spark task per shard/slice
```

> **`term` gotcha:** dynamically-mapped string fields are `text` (analyzed), so a `term` query on
> the bare field usually returns nothing. Query the exact `.keyword` sub-field
> (`{"term": {"region.keyword": "us"}}`) or use `match`. Standard Elasticsearch behavior; check with
> `GET /<index>/_mapping/field/<field>`.

**`read_index` is distributed** and serverless-safe: it opens one Elasticsearch Point-in-Time for a
consistent snapshot, then fans out `spark.range(num_slices).mapInPandas(...)`, one task per PIT
slice (defaulting to the index's shard count), each task paging its slice with `search_after`. The
data stays across executors (never collected to the driver), so it scales for full-index export.

- **PIT lifetime is a sliding window, not a total budget.** The returned DataFrame is **lazy**:
  Spark reads each slice when an action runs, not when `read_index` returns. Every page request
  re-sends `pit_keep_alive` and resets the Point-in-Time's expiry, so `pit_keep_alive` only has to
  cover the longest gap *between* consecutive reads of the PIT, not the whole job's wall-clock: the
  open-PIT-to-first-page gap (the read is lazy, so Spark may schedule the tasks late, plus
  serverless executor cold-start) and any gap between pages if a slow consumer applies backpressure.
  The `5m` default covers a normal scheduling gap; raise it only if one of those gaps runs longer.
  (The PIT can't be closed on the driver without killing the still-lazy read, so it's left to expire
  on its own once reads stop touching it.)
- **For small/bounded reads** (lookups, reference data), pass `num_slices=1` for a single unsliced
  reader; the returned DataFrame is still lazy and distributed, just one task instead of a fan-out.

### Read fidelity (write → read)

Reads honor the **same contract as writes**: `read_index(cfg, df.schema)` after `bulk_write(df,
cfg)` reproduces the original DataFrame, **except** the deltas the
[Datatype coverage](#datatype-coverage-write-transforms--read-inverse) table documents as one-way (decimal precision beyond
~15-17 sig figs, sub-millisecond timestamp truncation, float32 widening). Each read coercion is the
exact inverse of the write transform:

| Declared Spark type | Stored in ES | Read back as |
|---|---|---|
| `timestamp` / `date` | epoch-millis integer | `datetime` / `date` (UTC) |
| `timestamp_ntz` | epoch-millis integer | naive `datetime` (no tzinfo, the UTC wall-clock) |
| `binary` | base64 string | `bytes` |
| `decimal(p,s)` | float | `Decimal` (precision already lost on write) |
| `variant` / `interval` | JSON / interval string | string (re-parse with `parse_json` yourself) |
| scalars / `struct` / `array` / `map` | same shape | same, recursively |

**Why the schema is required:** an epoch-millis integer in `_source` could be a `timestamp`, a
`date`, or a genuine `long`; a base64 string could be `binary` or a real `keyword`. The stored value
alone is ambiguous, so the reader must be told the intended type. There is no mapping inference:
declare the schema explicitly. (ES also has no array type: a field declared `array<T>` is read as a
list even if ES returned a scalar.)

**A declared type that doesn't fit the stored value raises `ReadSchemaMismatch`** rather than
coercing, because every coercion here yields plausible-looking wrong data that nothing downstream can
detect:

| Stored in ES | Declared | Result |
|---|---|---|
| `["prod", "urgent"]` | `string` | raises: declare `array<string>` |
| `{"k": 1}` | `string` | raises: declare a matching `struct` |
| `3.7` | `int` | raises: lossy |
| `3.0` | `int` | `3` (exact, allowed) |
| `10000000000` | `int` | raises: exceeds the declared width's range (see below) |
| `"maybe"` | `boolean` | raises: no defined meaning (see below) |
| any | `char` / `varchar` / `void` | raises: unsupported declared type (see below) |

**Booleans stored as strings are parsed, not trusted to `bool()`.** Reading an index the connector
did not write, a field may hold the *string* `"false"`, and `bool("false")` is `True`, so the value
would invert with nothing to notice it. Strings are matched against the same allow-list the write side
uses: `"true"/"t"/"1"/"yes"/"y"` and `"false"/"f"/"0"/"no"/"n"`, case-insensitive and
whitespace-trimmed. Anything else raises. Real JSON booleans and `0`/`1` are unaffected.

**`char`, `varchar` and `void` cannot be declared.** Spark cannot carry `CharType`, `VarcharType`, or
`NullType` through the `mapInPandas` the reader uses, so they raise `ReadSchemaMismatch` naming the
fix (`StringType`, or the column's real type). They fail fast at coercion rather than surfacing later
as Spark's `Invalid return type in mapInPandas`, which names neither the field nor the cause.

**An integer value that exceeds the declared width raises, it is not wrapped.** A stored value that
fits an integer type but overflows the declared width (e.g. `10000000000` read as `int`, or `128`
read as `byte`) raises `ReadSchemaMismatch` naming a wider type. Without this, the returned value
must still be cast to the declared type by `mapInPandas`, and that cast either raises a cause-less
`ArrowInvalid` or, on runtimes where `spark.sql.execution.pandas.convertToArrowArraySafely` defaults
`false` (Spark 3.5 / DBR 15.x and earlier), **silently wraps** it (`10000000000` → `1410065408`). A
pure `read_index(cfg, df.schema)` round-trip never hits this (the declared width equals the width
written), so it only arises reading an index the connector did not write, or one whose values
outgrew an under-sized declared type. Declare `short`/`int`/`long`, or a `decimal`/`string` wide
enough to hold the value.

Watch the multi-value case: **ES has no array type**, so any field can hold multiple values under the
same mapping and `GET /_mapping` will not tell you which do. Declare `array<T>` wherever
multi-values are possible; a single stored scalar still reads back as a one-element list. Only lossy
numeric conversions are rejected, since JSON has one number type and an integral float arriving for an
`int` column is normal.

> **Dynamic-mapping gotcha:** if you don't pre-create an index mapping, ES infers each field's type
> from the **first** document it sees and keeps it. A later doc whose value doesn't fit is silently
> coerced for indexing (a `float` `1.5` into a field first seen as an integer indexes as `1`; a
> keyword string past `ignore_above`, default 256 chars, isn't indexed). This does **not** affect the
> connector round-trip: reads come from `_source`, which ES stores verbatim, so `read_index` returns
> your original value regardless. It only bites when you **query ES directly** (aggregations, sort,
> `term`), where you see the coerced/truncated *indexed* value, not `_source`. For heterogeneous or
> long-string fields, pre-create an explicit mapping. (Same root cause as the `term`/`.keyword`
> gotcha above.)

> **A `read_index` round-trip cannot verify a mapping.** Follows from the note above: reads come from
> `_source`, which is a verbatim copy of what you sent, so a write-then-read-back comparison compares
> your data against a copy of itself and **cannot fail** on a mapping mismatch. With `amount`
> dynamically mapped `long`, writing `100.75` gives `errors=0` and round-trips as `100.75` while the
> indexed value is `100` and `range: {amount: {gt: 100.5}}` finds nothing. To verify a mapping, read
> the **indexed** value: a search with `"fields": [...]`, an aggregation, or `GET /<index>/_mapping`
> compared against the schema you intended.

## Configuration

**You always create one of two config objects**, depending on the direction:

- **`EsWriteConfig`**: for `bulk_write` / `make_foreach_batch`.
- **`EsReadConfig`**: for `read_index`.

Both are frozen, serializable dataclasses (in `src/databricks_es_connector/config.py`) so they can
be shipped to Spark executors: the Elasticsearch client is built *inside* each partition, never on
the driver. Each one takes the same **connection fields** (hosts, auth, TLS, client tuning) plus the
fields specific to its direction. So a config is `connection + write behavior` or
`connection + read behavior`; the three tables below split exactly along that line.

> **`EsConfig` is an alias for `EsWriteConfig`**, so code using the older name keeps working. New
> code should use `EsWriteConfig`.

The package also exports the pure value transforms for direct use or testing: `coerce_value` and
`to_es_source` (Spark value → ES `_source`), `read_coerce` (the declared-type inverse), and
`sanitize_for_arrow` (Spark-side, called automatically by `bulk_write`). Plus `reconcile_or_raise` and
the exceptions `EsWriteError`, `AmbiguousDeleteFlag`, `ReadSchemaMismatch`.

### Connection (shared by both configs)

These fields are defined on the `EsConnection` base and accepted by both `EsWriteConfig` and
`EsReadConfig`.

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `hosts` | `str` | - | **Yes** | e.g. `"https://host:9200"`. Empty string raises `ValueError`. |
| `api_key` | `str \| None` | `None` | one-of\* | Preferred auth. Base64 `id:key` or an encoded key. |
| `basic_auth` | `tuple \| None` | `None` | one-of\* | `("user", "pass")`. Sandbox/dev only; prefer `api_key` in prod. |
| `verify_certs` | `bool` | `True` | No | Set `False` for self-signed sandbox boxes. Library default is secure. |
| `ca_certs` | `str \| None` | `None` | No | Path to a CA bundle when pinning. Mutually exclusive with `verify_certs=False`. |
| `http_compress` | `bool` | `True` | No | gzip both directions: compresses the request body on write and asks ES to gzip the response on read (`content-encoding` + `accept-encoding`). The egress-cost lever. |
| `request_timeout` | `int` | `60` | No | Per-request timeout (seconds). |
| `transport_max_retries` | `int` | `3` | No | Retries for a whole **HTTP request**: a connection reset, a load-balancer 503, a gateway timeout, or a 429 on the bulk call itself. Re-sends the entire request. Does **not** retry an individual rejected document, see [Retries on a write](#retries-on-a-write-two-layers). |
| `max_retries` | `int` | - | No | Shorthand that sets **both** retry levels at once: `transport_max_retries` and [`max_retries_per_doc`](#write-behavior-eswriteconfig). Cannot be combined with either of those (raises). Reading `cfg.max_retries` back gives the shared number, or `None` if the levels were set to different values. |
| `retry_on_timeout` | `bool` | `True` | No | Whether a **timed-out** request is retried. Narrower than it sounds: connection errors and the `retry_on_status` codes are retried regardless, so setting this `False` only stops retries on timeouts. It is separate because a timeout is ambiguous, the write may have been applied and only the response lost, so replaying it is a judgment call. To disable transport retries entirely, use `transport_max_retries=0`. |

\* **Auth is required**: you must set exactly one of `api_key` or `basic_auth`, or the
constructor raises `ValueError`. Setting `ca_certs` together with `verify_certs=False` also
raises (pick one).

**How many retry levels apply depends on the direction.** A **read** has only the transport level
above, so `transport_max_retries` and `max_retries` are equivalent there. A **write** has a second,
per-document level, and `max_retries=N` sets both at once, see
[Retries on a write](#retries-on-a-write-two-layers).

### Write behavior (`EsWriteConfig`)

`EsWriteConfig` = the connection fields above **plus** these write-specific fields.

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `index` | `str` | `""` | **Yes** | Target index. `bulk_write`/`build_action` raise if empty. |
| `id_field` | `str \| None` | `None` | No | Column used as the deterministic `_id` → idempotent upserts. If unset, ES assigns random IDs (replays duplicate). If set, the column must **exist** (`bulk_write` raises before writing) and be non-null in every row. |
| `chunk_size` | `int` | `500` | No | How many documents the connector groups into each `_bulk` HTTP request (the `N` in [Retries on a write](#retries-on-a-write-two-layers)). A chunk is also flushed early if it first reaches 100 MB (elasticsearch-py's `max_chunk_bytes` default, which matches Elasticsearch's own `http.max_content_length` request ceiling). Larger values mean fewer, bigger requests (higher throughput but more executor memory per request); smaller values mean more, smaller requests. `500` suits most workloads; lower it for very large documents. |
| `write_concurrency` | `int` | `1` | No | How many `_bulk` request streams run **in parallel within a single partition**. `1` (default) is one serial stream per partition, so write concurrency across the cluster is just the partition count. Raise it when the write is **latency-bound** — executors idle waiting on each bulk's ES round-trip, with CPU *and* network both under-utilized — rather than CPU- or bandwidth-bound; each partition then keeps `write_concurrency` requests in flight to fill that wait. Every stream is a full `streaming_bulk` with the same `chunk_size` and per-document retry, so error accounting is unchanged. Total requests hitting ES at once = (running partitions) × `write_concurrency`; raise it gradually and watch for 429s (if they climb, the ES cluster is the ceiling, not the client). Costs one executor thread and up to one chunk of in-flight docs per unit. |
| `max_retries_per_doc` | `int` | `3` | No | Retries for an individual document ES rejected with a retryable status, with exponential backoff (only the failed subset is re-sent). `elasticsearch-py`'s own default is **0**; this is the knob that actually covers a 429, since the connection-level `max_retries` cannot see it. |
| `retry_on_doc_status` | `tuple` | `(429,)` | No | Which per-document statuses to retry. `429` is ES's write queue being full, the one reliably transient case. Adding `503` (a shard briefly unavailable, e.g. during relocation) is the main sensible extension: `retry_on_doc_status=(429, 503)`. Do **not** add deterministic statuses like `400` (malformed doc) or `409` (version conflict): the retry fails identically and only delays the real error. Empty with a non-zero `max_retries_per_doc` raises. |
| `require_existing_index` | `bool` | `True` | No | Verify the index exists before writing. ES auto-creates a missing index, so a **typo'd index name** otherwise produces a brand-new dynamically-mapped index and a perfect-looking `written` count. One `indices.exists` call on the driver. Set `False` to allow auto-creation (e.g. with an index template). |

#### Retries on a write: two layers

A bulk request sends N documents in one HTTP call, and ES answers with a **single HTTP 200** plus a
per-item status in the response *body*. "The request succeeded" and "the documents succeeded" are
therefore different questions, so a write has two retry layers:

| | [`transport_max_retries`](#connection-shared-by-both-configs) | [`max_retries_per_doc`](#write-behavior-eswriteconfig) |
|---|---|---|
| Defined on | `EsConnection` | `EsWriteConfig` |
| Default | `3` | `3` |
| Retries on | `429, 502, 503, 504` (whole request) | `429` (per item, via `retry_on_doc_status`) |
| Retry unit | The entire request, all N documents | Only the failed subset |
| Catches | ES unreachable, LB 503, gateway timeout, connection reset | A document ES refused because its write queue is full |

A **429 means different things at each layer**: "the cluster rejected your request" versus "the
cluster rejected this document". `transport_max_retries` cannot see the second one, because the
per-item statuses live in a response body the transport layer never reads. That is why
`max_retries_per_doc` exists, and why leaving it at `0` while raising `transport_max_retries` warns:
rejected documents would get no retries at all.

**Setting them.** `max_retries=N` is shorthand for both levels at once:

```python
EsWriteConfig(..., max_retries=5)                                  # 5 at both levels
EsWriteConfig(..., transport_max_retries=5, max_retries_per_doc=2)  # or set each
```

Use one style or the other; combining `max_retries` with either individual field raises, so the
effective value is never in doubt.

**Keep `max_retries_per_doc` low.** Each per-document retry sleeps in the executor with exponential
backoff (2s, doubling, capped at 600s), so `8` can stall a partition for ~8.5 minutes. The transport
layer has no such cost, so if you want many transport retries, set the two levels separately rather
than raising `max_retries`.

**Doc shaping**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `drop_fields` | `tuple` | `()` | No | Columns the client chooses to **skip** to shrink payload (egress opt-out), e.g. `("raw_data", "unmapped")`. |
| `strict_drop_fields` | `bool` | `True` | No | Validate every `drop_fields` name against the DataFrame's columns and raise on an unknown one. A misspelled entry prunes **nothing**, shipping the field to ES while the caller believes it was withheld: since `drop_fields` is commonly a PII/egress control, it must fail **closed**. Set `False` only when deliberately reusing one config across DataFrames with differing columns. |

**Deletes**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `has_deletes` | `bool` | `False` | No | `False` = every row is an index/upsert. `True` routes rows whose `delete_flag_column` is truthy to an ES delete-by-`_id`. |
| `delete_flag_column` | `str \| None` | `None` | when `has_deletes` | Boolean-ish column; a truthy value deletes that `_id`. A null value = not a delete. The column is pruned from the `_source` of kept rows so it is never indexed. It must **exist** in the DataFrame: `bulk_write` raises if it does not (see below). |

> **A `delete_flag_column` that isn't in the DataFrame fails the write.** If the name is misspelled,
> every row reads as not-flagged, so each intended delete would be applied as an **upsert** and the
> documents would stay in Elasticsearch, reporting `deleted: 0`, `errors: 0` and a clean
> reconciliation. `bulk_write` therefore checks the column exists before writing anything. This is not
> optional: `has_deletes=True` has no meaning without a real flag column. (`id_field` and, under
> `strict_drop_fields`, `drop_fields` are validated in the same pass, for the same reason: a config
> field that names a column must fail closed when the name is wrong.)

> **Prefer a real boolean for the delete flag.** Strings are parsed against an allow-list in both
> directions, case-insensitive and whitespace-trimmed: `"true"/"t"/"1"/"yes"/"y"` delete,
> `"false"/"f"/"0"/"no"/"n"` and empty do not. Anything else raises `AmbiguousDeleteFlag` rather than
> defaulting, since a value like `"on"` or `"2"` reads as truthy to a human but has no defined meaning
> here. `.cast("boolean")` in Spark avoids the question entirely.

When `has_deletes=True` the constructor requires both `id_field` (you cannot delete without an
`_id`) and `delete_flag_column`, and raises `ValueError` otherwise. Setting `delete_flag_column`
while `has_deletes=False` also raises: a flag column that does nothing is a misconfiguration, not
a silent no-op.

Delete idempotency: deleting a doc that isn't in ES returns a 404, which the connector treats as
an expected no-op (not an error), so checkpoint replays, filtered rows, and re-processed deletes
stay clean. This suppression is scoped strictly to `delete` + `404`; a 404 on an index, or any
other status on a delete, is still counted in `errors`.

See [Deletes](#deletes) above for the recommended Change-Data-Feed pattern and the
cross-batch ordering caveat.

**Serialization mode**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `serialize_in_spark` | `bool` | `False` | No | Where each document's JSON is built. `False` (default) is the per-row Python path (`coerce_value` → `build_action` → `streaming_bulk` serializes on the executor). `True` builds the whole `_bulk` action line in **Spark** (`to_json`, in the JVM) and ships the pre-built NDJSON without re-serializing in Python — much faster on wide/large writes, with a slightly different fidelity contract (below). Opt-in so the default path's round-trip guarantee is untouched. Index/upsert only: combining it with `has_deletes` raises. |

#### Spark-native serialization (`serialize_in_spark=True`)

The default write path shapes and JSON-serializes every row **in Python** on the executor. That
per-row work is GIL-bound and is the throughput ceiling on wide or very large writes: the
transform-and-serialize step, not the Elasticsearch send, dominates wall time. `serialize_in_spark=True`
moves it into Spark — the entire `_bulk` action line (`{"index":{…}}\n{…_source…}`) is built once per
row with `to_json` in Catalyst (multi-core JVM, no GIL), and the executor only forwards the pre-built
NDJSON to `es.bulk(operations=…)`, which sends `str`/`bytes` lines verbatim (no Python re-encode).
Measured ≈5× faster on the transform+serialize segment (30M rows, 32 cores).

It is **opt-in** because `to_json` is a different serializer from `coerce_value`, so a few edge cases
differ from the default path's documented round-trip contract:

| Case | Default path (`coerce_value`) | `serialize_in_spark` (`to_json`) |
|---|---|---|
| `NaN` / `±inf` | JSON `null`, **counted** in `coerced_nonfinite` | JSON `null` (nulled in Spark at **any nesting depth**), but **not counted** (`coerced_nonfinite` is always `0`) |
| `decimal` | → `double` (precision lost past ~15–17 sig figs) | rendered at **full precision** (more faithful) |
| `float` (32-bit) | exact widened double (`0.10000000149…`) | short decimal repr (`0.1`) |
| null `id_field` value | whole partition **raises** | **fails closed**: the row's action line is emitted as null, so it surfaces as `unaccounted` and `reconcile_or_raise` fails the write (rather than shipping `"_id": null`, which could auto-assign an id and duplicate on replay) |
| everything else (nested struct/array/map, `binary`→base64, `timestamp`→epoch-millis, kept null fields) | — | **matches** |

Everything else is unchanged: `chunk_size`, per-document `429` retry (`max_retries_per_doc` /
`retry_on_doc_status`), the `written`/`deleted`/`errors`/`ignored`/`unaccounted` accounting, and
`reconcile_or_raise` all behave exactly as on the default path. Use the default path when you need the
full round-trip fidelity guarantee or delete routing; use `serialize_in_spark=True` for throughput on
large index/upsert exports where the differences above are acceptable.

### Read behavior (`EsReadConfig`)

`EsReadConfig` = the connection fields above **plus** these read-specific fields.

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| `index` | `str` | `""` | **Yes** | Source index. `read_index` raises if empty. |
| `id_field` | `str \| None` | `None` | No | If the declared schema names this column, it is filled from the ES `_id` when absent from `_source`. |
| `query` | `dict \| None` | `None` | No | Raw ES query DSL to filter the read. `None` = `match_all`. v1 accepts a raw DSL dict only, no Spark-predicate pushdown. (For `term` on string fields, note the `.keyword` gotcha in the Reading section.) |
| `num_slices` | `int \| None` | `None` | No | Parallelism for the distributed read. `None` defaults to the index's shard count. One slice per Spark task. |
| `strict_slices` | `bool` | `True` | No | When `num_slices` is `None` and the shard-count lookup fails (permissions, an alias spanning indices, a transient error), raise instead of degrading. The only correct fallback is a **single serial reader**: still complete, but it forfeits all parallelism, and a `logging` warning is easy to miss entirely on serverless, so a large export just looks mysteriously slow. Pass `num_slices` explicitly to skip the lookup, or set `False` to accept the serial read. |
| `batch_size` | `int` | `1000` | No | Docs per scroll/PIT page. Must be positive. |
| `pit_keep_alive` | `str` | `"5m"` | No | Point-in-Time lifetime, as a **sliding window**: each page re-sends it and resets the PIT's expiry, so it need only cover the longest gap between consecutive reads of the PIT (for `read_index`, the lazy open-to-first-page scheduling gap), not the whole job. See the Reading section. |
| `include_id` | `bool` | `True` | No | Expose the ES `_id` via the `id_field` column when the schema declares it. |

## Datatype coverage (write transforms + read inverse)

This table is the **write** transform (Spark → ES `_source`); the
[Read fidelity](#read-fidelity-write--read) table above is its inverse (ES `_source` → Spark) and
lists which of these transforms are one-way. Both directions share this one contract.

Every valid Spark column can be exported with **no caller pre-processing**: hand `bulk_write` any
DataFrame. Values are transformed on the way to Elasticsearch as follows. These transforms are
**by design**: they are the deltas to expect between the Spark row and the ES `_source`, not bugs.

| Spark type | ES `_source` value | Example (input → stored) |
|---|---|---|
| `string`, `boolean` | unchanged | `"hi"` → `"hi"`; `true` → `true` |
| `byte`/`short`/`int`/`long` | unchanged (all integer widths become one JSON number; width not preserved) | `5` → `5` |
| `double` | unchanged; non-finite values (`Infinity`/`-Infinity`/`NaN`) become `null` (no JSON representation) | `1.5` → `1.5`; `Infinity` → `null` |
| `float` (32-bit) | its **exact 32-bit value widened to double**, the stored number shows the float32 rounding, not the literal you typed (see note) | `0.1` (float) → `0.10000000149011612` |
| `decimal(p,s)` | **float** (precision lost beyond ~15-17 sig figs) | `Decimal("1.50")` → `1.5` |
| `date` / `timestamp` | **epoch milliseconds** (integer), floored to the millisecond (sub-ms precision dropped). The `timestamp` epoch is the true UTC instant, independent of `spark.sql.session.timeZone` (see below) | `2021-01-01T00:00:00Z` → `1609459200000` |
| `timestamp_ntz` | **epoch milliseconds** of the wall-clock read as UTC (see below) | `2021-06-01 12:00:00` → `1622548800000` |
| `binary` | **base64 string** | `b"\x01\x02"` → `"AQI="` |
| `struct` / `map` | nested object (recursed). Non-string `map` keys are rendered to strings (JSON keys must be strings) using the same transform as the value type. **A `decimal` key inherits the decimal precision loss, and for a key that means colliding keys drop entries (see note)** | `{a: 1}` → `{"a": 1}`; `map<int,_>` `{1: "x"}` → `{"1": "x"}` |
| `array` | array (recursed) | `[1, 2]` → `[1, 2]` |
| `null` (any type) | present as JSON `null` (the field is kept, its value is `null`) | `None` → `null` |
| `variant` | **string containing serialized JSON** (see below) | `{"k": 1}` → `"{\"k\":1}"` |
| `interval` | **string** (see below) | `INTERVAL '1 02:03:04' DAY TO SECOND` → `"INTERVAL '1 02:03:04' DAY TO SECOND"` |

**Why `float` looks "changed":** a Spark `FLOAT` is 32-bit and can't represent `0.1` exactly, it
holds the nearest float32, whose true value is `0.10000000149011612`. The connector stores that
exact value (widened to a 64-bit double) rather than reformatting it back to `0.1`, so the stored
number is faithful to what Spark actually held, not to the source literal. Use `DOUBLE` if you need
`0.1` to store as `0.1`.

**`decimal` map keys lose entries, not just digits:** a `map<decimal(p,s), V>` key goes through the
same `decimal` → float conversion as a decimal *value*, so it inherits the same ~15-17 significant
figure limit. For a **key** that limit is destructive rather than merely lossy: two distinct keys that
differ only beyond it render to the same string, and the later entry **overwrites** the earlier one,
so the map comes back with fewer entries than it went in with. The row still produces exactly one
document, so the write reports success and reconciliation stays clean, there is no error and no
count. For example, `decimal(38,0)` keys `10000000000000000000000000000000000001` and
`...002` both render `"1e+37"`, and `Decimal("1.000000000000000001")` and `Decimal("1.000000000000000002")`
both render `"1.0"`. Only `decimal` keys are affected: an `int`/`bigint` key never passes through a
float (Python integers are arbitrary precision), and `date`/`binary`/`string` keys render losslessly.
**Use a `string` or integer key type if your keys need more than ~15-17 significant figures.**

**Timestamp precision:** epoch-millis is floored to the containing millisecond, consistently for
pre- and post-epoch instants (matches Spark/Java `unix_millis`). Elasticsearch `date` is
millisecond-resolution by default; sub-millisecond (microsecond) precision from a Spark `timestamp`
is not preserved. Map the field as `date_nanos` and send nanos yourself if you need finer than ms.

**`timestamp` vs `timestamp_ntz`:** a regular Spark `timestamp` is an instant and converts to its
true epoch-millis. A `timestamp_ntz` (no time zone) carries a wall-clock value with no zone, so the
connector interprets it as **UTC**, deterministically, since executors can be in different zones.
The stored epoch-millis is that wall-clock time read as UTC, not as the cluster's local zone. If
your NTZ values are really in some other zone, convert them to a zoned `timestamp` in Spark before
export so the epoch is correct.

**Timezone independence (`timestamp`):** the epoch a `timestamp` stores is its **true UTC instant,
regardless of `spark.sql.session.timeZone`**: set the session to `America/New_York`, `Asia/Kolkata`,
or anything else and the same instant produces the same epoch. The connector guarantees this by
converting every `timestamp` column (at any nesting depth: inside `struct`/`array`/`map` too) to its
epoch-millis in Spark via `unix_millis` **before** the export, rather than letting Spark's Arrow
conversion render it to the session-local wall-clock. This matters because a naive
(session-local) render would otherwise be stored as if it were UTC, silently shifting the stored
instant by the session's offset. No caller action is needed, and the session is never mutated. Only
`timestamp` is converted this way; `timestamp_ntz` (defined as UTC above) and `date` (no
time-of-day) are already session-independent and are left untouched.

### Arrow-hostile types: `variant` and `interval` (handled automatically)

`VARIANT` and `INTERVAL` have no Apache Arrow representation, so they cannot cross into
`mapInPandas` directly. `bulk_write` handles this for you: any column whose type **contains** one
of these at **any nesting depth** is serialized to a **string** before export, so no caller action
is required; hand `bulk_write` the raw DataFrame. The string form depends on the type:

- **`variant`** (at any depth, e.g. `variant`, `struct<...,v:variant>`, `array<struct<...,v:variant>>`)
  → a **JSON string**. Example: a `variant` holding `{"k": 1, "nested": [2, 3]}` is stored as the
  string `"{\"k\":1,\"nested\":[2,3]}"`.
- **`interval`** (day-time or year-month) → its **Spark string form**. Example:
  `INTERVAL '1 02:03:04' DAY TO SECOND` is stored as the string `"INTERVAL '1 02:03:04' DAY TO SECOND"`.

Round-trip consequence to expect: such a column lands in ES as a **string**, not a queryable nested
object, so map it as `keyword`/`text`, not `object`.

When adding fields, add a matching entry to the ES index mapping or ES will dynamic-map (and guess)
the type.

## Developing the library

Work on the library locally (on your machine, not Databricks): set up an environment
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
dependency tree with pinned versions (direct and transitive) for security scanning
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

**Cutting a tagged release?** Follow [`RELEASING.md`](RELEASING.md): it enumerates the four required
steps (integration tests, rebuild wheel, run the release gates, attach the wheel to the tag). Two
hard gates run in step 3: `scripts/check_requirements_match.py` (requirements.txt matches the wheel's
resolved closure), `scripts/check_readme_sync.py` (every module / fixture / script is documented), and
`scripts/check_version_consistency.py` (every version/wheel reference agrees with `pyproject.toml`).

**When cutting a release, regenerate [`requirements.txt`](requirements.txt).** The build does
*not* read that file: the wheel declares only the abstract range `elasticsearch>=8,<9`, so a
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
through Delta Share: no code changes, only the `EsWriteConfig` / `EsReadConfig` endpoint/auth differ.

## Repo layout

```
src/databricks_es_connector/   # the library (this is what ships in the .whl)
  config.py                    #   EsConnection base + EsWriteConfig / EsReadConfig (EsConfig alias)
  transform.py                 #   pure-Python row shaping on WRITE (timestamps, numpy/arrays, decimal, binary, pruning)
  bulk.py                      #   executor-side mapInPandas bulk write (batch entry point)
  stream.py                    #   foreachBatch helper for Structured Streaming
  spark_prep.py                #   sanitize_for_arrow: Arrow-hostile types (VARIANT/INTERVAL) -> JSON string
  spark_serialize.py           #   serialize_in_spark: build the _bulk NDJSON in Spark (to_json), not per-row Python
  read_transform.py            #   pure-Python inverse coercion on READ (ES value + type -> Spark value)
  read.py                      #   read_index (distributed sliced-scroll PIT reader)
tests/                         # unit tests for the pure-Python layer (no Spark/ES needed)
integration_tests/             # live-Spark/ES tests run on Databricks serverless via dbx_test
  test_datatype_coverage.py    #   every Spark datatype + edge cases, round-tripped through ES
  test_bulk_write_roundtrip.py #   the bulk_write result contract (counts, total_input, error_samples)
  test_concurrency_roundtrip.py #  write_concurrency>1: threaded per-partition write, no loss/dupe/miscount
  test_deletes_roundtrip.py    #   has_deletes routing live: delete-by-id, delete-404 no-op
  test_preflight_column_guards.py # a config field naming a missing column fails before any write
  test_streaming_sink.py       #   make_foreach_batch on real Structured Streaming + restart idempotency
  test_read_roundtrip.py       #   write->read fidelity + distributed sliced read (multi-shard)
  test_timezone_utc.py         #   timestamp epoch is session-timezone-independent (UTC + non-UTC)
  test_dynamic_mapping_coercion.py #  ES dynamic-mapping coercion: _source faithful vs indexed value
  test_sanitize_for_arrow.py   #   the Spark-side VARIANT/INTERVAL serialization, no ES
  config/test_config.yml       #   dbx_test config (profile, wheel, serverless env)
scripts/                       # release + maintenance gates (not shipped in the .whl)
  check_requirements_match.py  #   requirements.txt == the wheel's resolved dependency closure
  check_readme_sync.py         #   every module / fixture / script is documented in the README(s)
  check_version_consistency.py #   __init__, HANDOFF header + every wheel reference match pyproject
.agents/skills/es-connector/   # maintainer skill: fidelity model, datatype contract, ES gotchas, release
RELEASING.md                   # the release checklist (integration tests, wheel, gates, tag)
```

Two test tiers: `tests/` is the fast, infra-free gate (pure-Python, run with `pytest`);
`integration_tests/` runs on a live serverless workspace + Elasticsearch to cover the Spark-side
code the unit tests can't reach, see [`integration_tests/README.md`](integration_tests/README.md).

The built wheel contains only `databricks_es_connector`; neither `tests/` nor `integration_tests/`
is part of the customer artifact.

## License & Attribution

**Copyright © Databricks, Inc.** — Developed and maintained by Databricks Forward Deployed Engineering. Available to support customers and the broader community in moving data between Databricks and Elasticsearch. For production support and customization, contact your Databricks account team.

---

**Built with 💜 by Databricks Forward Deployed Engineering**
