# databricks-es-connector

Serverless-safe bulk export from Databricks/Spark to Elasticsearch.

Schema-agnostic Python library: given a Spark DataFrame and an `EsConfig`, it writes
rows to an Elasticsearch index via the `elasticsearch-py` client, parallelized
across executors with `mapInPandas` (serverless-safe), with gzip request compression
and deterministic document IDs for idempotent upserts.

Built because the `elasticsearch-spark` connector cannot run on serverless compute
(no third-party Spark JARs), has no Spark 4 / DBR 17+ build, and offers no request
compression. The Python client has none of those limits.

> **Maturity: 0.1.0.** The mechanism is proven and every Spark datatype is exportable, but
> production hardening items (TLS/CA trust, API-key auth worked example, FIPS/FedRAMP, error
> dead-lettering, index templates/ILM) are not yet built. See [HANDOFF.md](HANDOFF.md) for the
> known limitations and pre-production checklist.

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
result = bulk_write(gold_df, cfg)   # -> {"written": N, "errors": M}
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

## Datatype coverage

Every Spark column can be exported and lands as usable data:

- **Handled automatically by `coerce_value`** (no caller action): all numerics (byte/short/int/
  long/float/double), `decimal` → float, `boolean`, `string`, `binary` → base64 string, `date`/
  `timestamp` → epoch-millis, `array`/`map`/`struct` (recursed), and null. Any unforeseen type
  falls back to its string form rather than failing the write.
- **Requires a one-line Spark-side cast first:** `INTERVAL` types cannot cross Arrow into
  `mapInPandas` at all (Spark raises `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION`). Call
  `cast_unsupported_to_string(df)` before `bulk_write` / the streaming source to cast those
  columns (top-level or nested) to string:

  ```python
  from databricks_es_connector import cast_unsupported_to_string, bulk_write
  bulk_write(cast_unsupported_to_string(gold_df), cfg)
  ```

Verified end-to-end (Spark → Arrow → bulk → ES) across every type above. Two fidelity notes:
`decimal` → float loses precision beyond ~15–17 significant figures (cast the column to string
in Spark if exactness matters); `binary` is stored base64-encoded. When adding fields, add a
matching entry to the ES index mapping or ES will dynamic-map (and guess) the type.

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
  spark_prep.py                #   cast_unsupported_to_string: Arrow-hostile types (intervals) -> string
tests/                         # unit tests for the pure-Python layer (no Spark/ES needed)
```

The built wheel contains only `databricks_es_connector`; `tests/` stays in the repo for
development but is never part of the customer artifact.

## License & Attribution

**Copyright © Databricks, Inc.** — Developed and maintained by Databricks Forward Deployed Engineering. Available to support customers and the broader community in connecting Databricks to Elasticsearch. For production support and customization, contact your Databricks account team.

---

**Built with 💜 by Databricks Forward Deployed Engineering**
