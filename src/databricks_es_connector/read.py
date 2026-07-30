"""Read an Elasticsearch index into a Spark DataFrame.

Two entry points, both requiring an explicit Spark `StructType` (v0.4.0 does no mapping inference —
several write transforms are one-way and can't be inverted from `_source` without the declared type,
see READ_DESIGN.md). Both share the coercion oracle `read_transform.read_coerce`, so they return
identical values; only the transport differs.

  - `read_index`         — DISTRIBUTED (Option B). Opens a Point-in-Time on the driver, fans out
                           `spark.range(num_slices).mapInPandas(...)`, one task per PIT slice, each
                           task building its own ES client and paging its slice with `search_after`.
                           Serverless-safe (same mechanism as the write path); data stays distributed
                           (never collected to the driver). The returned DataFrame is lazy, so the
                           PIT is left to expire via `pit_keep_alive` (see read_index docstring). For
                           full-index export.
  - `read_index_collect` — DRIVER-SIDE (Option A). Pages the whole index through the driver and
                           `createDataFrame`s the result. No executors involved. For bounded reads
                           (lookups / reference data) and as the simple, proven fallback.

pyspark is imported lazily so the pure-Python layers stay importable without Spark.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .config import EsReadConfig
from .read_transform import read_coerce

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.types import DataType, StructType

_log = logging.getLogger(__name__)


# --- pure helpers (no Spark) -----------------------------------------------------------------

def _spark_type_token(dtype: "DataType") -> str:
    """Render a pyspark DataType to the lowercase token read_coerce understands.

    Spark's own `simpleString()` already produces exactly the DDL form we parse (`timestamp`,
    `decimal(10,2)`, `struct<name:type,...>`, `array<...>`, `map<...>`), so we delegate to it.
    Kept as a seam so the coercion layer never imports pyspark.
    """
    return dtype.simpleString()


def _schema_field_tokens(schema: "StructType"):
    """[(field_name, type_token), ...] for the top-level fields of the declared schema."""
    return [(f.name, _spark_type_token(f.dataType)) for f in schema.fields]


def _coerce_hit(source: dict, doc_id, field_tokens, id_field: Optional[str], include_id: bool):
    """Turn one ES hit (_source + _id) into a row dict matching the declared schema."""
    row = {}
    for name, token in field_tokens:
        if include_id and id_field is not None and name == id_field and name not in source:
            # The id lives in _source by default (writes keep it), but honor _id if _source omitted it.
            row[name] = read_coerce(doc_id, token)
        else:
            row[name] = read_coerce(source.get(name), token)
    return row


def _slice_hits(es, pit_id, query, batch_size, keep_alive, slice_spec=None):
    """Yield raw hits over an ALREADY-OPEN PIT, optionally restricted to one slice.

    Pages with `search_after` (no from/size deep-pagination limit). `slice_spec`, when given, is
    `{"id": i, "max": n}` — an ES sliced scroll, so N parallel readers each pull a disjoint 1/N of
    the index. Does NOT open or close the PIT (the caller owns its lifecycle), so it works both for
    the driver path (one local PIT) and the distributed path (one PIT shared across executors).
    """
    search_after = None
    while True:
        # Native 8.x keyword form. With a PIT the index is scoped by the pit itself (no `index=`).
        # `_shard_doc` is the cheapest deterministic total order for search_after, valid only with a
        # PIT, and required per-slice for stable paging.
        kwargs = dict(
            size=batch_size,
            query=query,
            pit={"id": pit_id, "keep_alive": keep_alive},
            sort=[{"_shard_doc": "asc"}],
        )
        if slice_spec is not None:
            kwargs["slice"] = slice_spec
        if search_after is not None:
            kwargs["search_after"] = search_after
        resp = es.search(**kwargs)
        hits = resp["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            yield h
        search_after = hits[-1]["sort"]
        pit_id = resp.get("pit_id", pit_id)   # PIT id can be refreshed across pages


def _resolve_num_slices(es, index: str, configured: Optional[int]) -> int:
    """How many slices to fan out into: the caller's `num_slices`, else the index's shard count.

    Sliced scroll partitions best at (a multiple of) the shard count; defaulting to it gives one
    slice per shard. Falls back to 1 (a single unsliced reader) if the shard count can't be read.
    """
    if configured is not None:
        return max(1, int(configured))
    try:
        settings = es.indices.get_settings(index=index, flat_settings=True)
        # {index_name: {"settings": {"index.number_of_shards": "N"}}}
        one = next(iter(settings.values()))
        shards = int(one["settings"]["index.number_of_shards"])
        return max(1, shards)
    except Exception as exc:
        # Couldn't read the shard count (permissions, alias spanning indices, transient error).
        # Falling back to a single unsliced reader is CORRECT but SERIAL — for a large index that
        # silently forfeits the parallelism this reader exists for, so warn rather than hide it.
        # Pass an explicit num_slices to skip this lookup entirely.
        _log.warning(
            "read_index: could not read shard count for index %r (%s); falling back to a single "
            "unsliced slice. Pass num_slices explicitly to parallelize.", index, exc)
        return 1


# --- shared validation -----------------------------------------------------------------------

def _validate(cfg: EsReadConfig, schema) -> None:
    # Duck-type the schema (a non-empty `.fields`) rather than isinstance(StructType) — this matches
    # how _schema_field_tokens already reads the schema, and it keeps this module importable and
    # unit-testable without pyspark. A wrong type has no truthy `.fields` and hits the error; a real
    # StructType passes.
    if not getattr(schema, "fields", None):
        raise ValueError("read_index requires a non-empty Spark StructType schema (v0.4.0 has no "
                         "mapping inference — declare the schema explicitly)")
    if not cfg.index:
        raise ValueError("EsReadConfig.index is required to read")


# --- driver-side reader (Option A) -----------------------------------------------------------

def read_index_collect(spark: "SparkSession", cfg: EsReadConfig, schema: "StructType") -> "DataFrame":
    """Driver-side read: page the whole index through the driver, coerce to `schema`, and
    `createDataFrame`. No executors. Use for bounded reads (lookups / reference data). For
    full-index export use `read_index` (distributed)."""
    _validate(cfg, schema)
    query = cfg.query or {"match_all": {}}
    field_tokens = _schema_field_tokens(schema)

    from elasticsearch import Elasticsearch
    es = Elasticsearch(**cfg.client_kwargs())
    pit_id = es.open_point_in_time(index=cfg.index, keep_alive=cfg.pit_keep_alive)["id"]
    try:
        rows = [
            _coerce_hit(h.get("_source", {}), h.get("_id"), field_tokens, cfg.id_field, cfg.include_id)
            for h in _slice_hits(es, pit_id, query, cfg.batch_size, cfg.pit_keep_alive)
        ]
    finally:
        try:
            es.close_point_in_time(id=pit_id)
        except Exception:
            pass
    return spark.createDataFrame(rows, schema)


# --- distributed reader (Option B) -----------------------------------------------------------

def _make_slice_reader(cfg: EsReadConfig, pit_id: str, query: dict, field_tokens, num_slices: int):
    """Return a mapInPandas function: for each slice id it receives (via spark.range), read that
    PIT slice and yield a pandas DataFrame of coerced rows. Captures only serializable values
    (frozen cfg, the PIT id string, the query dict, the type tokens) — no Spark/ES objects."""
    col_names = [name for name, _ in field_tokens]

    def _read(iterator):
        import pandas as pd
        from elasticsearch import Elasticsearch

        es = Elasticsearch(**cfg.client_kwargs())
        for pdf in iterator:
            for slice_id in pdf["id"].tolist():
                # A sliced scroll needs max > 1; with a single slice, read the PIT unsliced.
                slice_spec = {"id": int(slice_id), "max": num_slices} if num_slices > 1 else None
                rows = [
                    _coerce_hit(h.get("_source", {}), h.get("_id"),
                                field_tokens, cfg.id_field, cfg.include_id)
                    for h in _slice_hits(es, pit_id, query, cfg.batch_size, cfg.pit_keep_alive,
                                         slice_spec=slice_spec)
                ]
                if rows:
                    # Column order fixed to the schema; a slice with no rows yields nothing.
                    yield pd.DataFrame(rows, columns=col_names)

    return _read


def read_index(spark: "SparkSession", cfg: EsReadConfig, schema: "StructType") -> "DataFrame":
    """Distributed read of an ES index into a Spark DataFrame against the caller-declared `schema`.

    Opens one Point-in-Time on the driver for a consistent snapshot, then fans out
    `spark.range(num_slices).mapInPandas(...)` — one task per PIT slice, each task building its own
    ES client and paging its slice with `search_after`, coercing via the documented inverse
    transforms. Serverless-safe (same `mapInPandas` mechanism as the write path); the data stays
    distributed across executors (never collected to the driver), so it scales for full-index export.
    `num_slices` defaults to the index's shard count.

    PIT lifecycle: the returned DataFrame is LAZY — Spark reads each slice when an action runs, not
    when this function returns. So we do NOT close the PIT here (a `finally` close would kill it
    before evaluation); it is left to expire via `pit_keep_alive`. **Set `pit_keep_alive` long enough
    to cover the whole downstream job** (the snapshot must outlive the last action that reads it).
    This is a deliberate trade-off: keeping the read distributed means the PIT can't be
    driver-closed. For small reads where an explicit close is preferable, use `read_index_collect`.

    For small/bounded reads, `read_index_collect` (driver-side, no executors) is simpler.
    """
    _validate(cfg, schema)
    query = cfg.query or {"match_all": {}}
    field_tokens = _schema_field_tokens(schema)

    from elasticsearch import Elasticsearch
    es = Elasticsearch(**cfg.client_kwargs())
    num_slices = _resolve_num_slices(es, cfg.index, cfg.num_slices)
    pit_id = es.open_point_in_time(index=cfg.index, keep_alive=cfg.pit_keep_alive)["id"]

    reader = _make_slice_reader(cfg, pit_id, query, field_tokens, num_slices)
    # One row per slice, one partition per slice, so each task handles exactly one slice. `range`
    # splits its contiguous id space [0, num_slices) into `numPartitions` equal chunks — with
    # num_slices rows in num_slices partitions that is deterministically one row each, and needs no
    # shuffle. (repartition() would round-robin, which only *tends* toward an even spread and could
    # co-locate two slices on one task under skew.) The result is a lazy distributed DataFrame —
    # data never crosses the driver.
    slices_df = spark.range(num_slices, numPartitions=num_slices)
    return slices_df.mapInPandas(reader, schema)
