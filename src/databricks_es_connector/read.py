"""Read an Elasticsearch index into a Spark DataFrame.

One entry point, `read_index`, requiring an explicit Spark `StructType` (v0.4.0 does no mapping
inference: several write transforms are one-way and can't be inverted from `_source` without the
declared type; see the README "Reading from Elasticsearch" section for the ambiguity rationale). It
uses the coercion oracle `read_transform.read_coerce` to invert the write transforms per declared
type.

`read_index` is DISTRIBUTED: it opens a Point-in-Time on the driver, then fans out
`spark.range(num_slices).mapInPandas(...)`, one task per PIT slice, each task building its own ES
client and paging its slice with `search_after`. Serverless-safe (same mechanism as the write path);
data stays distributed (never collected to the driver). The returned DataFrame is lazy; the PIT is
self-extended by each page's `keep_alive` (see the read_index docstring).

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

def _spark_type_token(dtype: "DataType"):
    """Render a pyspark DataType to the token read_transform.read_coerce understands.

    A scalar type becomes its lowercase `simpleString()` string ("timestamp", "decimal(10,2)", ...).
    A container becomes a tuple carrying its already-walked sub-token(s) -- ("array", elem),
    ("map", key, val), ("struct", [(name, sub), ...]) -- so the coercion layer never re-parses a
    `struct<...>` DDL string; we hand it the structure the DataType already gives us here.

    Containers are recognized structurally (elementType / keyType+valueType / fields) rather than by
    isinstance, so this stays importable without pyspark and the read.py unit tests can pass simple
    duck-typed DataType fakes. Anything without those attributes is a scalar -> simpleString().
    """
    fields = getattr(dtype, "fields", None)
    if fields is not None:                                        # StructType
        return ("struct", [(f.name, _spark_type_token(f.dataType)) for f in fields])
    if getattr(dtype, "elementType", None) is not None:          # ArrayType
        return ("array", _spark_type_token(dtype.elementType))
    if getattr(dtype, "keyType", None) is not None:              # MapType
        return ("map", _spark_type_token(dtype.keyType), _spark_type_token(dtype.valueType))
    return dtype.simpleString()                                   # scalar


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
    `{"id": i, "max": n}`, an ES sliced scroll, so N parallel readers each pull a disjoint 1/N of
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


def _resolve_num_slices(es, index: str, configured: Optional[int],
                        strict: bool = True) -> int:
    """How many slices to fan out into: the caller's `num_slices`, else the index's shard count.

    Sliced scroll partitions best at (a multiple of) the shard count; defaulting to it gives one
    slice per shard.

    If the shard count cannot be read (permissions, an alias spanning indices, a transient error),
    the only correct fallback is 1: a single unsliced reader. That still returns every document, but
    it is SERIAL, forfeiting the parallelism this reader exists for. Warning about it via `logging`
    is not enough on Databricks serverless, where driver log output is easy to miss entirely, so a
    large export would just look mysteriously slow. With `strict=True` (the default) the failure is
    raised instead, naming the fix; pass an explicit `num_slices` to skip the lookup entirely, or
    `strict_slices=False` to accept the serial fallback.
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
        if strict:
            raise RuntimeError(
                f"read_index: could not read the shard count for index {index!r} ({exc}). Falling "
                "back to a single unsliced reader would still be correct but SERIAL, silently "
                "forfeiting parallelism on what may be a large index. Pass num_slices explicitly, "
                "or set EsReadConfig.strict_slices=False to accept the serial fallback."
            ) from exc
        _log.warning(
            "read_index: could not read shard count for index %r (%s); falling back to a single "
            "unsliced slice. Pass num_slices explicitly to parallelize.", index, exc)
        return 1


# --- shared validation -----------------------------------------------------------------------

def _validate(cfg: EsReadConfig, schema) -> None:
    # Duck-type the schema (a non-empty `.fields`) rather than isinstance(StructType), this matches
    # how _schema_field_tokens already reads the schema, and it keeps this module importable and
    # unit-testable without pyspark. A wrong type has no truthy `.fields` and hits the error; a real
    # StructType passes.
    if not getattr(schema, "fields", None):
        raise ValueError("read_index requires a non-empty Spark StructType schema (v0.4.0 has no "
                         "mapping inference, declare the schema explicitly)")
    if not cfg.index:
        raise ValueError("EsReadConfig.index is required to read")


# --- distributed reader -----------------------------------------------------------------------

def _make_slice_reader(cfg: EsReadConfig, pit_id: str, query: dict, field_tokens, num_slices: int):
    """Return a mapInPandas function: for each slice id it receives (via spark.range), read that
    PIT slice and yield a pandas DataFrame of coerced rows. Captures only serializable values
    (frozen cfg, the PIT id string, the query dict, the type tokens), no Spark/ES objects."""
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
    `spark.range(num_slices).mapInPandas(...)`, one task per PIT slice, each task building its own
    ES client and paging its slice with `search_after`, coercing via the documented inverse
    transforms. Serverless-safe (same `mapInPandas` mechanism as the write path); the data stays
    distributed across executors (never collected to the driver), so it scales for full-index export.
    `num_slices` defaults to the index's shard count.

    PIT lifecycle: the returned DataFrame is LAZY, Spark reads each slice when an action runs, not
    when this function returns. So we do NOT close the PIT here (a `finally` close would kill it
    before evaluation); it is left to expire via `pit_keep_alive`. Crucially, `pit_keep_alive` is a
    SLIDING window, not a total budget: `_slice_hits` re-sends it on every page and follows the
    refreshed `pit_id`, so each read resets the PIT's expiry (per the ES PIT API, the value "just
    needs to be long enough for the next request"). It therefore only has to cover the longest gap
    between consecutive touches of the PIT, NOT the whole job:
      - the open-PIT -> first-page gap (the read is lazy, so Spark may not schedule the tasks
        immediately, plus serverless executor cold-start), and
      - the largest gap between pages if a slow downstream consumer applies backpressure.
    The default (5m) covers a normal scheduling gap; raise it only if one of those gaps is longer.
    Keeping the read distributed is why the PIT can't be driver-closed; it expires on its own once
    reads stop touching it. For a small/bounded read, pass `num_slices=1` (a single unsliced reader).
    """
    _validate(cfg, schema)
    query = cfg.query or {"match_all": {}}
    field_tokens = _schema_field_tokens(schema)

    from elasticsearch import Elasticsearch
    es = Elasticsearch(**cfg.client_kwargs())
    try:
        num_slices = _resolve_num_slices(es, cfg.index, cfg.num_slices,
                                         strict=cfg.strict_slices)
        pit_id = es.open_point_in_time(index=cfg.index, keep_alive=cfg.pit_keep_alive)["id"]
    finally:
        # The driver client's only jobs are resolving the slice count and opening the PIT; each
        # executor builds its own client. Close this one now (best-effort) so it doesn't leak, but
        # do NOT close the PIT: the returned DataFrame is lazy and the executors read it later.
        try:
            es.close()
        except Exception:
            pass

    reader = _make_slice_reader(cfg, pit_id, query, field_tokens, num_slices)
    # One row per slice, one partition per slice, so each task handles exactly one slice. `range`
    # splits its contiguous id space [0, num_slices) into `numPartitions` equal chunks, with
    # num_slices rows in num_slices partitions that is deterministically one row each, and needs no
    # shuffle. (repartition() would round-robin, which only *tends* toward an even spread and could
    # co-locate two slices on one task under skew.) The result is a lazy distributed DataFrame,
    # data never crosses the driver.
    slices_df = spark.range(num_slices, numPartitions=num_slices)
    return slices_df.mapInPandas(reader, schema)
